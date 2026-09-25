#!/usr/bin/env python3
"""
Inter-annotator agreement: the ceiling for every "arm vs REF" number.

Protocol (to be run by a person; nothing here can substitute for it):
  1. Draw a fixed random subset of the organelle images, e.g.
       python experiments/annotator_agreement.py --sample 50 --seed 0
     which writes results/annotator/subset.txt. Use held-out images only if
     the second annotation will also be used to test segmenters.
  2. A second annotator, blind to the first annotation and to any pipeline
     output, draws foreground masks for those images with the same tool and
     instructions as the original annotation. Save them as <stem>.png
     (binary, same size) in one directory.
  3. Run
       python experiments/annotator_agreement.py --second /path/to/masks2
     It treats annotator 1 (nmi_data/seg) as REF and annotator 2 as an arm and
     reports exactly the statistics used for the pipeline arms (Seg-IoU;
     descriptor bias, CCC, MdAPE at every pruning length L; junction and
     endpoint F1). An arm whose agreement with REF is within these numbers is
     as good as a second human.

  --selftest compares the annotation with a copy perturbed by 1 px boundary
  noise, only to check that the script runs; it is not a result.

Real mitochondria (T15-B; the organelle set is simulated, so its ceiling is
the simulator truth, not a second annotator):
  1. python experiments/annotator_agreement.py --sample-real 50 --seed 0
     draws whole test FRAMES (never splitting a frame) from EP1-UiT-Rat and
     EP-UiT-Human, alternating datasets in a seeded order, until the frames
     cover >= 50 of the manifest's test tiles; writes
     results/annotator/subset_real.txt (frame lines "DATASET FRAME", then the
     tile ids) and, with --package DIR, copies the raw frames (as acquired, no
     masks, no contrast change) plus PROTOCOL.md into DIR.
  2. The colleague annotates the whole frames, blind to the existing masks,
     and returns <FRAME>.png binary masks at the frame's size.
  3. python experiments/annotator_agreement.py --second-real DIR \
         --out results/paper/annotator
     tiles annotator 2's masks on the manifest's tile grid and writes
     real_mito.csv (per tile) and real_mito_summary.csv (per dataset, per L),
     annotator 1 = the existing expert masks.
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

IMAGES = '/mnt/nas1/nba055-2/idea_1/nmi_data/org'
MASKS = '/mnt/nas1/nba055-2/idea_1/nmi_data/seg'
LS = [0, 2, 5, 10, 'auto']
REAL = '/mnt/nas1/nba055-2/idea_1/real_mito'
SRC = '/mnt/nas1/nba055-2/data/Mito Segmentation Dataset'
REAL_SRC = {'UIT': ('EP1-UiT-Rat', 'image'), 'HUMAN': ('EP-UiT-Human', 'image')}


def compare(stems, second_of, out, first_of=None, prefix=''):
    import cv2
    import pandas as pd
    import downstream_morphometry as dm
    first_of = first_of or (lambda s: cv2.imread(os.path.join(MASKS, s + '.png'), cv2.IMREAD_GRAYSCALE))
    rows = []
    for s in stems:
        a = first_of(s)
        b = second_of(s, a)
        if b is None:
            continue
        inter = np.logical_and(a > 0, b > 0).sum()
        union = np.logical_or(a > 0, b > 0).sum()
        ta, tb = dm.pixel_arm_table(a), dm.pixel_arm_table(b)
        for L in LS:
            da, db = dm.descriptors_at(ta, L)[0], dm.descriptors_at(tb, L)[0]
            r = {'stem': s, 'L': L, 'iou': inter / union if union else 1.0}
            for k in dm.DESCRIPTORS:
                r[f'{k}_1'], r[f'{k}_2'] = da[k], db[k]
            ja, ea = dm.vertex_positions(ta, L)
            jb, eb = dm.vertex_positions(tb, L)
            for tol in (3, 5):
                r[f'junc_f1_{tol}'] = dm.point_f1(jb, ja, tol)[2]
                r[f'end_f1_{tol}'] = dm.point_f1(eb, ea, tol)[2]
            rows.append(r)
    P = pd.DataFrame(rows)
    if prefix:
        P.insert(0, 'dataset', [x.split('_')[0] for x in P.stem])
    P.to_csv(os.path.join(out, (prefix or 'per_image') + '.csv'), index=False)
    S = []
    keys = ['dataset', 'L'] if prefix else ['L']
    for key, g in P.groupby(keys, sort=False):
        r = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        r.update({'n': len(g), 'iou_mean': g.iou.mean()})
        for k in dm.DESCRIPTORS:
            x, y = g[f'{k}_2'].to_numpy(float), g[f'{k}_1'].to_numpy(float)
            ok = np.isfinite(x) & np.isfinite(y)
            ag = dm.agreement(x[ok], y[ok])
            r[f'{k}_bias_pct'], r[f'{k}_ccc'], r[f'{k}_mdape'] = ag['bias_pct'], ag['ccc'], ag['mdape']
        for tol in (3, 5):
            r[f'junc_f1_{tol}'] = g[f'junc_f1_{tol}'].mean()
            r[f'end_f1_{tol}'] = g[f'end_f1_{tol}'].mean()
        S.append(r)
    S = pd.DataFrame(S)
    S.to_csv(os.path.join(out, (prefix + '_summary' if prefix else 'summary') + '.csv'), index=False)
    import subprocess
    h = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True,
                       cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
    open(os.path.join(out, 'commit.txt'), 'w').write(h + '\n')
    print(S.round(3).T.to_string())


def _manifest_test():
    import pandas as pd
    M = pd.read_csv(os.path.join(REAL, 'manifest.csv'))
    return M[(M.split == 'test') & M.dataset.isin(list(REAL_SRC))]


def sample_real(n, seed, out, package):
    import shutil
    M = _manifest_test()
    rng = np.random.default_rng(seed)
    order = {d: list(rng.permutation(sorted(g.group.unique()))) for d, g in M.groupby('dataset')}
    frames, tiles, turn = [], [], sorted(order)
    while len(tiles) < n and any(order.values()):
        for d in turn:
            if order[d] and len(tiles) < n:
                f = order[d].pop(0)
                frames.append((d, f))
                tiles += sorted(M[M.group == f].id)
    with open(os.path.join(out, 'subset_real.txt'), 'w') as fh:
        fh.write(''.join(f'# {d} {f}\n' for d, f in frames) + '\n'.join(tiles) + '\n')
    print(f'{len(frames)} frames, {len(tiles)} tiles -> {out}/subset_real.txt')
    for d, f in frames:
        print(f'  {d} {f}: {sum(1 for t in tiles if t.startswith(f + "_"))} tiles')
    if not package:
        return
    os.makedirs(package, exist_ok=True)
    for d, f in frames:
        ds, sub = REAL_SRC[d]
        raw = f.split('_', 1)[1]
        src = glob.glob(os.path.join(SRC, ds, sub, raw + '.*'))[0]
        shutil.copy2(src, os.path.join(package, f + os.path.splitext(src)[1]))
    open(os.path.join(package, 'PROTOCOL.md'), 'w').write(PROTOCOL.format(
        frames='\n'.join(f'- `{f}` ({d})' for d, f in frames)))
    print(f'package: {package}')


PROTOCOL = """# Second annotation of real mitochondria: protocol

You are annotating mitochondria so that we can measure how much two experts
disagree. Please work **without looking at any existing masks or pipeline
output** for these images.

## Images

{frames}

The files are the raw frames as acquired (no contrast change). Annotate each
whole frame.

## Match the original annotation

Use the same tool, zoom level, brush and pixel size as the original
annotation of these datasets:

- Tool / software: **TODO (copy from the original annotation instructions)**
- Zoom and brush size: **TODO**
- Display contrast: **TODO**
- What counts as a mitochondrion, and how to handle touching, faint or
  out-of-focus structures: **TODO**

## Output

For each frame, one binary PNG mask with the same file name stem and the same
size as the frame (`<frame>.png`; mitochondria = 255, background = 0). Return
the folder of masks.
"""


def compare_real(second, out):
    import cv2
    import re
    sub_p = os.path.join('results', 'annotator', 'subset_real.txt')
    tiles = [t for t in open(sub_p).read().split('\n') if t and not t.startswith('#')]
    cache = {}

    def frame_mask(tile):
        f = re.sub(r'_y\d+x\d+$', '', tile)
        if f not in cache:
            p = glob.glob(os.path.join(second, f + '.*'))
            cache[f] = cv2.imread(p[0], cv2.IMREAD_GRAYSCALE) if p else None
        return cache[f]

    def first_of(t):
        return cv2.imread(os.path.join(REAL, t.split('_')[0], 'masks', t + '.png'), cv2.IMREAD_GRAYSCALE)

    def second_of(t, m):
        fm = frame_mask(t)
        if fm is None:
            return None
        y, x = map(int, re.search(r'_y(\d+)x(\d+)$', t).groups())
        return ((fm[y:y + 256, x:x + 256] > 0) * 255).astype(np.uint8)
    compare(tiles, second_of, out, first_of=first_of, prefix='real_mito')


def main():
    import cv2
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sample', type=int, default=0, help='write a random subset of N stems')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--second', default=None, help='directory of annotator-2 masks')
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--sample-real', type=int, default=0, help='draw whole real test frames covering >= N tiles')
    ap.add_argument('--package', default=None, help='with --sample-real: export raw frames + protocol here')
    ap.add_argument('--second-real', default=None, help='directory of annotator-2 frame masks')
    ap.add_argument('--out', default='results/annotator')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.sample_real:
        sample_real(a.sample_real, a.seed, a.out, a.package)
        return
    if a.second_real:
        compare_real(a.second_real, a.out)
        return
    stems = sorted(os.path.splitext(os.path.basename(p))[0]
                   for p in glob.glob(os.path.join(MASKS, '*.png')))
    if a.sample:
        sub = sorted(np.random.default_rng(a.seed).choice(stems, a.sample, replace=False))
        open(os.path.join(a.out, 'subset.txt'), 'w').write('\n'.join(sub) + '\n')
        print(f'wrote {len(sub)} stems to {a.out}/subset.txt')
        return
    sub_p = os.path.join(a.out, 'subset.txt')
    sub = open(sub_p).read().split() if os.path.exists(sub_p) else stems
    if a.selftest:
        rng = np.random.default_rng(0)

        def second_of(s, m):
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            band = cv2.dilate((m > 0).astype(np.uint8), k) - cv2.erode((m > 0).astype(np.uint8), k)
            out = (m > 0).astype(np.uint8)
            flip = (band > 0) & (rng.random(m.shape) < 0.3)
            out[flip] = 1 - out[flip]
            return out * 255
        compare(sub[:30], second_of, a.out)
        return
    if not a.second:
        ap.error('give --second DIR (annotator-2 masks), or --sample N, or --selftest')

    def second_of(s, m):
        p = os.path.join(a.second, s + '.png')
        return cv2.imread(p, cv2.IMREAD_GRAYSCALE) if os.path.exists(p) else None
    compare(sub, second_of, a.out)


if __name__ == '__main__':
    main()
