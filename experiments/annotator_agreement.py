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
LS = [0, 2, 5, 10]


def compare(stems, second_of, out):
    import cv2
    import pandas as pd
    import downstream_morphometry as dm
    rows = []
    for s in stems:
        a = cv2.imread(os.path.join(MASKS, s + '.png'), cv2.IMREAD_GRAYSCALE)
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
    P.to_csv(os.path.join(out, 'per_image.csv'), index=False)
    S = []
    for L, g in P.groupby('L'):
        r = {'L': L, 'n': len(g), 'iou_mean': g.iou.mean()}
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
    S.to_csv(os.path.join(out, 'summary.csv'), index=False)
    print(S.round(3).T.to_string())


def main():
    import cv2
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sample', type=int, default=0, help='write a random subset of N stems')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--second', default=None, help='directory of annotator-2 masks')
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--out', default='results/annotator')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
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
