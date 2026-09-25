#!/usr/bin/env python3
"""
T15-A — inter-observer ceiling on public retinal-vessel data.

STARE ships two manual segmentations of its 20 images (Hoover: `ah`, Kouznetsova:
`vk`); DRIVE ships a second observer for its 20 TEST images (`2nd_manual`), which
needs the registered DRIVE download (--drive-test; skipped if absent).

Per image, with the descriptor code used for every arm in this paper
(downstream_morphometry: degree-based junctions, one-diameter rule):
  OBS2   observer 2's mask            vs observer 1
  PIPE   the v7 structure layer of the pipeline (for_curvilinear preset, the
         retrained polarity-agnostic U-Net), decoded from its bytes, vs observer 1
Junction / endpoint F1 at 3 and 5 px against observer 1.
`split` marks whether the image was in the U-Net's training or validation split
(train_unet_multidomain.build_splits(seed=0)); PIPE rows on training images are
optimistic, so the summary is given for all images and for validation images.

Writes <out>/vessels.csv (per image x comparison) and <out>/vessels_summary.csv.

Usage (from repo root):
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 python experiments/interobserver_vessels.py \
        --out results/paper/annotator [--drive-test /path/to/DRIVE/test]
"""
import argparse
import contextlib
import glob
import io
import multiprocessing as mp
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

EXT = '/mnt/nas1/nba055-2/idea_1/ext_datasets'
PREP = f'{EXT}/prepared'
TOLS = (3, 5)


def read_mask(p):
    if p.lower().endswith('.gif'):
        from PIL import Image
        return (np.asarray(Image.open(p).convert('L')) > 127).astype(np.uint8)
    return (cv2.imread(p, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)


def stare_items():
    out = []
    for p in sorted(glob.glob(f'{PREP}/stare/images/*.png')):
        i = os.path.splitext(os.path.basename(p))[0]
        o1, o2 = f'{EXT}/stare/labels/{i}.ah.ppm', f'{EXT}/stare/labels-vk/{i}.vk.ppm'
        if os.path.exists(o1) and os.path.exists(o2):
            out.append(('STARE', i, p, o1, o2))
    return out


def drive_items(test_dir):
    out = []
    if not test_dir or not os.path.isdir(test_dir):
        return out
    for p in sorted(glob.glob(os.path.join(test_dir, 'images', '*_test.tif'))):
        n = os.path.basename(p).split('_')[0]
        o1 = os.path.join(test_dir, '1st_manual', f'{n}_manual1.gif')
        o2 = os.path.join(test_dir, '2nd_manual', f'{n}_manual2.gif')
        if os.path.exists(o1) and os.path.exists(o2):
            out.append(('DRIVE', n, p, o1, o2))
    return out


def load_image(dataset, p):
    if dataset == 'DRIVE':                       # as prep_ext_datasets: green channel
        return cv2.imread(p, cv2.IMREAD_COLOR)[:, :, 1]
    return cv2.imread(p, cv2.IMREAD_GRAYSCALE)


def pipeline_table(img):
    import downstream_morphometry as dm
    from nanograph_v4 import NanographConfig, graph_branch as gb
    from nanograph_v4.api import nanograph_encode
    cfg = NanographConfig().for_curvilinear()
    with contextlib.redirect_stdout(io.StringIO()):
        r = nanograph_encode(img, config=cfg, verbose=False)
    sb = gb.encode_structure(r.structure)
    tab = dm.graph_arm_table(*gb.structure_to_arrays(gb.decode_structure(sb), True))
    return tab, len(sb), r.mask


def work(item):
    import torch
    import downstream_morphometry as dm
    torch.set_num_threads(1)
    dataset, iid, ip, o1p, o2p = item
    o1, o2 = read_mask(o1p), read_mask(o2p)
    img = load_image(dataset, ip)
    t1, t2 = dm.pixel_arm_table(o1), dm.pixel_arm_table(o2)
    tp, sbytes, pmask = pipeline_table(img)
    ref = dm.descriptors_at(t1, 'auto', 'degree')[0]
    j1, e1 = dm.vertex_positions(t1, 'auto')
    rows = []
    for arm, tab, extra in (('OBS2', t2, {'iou': iou(o2, o1)}),
                            ('PIPE', tp, {'iou': iou(pmask, o1), 'structure_bytes': sbytes})):
        d = dm.descriptors_at(tab, 'auto', 'degree')[0]
        ja, ea = dm.vertex_positions(tab, 'auto')
        r = {'dataset': dataset, 'id': iid, 'arm': arm, **extra}
        for k in dm.DESCRIPTORS:
            r[k], r[k + '_ref'] = d[k], ref[k]
        for tol in TOLS:
            r[f'junc_f1_{tol}'] = dm.point_f1(ja, j1, tol)[2]
            r[f'end_f1_{tol}'] = dm.point_f1(ea, e1, tol)[2]
        rows.append(r)
    return rows


def iou(a, b):
    a, b = a > 0, b > 0
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else np.nan


def splits():
    from train_unet_multidomain import build_splits
    s = build_splits(seed=0)
    out = {}
    for ds, key in (('stare', 'STARE'), ('drive', 'DRIVE')):
        for part in ('train', 'val'):
            for ip, _ in s.get(ds, {}).get(part, []):
                out[(key, os.path.splitext(os.path.basename(ip))[0])] = part
    return out


def summarise(df):
    import pandas as pd
    import downstream_morphometry as dm
    rows = []
    for (ds, arm), g0 in df.groupby(['dataset', 'arm']):
        for subset, g in (('all', g0), ('val', g0[g0.split == 'val'])):
            if not len(g):
                continue
            r = {'dataset': ds, 'arm': arm, 'subset': subset, 'n': len(g), 'iou': g.iou.mean()}
            for k in dm.DESCRIPTORS:
                ag = dm.agreement(g[k].to_numpy(float), g[k + '_ref'].to_numpy(float))
                r[f'{k}_ccc'], r[f'{k}_mdape'], r[f'{k}_bias_pct'] = ag['ccc'], ag['mdape'], ag['bias_pct']
            for c in [c for c in g.columns if c.startswith(('junc_f1', 'end_f1'))]:
                r[c] = g[c].mean()
            rows.append(r)
    return pd.DataFrame(rows)


def stamp_commit(out):
    import subprocess
    h = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True,
                       cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
    open(os.path.join(out, 'commit.txt'), 'w').write(h + '\n')


def main():
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results/paper/annotator')
    ap.add_argument('--drive-test', default=os.environ.get('DRIVE_TEST'))
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    items = stare_items() + drive_items(a.drive_test)
    if a.limit:
        items = items[:a.limit]
    print(f'{len(items)} images:', {d: sum(1 for x in items if x[0] == d) for d in ('STARE', 'DRIVE')}, flush=True)
    with mp.get_context('spawn').Pool(a.workers) as pool:
        rows = [r for rs in pool.imap_unordered(work, items) for r in rs]
    df = pd.DataFrame(rows)
    sp = splits()
    df['split'] = [sp.get((d, i), 'unused') for d, i in zip(df.dataset, df.id)]
    os.makedirs(a.out, exist_ok=True)
    df.sort_values(['dataset', 'id', 'arm']).to_csv(os.path.join(a.out, 'vessels.csv'), index=False)
    stamp_commit(a.out)
    S = summarise(df)
    S.to_csv(os.path.join(a.out, 'vessels_summary.csv'), index=False)
    cols = ['dataset', 'arm', 'subset', 'n', 'iou'] + [f'{k}_ccc' for k in ('n_branches', 'n_junctions',
                                                                            'total_length_px', 'mean_width_px')] \
        + ['junc_f1_3', 'end_f1_3']
    pd.set_option('display.width', 220)
    print(S[cols].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
