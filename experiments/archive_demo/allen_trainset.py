#!/usr/bin/env python3
"""
T18 — training data for a U-Net distilled from Allen's TOMM20 segmentation.

Input image: the +-1 um slab projection of allen_check.py, with everything
outside the cell's own mask set to the in-cell 2nd-percentile intensity
(Allen's struct_segmentation of a crop covers that cell only, so neighbours'
mitochondria would otherwise be unlabelled foreground). The same masking is
applied at inference (`masked_input`).
Label: Allen's 3-D struct_segmentation max-projected over the same slab.

Split by plate (seeded): 36 train / 8 val / 8 test plates of the 52. Per
split, all mitotic cells up to --mitotic-cap per stage plus interphase cells
to reach --n-<split>. Each cell is cut into 256x256 tiles (corner-anchored,
padded with the background value when smaller) that contain labelled
foreground.

Writes <out>/{images,masks}/<CellId>_<k>.png and <out>/manifest.csv in the
format of real_mito/manifest.csv (dataset=ALLEN, group=plate, split, id, fg),
plus <out>/cells.csv (the cells per split) so evaluation can run per cell.

Labels are the output of Allen's segmentation workflow, not human
annotation: a network trained on them can at best reproduce that workflow.
"""
import argparse
import ast
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from allen_check import fetch, slab_projection  # noqa: E402

TILE = 256


def masked_input(im, cell):
    """Outside-cell pixels -> in-cell 2nd percentile (train and inference)."""
    bg = np.percentile(im[cell], 2) if cell.any() else 0
    out = im.copy()
    out[~cell] = np.uint8(bg)
    return out


def plate_split(plates, seed=0):
    p = list(np.random.default_rng(seed).permutation(sorted(plates)))
    return {q: ('val' if i < 8 else 'test' if i < 16 else 'train') for i, q in enumerate(p)}


def tiles(im, lab, fill):
    H, W = im.shape
    ph, pw = max(TILE - H, 0), max(TILE - W, 0)
    if ph or pw:
        im = np.pad(im, ((0, ph), (0, pw)), constant_values=fill)
        lab = np.pad(lab, ((0, ph), (0, pw)))
        H, W = im.shape
    for y in sorted({0, H - TILE}):
        for x in sorted({0, W - TILE}):
            yield y, x, im[y:y + TILE, x:x + TILE], lab[y:y + TILE, x:x + TILE]


def work(args):
    row, cache, out = args
    import cv2
    import tifffile
    cid = row['CellId']
    try:
        rp, sp = os.path.join(cache, f'{cid}_raw.ome.tif'), os.path.join(cache, f'{cid}_seg.ome.tif')
        fetch(row['crop_raw_url'], rp)
        fetch(row['crop_seg_url'], sp)
        im, cell, lab, _ = slab_projection(tifffile.imread(rp), tifffile.imread(sp),
                                           ast.literal_eval(row['name_dict']),
                                           ast.literal_eval(row['scale_micron'])[-1])
        x = masked_input(im, cell)
        fill = int(x[~cell].max()) if (~cell).any() else 0
        rows = []
        for k, (y0, x0, ti, tl) in enumerate(tiles(x, lab.astype(np.uint8), fill)):
            if tl.sum() == 0:
                continue
            tid = f'{cid}_{k}'
            cv2.imwrite(os.path.join(out, 'images', tid + '.png'), ti)
            cv2.imwrite(os.path.join(out, 'masks', tid + '.png'), tl * 255)
            rows.append({'dataset': 'ALLEN', 'group': f"ALLEN_{row['PlateId']}", 'split': row['split'],
                         'id': tid, 'fg': float(tl.mean())})
        return rows, ''
    except Exception as ex:
        return [], f'{cid}: {type(ex).__name__}: {ex}'


def main():
    import multiprocessing as mp
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/mnt/nas1/nba055-2/idea_1/archive_demo/allen')
    ap.add_argument('--out', default='/mnt/nas1/nba055-2/idea_1/archive_demo/allen/trainset')
    ap.add_argument('--n-train', type=int, default=1500)
    ap.add_argument('--n-val', type=int, default=250)
    ap.add_argument('--n-test', type=int, default=300)
    ap.add_argument('--mitotic-cap', type=int, default=60, help='per stage and split')
    ap.add_argument('--workers', type=int, default=6)
    a = ap.parse_args()
    M = pd.read_csv(os.path.join(a.root, 'manifest_tomm20.csv'))
    M = M[M.outlier.astype(str) != 'Yes']
    sp = plate_split(M.PlateId.unique())
    M['split'] = M.PlateId.map(sp)
    picks = []
    for s, n in (('train', a.n_train), ('val', a.n_val), ('test', a.n_test)):
        g = M[M.split == s]
        mit = pd.concat(h.sample(min(a.mitotic_cap, len(h)), random_state=0)
                        for st, h in g[g.cell_stage != 'M0'].groupby('cell_stage'))
        inter = g[g.cell_stage == 'M0'].sample(max(n - len(mit), 0), random_state=0)
        picks.append(pd.concat([mit, inter]))
    C = pd.concat(picks)
    for d in ('images', 'masks'):
        os.makedirs(os.path.join(a.out, d), exist_ok=True)
    C.to_csv(os.path.join(a.out, 'cells.csv'), index=False)
    print(C.groupby(['split', 'cell_stage']).size().unstack(0).to_string(), flush=True)
    cache = os.path.join(a.root, 'cells')
    with mp.get_context('spawn').Pool(a.workers) as pool:
        res = pool.map(work, [(r, cache, a.out) for r in C.to_dict('records')], chunksize=4)
    rows = [r for rs, _ in res for r in rs]
    errs = [e for _, e in res if e]
    pd.DataFrame(rows).to_csv(os.path.join(a.out, 'manifest.csv'), index=False)
    open(os.path.join(a.out, 'errors.txt'), 'w').write('\n'.join(errs) + '\n')
    D = pd.DataFrame(rows)
    print(D.groupby('split').agg(tiles=('id', 'size'), plates=('group', 'nunique'), fg=('fg', 'mean')).to_string())
    print(f'errors: {len(errs)}')


if __name__ == '__main__':
    main()
