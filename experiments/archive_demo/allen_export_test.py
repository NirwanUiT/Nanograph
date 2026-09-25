#!/usr/bin/env python3
"""
T18 bake-off inputs: the held-out-plate test cells of allen_trainset.py as
2-D files, so every segmenter (in its own environment) predicts from
identical inputs and writes <pred_dir>/<CellId>.png (0/255, cell-crop size).

<out>/img/<CellId>.png         slab image, outside-cell set to background
                                (allen_trainset.masked_input; the training input)
<out>/img_raw/<CellId>.png     slab image, unmasked
<out>/allen/<CellId>.png       Allen's struct segmentation over the slab (reference)
<out>/cell/<CellId>.png        projected cell mask
<out>/cells.csv                the test cells (stage, plate, voxel size)
"""
import argparse
import ast
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from allen_check import fetch, slab_projection  # noqa: E402
from allen_trainset import masked_input  # noqa: E402

ROOT = '/mnt/nas1/nba055-2/idea_1/archive_demo/allen'


def work(args):
    row, out = args
    import cv2
    import tifffile
    cid = row['CellId']
    rp, sp = f'{ROOT}/cells/{cid}_raw.ome.tif', f'{ROOT}/cells/{cid}_seg.ome.tif'
    fetch(row['crop_raw_url'], rp)
    fetch(row['crop_seg_url'], sp)
    im, cell, aref, _ = slab_projection(tifffile.imread(rp), tifffile.imread(sp),
                                        ast.literal_eval(row['name_dict']), ast.literal_eval(row['scale_micron'])[-1])
    cv2.imwrite(f'{out}/img/{cid}.png', masked_input(im, cell))
    cv2.imwrite(f'{out}/img_raw/{cid}.png', im)
    cv2.imwrite(f'{out}/allen/{cid}.png', aref.astype(np.uint8) * 255)
    cv2.imwrite(f'{out}/cell/{cid}.png', cell.astype(np.uint8) * 255)
    return cid


def main():
    import multiprocessing as mp
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=f'{ROOT}/testset')
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--cells', default=None, help='comma list of CellIds from the TOMM20 manifest (instead of the test split)')
    a = ap.parse_args()
    if a.cells:
        C = pd.read_csv(f'{ROOT}/manifest_tomm20.csv')
        C = C[C.CellId.isin([int(c) for c in a.cells.split(',')])]
    else:
        C = pd.read_csv(f'{ROOT}/trainset/cells.csv')
        C = C[C.split == 'test']
    for d in ('img', 'img_raw', 'allen', 'cell'):
        os.makedirs(os.path.join(a.out, d), exist_ok=True)
    C.to_csv(os.path.join(a.out, 'cells.csv'), index=False)
    with mp.get_context('spawn').Pool(a.workers) as pool:
        done = pool.map(work, [(r, a.out) for r in C.to_dict('records')])
    print(f'exported {len(done)} test cells to {a.out}')


if __name__ == '__main__':
    main()
