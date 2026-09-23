#!/usr/bin/env python3
"""
T7 — reproducible preparation of the Zenodo 7724799 "MITO" mitochondria set into
256x256 maximum-intensity-projection (MIP) tiles used for the failure-case
diagnostics in the paper.

Raw layout (public_mito/MITO/):
    M{1,2}_U2OS/M{1,2}_NNN/{fluo.tif, label.tif, ratio.tif}
Each fluo.tif / label.tif is a 24-page z-stack at 1200x1200 (uint16 fluo,
uint8 {0,255} label). 41 fields total across the two microscopes.

Per-field prep:
  1. image  = max-intensity projection over the 24 z-pages of fluo.tif, then a
     (0.5, 99.8) percentile contrast stretch to uint8.
  2. mask   = the per-slice binary annotations are collapsed to a single 2-D
     mask by MAX-PROJECTION UNION across the 24 label pages (a pixel is
     foreground if it is annotated in ANY z-slice), matching the MIP image.
  3. tile the 1200x1200 projection on a stride-256 grid (no partial edge tiles)
     and keep only tiles whose foreground fraction is in [0.01, 0.20] — this
     drops empty background tiles and near-saturated tiles, leaving a sparse
     curvilinear-foreground regime. Yields 228 tiles.

Outputs: public_mito/mito_mip_tiles/{images,masks}/<field>_y<Y>x<X>.png

This script reproduces the exact tile set used for results/paper/mito_diag/*.
"""
import os
import glob

import cv2
import numpy as np
import tifffile

RAW = os.environ.get('MITO_RAW', '/mnt/nas1/nba055-2/idea_1/public_mito/MITO')
OUT = os.environ.get('MITO_OUT', '/mnt/nas1/nba055-2/idea_1/public_mito/mito_mip_tiles')

TILE = 256
LO_FRAC, HI_FRAC = 0.01, 0.20
PLO, PHI = 0.5, 99.8


def _mip(path):
    """Max-intensity projection over the z-stack pages of a multipage TIFF."""
    stk = tifffile.imread(path)
    if stk.ndim == 2:
        return stk
    return stk.max(axis=0)


def main():
    img_dir = os.path.join(OUT, 'images')
    msk_dir = os.path.join(OUT, 'masks')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(msk_dir, exist_ok=True)

    n = 0
    for fluo in sorted(glob.glob(os.path.join(RAW, 'M*_U2OS', 'M*_*', 'fluo.tif'))):
        field = os.path.basename(os.path.dirname(fluo))
        try:
            mip = _mip(fluo).astype(np.float64)
            gt = (_mip(fluo.replace('fluo', 'label')) > 0).astype(np.uint8) * 255
        except Exception as e:                       # unreadable field -> skip
            print('skip', field, e)
            continue

        lo, hi = np.percentile(mip, (PLO, PHI))
        a8 = np.clip((mip - lo) / (hi - lo + 1e-9) * 255, 0, 255).astype(np.uint8)

        H, W = a8.shape
        for y in range(0, H - (TILE - 1), TILE):
            for x in range(0, W - (TILE - 1), TILE):
                g = gt[y:y + TILE, x:x + TILE]
                if LO_FRAC <= (g > 0).mean() <= HI_FRAC:
                    cv2.imwrite(os.path.join(img_dir, f'{field}_y{y}x{x}.png'),
                                a8[y:y + TILE, x:x + TILE])
                    cv2.imwrite(os.path.join(msk_dir, f'{field}_y{y}x{x}.png'), g)
                    n += 1
    print('MIP tiles:', n)


if __name__ == '__main__':
    main()
