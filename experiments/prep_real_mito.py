#!/usr/bin/env python3
"""
Real annotated mitochondria data -> 256x256 uint8 tiles + manifest (T13).

Sources (all with expert/manual foreground masks):
  UIT    EP1-UiT-Rat: 32 epifluorescence frames, 1024x1024
  HUMAN  EP-UiT-Human: 4 epifluorescence frames, 1024x1024 (greyscale maps; >=128 = fg)
  CBMI   CBMI MITO ("Fluorescence Microscopy Images v2"): 256x256 uint16 tiles,
         provided train/val/test splits (disjoint source frames)
  MITO   Zenodo 7724799: 2 cells (M1, M2) x ~20 z-slices, 1200x1200

Each frame is contrast-stretched to uint8 with its (0.5, 99.8) percentiles
(the pipeline's input format), tiled on a 256 stride (no partial tiles) and
tiles with < 0.5 % foreground are dropped. Splits are by frame or cell, never
by tile:
  UIT    frames shuffled (seed 0): 24 train / 4 val / 4 test
  CBMI   as provided
  MITO   cell M1 -> train (its last 4 slices -> val), cell M2 -> test
  HUMAN  test only
Writes <out>/<dataset>/{images,masks}/<id>.png and <out>/manifest.csv
(dataset, group, split, id, fg).
"""
import argparse
import glob
import os
import re

import cv2
import numpy as np
import pandas as pd

ROOT = '/mnt/nas1/nba055-2/data/Mito Segmentation Dataset'
CBMI = os.path.join(ROOT, 'Online IEEE', 'Fluorescence Microscopy Images v2', 'cbmi_MITO')
MITO = '/mnt/nas1/nba055-2/idea_1/public_mito/mito_zenodo'
MIN_FG = 0.005


def stretch(img):
    img = img.astype(np.float64)
    lo, hi = np.percentile(img, 0.5), np.percentile(img, 99.8)
    return np.clip((img - lo) / max(hi - lo, 1e-9) * 255, 0, 255).astype(np.uint8)


def tiles(img, msk, size=256):
    H, W = img.shape
    for y in range(0, H - size + 1, size):
        for x in range(0, W - size + 1, size):
            yield y, x, img[y:y + size, x:x + size], msk[y:y + size, x:x + size]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/mnt/nas1/nba055-2/idea_1/real_mito')
    a = ap.parse_args()
    rows = []

    def emit(ds, group, split, tid, img, msk):
        fg = float((msk > 0).mean())
        if fg < MIN_FG:
            return
        d = os.path.join(a.out, ds)
        os.makedirs(os.path.join(d, 'images'), exist_ok=True)
        os.makedirs(os.path.join(d, 'masks'), exist_ok=True)
        cv2.imwrite(os.path.join(d, 'images', tid + '.png'), img)
        cv2.imwrite(os.path.join(d, 'masks', tid + '.png'), (msk > 0).astype(np.uint8) * 255)
        rows.append({'dataset': ds, 'group': group, 'split': split, 'id': tid, 'fg': fg})

    # UIT (rat): split by frame
    frames = sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(ROOT, 'EP1-UiT-Rat', 'image')))
    rng = np.random.default_rng(0)
    order = list(rng.permutation(frames))
    split_of = {f: ('test' if i < 4 else 'val' if i < 8 else 'train') for i, f in enumerate(order)}
    for f in frames:
        img = cv2.imread(glob.glob(os.path.join(ROOT, 'EP1-UiT-Rat', 'image', f + '.*'))[0], -1)
        msk = cv2.imread(glob.glob(os.path.join(ROOT, 'EP1-UiT-Rat', 'segment', f + '.*'))[0], -1)
        img = img if img.ndim == 2 else img[..., 0]
        msk = msk if msk.ndim == 2 else msk[..., 0]
        for y, x, ti, tm in tiles(stretch(img), msk):
            emit('UIT', f'UIT_{f}', split_of[f], f'UIT_{f}_y{y}x{x}', ti, tm)

    # HUMAN: test only
    for p in sorted(glob.glob(os.path.join(ROOT, 'EP-UiT-Human', 'image', '*'))):
        f = os.path.splitext(os.path.basename(p))[0]
        img = cv2.imread(p, -1)
        msk = cv2.imread(glob.glob(os.path.join(ROOT, 'EP-UiT-Human', 'annotation', f + '.*'))[0], -1)
        img = img if img.ndim == 2 else img[..., 0]
        msk = msk if msk.ndim == 2 else msk[..., 0]
        # EP-UiT-Human "annotations" are greyscale maps (0-255, bimodal: faint
        # halo values near 0, structure near 255), not binary masks. Foreground
        # = value >= 128 (the midpoint), fixed before any evaluation on it.
        msk = (msk >= 128).astype(np.uint8)
        for y, x, ti, tm in tiles(stretch(img), msk):
            emit('HUMAN', f'HUMAN_{f}', 'test', f'HUMAN_{f}_y{y}x{x}', ti, tm)

    # CBMI: provided splits, source frame = name without the _h*_w* offset
    for split in ('train', 'val', 'test'):
        for p in sorted(glob.glob(os.path.join(CBMI, split, 'images', '*'))):
            name = os.path.basename(p)
            m = cv2.imread(os.path.join(CBMI, split, 'masks', name), -1)
            img = cv2.imread(p, -1)
            img = img if img.ndim == 2 else img[..., 0]
            m = m if m.ndim == 2 else m[..., 0]
            src = re.sub(r'_h\d+_w\d+\.tif$', '', name)
            emit('CBMI', f'CBMI_{src}', split, 'CBMI_' + os.path.splitext(name)[0], stretch(img), m)

    # MITO: split by cell
    for p in sorted(glob.glob(os.path.join(MITO, 'images', '*.png'))):
        f = os.path.splitext(os.path.basename(p))[0]
        cell, sl = f.split('_')
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        msk = cv2.imread(os.path.join(MITO, 'masks', f + '.png'), cv2.IMREAD_GRAYSCALE)
        if cell == 'M2':
            split = 'test'
        else:
            split = 'val' if int(sl) > 17 else 'train'
        for y, x, ti, tm in tiles(stretch(img), msk):
            emit('MITO', f'MITO_{cell}', split, f'MITO_{f}_y{y}x{x}', ti, tm)

    M = pd.DataFrame(rows)
    M.to_csv(os.path.join(a.out, 'manifest.csv'), index=False)
    print(M.groupby(['dataset', 'split']).agg(tiles=('id', 'size'), groups=('group', 'nunique'),
                                              fg=('fg', 'mean')).round(3).to_string())


if __name__ == '__main__':
    main()
