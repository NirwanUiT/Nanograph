#!/usr/bin/env python3
"""
Fine-tune micro-SAM (vit_b, initialised from the vit_b_lm generalist) with its
instance-segmentation decoder on one training pool:
  allen  Allen TOMM20 tiles (allen_trainset.py; train / val splits by plate)
  real   UiT-Rat + CBMI + MITO tiles (real_mito manifest; train / val as provided)
Binary masks are split into instances (8-connected components) for SAM's
object-level training. Fixed budget of --iterations; best checkpoint on val.

Output: <save-root>/checkpoints/microsam_<pool>/best.pt
Runs in the micro-SAM environment on a GPU.
"""
import argparse
import os

import numpy as np
import pandas as pd

ALLEN = '/mnt/nas1/nba055-2/idea_1/archive_demo/allen/trainset'
REAL = '/mnt/nas1/nba055-2/idea_1/real_mito'


def pairs(pool, split):
    if pool == 'allen':
        M = pd.read_csv(f'{ALLEN}/manifest.csv')
        M = M[M.split == split]
        return [(f'{ALLEN}/images/{i}.png', f'{ALLEN}/masks/{i}.png', i) for i in M.id]
    M = pd.read_csv(f'{REAL}/manifest.csv')
    M = M[M.dataset.isin(['UIT', 'CBMI', 'MITO']) & (M.split == split)]
    return [(f'{REAL}/{d}/images/{i}.png', f'{REAL}/{d}/masks/{i}.png', i) for d, i in zip(M.dataset, M.id)]


def instance_labels(prs, out):
    import cv2
    import tifffile
    os.makedirs(out, exist_ok=True)
    raws, labs = [], []
    for ip, mp, i in prs:
        lp = os.path.join(out, f'{i}.tif')
        if not os.path.exists(lp):
            m = (cv2.imread(mp, cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
            _, lab = cv2.connectedComponents(m, connectivity=8)
            tifffile.imwrite(lp, lab.astype(np.uint16))
        raws.append(ip)
        labs.append(lp)
    return raws, labs


def main():
    import micro_sam.training as st
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', required=True, choices=['allen', 'real'])
    ap.add_argument('--save-root', default='/mnt/nas1/nba055-2/idea_1/archive_demo/microsam')
    ap.add_argument('--iterations', type=int, default=10000)
    ap.add_argument('--batch', type=int, default=2)
    ap.add_argument('--objects', type=int, default=10)
    ap.add_argument('--dry', action='store_true', help='build labels and loaders, draw one batch, stop')
    a = ap.parse_args()
    lab_dir = os.path.join(a.save_root, f'labels_{a.pool}')
    tr = instance_labels(pairs(a.pool, 'train'), lab_dir)
    va = instance_labels(pairs(a.pool, 'val'), lab_dir)
    common = dict(raw_key=None, label_key=None, patch_shape=(256, 256), with_segmentation_decoder=True,
                  min_size=10, num_workers=4)
    tl = st.default_sam_loader(raw_paths=tr[0], label_paths=tr[1], batch_size=a.batch, shuffle=True,
                               is_train=True, **common)
    vl = st.default_sam_loader(raw_paths=va[0], label_paths=va[1], batch_size=a.batch, shuffle=False,
                               is_train=False, **common)
    print(f'pool {a.pool}: {len(tr[0])} train / {len(va[0])} val tiles', flush=True)
    if a.dry:
        x, y = next(iter(tl))
        print('batch', tuple(x.shape), tuple(y.shape), 'label channels', y.shape[1])
        return
    st.train_sam(name=f'microsam_{a.pool}', model_type='vit_b_lm', train_loader=tl, val_loader=vl,
                 n_iterations=a.iterations, n_objects_per_batch=a.objects, with_segmentation_decoder=True,
                 device='cuda', save_root=a.save_root)


if __name__ == '__main__':
    main()
