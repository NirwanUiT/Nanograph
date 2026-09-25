#!/usr/bin/env python3
"""
T18 bake-off figure: per cell, the slab image, Allen's segmentation, and each
segmenter's mask against Allen's (white both, magenta segmenter only, green
Allen only) with the segmenter's decoded structure-layer centreline in cyan.
Captions: Dice, clDice (vs Allen), separate mitochondria, structure bytes.

Segmenters: real_mito (computed here), plus any --pred name=DIR of
<CellId>.png predictions (Nellie, nnU-Net, micro-SAM, allen_ft, ...).

Usage:
    python experiments/archive_demo/allen_compare_show.py --test DIR \
        --pred nellie=DIR/preds/nellie [--pred nnunet=...] --out fig.png
"""
import argparse
import contextlib
import io
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from allen_eval_seg import cldice  # noqa: E402

STAGE = {'M0': 'interphase', 'M1M2': 'prophase', 'M3': 'metaphase', 'M4M5': 'anaphase',
         'M6M7_single': 'telophase (1 daughter)', 'M6M7_complete': 'telophase (2 daughters)'}


def structure(mask):
    import cv2
    import downstream_morphometry as dm
    from skimage.morphology import skeletonize
    from nanograph_v4 import graph_branch as gb
    sk = skeletonize(mask > 0)
    if sk.sum() < 2:
        return None, 0, 0
    dt = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    sb = gb.encode_structure(gb.branch_structure(sk, dt))
    st = gb.decode_structure(sb)
    arr = gb.structure_to_arrays(st, True)
    d = dm.descriptors_at(dm.graph_arm_table(*arr), 'auto', 'degree')[0]
    return arr, len(sb), int(d['n_components'])


def main():
    import cv2
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    from nanograph_v4 import NanographConfig
    from nanograph_v4.detect import detect_polarity
    from nanograph_v4.segment import learned_segment
    from nanograph_v4.unet_seg import load_unet
    ap = argparse.ArgumentParser()
    ap.add_argument('--test', required=True)
    ap.add_argument('--pred', action='append', default=[])
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    preds = dict(p.split('=', 1) for p in a.pred)
    cfg = NanographConfig().for_real_mito()
    net = load_unet(cfg.segment.learned_ckpt, 'cpu')
    C = pd.read_csv(os.path.join(a.test, 'cells.csv'))
    order = list(STAGE)
    C = C.assign(o=C.cell_stage.map(order.index)).sort_values('o')
    names = ['real_mito'] + list(preds)
    label = {'real_mito': 'ours now (real-mito U-Net)', 'nellie': 'Nellie (zero-shot)', 'nnunet': 'nnU-Net (Allen-trained)',
             'allen_ft': 'ours, Allen-trained', 'microsam': 'micro-SAM', 'microsam_ft': 'micro-SAM (Allen-trained)',
             'microsam_zs': 'micro-SAM (zero-shot)'}
    ncol = 2 + len(names)
    fig, ax = plt.subplots(len(C), ncol, figsize=(3.1 * ncol, 3.3 * len(C)))
    ax = np.atleast_2d(ax)
    for i, r in enumerate(C.itertuples()):
        cid = r.CellId
        rd = lambda d: cv2.imread(os.path.join(d, f'{cid}.png'), cv2.IMREAD_GRAYSCALE)
        img, cell, aref = rd(f'{a.test}/img_raw'), rd(f'{a.test}/cell') > 0, rd(f'{a.test}/allen') > 0
        x = 255 - img if detect_polarity(img, cfg=cfg) else img
        with contextlib.redirect_stdout(io.StringIO()):
            masks = {'real_mito': (learned_segment(net, x, cfg=cfg, device='cpu') > 0) & cell}
        for n, d in preds.items():
            m = rd(d)
            masks[n] = (m > 0) & cell if m is not None else np.zeros_like(cell)
        ys, xs = np.nonzero(cell)
        box = (xs.min() - 3, xs.max() + 3, ys.max() + 3, ys.min() - 3)
        ax[i, 0].imshow(img, cmap='gray')
        ax[i, 0].set_ylabel(STAGE.get(r.cell_stage, r.cell_stage), fontsize=9)
        _, ab, an = structure(aref)
        ax[i, 1].imshow(np.dstack([aref] * 3).astype(float))
        ax[i, 1].set_xlabel(f'{an} mitochondria | {ab} B', fontsize=8)
        for j, n in enumerate(names):
            m = masks[n]
            rgb = np.zeros(m.shape + (3,))
            rgb[m & aref] = 1
            rgb[m & ~aref] = (0.85, 0.1, 0.85)
            rgb[aref & ~m] = (0.1, 0.8, 0.1)
            A = ax[i, 2 + j]
            A.imshow(rgb)
            arr, b, nc = structure(m)
            if arr is not None:
                pos, _, edges, _ = arr
                for u, v in edges:
                    A.plot([pos[u][1], pos[v][1]], [pos[u][0], pos[v][0]], color='#00c8ff', lw=0.7)
            dice = 2 * (m & aref).sum() / max(m.sum() + aref.sum(), 1)
            A.set_xlabel(f'Dice {dice:.2f} | clDice {cldice(m, aref):.2f}\n{nc} mitochondria | {b} B', fontsize=8)
            if i == 0:
                A.set_title(label.get(n, n), fontsize=9)
        for j in range(ncol):
            A = ax[i, j]
            A.contour(cell, [0.5], colors=['#5a8fd6'], linewidths=0.5)
            A.set_xlim(box[0], box[1])
            A.set_ylim(box[2], box[3])
            A.set_xticks([])
            A.set_yticks([])
    ax[0, 0].set_title('TOMM20 (±1 µm slab)', fontsize=9)
    ax[0, 1].set_title("Allen's segmentation (reference)", fontsize=9)
    fig.suptitle('white: agrees with Allen   magenta: this method only   green: Allen only   cyan: stored '
                 'structure-layer centreline', fontsize=9, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(a.out, dpi=120)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
