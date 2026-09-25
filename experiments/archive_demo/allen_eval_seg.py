#!/usr/bin/env python3
"""
T18 — segmenters on held-out Allen plates (the 'test' cells of allen_trainset.py).

Per cell and segmenter: Dice with Allen's struct_segmentation over the same
slab, and the descriptors of each segmenter's v7 structure layer (read from
its bytes, one-diameter rule) against the descriptors of Allen's mask pushed
through the same structure layer (so storage is identical and only the mask
differs). Allen's mask is a model output, not ground truth: for the
Allen-trained network this measures how well the workflow is reproduced.

Segmenters:
  real_mito    shipped real-mito U-Net, unmasked slab input (as in the pilot)
  allen_ft     U-Net fine-tuned on Allen tiles (masked input, allen_trainset.masked_input)
  nellie       Nellie (Lefebvre et al., Nat Methods 2025), if its env exists
               (NELLIE_PY=/path/to/python); skipped otherwise

Writes <out>/per_cell.csv and <out>/summary.csv.
"""
import argparse
import ast
import contextlib
import io
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from allen_check import slab_projection  # noqa: E402
from allen_trainset import masked_input  # noqa: E402

ROOT = '/mnt/nas1/nba055-2/idea_1/archive_demo/allen'
DESC = ['n_components', 'n_branches', 'n_junctions', 'total_length_px', 'mean_width_px', 'cycle_rank']


def structure_descriptors(mask):
    import cv2
    import downstream_morphometry as dm
    from skimage.morphology import skeletonize
    from nanograph_v4 import graph_branch as gb
    sk = skeletonize(mask > 0)
    if sk.sum() < 2:
        return {k: 0.0 for k in DESC}, 0
    dt = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    sb = gb.encode_structure(gb.branch_structure(sk, dt))
    t = dm.graph_arm_table(*gb.structure_to_arrays(gb.decode_structure(sb), True))
    d = dm.descriptors_at(t, 'auto', 'degree')[0]
    return {k: d[k] for k in DESC}, len(sb)


def work(args):
    row, ckpt = args
    import tifffile
    import torch
    from nanograph_v4 import NanographConfig
    from nanograph_v4.detect import detect_polarity
    from nanograph_v4.segment import learned_segment
    from nanograph_v4.unet_seg import load_unet
    torch.set_num_threads(1)
    global _NETS
    cfg = NanographConfig().for_real_mito()
    if '_NETS' not in globals():
        _NETS = {'real_mito': load_unet(cfg.segment.learned_ckpt, 'cpu')}
        if ckpt and os.path.exists(ckpt):
            _NETS['allen_ft'] = load_unet(ckpt, 'cpu')
    cid = row['CellId']
    im, cell, aref, _ = slab_projection(tifffile.imread(f'{ROOT}/cells/{cid}_raw.ome.tif'),
                                        tifffile.imread(f'{ROOT}/cells/{cid}_seg.ome.tif'),
                                        ast.literal_eval(row['name_dict']),
                                        ast.literal_eval(row['scale_micron'])[-1])
    masks = {'allen': aref}
    for name, net in _NETS.items():
        x = masked_input(im, cell) if name == 'allen_ft' else im
        x = 255 - x if detect_polarity(x, cfg=cfg) else x
        with contextlib.redirect_stdout(io.StringIO()):
            masks[name] = (learned_segment(net, x, cfg=cfg, device='cpu') > 0) & cell
    ref, rb = structure_descriptors(aref)
    rows = []
    for name, m in masks.items():
        d, b = structure_descriptors(m)
        dice = 2 * (m & aref).sum() / max(m.sum() + aref.sum(), 1)
        rows.append({'CellId': cid, 'cell_stage': row['cell_stage'], 'PlateId': row['PlateId'],
                     'segmenter': name, 'dice_vs_allen': float(dice), 'fg_in_cell': float(m[cell].mean()),
                     'structure_bytes': b, **d, **{k + '_allen': ref[k] for k in DESC}})
    return rows


def main():
    import multiprocessing as mp
    import pandas as pd
    import downstream_morphometry as dm
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='results/allen/unet/allen_finetune_s0.pt')
    ap.add_argument('--out', default='results/allen/seg_eval')
    ap.add_argument('--workers', type=int, default=6)
    a = ap.parse_args()
    C = pd.read_csv(f'{ROOT}/trainset/cells.csv')
    C = C[C.split == 'test']
    with mp.get_context('spawn').Pool(a.workers) as pool:
        R = pd.DataFrame([r for rs in pool.map(work, [(r, a.ckpt) for r in C.to_dict('records')]) for r in rs])
    os.makedirs(a.out, exist_ok=True)
    R.to_csv(os.path.join(a.out, 'per_cell.csv'), index=False)
    S = []
    for name, g in R.groupby('segmenter'):
        r = {'segmenter': name, 'n': len(g), 'dice_median': g.dice_vs_allen.median(),
             'fg_in_cell': g.fg_in_cell.median(), 'structure_bytes_median': g.structure_bytes.median()}
        for k in DESC:
            ag = dm.agreement(g[k].to_numpy(float), g[k + '_allen'].to_numpy(float))
            r[f'{k}_ccc'], r[f'{k}_bias_pct'] = ag['ccc'], ag['bias_pct']
        S.append(r)
    S = pd.DataFrame(S)
    S.to_csv(os.path.join(a.out, 'summary.csv'), index=False)
    pd.set_option('display.width', 220)
    print(S.round(3).T.to_string())


if __name__ == '__main__':
    main()
