#!/usr/bin/env python3
"""
T18 segmenter bake-off on held-out Allen plates (allen_export_test.py inputs).

Every segmenter is scored the same way per test cell, inside the cell mask:
  - Dice and clDice with Allen's struct segmentation over the same slab
    (Allen's mask is a model output, not ground truth);
  - junction / endpoint F1 (3 px) of the structure layers;
  - the descriptors of each segmenter's v7 structure layer (read from its
    bytes, one-diameter rule) against those of Allen's mask pushed through the
    same structure layer (storage identical, only the mask differs).

Segmenters:
  real_mito   shipped real-mito U-Net on the unmasked slab (as in the pilot)
  allen_ft    our clDice U-Net fine-tuned on Allen tiles (--ckpt), masked input
  <name>      any directory of predictions given as --pred name=DIR
              (<DIR>/<CellId>.png, 0/255, cell-crop size): nnU-Net, Nellie,
              micro-SAM, ... each produced in its own environment

Writes <out>/per_cell.csv and <out>/summary.csv (ranked by clDice).
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

TEST = '/mnt/nas1/nba055-2/idea_1/archive_demo/allen/testset'
DESC = ['n_components', 'n_branches', 'n_junctions', 'total_length_px', 'mean_width_px', 'cycle_rank']


def structure_table(mask):
    import cv2
    import downstream_morphometry as dm
    from skimage.morphology import skeletonize
    from nanograph_v4 import graph_branch as gb
    sk = skeletonize(mask > 0)
    if sk.sum() < 2:
        return None, 0
    dt = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    sb = gb.encode_structure(gb.branch_structure(sk, dt))
    return dm.graph_arm_table(*gb.structure_to_arrays(gb.decode_structure(sb), True)), len(sb)


def cldice(p, r):
    from skimage.morphology import skeletonize
    sp, sr = skeletonize(p), skeletonize(r)
    tprec = (sp & r).sum() / max(sp.sum(), 1)
    tsens = (sr & p).sum() / max(sr.sum(), 1)
    return 2 * tprec * tsens / max(tprec + tsens, 1e-9)


def work(args):
    row, ckpt, preds = args
    import cv2
    import torch
    import downstream_morphometry as dm
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
    rd = lambda d: cv2.imread(os.path.join(d, f'{cid}.png'), cv2.IMREAD_GRAYSCALE)
    cell, aref = rd(f'{TEST}/cell') > 0, rd(f'{TEST}/allen') > 0
    masks = {'allen': aref}
    for name, net in _NETS.items():
        x = rd(f'{TEST}/img' if name == 'allen_ft' else f'{TEST}/img_raw')
        x = 255 - x if detect_polarity(x, cfg=cfg) else x
        with contextlib.redirect_stdout(io.StringIO()):
            masks[name] = (learned_segment(net, x, cfg=cfg, device='cpu') > 0) & cell
    for name, d in preds.items():
        m = rd(d)
        if m is not None:
            masks[name] = (m > 0) & cell
    tref, _ = structure_table(aref)
    ref = dm.descriptors_at(tref, 'auto', 'degree')[0] if tref is not None else {k: 0.0 for k in DESC}
    jr, er = dm.vertex_positions(tref, 'auto') if tref is not None else ([], [])
    rows = []
    for name, m in masks.items():
        t, b = structure_table(m)
        d = dm.descriptors_at(t, 'auto', 'degree')[0] if t is not None else {k: 0.0 for k in DESC}
        ja, ea = dm.vertex_positions(t, 'auto') if t is not None else ([], [])
        rows.append({'CellId': cid, 'cell_stage': row['cell_stage'], 'PlateId': row['PlateId'], 'segmenter': name,
                     'dice': 2 * (m & aref).sum() / max(m.sum() + aref.sum(), 1), 'cldice': cldice(m, aref),
                     'junc_f1': dm.point_f1(ja, jr, 3)[2], 'end_f1': dm.point_f1(ea, er, 3)[2],
                     'fg_in_cell': float(m[cell].mean()), 'structure_bytes': b,
                     **{k: d[k] for k in DESC}, **{k + '_allen': ref[k] for k in DESC}})
    return rows


def main():
    import multiprocessing as mp
    import subprocess
    import pandas as pd
    import downstream_morphometry as dm
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='results/allen/unet/allen_finetune_s0.pt')
    ap.add_argument('--pred', action='append', default=[], help='name=DIR of <CellId>.png predictions')
    ap.add_argument('--out', default='results/allen/seg_eval')
    ap.add_argument('--workers', type=int, default=6)
    a = ap.parse_args()
    preds = dict(p.split('=', 1) for p in a.pred)
    C = pd.read_csv(f'{TEST}/cells.csv')
    with mp.get_context('spawn').Pool(a.workers) as pool:
        R = pd.DataFrame([r for rs in pool.map(work, [(r, a.ckpt, preds) for r in C.to_dict('records')])
                          for r in rs])
    os.makedirs(a.out, exist_ok=True)
    R.to_csv(os.path.join(a.out, 'per_cell.csv'), index=False)
    S = []
    for name, g in R.groupby('segmenter'):
        r = {'segmenter': name, 'n': len(g), 'cldice': g.cldice.median(), 'dice': g.dice.median(),
             'junc_f1': g.junc_f1.mean(), 'end_f1': g.end_f1.mean(), 'fg_in_cell': g.fg_in_cell.median(),
             'structure_bytes': g.structure_bytes.median()}
        for k in DESC:
            ag = dm.agreement(g[k].to_numpy(float), g[k + '_allen'].to_numpy(float))
            r[f'{k}_ccc'], r[f'{k}_bias_pct'] = ag['ccc'], ag['bias_pct']
        S.append(r)
    S = pd.DataFrame(S).sort_values('cldice', ascending=False)
    S.to_csv(os.path.join(a.out, 'summary.csv'), index=False)
    h = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True,
                       cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
    open(os.path.join(a.out, 'commit.txt'), 'w').write(h + '\n')
    pd.set_option('display.width', 220)
    print(S.round(3).T.to_string())


if __name__ == '__main__':
    main()
