#!/usr/bin/env python3
"""
Held-out comparison of organelle segmenters for downstream structure (T12 #4).

Each model's learned mask (as the pipeline makes it: U-Net -> threshold ->
light clean-up) on the 108 held-out images is compared with the annotation:
Seg-IoU / precision / recall, the downstream descriptors (bias, CCC, MdAPE
against REF at L = 0 and 5, degree-based junctions) and junction / endpoint
F1; plus the v7 structure layer built from that mask (bytes, descriptors).

Usage (from repo root):
    python experiments/eval_unet_topo.py name=path.pt [name=path.pt ...]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

IMAGES = '/mnt/nas1/nba055-2/idea_1/nmi_data/org'
MASKS = '/mnt/nas1/nba055-2/idea_1/nmi_data/seg'
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    import cv2
    import pandas as pd
    import torch
    import downstream_morphometry as dm
    from nanograph_v4 import NanographConfig, graph_branch as gb
    from nanograph_v4.detect import detect_polarity
    from nanograph_v4.segment import learned_segment
    from nanograph_v4.unet_seg import load_unet
    from skimage.morphology import skeletonize
    torch.set_num_threads(4)
    models = dict(a.split('=', 1) for a in sys.argv[1:])
    held = open(os.path.join(ROOT, 'results', 'paper', 'heldout_organelle.txt')).read().split()
    cfg = NanographConfig()
    rows = []
    for name, ck in models.items():
        m = load_unet(ck, device='cpu')
        for s in held:
            img = cv2.imread(os.path.join(IMAGES, s + '.png'), cv2.IMREAD_GRAYSCALE)
            gt = cv2.imread(os.path.join(MASKS, s + '.png'), cv2.IMREAD_GRAYSCALE) > 0
            x = 255 - img if detect_polarity(img, cfg=cfg) else img
            pm = learned_segment(m, x, cfg=cfg, device='cpu') > 0
            tp, fp, fn = (pm & gt).sum(), (pm & ~gt).sum(), (~pm & gt).sum()
            tref = dm.pixel_arm_table(gt.astype(np.uint8))
            tseg = dm.pixel_arm_table(pm.astype(np.uint8))
            sk, dt = skeletonize(pm), cv2.distanceTransform(pm.astype(np.uint8), cv2.DIST_L2,
                                                            cv2.DIST_MASK_PRECISE)
            st = gb.branch_structure(sk, dt)
            blob = gb.encode_structure(st)
            t7 = dm.graph_arm_table(*gb.structure_to_arrays(gb.decode_structure(blob), path_lengths=True))
            for arm, t in (('SEG', tseg), ('GRAPH7', t7)):
                for L in (0, 5):
                    d = dm.descriptors_at(t, L)[0]
                    dr = dm.descriptors_at(tref, L)[0]
                    r = {'model': name, 'stem': s, 'arm': arm, 'L': L,
                         'iou': tp / max(tp + fp + fn, 1), 'precision': tp / max(tp + fp, 1),
                         'recall': tp / max(tp + fn, 1), 'bytes': len(blob) if arm == 'GRAPH7' else np.nan}
                    for k in dm.DESCRIPTORS:
                        r[k], r[k + '_ref'] = d[k], dr[k]
                    jr, er = dm.vertex_positions(tref, L)
                    ja, ea = dm.vertex_positions(t, L)
                    r['junc_f1_3'] = dm.point_f1(ja, jr, 3)[2]
                    r['end_f1_3'] = dm.point_f1(ea, er, 3)[2]
                    rows.append(r)
        print('done', name, flush=True)
    P = pd.DataFrame(rows)
    out = os.path.join(ROOT, 'results', 'v7', 'unet')
    os.makedirs(out, exist_ok=True)
    P.to_csv(os.path.join(out, 'heldout_per_image.csv'), index=False)
    S = []
    for (name, arm, L), g in P.groupby(['model', 'arm', 'L'], sort=False):
        r = {'model': name, 'arm': arm, 'L': L, 'n': len(g), 'iou': g.iou.mean(),
             'precision': g.precision.mean(), 'recall': g.recall.mean(), 'bytes': g.bytes.mean(),
             'junc_f1_3': g.junc_f1_3.mean(), 'end_f1_3': g.end_f1_3.mean()}
        for k in dm.DESCRIPTORS:
            x, y = g[k].to_numpy(float), g[k + '_ref'].to_numpy(float)
            ok = np.isfinite(x) & np.isfinite(y)
            ag = dm.agreement(x[ok], y[ok])
            r[f'{k}_bias_pct'], r[f'{k}_ccc'], r[f'{k}_mdape'] = ag['bias_pct'], ag['ccc'], ag['mdape']
        S.append(r)
    S = pd.DataFrame(S)
    S.to_csv(os.path.join(out, 'heldout_summary.csv'), index=False)
    cols = ['model', 'arm', 'L', 'iou', 'precision', 'recall', 'n_branches_bias_pct', 'n_branches_ccc',
            'n_junctions_bias_pct', 'n_junctions_ccc', 'cycle_rank_bias_pct', 'total_length_px_bias_pct',
            'mean_width_px_bias_pct', 'junc_f1_3', 'end_f1_3']
    print(S[cols].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
