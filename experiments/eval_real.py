#!/usr/bin/env python3
"""
Evaluate segmenters on the real test sets (T13).

For every checkpoint in results/real/unet/*.pt plus the shipped
simulation-trained model ('shipped_sim') and the clean simulation baseline:
per test tile, the learned mask as the pipeline makes it; pixel IoU /
precision / recall against the expert mask; the v7 structure layer built from
the predicted mask (bytes; descriptors at L = 5, degree-based junctions)
against the expert mask's skeleton (REF); junction / endpoint F1 (3 px).

Outputs results/real/eval_per_tile.csv and eval_summary.csv
(model x test dataset; mean and, for seeded families, mean/sd over seeds).
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL = '/mnt/nas1/nba055-2/idea_1/real_mito'
L = 5


def _eval_model(job):
    name, ck = job
    import cv2
    import pandas as pd
    import torch
    import downstream_morphometry as dm
    from nanograph_v4 import NanographConfig, graph_branch as gb
    from nanograph_v4.detect import detect_polarity
    from nanograph_v4.segment import learned_segment
    from nanograph_v4.unet_seg import load_unet
    from skimage.morphology import skeletonize
    torch.set_num_threads(1)
    cfg = NanographConfig()
    m = load_unet(ck, device='cpu')
    M = pd.read_csv(os.path.join(REAL, 'manifest.csv'))
    M = M[M.split == 'test']
    rows = []
    for d, tid in zip(M.dataset, M.id):
        img = cv2.imread(os.path.join(REAL, d, 'images', tid + '.png'), cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(os.path.join(REAL, d, 'masks', tid + '.png'), cv2.IMREAD_GRAYSCALE) > 0
        x = 255 - img if detect_polarity(img, cfg=cfg) else img
        pm = learned_segment(m, x, cfg=cfg, device='cpu') > 0
        tp, fp, fn = (pm & gt).sum(), (pm & ~gt).sum(), (~pm & gt).sum()
        r = {'model': name, 'dataset': d, 'id': tid, 'iou': tp / max(tp + fp + fn, 1),
             'precision': tp / max(tp + fp, 1), 'recall': tp / max(tp + fn, 1)}
        tref = dm.pixel_arm_table(gt.astype(np.uint8))
        if pm.sum() >= 2:
            dt = cv2.distanceTransform(pm.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            blob = gb.encode_structure(gb.branch_structure(skeletonize(pm), dt))
            t7 = dm.graph_arm_table(*gb.structure_to_arrays(gb.decode_structure(blob), path_lengths=True))
            r['bytes'] = len(blob)
        else:
            t7 = dm.pixel_arm_table(np.zeros_like(gt, np.uint8))
            r['bytes'] = 0
        d7, dr = dm.descriptors_at(t7, L)[0], dm.descriptors_at(tref, L)[0]
        for k in dm.DESCRIPTORS:
            r[k], r[k + '_ref'] = d7[k], dr[k]
        jr, er = dm.vertex_positions(tref, L)
        ja, ea = dm.vertex_positions(t7, L)
        r['junc_f1'] = dm.point_f1(ja, jr, 3)[2]
        r['end_f1'] = dm.point_f1(ea, er, 3)[2]
        rows.append(r)
    return rows


def main():
    import multiprocessing as mp
    import pandas as pd
    import downstream_morphometry as dm
    models = {'shipped_sim': os.path.join(ROOT, 'weights', 'unet_organelle_cldice.pt'),
              'clean_sim': os.path.join(ROOT, 'results', 'v7', 'unet', 'unet_clean_base.pt')}
    for p in sorted(glob.glob(os.path.join(os.environ.get('UNET_DIR', os.path.join(ROOT, 'results', 'real', 'unet')), '*.pt'))):
        models[os.path.splitext(os.path.basename(p))[0]] = p
    with mp.get_context('spawn').Pool(int(os.environ.get('WORKERS', 12))) as pool:
        rows = [r for rs in pool.imap_unordered(_eval_model, list(models.items())) for r in rs]
    P = pd.DataFrame(rows)
    out = os.environ.get('OUT_DIR', os.path.join(ROOT, 'results', 'real'))
    os.makedirs(out, exist_ok=True)
    P.to_csv(os.path.join(out, 'eval_per_tile.csv'), index=False)
    S = []
    for (name, d), g in P.groupby(['model', 'dataset']):
        r = {'model': name, 'dataset': d, 'n': len(g), 'iou': g.iou.mean(), 'precision': g.precision.mean(),
             'recall': g.recall.mean(), 'bytes': g.bytes.mean(), 'junc_f1': g.junc_f1.mean(),
             'end_f1': g.end_f1.mean()}
        for k in dm.DESCRIPTORS:
            x, y = g[k].to_numpy(float), g[k + '_ref'].to_numpy(float)
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() > 2 and y[ok].mean() != 0:
                ag = dm.agreement(x[ok], y[ok])
                r[f'{k}_bias_pct'], r[f'{k}_ccc'] = ag['bias_pct'], ag['ccc']
        S.append(r)
    S = pd.DataFrame(S)
    S['family'] = S.model.str.replace(r'_s\d$', '', regex=True)
    S.to_csv(os.path.join(out, 'eval_summary.csv'), index=False)
    cols = ['iou', 'precision', 'recall', 'n_branches_bias_pct', 'n_junctions_bias_pct',
            'total_length_px_bias_pct', 'n_branches_ccc', 'junc_f1', 'end_f1']
    F = S.groupby(['family', 'dataset'])[cols].agg(['mean', 'std'])
    F.to_csv(os.path.join(out, 'eval_family.csv'))
    pd.set_option('display.width', 250)
    print(S.groupby(['family', 'dataset'])[cols].mean().round(3).to_string())


if __name__ == '__main__':
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    main()
