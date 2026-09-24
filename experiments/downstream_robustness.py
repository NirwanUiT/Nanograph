#!/usr/bin/env python3
"""
Robustness checks on the T10/T11.2 downstream comparison.

  --check pruning  sensitivity to the pruning rule ('once' = the rule used in
                   T11.2, 'once_noguard', 'iterative') at L = 2, 5, 10, from
                   the branch tables saved by downstream_morphometry.py
  --check holm     Holm-adjusted p for the GRAPH-vs-JPEG paired tests, per
                   (junction definition, L) family and over all of them
  --check masks    the SEG arm ran the U-Net on CPU; the encoder (PRE) on GPU.
                   Compares the two masks and the SEG descriptors they give,
                   and recomputes the decomposition with the GPU mask.

Usage (from repo root):
    python experiments/downstream_robustness.py --check pruning holm masks
Writes results/paper/downstream/robustness/*.csv.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import downstream_morphometry as dm  # noqa: E402

COUNTS = ['n_components', 'total_length_px', 'n_branches', 'n_junctions', 'cycle_rank']


def _stems(out):
    P = pd.read_csv(os.path.join(out, 'per_image.csv'), dtype={'stem': str})
    d0 = P[P.junction_def == 't10']
    return sorted(set.intersection(*(set(d0[d0.arm == a].stem)
                                     for a in ['REF', 'GRAPH', 'JPEG', 'PRE', 'SEG'])))


def check_pruning(out, rdir):
    stems = _stems(out)
    tabs = {(s, a): json.load(open(os.path.join(out, 'branch_lengths', f'{s}_{a}.json')))
            for s in stems for a in ('REF', 'GRAPH', 'JPEG')}
    rows = []
    for prune in ('once', 'once_noguard', 'iterative'):
        for L in (2, 5, 10):
            v = {a: pd.DataFrame([dm.descriptors_at(tabs[s, a], L, 'degree', prune)[0]
                                  for s in stems]) for a in ('REF', 'GRAPH', 'JPEG')}
            for desc in COUNTS:
                y = v['REF'][desc].to_numpy(float)
                g, j = v['GRAPH'][desc].to_numpy(float), v['JPEG'][desc].to_numpy(float)
                ag, aj = dm.agreement(g, y), dm.agreement(j, y)
                rows.append({'prune': prune, 'L': L, 'descriptor': desc,
                             'graph_bias_pct': ag['bias_pct'], 'jpeg_bias_pct': aj['bias_pct'],
                             'graph_ccc': ag['ccc'], 'jpeg_ccc': aj['ccc'],
                             'graph_mdape': ag['mdape'], 'jpeg_mdape': aj['mdape'],
                             **dm.paired_error_test(g, j, y)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(rdir, 'pruning_rule.csv'), index=False)
    print(df[df.L == 5].round(3).to_string(index=False))


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, (len(p) - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


def check_holm(out, rdir):
    S = pd.read_csv(os.path.join(out, 'summary.csv'))
    rows = []
    for (jd, L), s in S.groupby(['junction_def', 'L'], sort=False):
        for desc in dm.DESCRIPTORS:
            r = s[(s.descriptor == desc) & (s.arm == 'GRAPH') & (s.ref == 'REF')].iloc[0]
            rows.append({'junction_def': jd, 'L': L, 'test': desc, 'p': r.wilcoxon_p,
                         'better': 'GRAPH' if r.wins_graph > r.wins_jpeg else 'JPEG'})
        b = s[(s.descriptor == 'branch_lengths') & (s.arm == 'GRAPH')].iloc[0]
        rows.append({'junction_def': jd, 'L': L, 'test': 'branch_lengths_w1', 'p': b.w1_wilcoxon_p,
                     'better': 'GRAPH' if b.w1_wins_graph > b.w1_wins_jpeg else 'JPEG'})
    df = pd.DataFrame(rows)
    df['p_holm_family'] = df.groupby(['junction_def', 'L'], sort=False).p.transform(holm)
    df['p_holm_all'] = holm(df.p)
    df.to_csv(os.path.join(rdir, 'holm.csv'), index=False)
    print(df.to_string(index=False))


def check_masks(out, rdir, images, masks):
    import cv2
    import torch
    from nanograph_v4.config import DEFAULT_CONFIG
    from nanograph_v4.unet_seg import load_unet
    stems = _stems(out)
    m_gpu = load_unet(DEFAULT_CONFIG.segment.learned_ckpt, device='cuda')
    m_cpu = load_unet(DEFAULT_CONFIG.segment.learned_ckpt, device='cpu')
    torch.set_num_threads(1)
    rows = []
    for s in stems:
        c = np.load(os.path.join(out, 'cache', s + '.npz'))
        seg = str(c['segmenter'])
        img = cv2.imread(os.path.join(images, s + '.png'), cv2.IMREAD_GRAYSCALE)
        mg = dm.segment_like_pipeline(img, seg, m_gpu, device='cuda') > 0
        mc = dm.segment_like_pipeline(img, seg, m_cpu, device='cpu') > 0
        u = np.logical_or(mg, mc).sum()
        tg = dm.pixel_arm_table(mg.astype(np.uint8))
        r = {'stem': s, 'segmenter': seg, 'iou_cpu_gpu': np.logical_and(mg, mc).sum() / u if u else 1.0,
             'pixels_differ': int((mg != mc).sum())}
        for jd, L in (('t10', 0), ('degree', 5)):
            dg = dm.descriptors_at(tg, L, jd)[0]
            for k in dm.DESCRIPTORS:
                r[f'segGPU_{jd}{L}_{k}'] = dg[k]
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(rdir, 'seg_masks_cpu_gpu.csv'), index=False)

    # decomposition with the GPU mask (the encoder's device) vs the CPU mask
    P = pd.read_csv(os.path.join(out, 'per_image.csv'), dtype={'stem': str})
    g = df.set_index('stem')
    out_rows = []
    for jd, L in (('t10', 0), ('degree', 5)):
        sel = P[(P.junction_def == jd) & (P.L == L)]
        W = {a: x.set_index('stem').loc[stems] for a, x in sel.groupby('arm')}
        for k in dm.DESCRIPTORS:
            ref, pre, segc = (W[a][k].to_numpy(float) for a in ('REF', 'PRE', 'SEG'))
            segg = g[f'segGPU_{jd}{L}_{k}'].to_numpy(float)
            ok = np.isfinite(ref) & np.isfinite(pre) & np.isfinite(segc) & np.isfinite(segg)
            out_rows.append({
                'junction_def': jd, 'L': L, 'descriptor': k,
                'segCPU_vs_segGPU_bias_pct': 100 * (segc[ok] - segg[ok]).mean() / segg[ok].mean(),
                'segCPU_vs_segGPU_ccc': dm.ccc(segc[ok], segg[ok]),
                'images_differ': int((np.abs(segc[ok] - segg[ok]) > 1e-9).sum()),
                'seg_vs_ref_cpu_pct': 100 * (segc[ok] - ref[ok]).mean() / ref[ok].mean(),
                'seg_vs_ref_gpu_pct': 100 * (segg[ok] - ref[ok]).mean() / ref[ok].mean(),
                'pre_vs_seg_cpu_pct': 100 * (pre[ok] - segc[ok]).mean() / segc[ok].mean(),
                'pre_vs_seg_gpu_pct': 100 * (pre[ok] - segg[ok]).mean() / segg[ok].mean()})
    dd = pd.DataFrame(out_rows)
    dd.to_csv(os.path.join(rdir, 'decomposition_gpu_mask.csv'), index=False)
    print(f"mask IoU CPU vs GPU: mean {df.iou_cpu_gpu.mean():.5f}, min {df.iou_cpu_gpu.min():.4f}; "
          f"identical masks {int((df.pixels_differ == 0).sum())}/{len(df)}; "
          f"median differing pixels {df.pixels_differ.median():.0f}")
    print(dd.round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--check', nargs='+', default=['pruning', 'holm', 'masks'])
    ap.add_argument('--images', default='/mnt/nas1/nba055-2/idea_1/nmi_data/org')
    ap.add_argument('--masks', default='/mnt/nas1/nba055-2/idea_1/nmi_data/seg')
    ap.add_argument('--out', default='results/paper/downstream')
    a = ap.parse_args()
    rdir = os.path.join(a.out, 'robustness')
    os.makedirs(rdir, exist_ok=True)
    if 'pruning' in a.check:
        check_pruning(a.out, rdir)
    if 'holm' in a.check:
        check_holm(a.out, rdir)
    if 'masks' in a.check:
        check_masks(a.out, rdir, a.images, a.masks)


if __name__ == '__main__':
    main()
