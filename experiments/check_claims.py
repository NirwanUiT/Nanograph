#!/usr/bin/env python3
"""Evaluate the 14 'claims to verify' (COPILOT_TASKS.md) on a results tree.

Usage (from repo root):
    python experiments/check_claims.py --runs results/paper --paper paper
Prints one markdown row per claim: number, HOLDS/FAILS, evidence.
"""
import argparse
import os
import re

import numpy as np
import pandas as pd


def macros(path):
    out = {}
    for line in open(path):
        m = re.match(r'\\newcommand\{\\(\w+)\}\{(.*)\}\s*$', line.strip())
        if m:
            out[m.group(1)] = m.group(2)
    return out


def num(s):
    """Parse a macro value: 0.04, 6.1\\times10^{-140}, 726/726, 53.8."""
    m = re.match(r'([-+\d.]+)\\times10\^\{(-?\d+)\}', s)
    if m:
        return float(m.group(1)) * 10 ** int(m.group(2))
    return float(s)


def csv(runs, *p):
    return pd.read_csv(os.path.join(runs, *p, 'metrics.csv'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', default='results/paper')
    ap.add_argument('--paper', default='paper')
    a = ap.parse_args()
    M = macros(os.path.join(a.paper, 'numbers.tex'))
    D, C = csv(a.runs, 'org_default'), csv(a.runs, 'org_classical')
    rows = []

    def claim(i, ok, ev):
        rows.append((i, 'HOLDS' if ok else 'FAILS', ev))

    claim(1, D.seg_iou.mean() > C.seg_iou.mean() and D.ng_bytes.mean() < C.ng_bytes.mean(),
          f'Seg-IoU {D.seg_iou.mean():.3f} (D) vs {C.seg_iou.mean():.3f} (C); '
          f'payload {D.ng_bytes.mean():.0f} B (D) vs {C.ng_bytes.mean():.0f} B (C)')
    claim(2, D.graph_n_cycles.mean() < C.graph_n_cycles.mean()
          and D.mean_width_px.mean() < C.mean_width_px.mean(),
          f'cycles {D.graph_n_cycles.mean():.2f} vs {C.graph_n_cycles.mean():.2f}; '
          f'width {D.mean_width_px.mean():.2f} vs {C.mean_width_px.mean():.2f}')
    p3, h3 = num(M['DGTIoUWinsP']), num(M['DHeldGTIoUWinsPct'])
    claim(3, p3 < 0.01 and h3 > 50,
          f"GT-IoU vs JPEG wins {M['DGTIoUWins']}, p = {M['DGTIoUWinsP']}; held-out wins {h3:.1f} %")
    p4 = num(M['CGTIoUWinsP'])
    claim(4, p4 > 0.05, f"classical GT-IoU vs JPEG wins {M['CGTIoUWins']}, p = {M['CGTIoUWinsP']}")
    claim(5, D.gt_fg_psnr.mean() < D.jpeg_gt_fg_psnr.mean(),
          f'GT-FG-PSNR {D.gt_fg_psnr.mean():.2f} (NG) vs {D.jpeg_gt_fg_psnr.mean():.2f} (JPEG)')
    b = num(M['DJpegBettim'])
    j0, g0, j1, g1 = (num(M[k]) for k in ('DJpegBetaZerom', 'DBetaZerom', 'DJpegBetaOnem', 'DBetaOnem'))
    claim(6, b <= 0.2 and j0 > g0 and j1 > g1,
          f'JPEG Betti {b:.3f}; beta0 {j0:.2f} vs {g0:.2f}; beta1 {j1:.2f} vs {g1:.2f}')
    tab = open(os.path.join(a.paper, 'tables', 'tab_topoall_body.tex')).read()
    betti = {m.group(1).strip(): float(m.group(2)) for m in
             re.finditer(r'^([^&\n]+?) & [^&]+ & \$([\d.]+)\\pm', tab, re.M)}
    others = {k: v for k, v in betti.items() if k != 'Temporal clip'}
    worst = max(others, key=others.get)
    claim(7, all(v <= 0.2 for v in others.values()),
          f'max JPEG Betti excl. clip {others[worst]:.3f} ({worst}); clip {betti.get("Temporal clip", float("nan")):.3f}')
    brr = num(M['DBRR'])
    claim(8, 0.0365 * 0.75 <= brr <= 0.0365 * 1.25,
          f"BRR {brr:.4f} (band 0.0274-0.0456); FewerX {M['DFewerX']}, RicherX {M['DRicherX']}")
    lh = num(M['LHeldSegIoUm'])
    claim(9, abs(lh - 0.875) <= 0.02, f'org_replace held-out Seg-IoU {lh:.3f}')
    pol = pd.read_csv(os.path.join(a.runs, 'polarity.csv'))
    dmax = (pol.as_is - pol.oracle).abs().max()
    claim(10, dmax <= 0.02, f'max |as_is - oracle| {dmax:.3f}')
    P = pd.read_csv(os.path.join(a.runs, 'seg_perturbation_ablation.csv'))
    base = P[P.kind == 'baseline'].set_index('filename')

    def arm(kind, level):
        g = P[(P.kind == kind) & (P.level == level)].set_index('filename')
        bb = base.loc[g.index]
        return ((g.beta0 != bb.beta0).mean() * 100,
                np.median(g.mean_width / bb.mean_width - 1) * 100,
                np.median((g.beta1 - bb.beta1).abs()))
    d0, dw, _ = arm('dilate', 1)
    n0, _, n1 = arm('boundary_noise', 1)
    bs = base.seg_iou.median()
    claim(11, abs(d0 - 8) <= 3 and abs(dw - 31) <= 5 and abs(n0 - 78) <= 5 and abs(n1 - 281) <= 30,
          f'dilate 1 px: beta0 changed {d0:.1f} %, width {dw:+.1f} %; boundary noise 1 px: '
          f'beta0 changed {n0:.1f} %, median |dbeta1| {n1:.0f}; baseline Seg-IoU median {bs:.3f}')
    st = csv(a.runs, 'cross_gt', 'stare')
    mt, ep = M['CgMtNoJpeg'], M['CgEpflNoJpeg']
    claim(12, not mt.startswith('0/') and not ep.startswith('0/')
          and st.jpeg_gt_fg_psnr.mean() > st.gt_fg_psnr.mean(),
          f'no-JPEG microtubules {mt}, EPFL {ep}; STARE JPEG GT-FG-PSNR '
          f'{st.jpeg_gt_fg_psnr.mean():.1f} vs {st.gt_fg_psnr.mean():.1f}')
    cl, cd = num(M['ClipLSegIoUm']), num(M['ClipSegIoUm'])
    claim(13, cl >= cd, f'clip Seg-IoU learned-replace {cl:.3f} vs default {cd:.3f}')
    ai = M['AblImproved']
    n, d = (int(x) for x in ai.split('/'))
    claim(14, n == d, f'bg grid + residual improve FG-PSNR on {ai}')

    print('| # | verdict | evidence |\n|---|---|---|')
    for i, v, ev in rows:
        print(f'| {i} | **{v}** | {ev} |')


if __name__ == '__main__':
    main()
