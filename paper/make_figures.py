#!/usr/bin/env python3
"""Regenerate every data-driven figure of the paper from evaluation outputs.

Usage (from repo root):
    python paper/make_figures.py --runs results/paper --out paper/figures
Reads the same layout as make_numbers.py. Figures that need raw images
(pipeline overview, qualitative panels, OOD failure, mitochondrial
generalisation) are produced by experiments/render_*.py, not here.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'experiments'))

BLUE, GREEN, GREY, RED, GOLD = '#1f5fa8', '#2e7d32', '#8a8a8a', '#b8322a', '#b8860b'
RAW = 256 * 256
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})


def load(path):
    p = os.path.join(path, 'metrics.csv')
    return pd.read_csv(p) if os.path.exists(p) else None


def self_iou(df):
    for c in ('self_iou', 'ng_recon_iou'):
        if c in df:
            return df[c]
    return df['ng_iou']


def save(fig, out, name):
    fig.tight_layout()
    fig.savefig(os.path.join(out, name), dpi=220)
    plt.close(fig)
    print('wrote', name)


def fig_codec(D, out):
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    ax[0].hist(D['ng_vs_raw'], 30, color=BLUE, alpha=.8, label='vs raw array')
    ax[0].hist(D['ng_vs_png'], 30, color=GREEN, alpha=.8, label='vs lossless PNG')
    ax[0].set_xlabel('payload reduction (×)'); ax[0].set_ylabel('images'); ax[0].legend(frameon=False)
    ax[0].set_title('(a) storage', loc='left')
    d = np.sort((D['gt_iou'] - D['jpeg_gt_iou']).to_numpy())
    ax[1].scatter(np.arange(len(d)), d, s=3, c=np.where(d > 0, GREEN, GREY))
    ax[1].axhline(0, color='k', lw=.5)
    ax[1].set_xlabel('images (sorted)'); ax[1].set_ylabel('GT-IoU − JPEG GT-IoU')
    ax[1].set_title('(b) per-image margin vs. byte-matched JPEG', loc='left')
    labs, vals, cols = [], [], []
    for lab, a, b, c in [('GT-IoU\nvs JPEG', 'gt_iou', 'jpeg_gt_iou', BLUE),
                         ('Self-IoU\nvs JPEG', None, 'jpeg_iou', GREEN),
                         ('Self-IoU\nvs WebP', None, 'webp_iou', GREEN),
                         ('Self-IoU\nvs J2K', None, 'jp2_iou', GREEN)]:
        x = D[a] if a else self_iou(D)
        if b not in D:
            continue
        m = x.notna() & D[b].notna()
        labs.append(lab); vals.append(100 * (x[m] > D[b][m]).mean()); cols.append(c)
    ax[2].bar(labs, vals, color=cols); ax[2].axhline(50, ls='--', color='k', lw=.6)
    ax[2].set_ylabel('images where Nanograph wins (%)'); ax[2].set_ylim(0, 100)
    ax[2].set_title('(c) win rates at matched bytes', loc='left')
    save(fig, out, 'fig_codec_rd.png')


def fig_seg(C, D, L, out):
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    ax[0].scatter(C['seg_recall'], C['seg_precision'], s=4, color=GOLD, label='classical cascade')
    if D is not None:
        ax[0].scatter(D['seg_recall'], D['seg_precision'], s=4, color=BLUE, label='default')
    ax[0].set_xlabel('recall'); ax[0].set_ylabel('precision'); ax[0].set_xlim(0, 1.02); ax[0].set_ylim(0, 1.02)
    ax[0].legend(frameon=False, markerscale=3); ax[0].set_title('(a) precision / recall', loc='left')
    ax[1].scatter(C['seg_iou'], C['gt_iou'], s=4, color=GOLD)
    ax[1].plot([0, 1], [0, 1], 'k--', lw=.6)
    ax[1].set_xlabel('Seg-IoU (mask vs. GT)'); ax[1].set_ylabel('GT-IoU (reconstruction vs. GT)')
    ax[1].set_title('(b) decoder as a shape prior (classical)', loc='left')
    bins = np.linspace(0, 1, 41)
    for df, lab, c in [(C, 'classical cascade', GOLD), (D, 'default', BLUE), (L, 'learned (replace)', GREEN)]:
        if df is not None:
            ax[2].hist(df['seg_iou'], bins, alpha=.6, color=c, label=f"{lab} ({df['seg_iou'].mean():.3f})")
    ax[2].set_xlabel('in-pipeline Seg-IoU'); ax[2].set_ylabel('images'); ax[2].legend(frameon=False)
    ax[2].set_title('(c) front ends, paired runs', loc='left')
    save(fig, out, 'fig_seg_bottleneck.png')


def fig_topology(D, out):
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    b0 = D['ng_seg_beta_0'] if 'ng_seg_beta_0' in D else D['graph_n_components']
    ax[0].hist(b0, bins=np.arange(b0.max() + 2) - .5, color=BLUE)
    ax[0].set_xlabel('$\\beta_0$ (components)'); ax[0].set_ylabel('images'); ax[0].set_title('(a)', loc='left')
    ax[1].hist(D['graph_n_cycles'], 30, color=RED)
    ax[1].set_xlabel('skeleton-graph cycle rank'); ax[1].set_title('(b)', loc='left')
    s = ax[2].scatter(D['mean_width_px'], D['total_edge_length'], c=b0, s=5, cmap='viridis')
    ax[2].set_xlabel('mean width (px)'); ax[2].set_ylabel('total network length (px)')
    fig.colorbar(s, ax=ax[2], label='$\\beta_0$'); ax[2].set_title('(c)', loc='left')
    save(fig, out, 'fig_topology.png')


CROSS = [('cells3d_membrane', 'Cells3D\nmembrane'), ('cells3d_nuclei', 'Cells3D\nnuclei'),
         ('retina', 'Retina\nvessels'), ('cell', 'Cells\n(fluor.)')]


def fig_cross(D, runs, out):
    sets = [('Organelles', D)] + [(lab, load(os.path.join(runs, 'cross', k))) for k, lab in CROSS]
    sets = [(l, d) for l, d in sets if d is not None]
    cols = ['#1f77b4', '#2ca02c', '#ffbf00', '#ff5722', '#9c27b0']
    panels = [('ng_fg_ssim', 'FG-SSIM'), ('ng_psnr', 'Full PSNR (dB)'), ('ng_bytes', 'Payload (bytes)'),
              ('ng_fg_psnr', 'FG-PSNR (dB)'), ('graph_n_components', 'Graph components'),
              ('graph_n_cycles', 'Skeleton-graph cycles')]
    fig, ax = plt.subplots(2, 3, figsize=(12, 6))
    for a, (c, lab) in zip(ax.ravel(), panels):
        m = [d[c].mean() for _, d in sets]; s = [d[c].std() for _, d in sets]
        a.bar(range(len(sets)), m, yerr=s, color=cols[:len(sets)], capsize=3)
        a.set_xticks(range(len(sets))); a.set_xticklabels([l for l, _ in sets], fontsize=7)
        a.set_ylabel(lab)
        if c in ('ng_bytes', 'graph_n_cycles'):
            a.set_yscale('log')
    save(fig, out, 'cross_dataset_comparison.png')
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for (l, d), c in zip(sets, cols):
        ax[0].scatter(d['graph_n_cycles'] + 1, d['ng_bytes'], s=5, color=c, label=l.replace('\n', ' '))
        ax[1].boxplot(d['ng_fg_ssim'], positions=[len(ax[1].lines) // 7], widths=.6)
    ax[0].set_xscale('log'); ax[0].set_yscale('log')
    ax[0].set_xlabel('skeleton-graph cycle rank + 1'); ax[0].set_ylabel('payload (bytes)')
    ax[0].legend(frameon=False, fontsize=7, markerscale=3); ax[0].set_title('(a)', loc='left')
    ax[1].set_xticks(range(len(sets))); ax[1].set_xticklabels([l for l, _ in sets], fontsize=7)
    ax[1].set_ylabel('FG-SSIM'); ax[1].set_title('(b)', loc='left')
    save(fig, out, 'fig_modality_scaling.png')


PRIOR = [('Skeleton', .0161, 2.0, (-10, -12)), ('Canny edge', .0203, 2.0, (6, -12)),
         ('Boundary', .0295, 1.3, (6, -4)), ('GA-CIR', .0100, 4.0, (6, 3)),
         ('GU-Net / GU-Net++', .0183, 2.0, (-20, 9))]


def fig_prior(D, out):
    aprr = (D['n_nodes'] / RAW).mean(); brr = (D['ng_bytes'] / RAW).mean(); bpp = brr / aprr
    fig, ax = plt.subplots(figsize=(5.2, 4))
    ax.tick_params(labelsize=8)
    x = np.logspace(-3, -1.3, 100)
    for b in (0.02, 0.04, 0.08):
        ax.plot(x, b / x, color=GREY, lw=.6, ls=':')
        ax.text(x[0] * 1.05, b / x[0], f'BRR={b}', fontsize=7, color=GREY, va='bottom', ha='left')
    for i, (n, a, p, off) in enumerate(PRIOR):
        c = RED if i >= 3 else GREY
        ax.scatter(a, p, color=c, s=30); ax.annotate(n, (a, p), fontsize=7, xytext=off, textcoords='offset points')
    ax.scatter(aprr, bpp, color=BLUE, s=60, zorder=3)
    ax.annotate('Nanograph', (aprr, bpp), fontsize=8, color=BLUE, xytext=(5, 3), textcoords='offset points')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('APRR (stored primitives per pixel)'); ax.set_ylabel('bytes per primitive')
    save(fig, out, 'fig_prior_positioning.png')


def fig_perturb(runs, out):
    p = os.path.join(runs, 'seg_perturbation_ablation.csv')
    if not os.path.exists(p):
        return
    d = pd.read_csv(p)
    b = d[d.kind == 'baseline'][['filename', 'beta0', 'beta1', 'mean_width', 'seg_iou']]
    m = d[d.kind != 'baseline'].merge(b, on='filename', suffixes=('', '_b'))
    m['b0chg'] = (m.beta0 != m.beta0_b) * 100
    m['ab1'] = (m.beta1 - m.beta1_b).abs()
    m['w'] = (m.mean_width / m.mean_width_b - 1) * 100
    arms = [('dilate', 'Dilation (1–5 px)'), ('erode', 'Erosion (1–5 px)'),
            ('drop_components', 'Component removal (5–40%)'), ('boundary_noise', 'Boundary noise (1–3 px band)')]
    rows = [('b0chg', 'images with\n$\\beta_0$ changed (%)', 'linear'),
            ('ab1', 'median $|\\Delta\\beta_1|$\n(cycles)', 'symlog'),
            ('w', 'median change in\nmean width (%)', 'linear')]
    op = b.seg_iou.mean()
    fig, ax = plt.subplots(3, 4, figsize=(13, 7.2), sharex='col', sharey='row')
    for j, (k, t) in enumerate(arms):
        g = m[m.kind == k].groupby('level').agg(seg=('seg_iou', 'median'), b0chg=('b0chg', 'mean'),
                                                 ab1=('ab1', 'median'), w=('w', 'median')).reset_index()
        for i, (c, lab, sc) in enumerate(rows):
            a = ax[i, j]
            a.plot(g.seg, g[c], 'o-', color=[BLUE, RED, GREEN][i])
            a.axvline(op, ls=':', color=GREY)
            a.set_xlim(0.95, -0.03 if k == 'erode' else min(g.seg.min() - 0.05, 0.3))
            if sc == 'symlog':
                a.set_yscale('symlog', linthresh=1)
            if c == 'w':
                a.axhline(0, color='k', lw=.5)
            if j == 0:
                a.set_ylabel(lab)
            if i == 0:
                a.set_title(t, fontsize=10)
            if i == 2:
                a.set_xlabel('Seg-IoU of degraded mask vs. GT')
            a.grid(alpha=.3)
    save(fig, out, 'fig_seg_perturb.png')


DS_LABELS = [('n_components', 'components'), ('total_length_px', 'total length'),
             ('mean_width_px', 'mean width'), ('n_branches', 'branches'),
             ('n_junctions', 'junctions'), ('cycle_rank', 'cycle rank')]


DS_SETTING = ('degree', 'auto')   # paper default: degree junctions, one-diameter rule


def fig_downstream(runs, out):
    """Descriptors read from the stored graph vs pixels (REF) vs byte-matched JPEG,
    at the paper's analysis setting DS_SETTING."""
    import json
    import downstream_morphometry as dm
    jd, Ls = DS_SETTING
    d = os.path.join(runs, 'downstream')
    p = os.path.join(d, 'per_image.csv')
    if not os.path.exists(p):
        print('skip fig_downstream (no downstream/per_image.csv)')
        return
    df = pd.read_csv(p, dtype={'stem': str})
    S = pd.read_csv(os.path.join(d, 'summary.csv'))
    if 'L' in df:
        df = df[(df.L.astype(str) == str(Ls)) & (df.get('junction_def', 't10') == jd)]
        S = S[(S.L.astype(str) == str(Ls)) & (S.get('junction_def', 't10') == jd)]
    W = {a: g.set_index('stem') for a, g in df.groupby('arm')}
    stems = sorted(set(W['REF'].index) & set(W['GRAPH'].index) & set(W['JPEG'].index))
    R, G, J = (W[a].loc[stems] for a in ('REF', 'GRAPH', 'JPEG'))
    arms = [('GRAPH', G, BLUE, 'o'), ('JPEG', J, GOLD, '^')]

    fig, ax = plt.subplots(1, 4, figsize=(15, 3.6))
    a = ax[0]
    for lab, X, c, m in arms:
        a.scatter(R.total_length_px, X.total_length_px, s=8, c=c, marker=m, alpha=.55,
                  lw=0, label=lab)
    lo, hi = 20, np.nanmax([R.total_length_px.max(), G.total_length_px.max(),
                            J.total_length_px.max()]) * 1.2
    a.plot([lo, hi], [lo, hi], color='k', lw=.7)
    a.set_xscale('log'); a.set_yscale('log')   # JPEG failures reach ~7x REF
    a.set_xlim(lo, hi); a.set_ylim(lo, hi)
    a.set_xlabel('REF total length (px)'); a.set_ylabel('arm total length (px)')
    a.legend(frameon=False, markerscale=2)
    a.set_title('(a) total length', loc='left')

    a = ax[1]
    for lab, X, c, m in arms:
        ok = X.mean_width_px.notna() & R.mean_width_px.notna()
        x, y = X.mean_width_px[ok], R.mean_width_px[ok]
        dd = x - y
        a.scatter((x + y) / 2, dd, s=8, c=c, marker=m, alpha=.55, lw=0, label=lab)
        b, sd = dd.mean(), dd.std()
        a.axhline(b, color=c, lw=1.2)
        for s in (-1.96, 1.96):
            a.axhline(b + s * sd, color=c, lw=.8, ls='--')
    a.axhline(0, color='k', lw=.5)
    a.set_xlabel('mean of arm and REF width (px)'); a.set_ylabel('arm − REF width (px)')
    a.legend(frameon=False, markerscale=2)
    a.set_title('(b) Bland–Altman, mean width', loc='left')

    a = ax[2]
    x = np.arange(len(DS_LABELS))
    for k, (lab, _, c, _) in enumerate(arms):
        v = [S[(S.descriptor == dk) & (S.arm == lab) & (S.ref == 'REF')].mdape.iloc[0]
             for dk, _ in DS_LABELS]
        bars = a.bar(x + (k - .5) * .38, v, .36, color=c, label=lab)
        a.bar_label(bars, fmt='%.0f', fontsize=7, padding=1, color='#333333')
    a.set_xticks(x); a.set_xticklabels([l for _, l in DS_LABELS], rotation=35, ha='right')
    a.set_ylabel('median |arm − REF| / REF (%)')
    a.legend(frameon=False)
    a.set_title('(c) MdAPE vs REF', loc='left')

    # (d) the image whose GRAPH-vs-REF Wasserstein distance is the median one
    a = ax[3]
    bd = pd.read_csv(os.path.join(d, 'branch_distances.csv'), dtype={'stem': str})
    if 'L' in bd:
        bd = bd[(bd.L.astype(str) == str(Ls)) & (bd.get('junction_def', 't10') == jd)]
    bd = bd.dropna(subset=['w1_GRAPH', 'w1_JPEG'])
    stem = bd.iloc[(bd.w1_GRAPH - bd.w1_GRAPH.median()).abs().argsort().iloc[0]].stem
    L = {}
    for arm in ('REF', 'GRAPH', 'JPEG'):
        with open(os.path.join(d, 'branch_lengths', f'{stem}_{arm}.json')) as f:
            js = json.load(f)
            L[arm] = (dm.descriptors_at(js, Ls, jd)[1] if 'branches' in js
                      else js['branch_lengths'])
    bins = np.linspace(0, max(max(v) for v in L.values() if v), 16)
    for arm, c in (('REF', GREY), ('GRAPH', BLUE), ('JPEG', GOLD)):
        a.hist(L[arm], bins, histtype='step', lw=1.6, color=c, label=f'{arm} (n={len(L[arm])})')
    a.set_xlabel('branch length (px)'); a.set_ylabel('branches')
    a.legend(frameon=False)
    a.set_title(f'(d) branch lengths, image {stem}', loc='left')
    save(fig, out, 'fig_downstream.png')


def _csv(path, **kw):
    for q in (path, path + '.gz'):
        if os.path.exists(q):
            return pd.read_csv(q, **kw)
    return None


def fig_layers(runs, out):
    """Bytes per stored object (organelle set, means) on a log axis."""
    import glob
    import struct
    D = load(os.path.join(runs, 'org_default'))
    sb, pb = [], []
    for f in glob.glob(os.path.join(runs, 'downstream', 'cache', '*.npz')):
        p = np.load(f)['payload_tagged'].tobytes()
        if p and p[0] == 7:
            sb.append(struct.unpack('<I', p[1:5])[0])
            pb.append(len(p))
    V = _csv(os.path.join(os.path.dirname(runs.rstrip('/')), 'v7', 'per_image.csv.gz'))
    if D is None or not sb or V is None:
        print('skip fig_layers'); return
    mask = V[(V.arm == 'SEG') & (V.L == 0)].bytes.mean()
    items = [('raw array', RAW, GREY), ('lossless PNG', D.png_bytes.mean(), GREY),
             ('byte-matched JPEG', D.jpeg_bytes.mean(), GOLD),
             ('Nanograph: structure + appearance', np.mean(pb), BLUE),
             ('lossless mask (bit-packed)', mask, GREY),
             ('Nanograph: structure layer', np.mean(sb), GREEN)]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    y = np.arange(len(items))[::-1]
    for yi, (lab, v, c) in zip(y, items):
        ax.barh(yi, v, color=c, height=0.62)
        ax.text(v * 1.08, yi, f'{v:,.0f} B', va='center', fontsize=8.5)
    ax.set_yticks(y); ax.set_yticklabels([i[0] for i in items])
    ax.set_xscale('log'); ax.set_xlim(80, RAW * 4)
    ax.set_xlabel('bytes per 256×256 image (mean over the organelle set)')
    ax.set_title('What each stored object costs', loc='left')
    save(fig, out, 'fig_layers.png')


TRUTH_DESC = [('n_components', 'components'), ('total_length_px', 'length'), ('mean_width_px', 'width'),
              ('n_branches', 'branches'), ('n_junctions', 'junctions')]


def fig_truth(runs, out):
    """Against the simulator's geometry: raw bias (a) and calibrated MdAPE (b)."""
    S = _csv(os.path.join(runs, 'sim_truth', 'summary.csv'))
    C = _csv(os.path.join(runs, 'sim_truth', 'calibrated.csv'))
    if S is None or C is None:
        print('skip fig_truth'); return
    S['L'] = S.L.astype(str); C['L'] = C.L.astype(str)
    arms = [('REF', "simulator's own mask", GREY), ('GRAPH7', 'Nanograph (stored graph)', BLUE),
            ('JPEG', 'byte-matched JPEG', GOLD)]
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
    x = np.arange(len(TRUTH_DESC)); w = 0.26
    for k, (arm, lab, c) in enumerate(arms):
        r = S[(S.arm == arm) & (S.L == 'auto')].iloc[0]
        ax[0].bar(x + (k - 1) * w, [r[f'{d}_bias_pct'] for d, _ in TRUTH_DESC], w, color=c, label=lab)
        v = [C[(C.L == 'auto') & (C.arm == arm) & (C.descriptor == d)].iloc[0].mdape_calibrated
             for d, _ in TRUTH_DESC]
        ax[1].bar(x + (k - 1) * w, v, w, color=c, label=lab)
    for a in ax:
        a.set_xticks(x); a.set_xticklabels([l for _, l in TRUTH_DESC])
    ax[0].axhline(0, color='k', lw=.6)
    ax[0].set_ylabel('bias vs true geometry (%)')
    ax[0].set_title('(a) raw bias against the simulated tubes', loc='left')
    ax[1].set_ylabel('median |error| after calibration (%)')
    ax[1].set_title('(b) after 2-fold linear calibration', loc='left')
    ax[1].legend(frameon=False, fontsize=8.5)
    save(fig, out, 'fig_truth.png')


def fig_clip(runs, out):
    """Temporal clip: components and length per frame, truth vs stored graph."""
    P = _csv(os.path.join(runs, 'clip_truth', 'per_image.csv'))
    if P is None:
        print('skip fig_clip'); return
    P['L'] = P.L.astype(str)
    P['frame'] = P.img.astype(str).str.replace('clip:', '', regex=False).astype(int)
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.4), sharex=True)
    for j, (col, lab) in enumerate([('n_components', 'mitochondria counted'), ('total_length_px', 'total length (px)')]):
        t = P[(P.arm == 'TRUE') & (P.L == '0')].sort_values('frame')
        ax[j].plot(t.frame, t[col], color='k', lw=1.8, label='truth')
        for L, c, ls, name in (('0', RED, '-', 'stored graph, no cleanup'), ('auto', BLUE, '-', 'stored graph, one-diameter rule')):
            g = P[(P.arm == 'GRAPH7') & (P.L == L)].sort_values('frame')
            ax[j].plot(g.frame, g[col], color=c, lw=1.3, ls=ls, label=name)
        ax[j].set_ylabel(lab); ax[j].set_xlabel('frame')
    ax[0].legend(frameon=False, fontsize=8.5)
    ax[0].set_title('(a) fragmentation counted as extra mitochondria', loc='left')
    ax[1].set_title('(b) length through time', loc='left')
    save(fig, out, 'fig_clip.png')


def fig_real(runs, out):
    """Real annotated mitochondria: segmentation (a); analysis-only bytes (b)."""
    names = [('UIT', 'UiT rat'), ('CBMI', 'CBMI'), ('MITO', 'MITO'), ('HUMAN', 'UiT human\n(untouched)')]
    R = _csv(os.path.join(runs, 'real', 'downstream', 'summary.csv'))
    Q = _csv(os.path.join(runs, 'real', 'downstream', 'jpeg_low_quality.csv'))
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
    x = np.arange(len(names))
    for k, (pre, lab, c) in enumerate([('default', 'simulation-trained', GREY), ('real-mito', 'real-trained', GREEN)]):
        v = []
        for d, _ in names:
            m = load(os.path.join(runs, 'real', pre, d))
            v.append(m.seg_iou.mean() if m is not None else np.nan)
        ax[0].bar(x + (k - .5) * .38, v, .36, color=c, label=lab)
    ax[0].set_xticks(x); ax[0].set_xticklabels([n for _, n in names])
    ax[0].set_ylabel('Seg-IoU vs expert mask'); ax[0].legend(frameon=False, fontsize=8.5)
    ax[0].set_title('(a) segmentation inside the pipeline', loc='left')
    if R is not None and Q is not None:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'experiments'))
        import downstream_morphometry as dm
        cols = [BLUE, GOLD, RED, GREEN]
        for (d, lab), c in zip(names, cols):
            h = Q[Q.dataset == d].groupby('q')
            b = h.bytes.mean()
            cc = h.apply(lambda g: dm.ccc(g.n_branches.to_numpy(float), g.n_branches_ref.to_numpy(float)))
            ax[1].plot(b.values, cc.values, 'o-', color=c, ms=4, lw=1.2, label=f'{lab.splitlines()[0]}: JPEG q1-q20')
            r = R[(R.dataset == d) & (R.L.astype(str) == 'auto') & (R.arm == 'GRAPH7')]
            if len(r):
                ax[1].plot(r.iloc[0].structure_bytes, r.iloc[0].n_branches_ccc, marker='*', ms=13, color=c,
                           markeredgecolor='k', ls='none')
        ax[1].set_xlabel('bytes stored for analysis'); ax[1].set_ylabel('branch-count CCC vs expert')
        ax[1].set_title('(b) structure layer (stars) vs JPEG (lines)', loc='left')
        ax[1].legend(frameon=False, fontsize=7.5, loc='lower right')
    save(fig, out, 'fig_real.png')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', default='results/paper')
    ap.add_argument('--out', default='paper/figures')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    D, C, L = (load(os.path.join(a.runs, k)) for k in ('org_default', 'org_classical', 'org_replace'))
    fig_codec(D, a.out)
    fig_seg(C, D, L, a.out)
    fig_topology(D, a.out)
    fig_cross(D, a.runs, a.out)
    fig_prior(D, a.out)
    fig_perturb(a.runs, a.out)
    fig_downstream(a.runs, a.out)
    fig_layers(a.runs, a.out)
    fig_truth(a.runs, a.out)
    fig_clip(a.runs, a.out)
    fig_real(a.runs, a.out)


if __name__ == '__main__':
    main()
