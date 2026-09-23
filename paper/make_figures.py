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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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


if __name__ == '__main__':
    main()
