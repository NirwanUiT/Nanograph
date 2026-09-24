#!/usr/bin/env python3
"""Generate paper/numbers.tex and paper/tables/*.tex from evaluation outputs.

Every number in the manuscript that depends on an evaluation run is a macro
defined here; tables that depend on runs are generated as bodies. Re-running
this script after new evaluations updates the paper with no hand edits.

Usage (from repo root):
    python paper/make_numbers.py --runs results/paper --out paper
Expected layout under --runs (each dir holds run_dataset.py's metrics.csv):
    org_default/  org_classical/  org_replace/
    ablation_bg_residual.csv
    heldout_organelle.txt             (one filename stem per line, 108 lines)
    cross/{cells3d_membrane,cells3d_nuclei,retina,cell}/
    cross_gt/{stare,drive,epfl_mito,microtubules}/
    mito/{temporal_clip,temporal_clip_replace,sted,mito_mip}/
    polarity.csv                       (dataset,as_is,oracle)
    commit.txt                         (git hash the runs were made at)
Missing inputs produce the macro value \\tbd, which renders visibly.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'experiments'))

import numpy as np
import pandas as pd
from scipy.stats import binomtest

RAW_BYTES = 256 * 256  # organelle frames are 256x256 8-bit


def load(path):
    p = os.path.join(path, 'metrics.csv') if os.path.isdir(path) else path
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    df['stem'] = df['filename'].astype(str).map(lambda s: os.path.splitext(os.path.basename(s))[0])
    return df


def self_iou_col(df):
    for c in ('self_iou', 'ng_recon_iou'):
        if c in df:
            return c
    if 'ng_iou' in df and not np.allclose(df['ng_iou'], 1.0):
        return 'ng_iou'
    return None


class Macros:
    def __init__(self):
        self.lines = []
        self.d = {}

    def put(self, name, value):
        self.d[name] = str(value)
        self.lines.append(f'\\newcommand{{\\{name}}}{{{value}}}')

    def pm(self, name, s, d):
        if s is None or len(s.dropna()) == 0:
            self.put(name, '\\tbd'); self.put(name + 'm', '\\tbd'); return
        s = s.dropna()
        self.put(name, f'{s.mean():.{d}f}\\pm{s.std():.{d}f}')
        self.put(name + 'm', f'{s.mean():.{d}f}')

    def wins(self, name, a, b, boot=True):
        if a is None or b is None:
            for k in ('', 'Pct', 'P', 'CI'):
                self.put(name + k, '\\tbd')
            return
        m = a.notna() & b.notna()
        a, b = a[m].to_numpy(), b[m].to_numpy()
        w, n = int((a > b).sum()), len(a)
        p = binomtest(w, n).pvalue if n else float('nan')
        self.put(name, f'{w}/{n}')
        self.put(name + 'Pct', f'{100 * w / n:.1f}' if n else '\\tbd')
        self.put(name + 'P', fmt_p(p))
        if boot and n:
            rng = np.random.default_rng(0)
            d = a - b
            bs = [rng.choice(d, n).mean() for _ in range(10000)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            self.put(name + 'CI', f'[{lo:+.4f},{hi:+.4f}]')
        else:
            self.put(name + 'CI', '\\tbd')


def fmt_p(p):
    if p != p:
        return '\\tbd'
    if p < 1e-3:
        e = int(np.floor(np.log10(p)))
        return f'{p / 10 ** e:.1f}\\times10^{{{e}}}'
    return f'{p:.2f}'


ORG_METRICS = [  # macro suffix, column, decimals
    ('Bytes', 'ng_bytes', 0), ('Nodes', 'n_nodes', 0), ('Edges', 'n_edges', 0),
    ('Time', 'total_time_ms', 0), ('FGPSNR', 'ng_fg_psnr', 2), ('FGSSIM', 'ng_fg_ssim', 4),
    ('GTFGPSNR', 'gt_fg_psnr', 2), ('GTFGSSIM', 'gt_fg_ssim', 4), ('FullPSNR', 'ng_psnr', 2),
    ('FullSSIM', 'ng_ssim', 3), ('SegIoU', 'seg_iou', 3), ('SegPrec', 'seg_precision', 3),
    ('SegRec', 'seg_recall', 3), ('GTIoU', 'gt_iou', 3), ('BetaZero', 'ng_seg_beta_0', 2),
    ('BetaOne', 'ng_seg_beta_1', 2), ('Cycles', 'graph_n_cycles', 2), ('Width', 'mean_width_px', 2),
    ('VsRaw', 'ng_vs_raw', 1), ('VsPNG', 'ng_vs_png', 1),
    ('JpegBytes', 'jpeg_bytes', 0), ('JpegFullPSNR', 'jpeg_psnr', 2), ('JpegFullSSIM', 'jpeg_ssim', 3),
    ('JpegFGPSNR', 'jpeg_fg_psnr', 2), ('JpegGTFGPSNR', 'jpeg_gt_fg_psnr', 2), ('JpegGTIoU', 'jpeg_gt_iou', 3),
    ('JpegBetti', 'jpeg_betti_preservation', 3), ('JpegBetaZero', 'jpeg_beta_0', 2),
    ('JpegBetaOne', 'jpeg_beta_1', 2),
    ('WebpBytes', 'webp_bytes', 0), ('WebpFullPSNR', 'webp_psnr', 2), ('WebpFullSSIM', 'webp_ssim', 3),
    ('WebpFGPSNR', 'webp_fg_psnr', 2),
    ('JtwoBytes', 'jp2_bytes', 0), ('JtwoFullPSNR', 'jp2_psnr', 2), ('JtwoFullSSIM', 'jp2_ssim', 3),
    ('JtwoFGPSNR', 'jp2_fg_psnr', 2),
]


def organelle(M, tag, df, heldout):
    col = lambda c: df[c] if (df is not None and c in df) else None
    for suf, c, d in ORG_METRICS:
        M.pm(tag + suf, col(c), d)
    si = self_iou_col(df) if df is not None else None
    M.pm(tag + 'SelfIoU', col(si) if si else None, 3)
    M.put(tag + 'N', str(len(df)) if df is not None else '\\tbd')
    M.wins(tag + 'GTIoUWins', col('gt_iou'), col('jpeg_gt_iou'))
    M.wins(tag + 'GTFGPSNRWins', col('gt_fg_psnr'), col('jpeg_gt_fg_psnr'), boot=False)
    M.wins(tag + 'FGPSNRWins', col('ng_fg_psnr'), col('jpeg_fg_psnr'), boot=False)
    for cod, cc in (('Jpeg', 'jpeg_iou'), ('Webp', 'webp_iou'), ('Jtwo', 'jp2_iou')):
        M.wins(tag + 'SelfIoUWins' + cod, col(si) if si else None, col(cc), boot=(cod == 'Jpeg'))
    # held-out subset (images the organelle U-Net did not train on)
    if df is not None and heldout is not None:
        h = df[df['stem'].isin(heldout)]
        M.put(tag + 'HeldN', str(len(h)))
        M.pm(tag + 'HeldSegIoU', h['seg_iou'], 3)
        M.wins(tag + 'HeldGTIoUWins', h['gt_iou'], h['jpeg_gt_iou'], boot=False)
    else:
        for k in ('HeldN', 'HeldSegIoU', 'HeldSegIoUm', 'HeldGTIoUWins', 'HeldGTIoUWinsPct',
                  'HeldGTIoUWinsP', 'HeldGTIoUWinsCI'):
            M.put(tag + k, '\\tbd')
    # storage budget (Eq. budget): APRR = nodes/pixel, BRR = bytes/raw byte
    if df is not None:
        aprr = (df['n_nodes'] / RAW_BYTES).mean()
        brr = (df['ng_bytes'] / RAW_BYTES).mean()
        bpp = brr / aprr
        M.put(tag + 'APRR', f'{aprr:.5f}')
        M.put(tag + 'BRR', f'{brr:.4f}')
        M.put(tag + 'BPP', f'{bpp:.1f}')
        M.put(tag + 'FewerX', f'{0.0183 / aprr:.1f}')   # vs GU-Net/GU-Net++ APRR
        M.put(tag + 'RicherX', f'{bpp / 2.0:.1f}')       # vs their 2.0 bytes/primitive
        M.put(tag + 'BRRvsGU', f'{100 * (brr / 0.0365 - 1):+.0f}')
    else:
        for k in ('APRR', 'BRR', 'BPP', 'FewerX', 'RicherX', 'BRRvsGU'):
            M.put(tag + k, '\\tbd')


def ablation(M, path):
    keys = [('FGPSNR', 'fg_psnr', 2), ('FGSSIM', 'fg_ssim', 4), ('FullPSNR', 'full_psnr', 2),
            ('Bytes', 'bytes', 0)]
    if not os.path.exists(path):
        for k, _, _ in keys:
            for s in ('With', 'Without', 'Delta'):
                M.put(f'Abl{k}{s}', '\\tbd'); M.put(f'Abl{k}{s}m', '\\tbd')
        M.put('AblImproved', '\\tbd'); M.put('AblBytesRatio', '\\tbd'); return
    a = pd.read_csv(path)
    for k, c, d in keys:
        M.pm(f'Abl{k}With', a[f'{c}_with'], d)
        M.pm(f'Abl{k}Without', a[f'{c}_without'], d)
        M.pm(f'Abl{k}Delta', a[f'{c}_with'] - a[f'{c}_without'], d)
    r = a['bytes_with'] / a['bytes_without']
    M.put('AblBytesRatio', f'{r.mean():.2f}\\pm{r.std():.2f}')
    M.put('AblImproved', f"{int((a['fg_psnr_with'] > a['fg_psnr_without']).sum())}/{len(a)}")


CROSS = [('cells3d_membrane', 'Cells3D mem.'), ('cells3d_nuclei', 'Cells3D nuc.'),
         ('retina', 'Retina'), ('cell', 'Cells')]


def pmstr(s, d):
    s = s.dropna()
    if not len(s):
        return '\\tbd'
    f = (lambda v: f'{v:,.0f}'.replace(',', '{,}')) if d == 0 else (lambda v: f'{v:.{d}f}')
    return f'${f(s.mean())}\\pm{f(s.std())}$'


def cross_table(runs, org, out):
    dfs = [('Organelles', org)] + [(lab, load(os.path.join(runs, 'cross', k))) for k, lab in CROSS]
    rows = [('FG-SSIM', 'ng_fg_ssim', 3), ('FG-PSNR (dB)', 'ng_fg_psnr', 2),
            ('Full PSNR (dB)', 'ng_psnr', 2), ('Full SSIM', 'ng_ssim', 3),
            ('Bytes', 'ng_bytes', 0), ('Graph $\\beta_0$ (components)', 'graph_n_components', 2),
            ('Skeleton-graph cycles', 'graph_n_cycles', 1), ('Encode time (ms)', 'total_time_ms', 0)]
    L = [' & ' + ' & '.join(l for l, _ in dfs) + ' \\\\',
         '$N$ & ' + ' & '.join(str(len(d)) if d is not None else '\\tbd' for _, d in dfs) + ' \\\\',
         '\\midrule']
    for lab, c, d in rows:
        L.append(lab + ' & ' + ' & '.join(pmstr(x[c], d) if (x is not None and c in x) else '\\tbd'
                                           for _, x in dfs) + ' \\\\')
    open(os.path.join(out, 'tables', 'tab_cross_body.tex'), 'w').write('\n'.join(L) + '\n')


def topo_all_table(runs, org, out):
    sets = [('Organelles', org)] + [(lab, load(os.path.join(runs, 'cross', k))) for k, lab in CROSS]
    sets += [('STARE', load(os.path.join(runs, 'cross_gt', 'stare'))),
             ('DRIVE', load(os.path.join(runs, 'cross_gt', 'drive'))),
             ('EPFL mito (EM)', load(os.path.join(runs, 'cross_gt', 'epfl_mito'))),
             ('Microtubules', load(os.path.join(runs, 'cross_gt', 'microtubules'))),
             ('Temporal clip', load(os.path.join(runs, 'mito', 'temporal_clip'))),
             ('STED TOM20', load(os.path.join(runs, 'mito', 'sted'))),
             ('MITO (MIP tiles)', load(os.path.join(runs, 'mito', 'mito_mip')))]
    L = []
    for i, (lab, d) in enumerate(sets):
        if i in (5, 9):
            L.append('\\midrule')
        if d is None or 'jpeg_betti_preservation' not in d:
            L.append(f'{lab} & \\tbd & \\tbd & \\tbd & \\tbd \\\\'); continue
        m = d['jpeg_bytes'].notna() if 'jpeg_bytes' in d else np.ones(len(d), bool)
        d = d[m]
        star = '' if m.all() else '$^{*}$'
        b0 = 'ng_seg_beta_0' if 'ng_seg_beta_0' in d else 'graph_n_components'
        b1 = 'ng_seg_beta_1' if 'ng_seg_beta_1' in d else 'graph_n_cycles'
        L.append(f"{lab} & {len(d)}{star} & {pmstr(d['jpeg_betti_preservation'], 3)} & "
                 f"{d['jpeg_beta_0'].mean():.1f} / {d[b0].mean():.1f} & "
                 f"{d['jpeg_beta_1'].mean():.1f} / {d[b1].mean():.1f} \\\\")
    open(os.path.join(out, 'tables', 'tab_topoall_body.tex'), 'w').write('\n'.join(L) + '\n')


def crossgt(M, runs):
    for k, tag in (('stare', 'Stare'), ('drive', 'Drive'), ('epfl_mito', 'Epfl'), ('microtubules', 'Mt')):
        d = load(os.path.join(runs, 'cross_gt', k))
        if d is None:
            for s in ('N', 'SegIoUm', 'GTFGPSNRm', 'JpegGTFGPSNRm', 'Bytesm', 'NoJpeg', 'Timem'):
                M.put('Cg' + tag + s, '\\tbd')
            continue
        M.put('Cg' + tag + 'N', str(len(d)))
        M.put('Cg' + tag + 'SegIoUm', f"{d['seg_iou'].mean():.2f}")
        m = d['jpeg_bytes'].notna()
        M.put('Cg' + tag + 'GTFGPSNRm', f"{d.loc[m, 'gt_fg_psnr'].mean():.1f}")
        M.put('Cg' + tag + 'JpegGTFGPSNRm', f"{d.loc[m, 'jpeg_gt_fg_psnr'].mean():.1f}")
        M.put('Cg' + tag + 'Bytesm', f"{d['ng_bytes'].mean() / 1000:.1f}")
        M.put('Cg' + tag + 'NoJpeg', f"{int((~m).sum())}/{len(d)}")
        M.put('Cg' + tag + 'Timem', f"{d['total_time_ms'].mean() / 1000:.1f}")


def mito(M, runs, out):
    spec = [('temporal_clip', 'Temporal clip (same acq.), default', 'Clip'),
            ('temporal_clip_replace', 'Temporal clip (same acq.), learned', 'ClipL'),
            ('sted', 'STED TOM20~\\cite{balakrishnan2024sted}, learned', 'Sted'),
            ('mito_mip', 'MITO confocal~\\cite{zhanghao2023mito}, default', 'Mito')]
    L = []
    for k, lab, tag in spec:
        d = load(os.path.join(runs, 'mito', k))
        if d is None:
            L.append(f'{lab} & \\tbd & \\tbd & \\tbd & \\tbd \\\\')
            for s in ('SegIoUm', 'GTIoUm', 'Nodesm', 'Bytesm', 'Timem', 'Prec', 'Rec'):
                M.put(tag + s, '\\tbd')
            continue
        seg = pmstr(d['seg_iou'], 3) if 'seg_iou' in d else '--- (no GT)'
        L.append(f"{lab} & {len(d)} & {seg} & {pmstr(d['ng_fg_psnr'], 1)} & {pmstr(d['ng_bytes'], 0)} \\\\")
        M.put(tag + 'SegIoUm', f"{d['seg_iou'].mean():.3f}" if 'seg_iou' in d else '--')
        M.put(tag + 'GTIoUm', f"{d['gt_iou'].mean():.3f}" if 'gt_iou' in d else '--')
        M.put(tag + 'Prec', f"{d['seg_precision'].mean():.2f}" if 'seg_precision' in d else '--')
        M.put(tag + 'Rec', f"{d['seg_recall'].mean():.2f}" if 'seg_recall' in d else '--')
        M.put(tag + 'Nodesm', f"{d['n_nodes'].mean():,.0f}".replace(',', '{,}'))
        M.put(tag + 'Bytesm', f"{d['ng_bytes'].mean() / 1000:.1f}")
        M.put(tag + 'Timem', f"{d['total_time_ms'].mean() / 1000:.0f}")
    open(os.path.join(out, 'tables', 'tab_mitogen_body.tex'), 'w').write('\n'.join(L) + '\n')


def polarity(M, runs):
    p = os.path.join(runs, 'polarity.csv')
    df = pd.read_csv(p) if os.path.exists(p) else None
    for k, tag in (('stare', 'Stare'), ('drive', 'Drive'), ('microtubules', 'Mt'), ('epfl', 'Epfl')):
        r = df[df['dataset'] == k] if df is not None else []
        if len(r):
            M.put('Pol' + tag + 'AsIs', f"{r['as_is'].iloc[0]:.3f}")
            M.put('Pol' + tag + 'Oracle', f"{r['oracle'].iloc[0]:.3f}")
        else:
            M.put('Pol' + tag + 'AsIs', '\\tbd'); M.put('Pol' + tag + 'Oracle', '\\tbd')


DS_DESC = [('n_components', 'NComponents'), ('total_length_px', 'TotalLength'),
           ('mean_width_px', 'MeanWidth'), ('n_branches', 'NBranches'),
           ('n_junctions', 'NJunctions'), ('cycle_rank', 'CycleRank')]
DS_ARMS = [('GRAPH', 'Graph'), ('JPEG', 'Jpeg'), ('PRE', 'Pre')]


def downstream(M, runs):
    """T10 macros: \\Ds<Arm><Descriptor>{CCC,MdAPE,Bias,BiasPct} (vs REF),
    \\DsWins<Descriptor> (GRAPH closer to REF than JPEG), \\DsTime*, \\DsWass*."""
    d = os.path.join(runs, 'downstream')
    sp, cp = os.path.join(d, 'summary.csv'), os.path.join(d, 'cost_summary.csv')
    S = pd.read_csv(sp) if os.path.exists(sp) else None
    if S is not None and 'L' in S:
        S = S[(S.L.astype(str) == '0') & (S.get('junction_def', 't10') == 't10')]   # T10 definition
    C = pd.read_csv(cp, index_col=0)['value'] if os.path.exists(cp) else None
    for dk, dn in DS_DESC:
        for arm, an in DS_ARMS:
            r = S[(S.descriptor == dk) & (S.arm == arm) & (S.ref == 'REF')] if S is not None else []
            if len(r):
                r = r.iloc[0]
                M.put(f'Ds{an}{dn}CCC', f'{r.ccc:.3f}')
                M.put(f'Ds{an}{dn}MdAPE', f'{r.mdape:.1f}')
                M.put(f'Ds{an}{dn}Bias', f'{r.bias:+.2f}')
                M.put(f'Ds{an}{dn}BiasPct', f'{r.bias_pct:+.1f}')
            else:
                for k in ('CCC', 'MdAPE', 'Bias', 'BiasPct'):
                    M.put(f'Ds{an}{dn}{k}', '\\tbd')
        r = S[(S.descriptor == dk) & (S.arm == 'GRAPH') & (S.ref == 'REF')] if S is not None else []
        if len(r):
            r = r.iloc[0]
            n = int(r.wins_graph + r.wins_jpeg + r.ties)
            M.put(f'DsWins{dn}', f'{int(r.wins_graph)}/{n}')
            M.put(f'DsLosses{dn}', f'{int(r.wins_jpeg)}/{n}')
            M.put(f'DsWinsP{dn}', fmt_p(r.wilcoxon_p))
        else:
            for k in ('DsWins', 'DsLosses', 'DsWinsP'):
                M.put(f'{k}{dn}', '\\tbd')
    for arm, an in (('GRAPH', 'Graph'), ('JPEG', 'Jpeg')):
        r = S[(S.descriptor == 'branch_lengths') & (S.arm == arm)] if S is not None else []
        M.put(f'DsWass{an}', f'{r.iloc[0].w1_mean:.2f}' if len(r) else '\\tbd')
        M.put(f'DsKS{an}', f'{r.iloc[0].ks_mean:.3f}' if len(r) else '\\tbd')
    if C is not None:
        M.put('DsN', f"{int(float(C['n_common']))}")
        M.put('DsTimeGraph', f"{1000 * float(C['graph_s_median']):.1f}")
        M.put('DsTimeJpeg', f"{1000 * float(C['jpeg_s_median']):.0f}")
        M.put('DsTimeRef', f"{1000 * float(C['ref_s_median']):.1f}")
        M.put('DsTimeRatio', f"{float(C['ratio_jpeg_over_graph']):.0f}")
        M.put('DsTimeRatioRef', f"{float(C['ratio_ref_over_graph']):.1f}")
    else:
        for k in ('DsN', 'DsTimeGraph', 'DsTimeJpeg', 'DsTimeRef', 'DsTimeRatio', 'DsTimeRatioRef'):
            M.put(k, '\\tbd')
    downstream_pruned(M, runs)


DS_PRUNE_L = 'auto'   # T13: the one-diameter analysis rule is the paper's default


def downstream_pruned(M, runs):
    """T11.2 macros at the chosen shared pruning length (degree-based junctions):
    \\DsP<Arm><Descriptor>{CCC,MdAPE,BiasPct}, \\DsPWins<Descriptor>/\\DsPLosses/\\DsPWinsP,
    \\DsPWass{Graph,Jpeg}, \\DsPruneL, \\DsRefSpurPct, \\DsRefSpurLenPct."""
    d = os.path.join(runs, 'downstream')
    sp, fp = os.path.join(d, 'summary.csv'), os.path.join(d, 'spur_fractions.csv')
    S = pd.read_csv(sp) if os.path.exists(sp) else None
    if S is not None and 'junction_def' in S:
        S = S[(S.junction_def == 'degree') & (S.L.astype(str) == str(DS_PRUNE_L))]
    else:
        S = None
    M.put('DsPruneL', 'one diameter' if DS_PRUNE_L == 'auto' else str(DS_PRUNE_L))
    for dk, dn in DS_DESC:
        for arm, an in DS_ARMS:
            r = S[(S.descriptor == dk) & (S.arm == arm) & (S.ref == 'REF')] if S is not None else []
            for k, col, fmt in (('CCC', 'ccc', '.3f'), ('MdAPE', 'mdape', '.1f'),
                                ('BiasPct', 'bias_pct', '+.1f')):
                M.put(f'DsP{an}{dn}{k}', format(r.iloc[0][col], fmt) if len(r) else '\\tbd')
        r = S[(S.descriptor == dk) & (S.arm == 'GRAPH') & (S.ref == 'REF')] if S is not None else []
        if len(r):
            r = r.iloc[0]
            n = int(r.wins_graph + r.wins_jpeg + r.ties)
            M.put(f'DsPWins{dn}', f'{int(r.wins_graph)}/{n}')
            M.put(f'DsPLosses{dn}', f'{int(r.wins_jpeg)}/{n}')
            M.put(f'DsPWinsP{dn}', fmt_p(r.wilcoxon_p))
        else:
            for k in ('DsPWins', 'DsPLosses', 'DsPWinsP'):
                M.put(f'{k}{dn}', '\\tbd')
    for arm, an in (('GRAPH', 'Graph'), ('JPEG', 'Jpeg')):
        r = S[(S.descriptor == 'branch_lengths') & (S.arm == arm)] if S is not None else []
        M.put(f'DsPWass{an}', f'{r.iloc[0].w1_mean:.2f}' if len(r) else '\\tbd')
    F = pd.read_csv(fp) if os.path.exists(fp) else None
    r = F[(F.arm == 'REF') & (F.L.astype(str) == '5')] if F is not None else []   # spur share at 5 px
    M.put('DsRefSpurPct', f'{100 * r.iloc[0].branch_frac_pooled:.1f}' if len(r) else '\\tbd')
    M.put('DsRefSpurLenPct', f'{100 * r.iloc[0].length_frac_mean:.1f}' if len(r) else '\\tbd')


def _csv(path, **kw):
    for q in (path, path + '.gz'):
        if os.path.exists(q):
            try:
                return pd.read_csv(q, **kw)
            except Exception:
                return None
    return None


def v7_sections(M, runs):
    """T13 macros: storage layers, stored-graph topology, truth-based and
    real-data results. Any missing input renders as \\tbd."""
    import glob
    import struct
    TBD = '\\tbd'
    put = lambda k, v, f: M.put(k, TBD if v is None or v != v else format(v, f))
    # --- storage layers (organelle payloads cached by the downstream run)
    sb, pb = [], []
    for f in glob.glob(os.path.join(runs, 'downstream', 'cache', '*.npz')):
        p = np.load(f)['payload_tagged'].tobytes()
        if p and p[0] == 7:
            sb.append(struct.unpack('<I', p[1:5])[0])
            pb.append(len(p))
    put('StructBytesm', np.mean(sb) if sb else None, '.0f')
    put('AppearBytesm', np.mean(pb) - np.mean(sb) if sb else None, '.0f')
    put('StructFracPct', 100 * np.mean(sb) / np.mean(pb) if sb else None, '.0f')
    V = _csv(os.path.join(os.path.dirname(runs.rstrip('/')), 'v7', 'per_image.csv.gz'))
    mb = V[(V.arm == 'SEG') & (V.L == 0)].bytes.mean() if V is not None else None
    put('MaskLosslessBytesm', mb, '.0f')
    put('StructVsMaskX', mb / np.mean(sb) if (sb and mb) else None, '.1f')
    # --- stored-graph topology vs the mask it encodes
    D = load(os.path.join(runs, 'org_default'))
    if D is not None and 'graph_n_components' in D:
        put('GraphCompEqPct', 100 * (D.graph_n_components == D.ng_seg_beta_0).mean(), '.1f')
        put('GraphCycEqPct', 100 * (D.graph_n_cycles == D.ng_seg_beta_1).mean(), '.1f')
    else:
        put('GraphCompEqPct', None, ''); put('GraphCycEqPct', None, '')
    # --- simulated truth (one-diameter rule)
    T = _csv(os.path.join(runs, 'sim_truth', 'summary.csv'))
    P = _csv(os.path.join(runs, 'sim_truth', 'per_image.csv'))
    C = _csv(os.path.join(runs, 'sim_truth', 'calibrated.csv'))
    arms = (('REF', 'Ref'), ('GRAPH7', 'Graph'), ('JPEG', 'Jpeg'))
    desc = (('n_components', 'Comp'), ('total_length_px', 'Len'), ('mean_width_px', 'Width'),
            ('n_branches', 'Br'), ('n_junctions', 'Junc'), ('cycle_rank', 'Cyc'))
    if T is not None:
        T['L'] = T.L.astype(str)
    for arm, an in arms:
        r = T[(T.arm == arm) & (T.L == 'auto')] if T is not None else []
        for d, dn in desc:
            put(f'Truth{an}{dn}Bias', r.iloc[0][f'{d}_bias_pct'] if len(r) else None, '+.0f')
            put(f'Truth{an}{dn}CCC', r.iloc[0][f'{d}_ccc'] if len(r) else None, '.2f')
        put(f'Truth{an}JuncFone', r.iloc[0]['junc_f1_3'] if len(r) else None, '.2f')
    if P is not None:
        P['L'] = P.L.astype(str)
        tr = P[(P.arm == 'TRUE') & (P.L == '0')]
        put('TruthN', len(tr), 'd')
        put('TruthTubesm', tr.n_tubes.mean(), '.1f')
        put('TruthCompm', tr.n_components.mean(), '.1f')
        put('TruthJuncm', tr.n_junctions.mean(), '.1f')
    else:
        for k in ('TruthN', 'TruthTubesm', 'TruthCompm', 'TruthJuncm'):
            put(k, None, '')
    if C is not None:
        C['L'] = C.L.astype(str)
    for d, dn in desc:
        for arm, an in arms:
            r = C[(C.L == 'auto') & (C.descriptor == d) & (C.arm == arm)] if C is not None else []
            put(f'Cal{an}{dn}', r.iloc[0].mdape_calibrated if len(r) else None, '.1f')
        r = C[(C.L == 'auto') & (C.descriptor == d) & (C.arm == 'GRAPH7')] if C is not None else []
        if len(r):
            M.put(f'CalWins{dn}', f'{int(r.iloc[0].closer_than_jpeg)}/{int(r.iloc[0].jpeg_closer)}')
            M.put(f'CalP{dn}', fmt_p(r.iloc[0].p))
        else:
            M.put(f'CalWins{dn}', TBD); M.put(f'CalP{dn}', TBD)
    # --- temporal clip truth: fixed L=0 vs one-diameter rule
    K = _csv(os.path.join(runs, 'clip_truth', 'summary.csv'))
    if K is not None:
        K['L'] = K.L.astype(str)
    for lab, L in (('Zero', '0'), ('Auto', 'auto')):
        r = K[(K.arm == 'GRAPH7') & (K.L == L)] if K is not None else []
        put(f'ClipTruthComp{lab}', r.iloc[0].n_components_bias_pct if len(r) else None, '+.0f')
        put(f'ClipTruthBr{lab}', r.iloc[0].n_branches_bias_pct if len(r) else None, '+.0f')
        put(f'ClipTruthLen{lab}', r.iloc[0].total_length_px_bias_pct if len(r) else None, '+.0f')
        put(f'ClipTruthJf{lab}', r.iloc[0].junc_f1_3 if len(r) else None, '.2f')
    # --- real data
    names = (('UIT', 'Uit'), ('CBMI', 'Cbmi'), ('MITO', 'Mito'), ('HUMAN', 'Human'))
    for d, dn in names:
        for pre, pn in (('default', 'Sim'), ('real-mito', 'Real')):
            m = load(os.path.join(runs, 'real', pre, d))
            put(f'RealSegIoU{pn}{dn}', m.seg_iou.mean() if m is not None else None, '.2f')
    R = _csv(os.path.join(runs, 'real', 'downstream', 'summary.csv'))
    Q = _csv(os.path.join(runs, 'real', 'downstream', 'jpeg_low_quality.csv'))
    for d, dn in names:
        r = R[(R.dataset == d) & (R.L.astype(str) == 'auto') & (R.arm == 'GRAPH7')] if R is not None else []
        put(f'RealStructBytes{dn}', r.iloc[0].structure_bytes if len(r) else None, '.0f')
        put(f'RealBrCCC{dn}', r.iloc[0].n_branches_ccc if len(r) else None, '.2f')
        put(f'RealLenCCC{dn}', r.iloc[0].total_length_px_ccc if len(r) else None, '.2f')
        for q, qn in ((1, 'Qone'), (20, 'Qtwenty')):
            h = Q[(Q.dataset == d) & (Q.q == q)] if Q is not None else []
            if len(h):
                import downstream_morphometry as dm
                put(f'RealJpeg{qn}Bytes{dn}', h.bytes.mean(), '.0f')
                put(f'RealJpeg{qn}BrCCC{dn}', dm.ccc(h.n_branches.to_numpy(float), h.n_branches_ref.to_numpy(float)), '.2f')
                put(f'RealJpeg{qn}LenCCC{dn}', dm.ccc(h.total_length_px.to_numpy(float),
                                                     h.total_length_px_ref.to_numpy(float)), '.2f')
            else:
                for k in ('Bytes', 'BrCCC', 'LenCCC'):
                    put(f'RealJpeg{qn}{k}{dn}', None, '')
    sbv = [float(M.d[f'RealStructBytes{dn}']) for _, dn in names if M.d[f'RealStructBytes{dn}'] != TBD]
    put('RealStructBytesMin', min(sbv) if sbv else None, '.0f')
    put('RealStructBytesMax', max(sbv) if sbv else None, '.0f')
    rat = [float(M.d[f'RealJpegQtwentyBytes{dn}']) / float(M.d[f'RealStructBytes{dn}'])
           for _, dn in names if TBD not in (M.d[f'RealJpegQtwentyBytes{dn}'], M.d[f'RealStructBytes{dn}'])]
    put('RealJpegRatioMin', min(rat) if rat else None, '.1f')
    put('RealJpegRatioMax', max(rat) if rat else None, '.1f')
    E = _csv(os.path.join(runs, 'real', 'segmenters', 'eval_summary.csv'))
    for fam, fn in (('shipped_sim', 'Sim'), ('ALL_joint', 'Real')):
        g = E[(E.family == fam) & (E.dataset == 'HUMAN')] if E is not None else []
        put(f'RealHumanBrBias{fn}', g.n_branches_bias_pct.mean() if len(g) else None, '+.0f')
        put(f'RealHumanJuncBias{fn}', g.n_junctions_bias_pct.mean() if len(g) else None, '+.0f')
        put(f'RealHumanBrCCC{fn}', g.n_branches_ccc.mean() if len(g) else None, '.2f')
        put(f'RealHumanIoU{fn}', g.iou.mean() if len(g) else None, '.2f')


def perturbation(M, runs):
    """Perturbation-study macros (\\Pert*), from seg_perturbation_ablation.csv."""
    d = _csv(os.path.join(runs, 'seg_perturbation_ablation.csv'))
    keys = ['PertN', 'PertBaseSegIoU', 'PertBaseComp', 'PertBaseCyc', 'PertDilOneBz', 'PertDilFiveBz',
            'PertDilFiveSeg', 'PertDilOneW', 'PertDilFiveW', 'PertDilCycLo', 'PertDilCycHi',
            'PertEroOneBz', 'PertEroBzLo', 'PertEroBzHi', 'PertEroWLo', 'PertEroWHi',
            'PertNoiseOneBz', 'PertNoiseOneCyc', 'PertNoiseThreeCyc', 'PertNoiseW',
            'PertDropW', 'PertDropCyc']
    if d is None:
        for k in keys:
            M.put(k, '\\tbd')
        return
    b = d[d.kind == 'baseline'][['filename', 'beta0', 'beta1', 'mean_width', 'seg_iou']]
    m = d[d.kind != 'baseline'].merge(b, on='filename', suffixes=('', '_b'))
    m['bz'] = (m.beta0 != m.beta0_b) * 100
    m['ab1'] = (m.beta1 - m.beta1_b).abs()
    m['w'] = (m.mean_width / m.mean_width_b.where(m.mean_width_b > 0) - 1) * 100
    g = m.groupby(['kind', 'level']).agg(bz=('bz', 'mean'), ab1=('ab1', 'median'), w=('w', 'median'),
                                          seg=('seg_iou', 'median'))
    G = lambda k, l, c: g.loc[(k, float(l)), c]
    M.put('PertN', f'{len(m):,}'.replace(',', '{,}'))
    M.put('PertBaseSegIoU', f'{b.seg_iou.mean():.3f}')
    M.put('PertBaseComp', f'{b.beta0.median():.0f}')
    M.put('PertBaseCyc', f'{b.beta1.median():.0f}')
    M.put('PertDilOneBz', f"{G('dilate', 1, 'bz'):.0f}")
    M.put('PertDilFiveBz', f"{G('dilate', 5, 'bz'):.0f}")
    M.put('PertDilFiveSeg', f"{G('dilate', 5, 'seg'):.2f}")
    M.put('PertDilOneW', f"{G('dilate', 1, 'w'):+.0f}")
    M.put('PertDilFiveW', f"{G('dilate', 5, 'w'):+.0f}")
    dc = [G('dilate', l, 'ab1') for l in range(1, 6)]
    M.put('PertDilCycLo', f'{min(dc):.0f}'); M.put('PertDilCycHi', f'{max(dc):.0f}')
    M.put('PertEroOneBz', f"{G('erode', 1, 'bz'):.0f}")
    eb = [G('erode', l, 'bz') for l in range(2, 6)]
    ew = [G('erode', l, 'w') for l in range(2, 6)]
    M.put('PertEroBzLo', f'{min(eb):.0f}'); M.put('PertEroBzHi', f'{max(eb):.0f}')
    M.put('PertEroWLo', f'{min(ew, key=abs):+.0f}'); M.put('PertEroWHi', f'{max(ew, key=abs):+.0f}')
    M.put('PertNoiseOneBz', f"{G('boundary_noise', 1, 'bz'):.0f}")
    M.put('PertNoiseOneCyc', f"{G('boundary_noise', 1, 'ab1'):.0f}")
    M.put('PertNoiseThreeCyc', f"{G('boundary_noise', 3, 'ab1'):.0f}")
    M.put('PertNoiseW', f"{np.median([G('boundary_noise', l, 'w') for l in (1, 2, 3)]):+.0f}")
    dw = [G('drop_components', l, 'w') for l in (0.05, 0.1, 0.2, 0.3, 0.4)]
    dy = [G('drop_components', l, 'ab1') for l in (0.05, 0.1, 0.2, 0.3, 0.4)]
    M.put('PertDropW', f'{max(abs(x) for x in dw):.0f}')
    M.put('PertDropCyc', f'{max(dy):.0f}')


REAL_FAMILIES = [('shipped_sim', 'Simulation only (shipped default)'),
                 ('ALL_scratch', 'Real, from scratch'),
                 ('ALL_finetune', 'Real, fine-tuned from simulation'),
                 ('ALL_joint', 'Real + simulation, joint (\\texttt{real-mito})')]


def real_seg_table(runs, out):
    """tables/tab_realseg_body.tex: standalone test-set IoU per training family
    (mean over seeds) and, on the untouched Human set, branch-count bias/CCC;
    last row: each dataset left out of joint training (LOO_<d>_joint)."""
    E = _csv(os.path.join(runs, 'real', 'segmenters', 'eval_summary.csv'))
    ds = ['UIT', 'CBMI', 'MITO', 'HUMAN']
    lines = []
    f = lambda v, fmt: '\\tbd' if v is None or v != v else format(v, fmt)
    for fam, lab in REAL_FAMILIES:
        g = E[E.family == fam] if E is not None else None
        iou = [g[g.dataset == d].iou.mean() if g is not None else None for d in ds]
        h = g[g.dataset == 'HUMAN'] if g is not None else None
        lines.append(' & '.join([lab] + [f(v, '.2f') for v in iou] +
                                [f(h.n_branches_bias_pct.mean() if h is not None else None, '+.0f') + '\\%',
                                 f(h.n_branches_ccc.mean() if h is not None else None, '.2f')]) + r'\\')
    loo = [E[(E.family == f'LOO_{d}_joint') & (E.dataset == d)].iou.mean() if E is not None and d != 'HUMAN'
           else None for d in ds]
    lines.append(' & '.join(['Joint, test dataset left out'] + [f(v, '.2f') for v in loo[:3]] + ['--', '--', '--'])
                 + r'\\')
    open(os.path.join(out, 'tables', 'tab_realseg_body.tex'), 'w').write('\n'.join(lines) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', default='results/paper')
    ap.add_argument('--out', default='paper')
    ap.add_argument('--org-default', default=None, help='override org_default dir')
    ap.add_argument('--org-classical', default=None)
    ap.add_argument('--org-replace', default=None)
    ap.add_argument('--ablation', default=None)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, 'tables'), exist_ok=True)
    held_p = os.path.join(a.runs, 'heldout_organelle.txt')
    held = set(open(held_p).read().split()) if os.path.exists(held_p) else None
    M = Macros()
    commit = os.path.join(a.runs, 'commit.txt')
    M.put('RunCommit', open(commit).read().strip()[:7] if os.path.exists(commit) else '\\tbd')
    D = load(a.org_default or os.path.join(a.runs, 'org_default'))
    C = load(a.org_classical or os.path.join(a.runs, 'org_classical'))
    Lr = load(a.org_replace or os.path.join(a.runs, 'org_replace'))
    organelle(M, 'D', D, held)
    organelle(M, 'C', C, held)
    organelle(M, 'L', Lr, held)
    ablation(M, a.ablation or os.path.join(a.runs, 'ablation_bg_residual.csv'))
    crossgt(M, a.runs)
    mito(M, a.runs, a.out)
    polarity(M, a.runs)
    downstream(M, a.runs)
    v7_sections(M, a.runs)
    perturbation(M, a.runs)
    real_seg_table(a.runs, a.out)
    cross_table(a.runs, D, a.out)
    topo_all_table(a.runs, D, a.out)
    hdr = ('% AUTO-GENERATED by paper/make_numbers.py -- do not edit by hand.\n'
           '\\providecommand{\\tbd}{\\textcolor{red}{[TBD]}}\n')
    open(os.path.join(a.out, 'numbers.tex'), 'w').write(hdr + '\n'.join(M.lines) + '\n')
    print(f'wrote {a.out}/numbers.tex ({len(M.lines)} macros) and tables/')


if __name__ == '__main__':
    main()
