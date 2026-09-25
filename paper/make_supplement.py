#!/usr/bin/env python3
"""
Supplementary tables from the paper-v3 outputs (numbers only, no prose).

  tables/supp_downstream_L.tex   organelle downstream, GRAPH vs JPEG vs REF, at
                                 every analysis setting (L = 0/2/5/10 px and the
                                 one-diameter rule), degree-based junctions
  tables/supp_truth_L.tex        simulated truth: bias and CCC per route and setting
  tables/supp_builder.tex        v7 branch builder vs the v6 pixel builder (ablation),
                                 against the simulated truth (one-diameter rule)
  tables/supp_real.tex           real data per dataset: structure layer and
                                 byte-matched JPEG vs the expert masks
  supplement.tex                 a standalone wrapper that inputs the above

Missing inputs give a table row of [TBD] so the build never silently drops one.

Usage (from repo root):
    python paper/make_supplement.py --runs results/paper --out paper
"""
import argparse
import os

import pandas as pd

DESC = [('n_components', 'Components'), ('total_length_px', 'Total length'), ('mean_width_px', 'Mean width'),
        ('n_branches', 'Branches'), ('n_junctions', 'Junctions'), ('cycle_rank', 'Cycle rank')]
SETTINGS = [('0', '0 px'), ('2', '2 px'), ('5', '5 px'), ('10', '10 px'), ('auto', 'one diameter')]
TBD = r'\tbd'


def _csv(p):
    for q in (p, p + '.gz'):
        if os.path.exists(q):
            return pd.read_csv(q)
    return None


def f(v, fmt):
    return TBD if v is None or v != v else format(v, fmt)


def table(caption, label, colspec, header, body):
    return '\n'.join([r'\begin{table}[h]', r'\centering', rf'\caption{{{caption}}}', rf'\label{{{label}}}',
                      r'\footnotesize', r'\begin{adjustbox}{max width=\linewidth}', rf'\begin{{tabular}}{{{colspec}}}',
                      r'\toprule', header + r'\\', r'\midrule'] + body +
                     [r'\bottomrule', r'\end{tabular}', r'\end{adjustbox}', r'\end{table}', ''])


def downstream_L(runs):
    S = _csv(os.path.join(runs, 'downstream', 'summary.csv'))
    if S is not None:
        S = S[(S.junction_def == 'degree') & (S.ref == 'REF')]
        S['L'] = S.L.astype(str)
    body = []
    for L, ln in SETTINGS:
        for i, (d, dn) in enumerate(DESC):
            g = S[(S.L == L) & (S.descriptor == d)] if S is not None else None
            G = g[g.arm == 'GRAPH'] if g is not None else []
            J = g[g.arm == 'JPEG'] if g is not None else []
            gv = G.iloc[0] if len(G) else None
            jv = J.iloc[0] if len(J) else None
            wins = (f'{int(gv.wins_graph)}/{int(gv.wins_graph + gv.wins_jpeg + gv.ties)}'
                    if gv is not None and gv.wins_graph == gv.wins_graph else TBD)
            body.append(' & '.join([ln if i == 0 else '', dn,
                                    f(gv.ccc if gv is not None else None, '.3f'),
                                    f(gv.mdape if gv is not None else None, '.1f'),
                                    f(gv.bias_pct if gv is not None else None, '+.1f'),
                                    f(jv.ccc if jv is not None else None, '.3f'),
                                    f(jv.mdape if jv is not None else None, '.1f'),
                                    f(jv.bias_pct if jv is not None else None, '+.1f'), wins]) + r'\\')
        body.append(r'\midrule')
    return table('Organelle downstream analysis at every cleanup setting: structure layer (GRAPH) and '
                 'byte-matched JPEG route against the reference mask (726 images; degree-based junctions). '
                 'The main text uses the one-diameter rule.', 'tab:supp-ds-L', 'llccccccc',
                 r'Cleanup & Descriptor & \multicolumn{3}{c}{Structure layer: CCC / MdAPE / bias \%} & '
                 r'\multicolumn{3}{c}{JPEG: CCC / MdAPE / bias \%} & Graph closer', body[:-1])


def truth_L(runs):
    T = _csv(os.path.join(runs, 'sim_truth', 'summary.csv'))
    if T is not None:
        T['L'] = T.L.astype(str)
    arms = [('REF', 'simulator mask'), ('GRAPH7', 'structure layer'), ('JPEG', 'JPEG route')]
    body = []
    for L, ln in SETTINGS:
        for i, (arm, an) in enumerate(arms):
            r = T[(T.L == L) & (T.arm == arm)] if T is not None else []
            r = r.iloc[0] if len(r) else None
            cells = [ln if i == 0 else '', an]
            for d, _ in DESC:
                cells.append(f"{f(r[d + '_bias_pct'] if r is not None else None, '+.0f')} / "
                             f"{f(r[d + '_ccc'] if r is not None else None, '.2f')}")
            body.append(' & '.join(cells) + r'\\')
        body.append(r'\midrule')
    return table('Against the true simulated geometry at every cleanup setting: bias (\\%) / CCC per descriptor.',
                 'tab:supp-truth-L', 'll' + 'c' * len(DESC),
                 'Cleanup & Route & ' + ' & '.join(dn for _, dn in DESC), body[:-1])


def builder(runs):
    T = _csv(os.path.join(runs, 'sim_truth', 'summary.csv'))
    C = _csv(os.path.join(runs, 'sim_truth', 'calibrated.csv'))
    body = []
    for arm, an in (('GRAPH7', 'v7 branch builder (paper)'), ('GRAPH6', 'v6 pixel builder (ablation)')):
        r = T[(T.L.astype(str) == 'auto') & (T.arm == arm)] if T is not None else []
        r = r.iloc[0] if len(r) else None
        cells = [an, f(r['bytes'] if r is not None and 'bytes' in r else None, '.0f')]
        for d, _ in DESC:
            c = C[(C.L.astype(str) == 'auto') & (C.arm == arm) & (C.descriptor == d)] if C is not None else []
            cells.append(f"{f(r[d + '_bias_pct'] if r is not None else None, '+.0f')} / "
                         f"{f(c.iloc[0].mdape_calibrated if len(c) else None, '.1f')}")
        body.append(' & '.join(cells) + r'\\')
    return table('Graph-builder ablation against the true simulated geometry (one-diameter rule): raw bias (\\%) / '
                 'median error after two-fold calibration (\\%). Bytes: mean payload.', 'tab:supp-builder',
                 'lc' + 'c' * len(DESC), 'Builder & Bytes & ' + ' & '.join(dn for _, dn in DESC), body)


def real(runs):
    R = _csv(os.path.join(runs, 'real', 'downstream', 'summary.csv'))
    names = [('UIT', 'UiT rat'), ('CBMI', 'CBMI'), ('MITO', 'MITO'), ('HUMAN', 'UiT human')]
    body = []
    for ds, dn in names:
        for i, (arm, an) in enumerate((('GRAPH7', 'structure layer'), ('JPEG', 'byte-matched JPEG'))):
            r = R[(R.dataset == ds) & (R.L.astype(str) == 'auto') & (R.arm == arm)] if R is not None else []
            r = r.iloc[0] if len(r) else None
            cells = [dn if i == 0 else '', an, f(r['n'] if r is not None else None, '.0f'),
                     f((r['structure_bytes'] if arm == 'GRAPH7' else r['bytes']) if r is not None else None, '.0f')]
            for d, _ in DESC:
                cells.append(f"{f(r[d + '_bias_pct'] if r is not None else None, '+.0f')} / "
                             f"{f(r[d + '_ccc'] if r is not None else None, '.2f')}")
            body.append(' & '.join(cells) + r'\\')
        body.append(r'\midrule')
    return table('Real mitochondria, per test set (\\texttt{real-mito} front end, one-diameter rule): bias (\\%) / '
                 'CCC against the expert masks. Bytes: structure layer, or the full payload that the JPEG matches.',
                 'tab:supp-real', 'llcc' + 'c' * len(DESC),
                 'Dataset & Route & $n$ & Bytes & ' + ' & '.join(dn for _, dn in DESC), body[:-1])


WRAPPER = r"""% AUTO-GENERATED by paper/make_supplement.py
\documentclass[10pt,a4paper]{article}
\usepackage[margin=2cm]{geometry}
\usepackage{booktabs,adjustbox,xcolor}
\providecommand{\tbd}{\textcolor{red}{[TBD]}}
\begin{document}
\section*{Supplementary tables}
\input{tables/supp_downstream_L}
\input{tables/supp_truth_L}
\input{tables/supp_builder}
\input{tables/supp_real}
\end{document}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', default='results/paper')
    ap.add_argument('--out', default='paper')
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, 'tables'), exist_ok=True)
    for name, fn in (('supp_downstream_L', downstream_L), ('supp_truth_L', truth_L),
                     ('supp_builder', builder), ('supp_real', real)):
        tex = fn(a.runs)
        open(os.path.join(a.out, 'tables', name + '.tex'), 'w').write(tex)
        print(f'wrote tables/{name}.tex ({tex.count(TBD)} TBD cells)')
    open(os.path.join(a.out, 'supplement.tex'), 'w').write(WRAPPER)


if __name__ == '__main__':
    main()
