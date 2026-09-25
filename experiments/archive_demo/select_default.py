#!/usr/bin/env python3
"""
Apply results/allen/SELECTION_RULE.md mechanically.

Inputs
  <realbench>/eval/summary.csv        benchmarks 1-4 (method x dataset: cldice, junc_f1, missing_frac, sec_per_tile)
  results/allen/seg_eval/summary.csv  benchmark 5 (segmenter: cldice = mean per cell, junc_f1)
  results/allen/preference.csv        benchmark 6 (method, win_rate), when it exists

Per benchmark: rank by clDice (higher better); candidates within 0.01 are
ordered by junction F1; a candidate missing, or failing on > 5 % of images,
takes the last rank. Candidates absent from a benchmark are listed, and the
decision is marked PROVISIONAL until every benchmark and every candidate is in.
Writes results/allen/selection.md.
"""
import os

import numpy as np
import pandas as pd

RB = '/mnt/nas1/nba055-2/idea_1/archive_demo/realbench/eval/summary.csv'
AL = 'results/allen/seg_eval/summary.csv'
PREF = 'results/allen/preference.csv'
OUT = 'results/allen/selection.md'
CANDIDATES = ['unet_sim', 'real_mito', 'allen_ft', 'nnunet_real', 'nnunet', 'microsam_zs', 'microsam_real',
              'microsam_ft', 'nellie', 'nellie_tuned_real', 'nellie_tuned_allen']
IN_DOMAIN = {('real_mito', 'UIT'), ('real_mito', 'CBMI'), ('real_mito', 'MITO'), ('nnunet_real', 'UIT'),
             ('nnunet_real', 'CBMI'), ('nnunet_real', 'MITO'), ('microsam_real', 'UIT'), ('microsam_real', 'CBMI'),
             ('microsam_real', 'MITO'), ('nellie_tuned_real', 'UIT'), ('nellie_tuned_real', 'CBMI'),
             ('nellie_tuned_real', 'MITO'), ('allen_ft', 'ALLEN'), ('nnunet', 'ALLEN'), ('microsam_ft', 'ALLEN'),
             ('nellie_tuned_allen', 'ALLEN')}


def rank(df):
    """df: method, score, tie (junc_f1), failed -> rank per method (1 = best)."""
    ok = df[~df.failed].sort_values('score', ascending=False).reset_index(drop=True)
    order = []
    i = 0
    while i < len(ok):                      # group candidates within 0.01 of the group's best
        j = i
        while j + 1 < len(ok) and ok.score[i] - ok.score[j + 1] <= 0.01:
            j += 1
        order += list(ok.iloc[i:j + 1].sort_values('tie', ascending=False).method)
        i = j + 1
    r = {m: k + 1 for k, m in enumerate(order)}
    for m in df[df.failed].method:
        r[m] = len(df)
    return r


TUNING = '/mnt/nas1/nba055-2/idea_1/archive_demo/realbench/nellie_tuning.json'
NELLIE_DEFAULT = 'otsu=False,min_r=0.25'


def aliases():
    """A tuned Nellie variant whose chosen setting IS the default is the same
    segmenter as `nellie`: merge it (identical masks must not take two ranks)."""
    import json
    if not os.path.exists(TUNING):
        return {}
    ch = json.load(open(TUNING))['chosen']
    return {f'nellie_tuned_{pool}': 'nellie' for pool, k in ch.items() if k == NELLIE_DEFAULT}


def main():
    benches, notes, speed = {}, [], {}
    alias = aliases()
    for a_, b_ in alias.items():
        notes.append(f'{a_}: tuning kept the default setting, so it is the same segmenter as {b_} (merged)')
    if os.path.exists(RB):
        S = pd.read_csv(RB)
        speed = S.groupby('method').sec_per_tile.median().to_dict()      # tie-break: seconds per tile
        for d, g in S.groupby('dataset'):
            benches[d] = pd.DataFrame({'method': g.method, 'score': g.cldice, 'tie': g.junc_f1,
                                       'failed': g.missing_frac > 0.05})
    else:
        notes.append('benchmarks 1-4 (real expert test sets): not run yet')
    if os.path.exists(AL):
        A = pd.read_csv(AL)
        A = A[A.segmenter != 'allen']
        benches['ALLEN'] = pd.DataFrame({'method': A.segmenter, 'score': A.cldice, 'tie': A.junc_f1, 'failed': False})
    else:
        notes.append('benchmark 5 (Allen held-out plates): not run yet')
    if os.path.exists(PREF):
        P = pd.read_csv(PREF)
        benches['PREFERENCE'] = pd.DataFrame({'method': P.method, 'score': P.win_rate, 'tie': 0.0, 'failed': False})
    else:
        notes.append('benchmark 6 (blinded preference): not run yet')
    benches = {b: df[~df.method.isin(alias)].reset_index(drop=True) for b, df in benches.items()}
    ranks = {b: rank(df) for b, df in benches.items()}
    present = sorted(set().union(*[set(df.method) for df in benches.values()])) if benches else []
    rows = []
    for m in present:
        rr = {b: ranks[b].get(m) for b in benches}
        missing = [b for b, v in rr.items() if v is None]
        for b in missing:                     # absent from a benchmark = last rank there
            rr[b] = len(present)
        rows.append({'method': m, **rr, 'mean_rank': np.mean(list(rr.values())), 'absent_from': ','.join(missing),
                     'sec': speed.get(m, np.inf)})
    T = pd.DataFrame(rows).sort_values(['mean_rank', 'sec']) if rows else pd.DataFrame()
    absent = [c for c in CANDIDATES if c not in present and c not in alias]
    final = not [n for n in notes if 'not run yet' in n] and not absent
    lines = ['# Default segmenter: selection (SELECTION_RULE.md applied mechanically)', '',
             f"**Status: {'FINAL' if final else 'PROVISIONAL'}**", '']
    lines += [f'- {n}' for n in notes]
    if absent:
        lines.append(f"- candidates not evaluated yet: {', '.join(absent)}")
    lines += ['', '## clDice per benchmark (* = in-domain for that candidate)', '']
    if benches:
        cols = list(benches)
        lines.append('| method | ' + ' | '.join(cols) + ' | mean rank | s/tile |')
        lines.append('|---|' + '---|' * (len(cols) + 2))
        for r in T.itertuples():
            cells = []
            for b in cols:
                df = benches[b]
                v = df[df.method == r.method].score
                s = f'{v.iloc[0]:.3f}' if len(v) else '–'
                cells.append(f'{s}{"*" if (r.method, b) in IN_DOMAIN else ""} (#{getattr(r, b)})')
            lines.append(f'| {r.method} | ' + ' | '.join(cells) + f' | {r.mean_rank:.2f} | {r.sec:.2f} |')
        lines += ['', 'Order: mean rank, then seconds per tile (SELECTION_RULE.md tie-break).',
                  '', f"Leader: **{T.iloc[0].method}** (mean rank {T.iloc[0].mean_rank:.2f})"
                  + ('' if final else ' - provisional')]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
