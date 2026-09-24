#!/usr/bin/env python3
"""Per-image diff of every run_dataset metrics.csv between two results trees.

Usage (from repo root):
    python experiments/diff_runs.py --old OLD/results/paper --new results/paper
For each run dir: images compared, images with any numeric column changed
beyond rounding (|d| > 1e-6 * max(1, |old|)), and the columns that changed.
Timing columns are excluded (they measure the machine, not the result).
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

SKIP = {'total_time_ms'}


def diff(old, new):
    o = pd.read_csv(old).set_index('filename')
    n = pd.read_csv(new).set_index('filename')
    idx = o.index.intersection(n.index)
    cols = [c for c in o.columns if c in n.columns and c not in SKIP
            and pd.api.types.is_numeric_dtype(o[c]) and pd.api.types.is_numeric_dtype(n[c])]
    a, b = o.loc[idx, cols].astype(float), n.loc[idx, cols].astype(float)
    both_nan = a.isna() & b.isna()
    changed = ~both_nan & ((a - b).abs() > 1e-6 * np.maximum(1, a.abs())).fillna(True)
    per_img = changed.any(axis=1)
    per_col = changed.sum(axis=0)
    return len(idx), int(per_img.sum()), per_col[per_col > 0].sort_values(ascending=False), \
        per_img[per_img].index.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--old', required=True)
    ap.add_argument('--new', default='results/paper')
    a = ap.parse_args()
    print('| run | images | changed | top changed columns (n images) |\n|---|---|---|---|')
    for p in sorted(glob.glob(os.path.join(a.new, '**', 'metrics.csv'), recursive=True)):
        rel = os.path.relpath(p, a.new)
        q = os.path.join(a.old, rel)
        if not os.path.exists(q):
            continue
        try:
            n, k, cols, _ = diff(q, p)
        except KeyError:                 # not a run_dataset metrics.csv
            continue
        top = ', '.join(f'{c} ({v})' for c, v in cols.head(6).items()) or '—'
        print(f'| {os.path.dirname(rel)} | {n} | {k} | {top} |')


if __name__ == '__main__':
    main()
