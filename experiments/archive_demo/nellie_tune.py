#!/usr/bin/env python3
"""
Tuned-Nellie variants (SELECTION_RULE.md addendum): choose ONE Nellie setting
per training pool on training images only, then predict every benchmark with it.

Pools
  real   UiT-Rat, CBMI, MITO training splits (60 seeded tiles each; pixel size
         = per-dataset calibration from real_bench.py)
  allen  60 seeded Allen training cells, prepared exactly like the test cells
         (unmasked slab, prediction and reference restricted to the cell)
Grid: otsu_thresh_intensity in {False, True} x min_radius_um in {0.15, 0.25, 0.35}.
Score: mean clDice against the pool's reference masks.

Outputs
  <realbench>/nellie_tuning.json                     chosen settings and grid scores
  <realbench>/preds/nellie_tuned_{real,allen}/<D>/   real test tiles
  <allen>/preds/nellie_tuned_{real,allen}/           Allen test cells

Runs in the base env; Nellie itself runs in its venv through allen_nellie.py.
"""
import itertools
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
PY = '/var/tmp/nba055/venvs/nellie/bin/python'
REAL = '/mnt/nas1/nba055-2/idea_1/real_mito'
RB = '/mnt/nas1/nba055-2/idea_1/archive_demo/realbench'
AL = '/mnt/nas1/nba055-2/idea_1/archive_demo/allen'
GRID = list(itertools.product([False, True], [0.15, 0.25, 0.35]))
ALLEN_UM = 0.108333
ENV = dict(os.environ, TMPDIR='/var/tmp/nba055/tmp')


def nellie(img_dir, out, um, otsu, r, ids=None, workers=6, label='X', timing=None):
    cmd = [PY, os.path.join(HERE, 'allen_nellie.py'), '--img-dir', img_dir, '--out', out, '--um', str(um),
           '--min-radius-um', str(r), '--workers', str(workers), '--label', label]
    if otsu:
        cmd.append('--otsu')
    if ids:
        cmd += ['--ids', ','.join(ids)]
    if timing:
        cmd += ['--timing', timing]
    subprocess.run(cmd, check=True, env=ENV, stdout=subprocess.DEVNULL)


def score(pred_dir, ref_dir, ids, cell_dir=None):
    import cv2
    from allen_eval_seg import cldice
    s = []
    for i in ids:
        p = cv2.imread(os.path.join(pred_dir, f'{i}.png'), cv2.IMREAD_GRAYSCALE)
        r = cv2.imread(os.path.join(ref_dir, f'{i}.png'), cv2.IMREAD_GRAYSCALE) > 0
        p = (p > 0) if p is not None else np.zeros_like(r)
        if cell_dir:
            c = cv2.imread(os.path.join(cell_dir, f'{i}.png'), cv2.IMREAD_GRAYSCALE) > 0
            p, r = p & c, r & c
        s.append(cldice(p, r))
    return float(np.mean(s))


def main():
    import pandas as pd
    px = json.load(open(f'{RB}/pixel_sizes.json'))
    rng = np.random.default_rng(0)
    M = pd.read_csv(f'{REAL}/manifest.csv')
    real_ids = {d: list(rng.choice(sorted(M[(M.dataset == d) & (M.split == 'train')].id), 60, replace=False))
                for d in ('UIT', 'CBMI', 'MITO')}
    # Allen training cells, prepared like the test cells
    C = pd.read_csv(f'{AL}/trainset/cells.csv')
    cells = sorted(rng.choice(C[C.split == 'train'].CellId, 60, replace=False))
    tune_dir = f'{AL}/tune_train'
    if not os.path.exists(f'{tune_dir}/cells.csv'):
        subprocess.run([sys.executable, os.path.join(HERE, 'allen_export_test.py'), '--out', tune_dir,
                        '--cells', ','.join(map(str, cells))], check=True, env=dict(ENV, CUDA_VISIBLE_DEVICES=''))
    grid = {'real': {}, 'allen': {}}
    for otsu, r in GRID:
        key = f'otsu={otsu},min_r={r}'
        sc = []
        for d, ids in real_ids.items():
            out = f'{RB}/tune/{key}/{d}'
            nellie(f'{REAL}/{d}/images', out, px[d]['um_per_px'], otsu, r, ids)
            sc.append(score(out, f'{REAL}/{d}/masks', ids))
        grid['real'][key] = {'per_dataset': dict(zip(real_ids, sc)), 'mean': float(np.mean(sc))}
        out = f'{AL}/tune/{key}'
        nellie(f'{tune_dir}/img_raw', out, ALLEN_UM, otsu, r)
        ids = [str(c) for c in cells]
        grid['allen'][key] = {'mean': score(out, f'{tune_dir}/allen', ids, f'{tune_dir}/cell')}
        print(key, 'real', round(grid['real'][key]['mean'], 3), 'allen', round(grid['allen'][key]['mean'], 3),
              flush=True)
    best = {pool: max(grid[pool], key=lambda k: grid[pool][k]['mean']) for pool in grid}
    json.dump({'grid': grid, 'chosen': best}, open(f'{RB}/nellie_tuning.json', 'w'), indent=1)
    print('chosen:', best)
    for pool, key in best.items():
        otsu = key.split(',')[0].split('=')[1] == 'True'
        r = float(key.split('=')[-1])
        name = f'nellie_tuned_{pool}'
        for d in ('UIT', 'CBMI', 'MITO', 'HUMAN'):
            nellie(f'{RB}/{d}/img_raw', f'{RB}/preds/{name}/{d}', px[d]['um_per_px'], otsu, r,
                   label=d, timing=f'{RB}/preds/{name}/timing.csv')
        nellie(f'{AL}/testset/img_raw', f'{AL}/preds/{name}', ALLEN_UM, otsu, r)


if __name__ == '__main__':
    main()
