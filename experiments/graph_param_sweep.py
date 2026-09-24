#!/usr/bin/env python3
"""
T11.3 — is the graph-construction under-count a tunable default?

One-at-a-time sweep of the graph-construction parameters on a fixed random
subset of organelle images (seed 0). For every setting: encode, decode the
graph from the payload, compute the T10 descriptors (at every shared pruning
length L of T11.2) and compare with REF (annotation skeleton); record payload
bytes and FG-PSNR of the decoded render. Shipped defaults are not changed.

Usage (from repo root):
    CUDA_VISIBLE_DEVICES= python experiments/graph_param_sweep.py \
        --n 150 --workers 12 --out results/paper/graph_sweep
"""
import argparse
import contextlib
import glob
import io
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# (setting name, parameter, value); the default appears once as 'default'.
SWEEP = [('default', None, None)]
SWEEP += [(f'spur_min_length={v}', 'graph.spur_min_length', v) for v in (0, 2, 10)]
SWEEP += [('bridge=off', 'graph.bridge_gaps', False)]
SWEEP += [(f'bridge_max_gap={v}', 'graph.bridge_max_gap', v) for v in (6.0, 20.0)]
SWEEP += [(f'min_component_nodes={v}', 'graph.min_component_nodes', v) for v in (0, 6, 10)]
SWEEP += [(f'spacing={v}', 'detect.sparse_spacing', v) for v in (2, 5, 8)]
DEFAULTS = {'graph.spur_min_length': 5, 'graph.bridge_gaps': True, 'graph.bridge_max_gap': 12.0,
            'graph.min_component_nodes': 3, 'detect.sparse_spacing': 3}


def _one(job):
    stem, img_dir, setting, param, value = job
    import cv2
    from nanograph_v4 import nanograph_encode, NanographConfig
    import downstream_morphometry as dm
    cfg = NanographConfig()
    if param:
        sec, key = param.split('.')
        setattr(getattr(cfg, sec), key, value)
    img = cv2.imread(os.path.join(img_dir, stem + '.png'), cv2.IMREAD_GRAYSCALE)
    with contextlib.redirect_stdout(io.StringIO()):
        r = nanograph_encode(img, verbose=False, config=cfg)
    table = dm.graph_arm_table(*dm._decoded_graph_arrays(r.compressed))
    rows = []
    for jd, L in dm.SETTINGS:
        d, _, _ = dm.descriptors_at(table, L, jd)
        rows.append({'stem': stem, 'setting': setting, 'param': param or '',
                     'value': value, 'junction_def': jd, 'L': L, 'payload_bytes': len(r.compressed),
                     'fg_psnr': r.psnr_fg, 'n_points': r.n_points, **d})
    return rows


def _ref(job):
    stem, mask_dir = job
    import cv2
    import downstream_morphometry as dm
    gt = cv2.imread(os.path.join(mask_dir, stem + '.png'), cv2.IMREAD_GRAYSCALE)
    table = dm.pixel_arm_table(gt)
    return [{'stem': stem, 'junction_def': jd, 'L': L, **dm.descriptors_at(table, L, jd)[0]}
            for jd, L in dm.SETTINGS]


def main():
    import multiprocessing as mp
    import pandas as pd
    import downstream_morphometry as dm
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--images', default='/mnt/nas1/nba055-2/idea_1/nmi_data/org')
    ap.add_argument('--masks', default='/mnt/nas1/nba055-2/idea_1/nmi_data/seg')
    ap.add_argument('--n', type=int, default=150)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--out', default='results/paper/graph_sweep')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    stems = sorted(os.path.splitext(os.path.basename(p))[0]
                   for p in glob.glob(os.path.join(a.images, '*.png')))
    stems = sorted(np.random.default_rng(0).choice(stems, a.n, replace=False).tolist())
    open(os.path.join(a.out, 'subset.txt'), 'w').write('\n'.join(stems) + '\n')

    ctx = mp.get_context('spawn')
    jobs = [(s, a.images, name, p, v) for name, p, v in SWEEP for s in stems]
    rows = []
    with ctx.Pool(a.workers) as pool:
        ref = [r for rs in pool.map(_ref, [(s, a.masks) for s in stems]) for r in rs]
        for i, rs in enumerate(pool.imap_unordered(_one, jobs, chunksize=4)):
            rows.extend(rs)
            if i % 150 == 0:
                print(f'[{i + 1}/{len(jobs)}]', flush=True)
    df = pd.DataFrame(rows)
    R = pd.DataFrame(ref)
    df.to_csv(os.path.join(a.out, 'per_image.csv'), index=False)
    R.to_csv(os.path.join(a.out, 'ref.csv'), index=False)

    out = []
    for (setting, jd, L), g in df.groupby(['setting', 'junction_def', 'L'], sort=False):
        m = g.merge(R[(R.junction_def == jd) & (R.L == L)], on='stem', suffixes=('', '_ref'))
        r = {'setting': setting, 'param': g.param.iloc[0], 'value': g.value.iloc[0],
             'junction_def': jd, 'L': L,
             'n': len(m), 'payload_bytes': m.payload_bytes.mean(), 'fg_psnr': m.fg_psnr.mean(),
             'n_points': m.n_points.mean()}
        for dsc in dm.DESCRIPTORS:
            x, y = m[dsc].to_numpy(float), m[dsc + '_ref'].to_numpy(float)
            ok = np.isfinite(x) & np.isfinite(y)
            ag = dm.agreement(x[ok], y[ok])
            r[f'{dsc}_bias_pct'] = ag['bias_pct']
            r[f'{dsc}_ccc'] = ag['ccc']
            r[f'{dsc}_mdape'] = ag['mdape']
        out.append(r)
    S = pd.DataFrame(out)
    S.to_csv(os.path.join(a.out, 'summary.csv'), index=False)
    print(S[(S.junction_def == 'degree') & (S.L == 5)][['setting', 'payload_bytes', 'fg_psnr', 'n_branches_bias_pct',
                       'n_junctions_bias_pct', 'total_length_px_bias_pct']].to_string())


if __name__ == '__main__':
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    main()
