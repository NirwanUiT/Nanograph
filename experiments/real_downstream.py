#!/usr/bin/env python3
"""
Downstream comparison on REAL annotated mitochondria (T13): descriptors read
from the stored graph vs the same analysis on a byte-matched JPEG, both
against the expert mask.

Per real test tile (real_mito/manifest.csv, split 'test'):
  REF     expert mask -> skeleton (skan)
  GRAPH7  decoded v7 structure layer, pipeline with --preset real-mito
  SEG     the pipeline's own (real-trained) mask
  JPEG    byte-matched JPEG (budget = v7 payload) -> the same real-trained
          U-Net (replace mode) -> skeleton
Descriptors under the one-diameter rule ('auto') and at L = 5; paired
per-tile |error| tests GRAPH7 vs JPEG. Tiles of a frame are correlated; the
report also gives frame-level means.

Usage (from repo root):
    python experiments/real_downstream.py --workers 12
"""
import contextlib
import io
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
REAL = '/mnt/nas1/nba055-2/idea_1/real_mito'
SETTINGS = [('5', 5, 0.0), ('auto', 'auto', 'auto')]


def _one(job):
    d, tid, group = job
    import cv2
    import torch
    import downstream_morphometry as dm
    from nanograph_v4 import nanograph_encode, NanographConfig
    from nanograph_v4.evaluate import jpeg_for_budget
    from nanograph_v4.segment import learned_segment
    from nanograph_v4.detect import detect_polarity
    from nanograph_v4.unet_seg import load_unet
    torch.set_num_threads(1)
    cfg = NanographConfig().for_real_mito()
    global _M
    if '_M' not in globals():
        _M = load_unet(cfg.segment.learned_ckpt, device='cpu')
    img = cv2.imread(os.path.join(REAL, d, 'images', tid + '.png'), cv2.IMREAD_GRAYSCALE)
    gt = cv2.imread(os.path.join(REAL, d, 'masks', tid + '.png'), cv2.IMREAD_GRAYSCALE)
    with contextlib.redirect_stdout(io.StringIO()):
        r = nanograph_encode(img, verbose=False, config=cfg)
    tabs = {'REF': dm.pixel_arm_table(gt), 'SEG': dm.pixel_arm_table(r.mask),
            'GRAPH7': dm.graph_arm_table(*dm._decoded_graph_arrays(r.compressed))}
    nbytes = {'GRAPH7': len(r.compressed),
              'GRAPH7_structure': r.compression_stats.get('structure_bytes', np.nan)}
    _, jb = jpeg_for_budget(img, len(r.compressed))
    if jb:
        dec = cv2.imdecode(np.frombuffer(jb, np.uint8), cv2.IMREAD_GRAYSCALE)
        x = 255 - dec if detect_polarity(dec, cfg=cfg) else dec
        tabs['JPEG'] = dm.pixel_arm_table(learned_segment(_M, x, cfg=cfg, device='cpu'))
        nbytes['JPEG'] = len(jb)
    rows = []
    for arm, t in tabs.items():
        for lab, L, m in SETTINGS:
            dd = dm.descriptors_at(t, L, min_len=m)[0]
            row = {'dataset': d, 'group': group, 'id': tid, 'arm': arm, 'L': lab,
                   'bytes': nbytes.get(arm, np.nan), 'structure_bytes': nbytes['GRAPH7_structure'], **dd}
            jr, _ = dm.vertex_positions(tabs['REF'], L, min_len=m)
            ja, _ = dm.vertex_positions(t, L, min_len=m)
            row['junc_f1'] = dm.point_f1(ja, jr, 3)[2]
            rows.append(row)
    return rows


def main():
    import multiprocessing as mp
    import pandas as pd
    import downstream_morphometry as dm
    workers = int(os.environ.get('WORKERS', 12))
    M = pd.read_csv(os.path.join(REAL, 'manifest.csv'))
    M = M[M.split == 'test']
    jobs = list(zip(M.dataset, M.id, M.group))
    with mp.get_context('spawn').Pool(workers) as pool:
        rows = [r for rs in pool.imap_unordered(_one, jobs, chunksize=2) for r in rs]
    P = pd.DataFrame(rows)
    out = os.environ.get('OUT_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                  'results', 'real', 'real_downstream'))
    os.makedirs(out, exist_ok=True)
    P.to_csv(os.path.join(out, 'per_tile.csv'), index=False)
    S = []
    for (d, L), g in P.groupby(['dataset', 'L']):
        W = {a: x.set_index('id') for a, x in g.groupby('arm')}
        ids = sorted(set.intersection(*(set(W[a].index) for a in ('REF', 'GRAPH7', 'JPEG'))))
        for arm in ('GRAPH7', 'SEG', 'JPEG'):
            r = {'dataset': d, 'L': L, 'arm': arm, 'n': len(ids), 'bytes': W[arm].loc[ids].bytes.mean(),
                 'structure_bytes': W['GRAPH7'].loc[ids].structure_bytes.mean(),
                 'junc_f1': W[arm].loc[ids].junc_f1.mean()}
            for k in dm.DESCRIPTORS:
                x, y = W[arm].loc[ids, k].to_numpy(float), W['REF'].loc[ids, k].to_numpy(float)
                ok = np.isfinite(x) & np.isfinite(y)
                if ok.sum() > 2 and y[ok].mean() != 0:
                    ag = dm.agreement(x[ok], y[ok])
                    r[f'{k}_bias_pct'], r[f'{k}_ccc'], r[f'{k}_mdape'] = ag['bias_pct'], ag['ccc'], ag['mdape']
                if arm == 'GRAPH7':
                    j = W['JPEG'].loc[ids, k].to_numpy(float)
                    ok3 = ok & np.isfinite(j)
                    t = dm.paired_error_test(x[ok3], j[ok3], y[ok3])
                    r[f'{k}_vs_jpeg'] = f"{t['wins_graph']}/{t['wins_jpeg']} p={t['wilcoxon_p']:.1e}"
            S.append(r)
    S = pd.DataFrame(S)
    S.to_csv(os.path.join(out, 'summary.csv'), index=False)
    pd.set_option('display.width', 250)
    cols = ['dataset', 'arm', 'n', 'bytes', 'structure_bytes', 'n_components_bias_pct', 'total_length_px_bias_pct',
            'n_branches_bias_pct', 'n_junctions_bias_pct', 'n_branches_ccc', 'total_length_px_ccc', 'junc_f1']
    for L in ('auto', '5'):
        print(f'\n== L = {L}')
        print(S[S.L == L][cols].round(3).to_string(index=False))
        v = S[(S.L == L) & (S.arm == 'GRAPH7')]
        print(v[['dataset'] + [c for c in S.columns if c.endswith('_vs_jpeg')]].to_string(index=False))


if __name__ == '__main__':
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    main()
