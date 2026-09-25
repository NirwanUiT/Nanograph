#!/usr/bin/env python3
"""
T18 pilot — does mitochondrial fragmentation across the cell cycle show up in
descriptors read ONLY from Nanograph structure layers?

Stratified sample of the TOMM20 manifest (allen_check.py): N cells per
cell_stage (seed 0). Per cell: the +-1 um slab projection of allen_check.py,
the real-mito segmenter's mask inside the cell, its v7 structure layer
(stored as <CellId>.ngs), and descriptors decoded from those bytes alone
(one-diameter rule), converted to um with the dataset's voxel size.

This is a pilot for the effect's direction and size, not the T18 analysis
(no mixed model, no pixel-route comparison).

Writes <out>/pilot/per_cell.csv and <out>/pilot/fig_pilot.png.
"""
import argparse
import contextlib
import io
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from allen_check import fetch, slab_projection  # noqa: E402

STAGES = ['M0', 'M1M2', 'M3', 'M4M5', 'M6M7_single', 'M6M7_complete']


def encode_cell(args):
    row, cache, ngdir = args
    import ast
    import cv2
    import tifffile
    import torch
    from skimage.morphology import skeletonize
    from nanograph_v4 import NanographConfig, graph_branch as gb
    from nanograph_v4.detect import detect_polarity
    from nanograph_v4.segment import learned_segment
    from nanograph_v4.unet_seg import load_unet
    torch.set_num_threads(1)
    global _NET
    cfg = NanographConfig().for_real_mito()
    if '_NET' not in globals():
        _NET = load_unet(cfg.segment.learned_ckpt, 'cpu')
    cid = row['CellId']
    try:
        rp, sp = os.path.join(cache, f'{cid}_raw.ome.tif'), os.path.join(cache, f'{cid}_seg.ome.tif')
        fetch(row['crop_raw_url'], rp)
        fetch(row['crop_seg_url'], sp)
        names = ast.literal_eval(row['name_dict'])
        um = ast.literal_eval(row['scale_micron'])[-1]
        im, cell, _, _ = slab_projection(tifffile.imread(rp), tifffile.imread(sp), names, um)
        x = 255 - im if detect_polarity(im, cfg=cfg) else im
        with contextlib.redirect_stdout(io.StringIO()):
            p = (learned_segment(_NET, x, cfg=cfg, device='cpu') > 0) & cell
        dt = cv2.distanceTransform(p.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        sb = gb.encode_structure(gb.branch_structure(skeletonize(p), dt))
        open(os.path.join(ngdir, f'{cid}.ngs'), 'wb').write(sb)
        return {'CellId': cid, 'cell_stage': row['cell_stage'], 'PlateId': row['PlateId'],
                'um_per_px': um, 'cell_area_um2': float(cell.sum()) * um * um,
                'structure_bytes': len(sb), 'raw_bytes': os.path.getsize(rp), 'error': ''}
    except Exception as ex:
        return {'CellId': cid, 'cell_stage': row['cell_stage'], 'error': f'{type(ex).__name__}: {ex}'}


def measure(ngdir, cid, um):
    """Descriptors from the stored bytes only."""
    import downstream_morphometry as dm
    from nanograph_v4 import graph_branch as gb
    st = gb.decode_structure(open(os.path.join(ngdir, f'{cid}.ngs'), 'rb').read())
    tab = dm.graph_arm_table(*gb.structure_to_arrays(st, True))
    out = dm.descriptors_at(tab, 'auto', 'degree')
    d, bl = out[0], out[1]
    L = d['total_length_px'] * um
    return {'n_components': d['n_components'], 'n_branches': d['n_branches'],
            'n_junctions': d['n_junctions'], 'total_length_um': L,
            'mean_width_um': d['mean_width_px'] * um,
            'mean_branch_um': float(np.mean(bl)) * um if len(bl) else np.nan,
            'components_per_100um': 100 * d['n_components'] / L if L > 0 else np.nan,
            'junctions_per_100um': 100 * d['n_junctions'] / L if L > 0 else np.nan}


def main():
    import multiprocessing as mp
    import time
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/mnt/nas1/nba055-2/idea_1/archive_demo/allen')
    ap.add_argument('--per-stage', type=int, default=30)
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    M = pd.read_csv(os.path.join(a.out, 'manifest_tomm20.csv'))
    S = pd.concat(g.sample(min(a.per_stage, len(g)), random_state=0) for _, g in M.groupby('cell_stage'))
    pdir, cache, ngdir = (os.path.join(a.out, d) for d in ('pilot', 'cells', 'ng_structure'))
    for d in (pdir, cache, ngdir):
        os.makedirs(d, exist_ok=True)
    with mp.get_context('spawn').Pool(a.workers) as pool:
        R = pd.DataFrame(pool.map(encode_cell, [(r, cache, ngdir) for r in S.to_dict('records')]))
    ok = R[R.error == ''].copy()
    t0 = time.perf_counter()
    D = pd.DataFrame([measure(ngdir, c, u) for c, u in zip(ok.CellId, ok.um_per_px)], index=ok.index)
    t_query = time.perf_counter() - t0
    ok = pd.concat([ok, D], axis=1)
    ok.to_csv(os.path.join(pdir, 'per_cell.csv'), index=False)
    print(f'{len(ok)}/{len(R)} cells encoded; errors: {int((R.error != "").sum())}')
    print(f'structure layers: {ok.structure_bytes.sum() / 1e3:.0f} kB total for {len(ok)} cells '
          f'(raw 3-D crops {ok.raw_bytes.sum() / 1e9:.1f} GB); '
          f'all descriptors read from the stored layers in {t_query:.2f} s')
    cols = ['components_per_100um', 'junctions_per_100um', 'mean_branch_um', 'total_length_um', 'mean_width_um']
    g = ok.groupby('cell_stage')[cols].median().reindex(STAGES)
    g.insert(0, 'n', ok.groupby('cell_stage').size().reindex(STAGES))
    print(g.round(2).to_string())
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for a_, (c, lab) in zip(ax, [('components_per_100um', 'separate mitochondria per 100 µm'),
                                 ('mean_branch_um', 'mean branch length (µm)'),
                                 ('junctions_per_100um', 'junctions per 100 µm')]):
        data = [ok[ok.cell_stage == s][c].dropna() for s in STAGES]
        a_.boxplot(data, labels=[s.replace('_', '\n') for s in STAGES], showfliers=False)
        for i, v in enumerate(data):
            a_.scatter(np.full(len(v), i + 1) + np.random.default_rng(i).uniform(-.15, .15, len(v)), v,
                       s=6, alpha=.5, color='#1f5fa8')
        a_.set_title(lab, loc='left', fontsize=10)
        a_.tick_params(axis='x', labelsize=8)
    fig.suptitle('TOMM20 hiPSC cells: descriptors read only from Nanograph structure layers '
                 '(pilot, M0 = interphase)', fontsize=10, x=0.01, ha='left')
    fig.tight_layout()
    fig.savefig(os.path.join(pdir, 'fig_pilot.png'), dpi=150)


if __name__ == '__main__':
    main()
