#!/usr/bin/env python3
"""
Analysis-only storage on real data: low-quality JPEG vs the structure layer (T13).

For every real test tile, JPEG at quality 1/5/10/20 is decoded, segmented with
the real-mito network (the same front end as the structure layer) and
analysed with the one-diameter rule; the expert mask is analysed the same way.
Writes <OUT_DIR>/jpeg_low_quality.csv (dataset, id, q, bytes, descriptors,
descriptors_ref) and prints, per dataset, the JPEG rows next to the structure
layer's agreement from <OUT_DIR>/per_tile.csv (written by real_downstream.py).

Usage (from repo root):
    OUT_DIR=results/paper/real/downstream python experiments/real_jpeg_floor.py
"""
import multiprocessing as mp
import os
import sys

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import downstream_morphometry as dm  # noqa: E402
from nanograph_v4 import NanographConfig  # noqa: E402
from nanograph_v4.detect import detect_polarity  # noqa: E402
from nanograph_v4.segment import learned_segment  # noqa: E402
from nanograph_v4.unet_seg import load_unet  # noqa: E402

REAL = '/mnt/nas1/nba055-2/idea_1/real_mito'
OUT = os.environ.get('OUT_DIR', 'results/real/real_downstream')
WORKERS = int(os.environ.get('WORKERS', 12))
QUALITIES = (1, 5, 10, 20)
SHOW = ['n_branches', 'n_junctions', 'total_length_px']


def describe(mask):
    return dm.descriptors_at(dm.pixel_arm_table(mask), 'auto', min_len='auto')[0]


def work(chunk):
    torch.set_num_threads(1)
    cfg = NanographConfig().for_real_mito()
    net = load_unet(cfg.segment.learned_ckpt, 'cpu')
    rows = []
    for d, tid in chunk:
        img = cv2.imread(f'{REAL}/{d}/images/{tid}.png', 0)
        ref = describe(cv2.imread(f'{REAL}/{d}/masks/{tid}.png', 0))
        for q in QUALITIES:
            _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, q])
            dec = cv2.imdecode(buf, 0)
            x = 255 - dec if detect_polarity(dec, cfg=cfg) else dec
            dj = describe(learned_segment(net, x, cfg=cfg, device='cpu'))
            rows.append({'dataset': d, 'id': tid, 'q': q, 'bytes': len(buf),
                         **{k: dj[k] for k in dm.DESCRIPTORS},
                         **{k + '_ref': ref[k] for k in dm.DESCRIPTORS}})
    return rows


def summary(x, y):
    ag = dm.agreement(np.asarray(x, float), np.asarray(y, float))
    return f"bias {ag['bias_pct']:+.0f}% CCC {ag['ccc']:.2f}"


def main():
    man = pd.read_csv(f'{REAL}/manifest.csv')
    man = man[man.split == 'test']
    jobs = list(zip(man.dataset, man.id))
    with mp.get_context('spawn').Pool(WORKERS) as pool:
        rows = [r for rs in pool.map(work, [jobs[i::WORKERS] for i in range(WORKERS)]) for r in rs]
    P = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    P.to_csv(os.path.join(OUT, 'jpeg_low_quality.csv'), index=False)

    R = pd.read_csv(os.path.join(OUT, 'per_tile.csv'))
    R = R[R.L.astype(str) == 'auto']
    for d in ['UIT', 'CBMI', 'MITO', 'HUMAN']:
        W = {a: x.set_index('id') for a, x in R[R.dataset == d].groupby('arm')}
        ids, g = W['REF'].index, W['GRAPH7']
        print(f"\n{d}: structure layer {g.structure_bytes.mean():.0f} B | " +
              ' | '.join(f"{k} {summary(g.loc[ids, k], W['REF'].loc[ids, k])}" for k in SHOW))
        for q, h in P[P.dataset == d].groupby('q'):
            print(f'  JPEG q{q:<3d} {h.bytes.mean():6.0f} B | ' +
                  ' | '.join(f"{k} {summary(h[k], h[k + '_ref'])}" for k in SHOW))


if __name__ == '__main__':
    main()
