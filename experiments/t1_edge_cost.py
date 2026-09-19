#!/usr/bin/env python3
"""T1 acceptance: byte cost of the v6 edge section on the 726 organelle images.

Reports the raw (pre-zlib) edge-section size and the end-to-end payload delta
(store_edges on vs off, same image, same config otherwise).
"""
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nanograph_v4 import nanograph_encode, NanographConfig

ORG = '/mnt/nas1/nba055-2/idea_1/nmi_data/org'

raw_sizes, deltas = [], []
paths = sorted(glob.glob(os.path.join(ORG, '*.png')))
for i, p in enumerate(paths):
    img = cv2.imread(p, 0)
    c_on = NanographConfig()
    c_off = NanographConfig()
    c_off.compress.store_edges = False
    r_on = nanograph_encode(img, sam_model=None, verbose=False, optimize=False, config=c_on)
    r_off = nanograph_encode(img, sam_model=None, verbose=False, optimize=False, config=c_off)
    from nanograph_v4.compress import _encode_edge_section
    rows = r_on.points[:, 0].astype(np.int32)
    order = np.argsort(rows)
    inv = np.empty(len(rows), dtype=np.int64)
    inv[order] = np.arange(len(rows))
    raw_sizes.append(len(_encode_edge_section(
        [(e.source, e.target) for e in r_on.graph.edges], inv)))
    deltas.append(r_on.compressed_bytes - r_off.compressed_bytes)
    if i % 100 == 0:
        print(f'[{i+1}/{len(paths)}]', flush=True)

raw_sizes = np.array(raw_sizes); deltas = np.array(deltas)
print(f'raw edge section: {raw_sizes.mean():.1f} ± {raw_sizes.std(ddof=1):.1f} bytes')
print(f'compressed payload delta: {deltas.mean():.1f} ± {deltas.std(ddof=1):.1f} bytes')
print(f'n = {len(raw_sizes)}')
