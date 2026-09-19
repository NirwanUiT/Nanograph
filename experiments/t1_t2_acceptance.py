#!/usr/bin/env python3
"""T1+T2 acceptance over the 726 organelle images.

T1: mean±sd byte cost of the v6 edge section (raw section size, plus the
end-to-end payload delta from a second encode with store_edges=False).
T2: decoded quantised width/intensity must equal encoder-side quantised
values EXACTLY on every image (mismatch count must be 0).
"""
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nanograph_v4 import nanograph_encode, NanographConfig
from nanograph_v4.compress import decompress_nanograph, _encode_edge_section

ORG = '/mnt/nas1/nba055-2/idea_1/nmi_data/org'

raw_sizes, deltas = [], []
mismatch_images = 0
n_imgs = 0
paths = sorted(glob.glob(os.path.join(ORG, '*.png')))
for i, p in enumerate(paths):
    img = cv2.imread(p, 0)
    cfg = NanographConfig()
    r = nanograph_encode(img, sam_model=None, verbose=False, optimize=False, config=cfg)
    c_off = NanographConfig(); c_off.compress.store_edges = False
    r_off = nanograph_encode(img, sam_model=None, verbose=False, optimize=False, config=c_off)
    deltas.append(r.compressed_bytes - r_off.compressed_bytes)

    rows = r.points[:, 0].astype(np.int32)
    order = np.argsort(rows)
    inv = np.empty(len(rows), dtype=np.int64); inv[order] = np.arange(len(rows))
    raw_sizes.append(len(_encode_edge_section(
        [(e.source, e.target) for e in r.graph.edges], inv)))

    # T2 exactness
    qs = cfg.compress.width_quant_scale
    w_q_enc = np.clip(np.round(r.widths * qs), 0, 255).astype(np.uint8)[order]
    i_q_enc = np.clip(np.round(r.intensities * 255), 0, 255).astype(np.uint8)[order]
    (dp, di, dw, *_rest) = decompress_nanograph(r.compressed, cfg=cfg)
    w_q_dec = np.clip(np.round(dw * qs), 0, 255).astype(np.uint8)
    i_q_dec = np.clip(np.round(di * 255), 0, 255).astype(np.uint8)
    if not (np.array_equal(w_q_enc, w_q_dec) and np.array_equal(i_q_enc, i_q_dec)):
        mismatch_images += 1
        print(f'MISMATCH {os.path.basename(p)}')
    n_imgs += 1
    if i % 100 == 0:
        print(f'[{i+1}/{len(paths)}]', flush=True)

raw_sizes = np.array(raw_sizes); deltas = np.array(deltas)
print(f'T1 raw edge section: {raw_sizes.mean():.1f} ± {raw_sizes.std(ddof=1):.1f} bytes (n={n_imgs})')
print(f'T1 compressed payload delta (edges on - off): {deltas.mean():.1f} ± {deltas.std(ddof=1):.1f} bytes')
print(f'T2 attribute mismatch images: {mismatch_images} / {n_imgs}')
