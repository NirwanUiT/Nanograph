#!/usr/bin/env python3
"""Run one U-Net checkpoint over a folder of PNGs -> <out>/<stem>.png (0/255),
optionally restricted to <cell-dir>/<stem>.png, with seconds per image in
<out>/timing.csv. Same polarity handling and segmentation call as real_bench.py."""
import argparse
import contextlib
import glob
import io
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))


def main():
    import cv2
    import pandas as pd
    import torch
    from nanograph_v4 import NanographConfig
    from nanograph_v4.detect import detect_polarity
    from nanograph_v4.segment import learned_segment
    from nanograph_v4.unet_seg import load_unet
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--img-dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--cell-dir', default=None)
    a = ap.parse_args()
    torch.set_num_threads(1)
    cfg = NanographConfig().for_real_mito()
    net = load_unet(a.ckpt, 'cpu')
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for p in sorted(glob.glob(os.path.join(a.img_dir, '*.png'))):
        stem = os.path.basename(p)[:-4]
        x = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        t0 = time.perf_counter()
        x = 255 - x if detect_polarity(x, cfg=cfg) else x
        with contextlib.redirect_stdout(io.StringIO()):
            m = learned_segment(net, x, cfg=cfg, device='cpu') > 0
        rows.append({'id': stem, 'seconds': time.perf_counter() - t0})
        if a.cell_dir:
            m &= cv2.imread(os.path.join(a.cell_dir, f'{stem}.png'), cv2.IMREAD_GRAYSCALE) > 0
        cv2.imwrite(os.path.join(a.out, f'{stem}.png'), m.astype(np.uint8) * 255)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, 'timing.csv'), index=False)
    print(len(rows), 'images, median s/image', np.median([r['seconds'] for r in rows]))


if __name__ == '__main__':
    main()
