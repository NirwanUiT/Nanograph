#!/usr/bin/env python3
"""Background-grid + DCT-residual ablation over the full organelle set.

Systematic version of the single-image-pair comparison in tab:ablation of the
revised manuscript: every image is encoded twice (default config vs. background
grid and foreground residual disabled) and per-image FG-PSNR / full PSNR /
bytes are written to CSV. Metric fields are the encode-result fields used by
run_dataset.py; no metric definitions are changed.

Usage (from repo root):
    python experiments/ablation_bg_residual.py \
        --images /mnt/nas1/nba055-2/idea_1/nmi_data/org \
        --out ablation_bg_residual.csv
"""
import argparse
import csv
import glob
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nanograph_v4 import nanograph_encode, NanographConfig


def cfg_with():
    return NanographConfig()


def cfg_without():
    c = NanographConfig()
    c.recon.use_fg_residual = False
    c.compress.store_bg_grid = False
    c.compress.store_fg_residual = False
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', required=True)
    ap.add_argument('--out', default='ablation_bg_residual.csv')
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.images, '*.png')))
    print(f'{len(paths)} images')
    rows = []
    for i, p in enumerate(paths):
        stem = os.path.splitext(os.path.basename(p))[0]
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        try:
            r_with = nanograph_encode(img, sam_model=None, verbose=False, config=cfg_with())
            r_wo = nanograph_encode(img, sam_model=None, verbose=False, config=cfg_without())
        except Exception as e:
            print(f'  FAIL {stem}: {e}')
            continue
        rows.append(dict(
            filename=stem,
            fg_psnr_with=r_with.psnr_fg, fg_psnr_without=r_wo.psnr_fg,
            fg_ssim_with=r_with.ssim_fg, fg_ssim_without=r_wo.ssim_fg,
            full_psnr_with=r_with.psnr_full, full_psnr_without=r_wo.psnr_full,
            bytes_with=r_with.compressed_bytes, bytes_without=r_wo.compressed_bytes,
        ))
        if i % 50 == 0:
            print(f'[{i+1}/{len(paths)}] {stem} '
                  f'FG-PSNR {r_wo.psnr_fg:.2f}->{r_with.psnr_fg:.2f} '
                  f'bytes {r_wo.compressed_bytes}->{r_with.compressed_bytes}', flush=True)

    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'wrote {args.out} ({len(rows)} rows)')

    import statistics as st
    for k in ['fg_psnr', 'full_psnr', 'bytes']:
        a = [r[f'{k}_with'] for r in rows]
        b = [r[f'{k}_without'] for r in rows]
        d = [x - y for x, y in zip(a, b)]
        print(f'{k:10s} with {st.mean(a):10.2f}±{st.stdev(a):8.2f}  '
              f'without {st.mean(b):10.2f}±{st.stdev(b):8.2f}  '
              f'delta {st.mean(d):+9.2f}±{st.stdev(d):7.2f}')


if __name__ == '__main__':
    main()
