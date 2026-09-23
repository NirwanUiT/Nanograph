#!/usr/bin/env python3
"""
A2 — quantify the cost of T3 (metrics on the decoded payload vs the
pre-compression render) and rule out a decode-path bug.

  --panel N   : save N pre/decoded/|diff| panels + report max abs pixel diff.
  --full DIR  : over every image in DIR, report mean/max of
                pre_gt_iou-gt_iou, pre_fg_psnr-fg_psnr, pre_psnr-psnr,
                pre_ssim-ssim  (needs GT masks via --masks for the IoU term).
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nanograph_v4 import nanograph_encode, NanographConfig


def _otsu_iou(render, gt_bin):
    u8 = (render * 255).astype(np.uint8)
    if u8.max() == 0:
        pred = np.zeros_like(u8)
    else:
        _, pred = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inter = np.logical_and(pred > 0, gt_bin > 0).sum()
    union = np.logical_or(pred > 0, gt_bin > 0).sum()
    return inter / union if union else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', default='/mnt/nas1/nba055-2/idea_1/nmi_data/org')
    ap.add_argument('--masks', default='/mnt/nas1/nba055-2/idea_1/nmi_data/seg')
    ap.add_argument('--out', default='results/paper/a2')
    ap.add_argument('--panel', type=int, default=0)
    ap.add_argument('--full', action='store_true')
    ap.add_argument('--preset', default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cfg = NanographConfig()
    if args.preset == 'learned-replace':
        cfg.segment.learned_mode = 'replace'
    elif args.preset == 'classical':
        cfg.segment.use_learned = False

    paths = sorted(glob.glob(os.path.join(args.images, '*.png')))

    # --- panel mode: pre vs decoded render + |diff| ---
    if args.panel > 0:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for ip in paths[:args.panel]:
            stem = os.path.splitext(os.path.basename(ip))[0]
            img = cv2.imread(ip, cv2.IMREAD_GRAYSCALE)
            res = nanograph_encode(img, config=cfg, verbose=False)
            dec = res.reconstruction
            pre = res.pre_reconstruction
            diff = np.abs(pre - dec)
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            ax[0].imshow(pre, cmap='gray', vmin=0, vmax=1); ax[0].set_title('pre-compression')
            ax[1].imshow(dec, cmap='gray', vmin=0, vmax=1); ax[1].set_title('decoded payload')
            ax[2].imshow(diff, cmap='hot', vmin=0, vmax=max(diff.max(), 1e-6))
            ax[2].set_title(f'|diff|  max={diff.max():.4f} ({diff.max()*255:.1f}/255)')
            for a in ax:
                a.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(args.out, f'a2_panel_{stem}.png'), dpi=110)
            plt.close()
            print(f'{stem}: max|Δ|={diff.max():.5f} ({diff.max()*255:.2f}/255)  '
                  f'mean|Δ|={diff.mean():.6f}')

    # --- full mode: scalar deltas over the whole set ---
    if args.full:
        dg, dfp, dp, ds = [], [], [], []
        for i, ip in enumerate(paths):
            stem = os.path.splitext(os.path.basename(ip))[0]
            img = cv2.imread(ip, cv2.IMREAD_GRAYSCALE)
            gtp = os.path.join(args.masks, stem + '.png')
            gt = cv2.imread(gtp, cv2.IMREAD_GRAYSCALE) if os.path.exists(gtp) else None
            res = nanograph_encode(img, config=cfg, verbose=False)
            orig = img.astype(float) / 255.0
            dec, pre = res.reconstruction, res.pre_reconstruction
            fg = res.mask > 0
            dp.append(psnr(orig, pre) - psnr(orig, dec))
            ds.append(ssim(orig, pre, data_range=1.0) - ssim(orig, dec, data_range=1.0))
            try:
                dfp.append(psnr(orig[fg], pre[fg]) - psnr(orig[fg], dec[fg]))
            except Exception:
                pass
            if gt is not None and gt.max() > 0:
                dg.append(_otsu_iou(pre, gt) - _otsu_iou(dec, gt))
            if i % 100 == 0:
                print(f'  {i}/{len(paths)}')

        def stat(name, arr):
            if not arr:
                print(f'  {name}: no data')
                return
            a = np.array(arr)
            print(f'  {name:26s} mean={a.mean():+.5f}  max|Δ|={a[np.argmax(np.abs(a))]:+.5f}  n={len(a)}')
        print('--- A2 full (pre-compression minus decoded) ---')
        stat('pre_gt_iou - gt_iou', dg)
        stat('pre_fg_psnr - fg_psnr', dfp)
        stat('pre_psnr - psnr', dp)
        stat('pre_ssim - ssim', ds)


if __name__ == '__main__':
    main()
