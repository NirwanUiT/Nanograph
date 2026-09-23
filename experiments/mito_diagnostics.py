#!/usr/bin/env python3
"""
T7 — mitochondria (Zenodo 7724799 MIP tiles) failure-case diagnostics.

For each forced segmenter {otsu, frangi, meijering, learned} this drives the
SAME pipeline as `run_dataset.py --force-segmenter X` (cfg.segment.force_segmenter),
reading back the selected foreground mask (result.mask), and reports:

  Seg-IoU, precision, recall  (mask vs the MIP-union GT)
  precision@dil20             : precision restricted to a 20-px dilation of the
                                annotated foreground — i.e. of the predicted
                                pixels that fall within 20 px of any GT pixel,
                                what fraction are true positives. This isolates
                                "near-miss" localisation error from gross flooding.

It also writes 10 TP/FP/FN overlays (green=TP, red=FN, blue=FP) for the best
segmenter, so the sparse-foreground failure mode is visible.

Outputs under results/paper/mito_diag/:
  <segmenter>/metrics.csv     per-tile rows
  summary.csv                 per-segmenter aggregates
  overlays/                   10 overlays
  commit.txt
"""
import argparse
import glob
import os
import subprocess
import sys

import cv2
import numpy as np

# nanograph_v4 is a symlink in the repo parent dir (three levels up from experiments/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nanograph_v4 import nanograph_encode, NanographConfig

SEGMENTERS = ['otsu', 'frangi', 'meijering', 'learned']
DIL_PX = 20


def _iou_pr(pred, gt):
    p = pred > 0
    g = gt > 0
    tp = int(np.logical_and(p, g).sum())
    fp = int(np.logical_and(p, ~g).sum())
    fn = int(np.logical_and(~p, g).sum())
    union = tp + fp + fn
    iou = tp / union if union else 1.0
    prec = tp / (tp + fp) if (tp + fp) else 1.0
    rec = tp / (tp + fn) if (tp + fn) else 1.0
    return iou, prec, rec, tp, fp, fn


def _precision_in_dilation(pred, gt, dil_px):
    """Precision computed only over predicted pixels within dil_px of GT fg."""
    g = (gt > 0).astype(np.uint8)
    if g.sum() == 0:
        return float('nan')
    k = 2 * dil_px + 1
    band = cv2.dilate(g, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
    p_in = (pred > 0) & band
    denom = int(p_in.sum())
    if denom == 0:
        return float('nan')
    tp_in = int((p_in & (gt > 0)).sum())
    return tp_in / denom


def _overlay(img_gray, pred, gt):
    base = np.stack([img_gray // 3] * 3, axis=-1).astype(np.uint8)
    p = pred > 0
    g = gt > 0
    base[p & g] = [0, 200, 0]      # TP green
    base[(~p) & g] = [200, 0, 0]   # FN red
    base[p & (~g)] = [0, 0, 200]   # FP blue
    return base


def _cfg_for(seg):
    cfg = NanographConfig()
    cfg.segment.force_segmenter = seg
    if seg == 'learned':
        cfg.segment.use_learned = True
        cfg.segment.learned_mode = 'replace'
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', default='/mnt/nas1/nba055-2/idea_1/public_mito/mito_mip_tiles/images')
    ap.add_argument('--masks', default='/mnt/nas1/nba055-2/idea_1/public_mito/mito_mip_tiles/masks')
    ap.add_argument('--out', default='results/paper/mito_diag')
    ap.add_argument('--overlay-seg', default='otsu', help='segmenter used for the 10 overlays')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ov_dir = os.path.join(args.out, 'overlays')
    os.makedirs(ov_dir, exist_ok=True)
    try:
        h = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
        with open(os.path.join(args.out, 'commit.txt'), 'w') as f:
            f.write(h + '\n')
    except Exception:
        pass

    img_paths = sorted(glob.glob(os.path.join(args.images, '*.png')))
    print(f'{len(img_paths)} MIP tiles')

    summary = []
    n_overlays = 0
    for seg in SEGMENTERS:
        cfg = _cfg_for(seg)
        seg_dir = os.path.join(args.out, seg)
        os.makedirs(seg_dir, exist_ok=True)
        rows = []
        for ip in img_paths:
            stem = os.path.splitext(os.path.basename(ip))[0]
            gt = cv2.imread(os.path.join(args.masks, stem + '.png'), cv2.IMREAD_GRAYSCALE)
            img = cv2.imread(ip, cv2.IMREAD_GRAYSCALE)
            if gt is None or img is None:
                continue
            try:
                res = nanograph_encode(img, config=cfg, verbose=False)
                pred = res.mask
                if pred.shape != gt.shape:
                    pred = cv2.resize(pred.astype(np.uint8), (gt.shape[1], gt.shape[0]),
                                      interpolation=cv2.INTER_NEAREST)
            except Exception as e:
                print('  fail', seg, stem, e)
                continue
            iou, prec, rec, tp, fp, fn = _iou_pr(pred, gt)
            pdil = _precision_in_dilation(pred, gt, DIL_PX)
            rows.append((stem, iou, prec, rec, pdil, tp, fp, fn))

            if seg == args.overlay_seg and n_overlays < 10:
                cv2.imwrite(os.path.join(ov_dir, f'{stem}_{seg}.png'),
                            _overlay(img, pred, gt)[:, :, ::-1])  # RGB->BGR for cv2
                n_overlays += 1

        with open(os.path.join(seg_dir, 'metrics.csv'), 'w') as f:
            f.write('stem,seg_iou,precision,recall,precision_dil20,tp,fp,fn\n')
            for r in rows:
                f.write('%s,%.6f,%.6f,%.6f,%.6f,%d,%d,%d\n' % r)

        arr = np.array([[r[1], r[2], r[3]] for r in rows], dtype=float)
        pdil_vals = np.array([r[4] for r in rows], dtype=float)
        mi, mp, mr = arr.mean(0) if len(arr) else (0, 0, 0)
        mpd = np.nanmean(pdil_vals) if len(pdil_vals) else float('nan')
        summary.append((seg, len(rows), mi, mp, mr, mpd))
        print(f'{seg:10s} n={len(rows):3d} IoU={mi:.3f} P={mp:.3f} R={mr:.3f} P@dil20={mpd:.3f}')

    with open(os.path.join(args.out, 'summary.csv'), 'w') as f:
        f.write('segmenter,n,seg_iou,precision,recall,precision_dil20\n')
        for s in summary:
            f.write('%s,%d,%.6f,%.6f,%.6f,%.6f\n' % s)
    print('wrote', os.path.join(args.out, 'summary.csv'))


if __name__ == '__main__':
    main()
