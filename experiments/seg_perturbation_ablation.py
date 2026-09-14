#!/usr/bin/env python3
"""Segmentation-perturbation ablation (review item C).

For each organelle image: obtain the pipeline's selected mask M*, degrade it
(dilation / erosion / component removal / boundary noise), rebuild the graph
from each degraded mask exactly as the pipeline does (skeletonise -> extract
points -> build_nanograph), and record graph invariants vs. the degraded
mask's Seg-IoU against ground truth. No re-encode; graph construction only.

Usage (from repo root):
    python experiments/seg_perturbation_ablation.py \
        --images /mnt/nas1/nba055-2/idea_1/nmi_data/org \
        --masks  /mnt/nas1/nba055-2/idea_1/nmi_data/seg \
        --out seg_perturbation_ablation.csv
"""
import argparse
import csv
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nanograph_v4 import nanograph_encode, NanographConfig
from nanograph_v4.skeleton import skeletonize_and_classify, extract_nanograph_points, \
    compute_skeleton_orientations
from nanograph_v4.graph import build_nanograph


def iou(a, b):
    a, b = a > 0, b > 0
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


def graph_metrics(mask, img):
    """Mask -> graph via the pipeline's own construction; return invariants."""
    if mask.sum() == 0:
        return None
    skel, ep, jn = skeletonize_and_classify(mask)
    if skel.sum() == 0:
        return None
    dt = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    ori = compute_skeleton_orientations(skel)
    imgf = img.astype(np.float64) / 255.0
    pts, inten, widths, _, types, oris = extract_nanograph_points(
        skel, ep, jn, imgf, dt, imgf, orientation_map=ori)
    if len(pts) == 0:
        return None
    pts = np.asarray(pts)
    g = build_nanograph(pts, inten, widths, oris, types, skel, dt, imgf,
                        mask.shape, cfg=NanographConfig())
    s = g.summary()
    b0 = s['n_components']
    b1 = max(0, s['n_edges'] - s['n_nodes'] + b0)  # cycle rank
    return dict(beta0=b0, beta1=b1, n_nodes=s['n_nodes'],
                total_edge_length=s['total_edge_length'],
                mean_width=s['mean_width'])


def perturbations(mask, rng):
    """Yield (kind, level, degraded_mask)."""
    m = (mask > 0).astype(np.uint8)
    for r in range(1, 6):
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        yield 'dilate', r, cv2.dilate(m, k)
        yield 'erode', r, cv2.erode(m, k)
    n_lbl, lbl = cv2.connectedComponents(m)
    comps = list(range(1, n_lbl))
    for frac in (0.05, 0.10, 0.20, 0.30, 0.40):
        drop = set(rng.choice(comps, size=max(1, int(round(frac * len(comps)))),
                              replace=False)) if comps else set()
        dm = m.copy()
        for c in drop:
            dm[lbl == c] = 0
        yield 'drop_components', frac, dm
    for band in (1, 2, 3):
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band + 1, 2 * band + 1))
        boundary_zone = cv2.dilate(m, k) - cv2.erode(m, k)
        flips = (boundary_zone > 0) & (rng.random(m.shape) < 0.5)
        nm = m.copy()
        nm[flips] = 1 - nm[flips]
        yield 'boundary_noise', band, nm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', required=True)
    ap.add_argument('--masks', required=True)
    ap.add_argument('--out', default='seg_perturbation_ablation.csv')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.images, '*.png')))
    if args.limit:
        paths = paths[:args.limit]
    print(f'{len(paths)} images')
    rng = np.random.default_rng(args.seed)
    cfg = NanographConfig()
    rows = []
    for i, p in enumerate(paths):
        stem = os.path.splitext(os.path.basename(p))[0]
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(os.path.join(args.masks, stem + '.png'), cv2.IMREAD_GRAYSCALE)
        if img is None or gt is None:
            continue
        try:
            r = nanograph_encode(img, sam_model=None, verbose=False,
                                 optimize=False, config=cfg)
        except Exception as e:
            print(f'  encode FAIL {stem}: {e}')
            continue
        mstar = (r.mask > 0).astype(np.uint8)
        base = graph_metrics(mstar, img)
        if base is None:
            continue
        rows.append(dict(filename=stem, kind='baseline', level=0,
                         seg_iou=iou(mstar, gt), **base))
        for kind, level, dm in perturbations(mstar, rng):
            gm = graph_metrics(dm, img)
            row = dict(filename=stem, kind=kind, level=level,
                       seg_iou=iou(dm, gt))
            row.update(gm if gm is not None else
                       dict(beta0=0, beta1=0, n_nodes=0,
                            total_edge_length=0.0, mean_width=0.0))
            rows.append(row)
        if i % 25 == 0:
            print(f'[{i+1}/{len(paths)}] {stem} base b0={base["beta0"]} '
                  f'b1={base["beta1"]} segIoU={rows[-19]["seg_iou"]:.3f}', flush=True)

    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'wrote {args.out} ({len(rows)} rows)')


if __name__ == '__main__':
    main()
