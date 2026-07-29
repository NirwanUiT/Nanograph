#!/usr/bin/env python3
"""
Nanograph v5 — Full dataset evaluation on sparse organelle images.

Usage:
    python run_dataset.py \
        --images /mnt/nas1/nba055-2/idea_1/nmi_data/org \
        --masks  /mnt/nas1/nba055-2/idea_1/nmi_data/seg \
        --outdir ./dataset_results \
        --n-samples 10
"""

import argparse
import glob
import os
import sys
import time
import random
import csv
import traceback

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanograph_v4 import nanograph_encode, NanographConfig, nanograph_morphometry
from nanograph_v4.evaluate import evaluate_result_with_gt, evaluate_result
from nanograph_v4.utils import graph_topology_score


def plot_sample(result, img_raw, gt_mask, title, save_path):
    """3×3 visualization: original / seg / GT / recon / error / overlay / skel / pts / graph."""
    fig, axes = plt.subplots(3, 3, figsize=(21, 21))

    # Row 1
    axes[0, 0].set_title(f'Original ({img_raw.shape[0]}×{img_raw.shape[1]})')
    axes[0, 0].imshow(img_raw, cmap='gray'); axes[0, 0].axis('off')

    axes[0, 1].set_title(f'Segmentation ({result.segmenter})\nType: {result.image_type}')
    axes[0, 1].imshow(result.mask, cmap='gray'); axes[0, 1].axis('off')

    if gt_mask is not None:
        axes[0, 2].set_title('Ground Truth Mask')
        axes[0, 2].imshow(gt_mask, cmap='gray')
    else:
        axes[0, 2].set_title('GT (N/A)')
    axes[0, 2].axis('off')

    # Row 2
    axes[1, 0].set_title(f'Reconstruction\nPSNR={result.psnr_full:.2f} SSIM={result.ssim_full:.4f}')
    axes[1, 0].imshow(result.reconstruction, cmap='gray'); axes[1, 0].axis('off')

    err = np.abs(img_raw.astype(float) / 255.0 - result.reconstruction)
    axes[1, 1].set_title(f'Error Map\nFG-PSNR={result.psnr_fg:.2f} FG-SSIM={result.ssim_fg:.4f}')
    axes[1, 1].imshow(err, cmap='hot', vmin=0, vmax=0.5); axes[1, 1].axis('off')

    if gt_mask is not None:
        gt_bin = (gt_mask > 0)
        seg_bin = (result.mask > 0)
        if gt_bin.shape != seg_bin.shape:
            gt_bin = cv2.resize(gt_mask.astype(np.uint8),
                                (seg_bin.shape[1], seg_bin.shape[0]),
                                interpolation=cv2.INTER_NEAREST) > 0
        overlay = np.stack([img_raw // 3] * 3, axis=-1)
        tp = seg_bin & gt_bin
        fn = (~seg_bin) & gt_bin
        fp = seg_bin & (~gt_bin)
        overlay[tp] = [0, 200, 0]
        overlay[fn] = [200, 0, 0]
        overlay[fp] = [0, 0, 200]
        tp_n, fn_n, fp_n = int(tp.sum()), int(fn.sum()), int(fp.sum())
        prec = tp_n / max(tp_n + fp_n, 1)
        rec = tp_n / max(tp_n + fn_n, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        axes[1, 2].set_title(f'Seg vs GT  (G=TP R=FN B=FP)\nP={prec:.3f} R={rec:.3f} F1={f1:.3f}')
        axes[1, 2].imshow(overlay)
    else:
        axes[1, 2].set_title('Seg vs GT (no GT)')
    axes[1, 2].axis('off')

    # Row 3
    skel_ov = cv2.cvtColor(img_raw, cv2.COLOR_GRAY2RGB)
    skel_ov[result.skeleton > 0] = [0, 255, 0]
    axes[2, 0].set_title(f'Skeleton ({np.count_nonzero(result.skeleton):,} px)')
    axes[2, 0].imshow(skel_ov); axes[2, 0].axis('off')

    pt_img = np.stack([img_raw // 3] * 3, axis=-1)
    colors = {'endpoint': [255, 0, 0], 'junction': [0, 0, 255], 'sampled': [0, 255, 0]}
    for pt, t in zip(result.points.astype(int), result.types):
        y, x = pt
        if 0 <= y < img_raw.shape[0] and 0 <= x < img_raw.shape[1]:
            c = colors.get(str(t), [255, 255, 255])
            pt_img[max(0, y-2):y+3, max(0, x-2):x+3] = c
    axes[2, 1].set_title(f'Nanograph Points ({result.n_points:,})')
    axes[2, 1].imshow(pt_img); axes[2, 1].axis('off')

    if result.graph is not None:
        graph_img = cv2.cvtColor(img_raw, cv2.COLOR_GRAY2RGB)
        for edge in result.graph.edges:
            src = result.graph.nodes[edge.source].position
            tgt = result.graph.nodes[edge.target].position
            cv2.line(graph_img, (src[1], src[0]), (tgt[1], tgt[0]), (0, 200, 200), 1)
        for node in result.graph.nodes:
            y, x = node.position
            color = {'endpoint': (255, 0, 0), 'junction': (0, 0, 255),
                     'sampled': (0, 180, 0)}.get(node.node_type, (255, 255, 255))
            cv2.circle(graph_img, (x, y), 2, color, -1)
        gs = result.graph.summary()
        axes[2, 2].set_title(f'Graph ({gs["n_nodes"]}N, {gs["n_edges"]}E, {gs["n_components"]}C)')
        axes[2, 2].imshow(graph_img)
    else:
        axes[2, 2].set_title('Graph (N/A)')
    axes[2, 2].axis('off')

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Nanograph v5 dataset evaluation')
    parser.add_argument('--images', required=True, help='Directory of original images')
    parser.add_argument('--masks', default=None, help='Directory of GT segmentation masks (optional)')
    parser.add_argument('--outdir', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset_results'),
                        help='Output directory')
    parser.add_argument('--n-samples', type=int, default=10,
                        help='Number of random sample visualizations to save')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling')
    parser.add_argument('--sam', default=None, help='Path to SAM checkpoint (optional)')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    samples_dir = os.path.join(args.outdir, 'samples')
    os.makedirs(samples_dir, exist_ok=True)

    # Collect images
    image_paths = sorted(glob.glob(os.path.join(args.images, '*.png')))
    if not image_paths:
        print(f'No .png images found in {args.images}')
        return
    n_total = len(image_paths)
    print(f'Found {n_total} images in {args.images}')
    print(f'Masks directory: {args.masks or "(none)"}')
    print(f'Output: {args.outdir}')
    print(f'Sample visualizations: {args.n_samples}')

    # Pick random sample indices
    random.seed(args.seed)
    sample_indices = set(random.sample(range(n_total), min(args.n_samples, n_total)))
    print(f'Sample indices: {sorted(sample_indices)}')

    # Load SAM if provided
    sam_model = None
    if args.sam:
        import torch
        from nanograph_v4.test_v4 import load_sam_model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        sam_model = load_sam_model(args.sam, device)

    cfg = NanographConfig()
    print(f'\nNanograph v5 — {cfg.param_count()} params')
    print(f'='*70)

    all_rows = []
    failures = []
    t_start = time.time()

    for i, img_path in enumerate(image_paths):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        is_sample = i in sample_indices

        # Progress every 50 images or on samples
        if i % 50 == 0 or is_sample:
            elapsed = time.time() - t_start
            rate = (i / elapsed) if elapsed > 0 and i > 0 else 0
            eta = ((n_total - i) / rate) if rate > 0 else 0
            print(f'\n[{i+1}/{n_total}] {stem}'
                  f'  ({rate:.1f} img/s, ETA {eta/60:.0f}m)'
                  f'{"  ** SAMPLE **" if is_sample else ""}')

        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f'  ERROR: Cannot read {img_path}')
                failures.append((stem, 'cannot read'))
                continue

            # Load GT mask
            gt_mask = None
            if args.masks:
                mask_path = os.path.join(args.masks, stem + '.png')
                if os.path.exists(mask_path):
                    gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            # Encode — verbose only for samples
            result = nanograph_encode(img, sam_model=sam_model,
                                      verbose=is_sample, config=cfg)

            # Evaluate
            if gt_mask is not None:
                row = evaluate_result_with_gt(result, img, gt_mask, cfg=cfg)
            else:
                row = evaluate_result(result, img, cfg=cfg)
                row['gt_available'] = False

            row['filename'] = os.path.basename(img_path)
            row['stem'] = stem

            # Graph topology
            if result.graph is not None:
                topo = graph_topology_score(result.graph)
                row['graph_n_components'] = topo['n_components']
                row['graph_n_cycles'] = topo['n_cycles']
                row['graph_n_junctions'] = topo['n_junctions']
                row['graph_n_endpoints'] = topo['n_endpoints']
                row['graph_mean_degree'] = topo['mean_degree']
                row['graph_complexity'] = topo['graph_complexity']

            # Morphometry
            if result.graph is not None:
                morph = nanograph_morphometry(result.graph)
                row['n_nodes'] = morph.get('n_nodes', 0)
                row['n_edges'] = morph.get('n_edges', 0)
                row['total_edge_length'] = morph.get('total_edge_length', 0)
                row['mean_width_px'] = morph.get('mean_width_px', 0)
                row['mean_curvature'] = morph.get('mean_curvature', 0)

            all_rows.append(row)

            # Sample visualization
            if is_sample:
                save_path = os.path.join(samples_dir, f'{stem}.png')
                try:
                    plot_sample(result, img, gt_mask,
                                f'Nanograph v5 — {stem}  |  '
                                f'{row["ng_bytes"]:,}B  FG-SSIM={row["ng_fg_ssim"]:.4f}  '
                                f'GT-IoU={row.get("gt_iou", 0):.4f}',
                                save_path)
                    print(f'  Sample saved: {save_path}')
                except Exception as e:
                    print(f'  WARNING: plot failed: {e}')

                # Print key metrics for sample
                print(f'  Bytes: {row["ng_bytes"]:,}  |  '
                      f'FG-SSIM: {row["ng_fg_ssim"]:.4f}  |  '
                      f'GT-IoU: {row.get("gt_iou", 0):.4f}  |  '
                      f'Seg-F1: {row.get("seg_f1", 0):.4f}  |  '
                      f'Topo-Q: {row.get("ng_topo_q", 0):.4f}')

        except Exception as e:
            print(f'  ERROR [{stem}]: {e}')
            traceback.print_exc()
            failures.append((stem, str(e)))
            continue

    elapsed_total = time.time() - t_start
    n_done = len(all_rows)

    if not all_rows:
        print('No results collected.')
        return

    # ====================== SAVE CSV ======================
    # Determine all keys
    all_keys = []
    for r in all_rows:
        for k in r:
            if k not in all_keys:
                all_keys.append(k)

    # Priority columns
    priority = [
        'filename', 'gt_iou', 'seg_iou', 'seg_f1', 'seg_precision', 'seg_recall',
        'gt_fg_ssim', 'gt_fg_psnr', 'ng_fg_ssim', 'ng_fg_psnr', 'ng_iou',
        'ng_topo_q', 'topo_preservation',
        'ng_psnr', 'ng_ssim', 'ng_bytes',
        'graph_n_components', 'graph_n_cycles', 'graph_n_junctions',
        'graph_n_endpoints', 'graph_mean_degree', 'graph_complexity',
        'n_nodes', 'n_edges', 'total_edge_length', 'mean_width_px', 'mean_curvature',
        'delta_gt_iou', 'delta_gt_fg_psnr', 'delta_gt_fg_ssim',
        'delta_fg_ssim', 'delta_iou',
        'total_time_ms',
    ]
    ordered = [k for k in priority if k in all_keys]
    ordered += [k for k in all_keys if k not in ordered]

    csv_path = os.path.join(args.outdir, 'metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=ordered, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_rows)
    print(f'\nCSV saved: {csv_path}')

    # Also write aligned text table
    txt_path = os.path.join(args.outdir, 'metrics.txt')
    with open(txt_path, 'w') as f:
        # Header
        f.write('\t'.join(ordered) + '\n')
        for row in all_rows:
            vals = []
            for k in ordered:
                v = row.get(k, '')
                if isinstance(v, float):
                    vals.append(f'{v:.6f}')
                else:
                    vals.append(str(v))
            f.write('\t'.join(vals) + '\n')
    print(f'TXT saved: {txt_path}')

    # ====================== SUMMARY ======================
    summary_lines = []
    summary_lines.append(f'NANOGRAPH v5 — DATASET EVALUATION SUMMARY')
    summary_lines.append(f'=' * 70)
    summary_lines.append(f'  Images processed: {n_done}/{n_total}')
    summary_lines.append(f'  Failures: {len(failures)}')
    summary_lines.append(f'  Total time: {elapsed_total:.1f}s  '
                         f'({elapsed_total/n_done:.2f}s/image)')
    summary_lines.append('')
    summary_lines.append(f'  {"Metric":<28} {"Mean":>9} {"Std":>9} '
                         f'{"Min":>9} {"Max":>9} {"Median":>9}')
    summary_lines.append(f'  {"-"*74}')

    stat_keys = [
        ('gt_iou', 'GT IoU'),
        ('seg_iou', 'Seg IoU'),
        ('seg_f1', 'Seg F1'),
        ('seg_precision', 'Seg Precision'),
        ('seg_recall', 'Seg Recall'),
        ('gt_fg_ssim', 'GT FG-SSIM'),
        ('gt_fg_psnr', 'GT FG-PSNR'),
        ('ng_fg_ssim', 'FG-SSIM'),
        ('ng_fg_psnr', 'FG-PSNR'),
        ('ng_iou', 'IoU (recon)'),
        ('ng_topo_q', 'Topology Q'),
        ('topo_preservation', 'Topo Preservation'),
        ('ng_psnr', 'Full PSNR'),
        ('ng_ssim', 'Full SSIM'),
        ('ng_bytes', 'Bytes'),
        ('graph_n_components', 'Graph Components'),
        ('graph_n_cycles', 'Graph Cycles'),
        ('graph_complexity', 'Graph Complexity'),
        ('n_nodes', 'Nodes'),
        ('n_edges', 'Edges'),
        ('total_edge_length', 'Total Edge Length'),
        ('mean_width_px', 'Mean Width (px)'),
        ('mean_curvature', 'Mean Curvature'),
        ('total_time_ms', 'Time (ms)'),
    ]
    for key, label in stat_keys:
        vals = [r[key] for r in all_rows
                if key in r and isinstance(r[key], (int, float))
                and not np.isnan(r[key]) and not np.isinf(r[key])]
        if vals:
            summary_lines.append(
                f'  {label:<28} {np.mean(vals):>9.4f} {np.std(vals):>9.4f} '
                f'{np.min(vals):>9.4f} {np.max(vals):>9.4f} {np.median(vals):>9.4f}')

    # Wins vs JPEG
    summary_lines.append('')
    summary_lines.append(f'  WINS vs byte-matched JPEG ({n_done} images):')
    for delta_key, label in [
        ('delta_gt_iou', 'GT-IoU'),
        ('delta_gt_fg_ssim', 'GT-FG-SSIM'),
        ('delta_gt_fg_psnr', 'GT-FG-PSNR'),
        ('delta_fg_ssim', 'FG-SSIM'),
        ('delta_iou', 'IoU'),
    ]:
        vals = [r.get(delta_key, 0) for r in all_rows]
        wins = sum(1 for v in vals if v > 0)
        ties = sum(1 for v in vals if v == 0)
        losses = n_done - wins - ties
        mean_delta = np.mean([v for v in vals if isinstance(v, (int, float))])
        summary_lines.append(
            f'    {label:<16}: {wins}/{n_done} wins ({100*wins/n_done:.0f}%)  '
            f'mean Δ={mean_delta:+.4f}')

    # Failures
    if failures:
        summary_lines.append(f'\n  FAILURES ({len(failures)}):')
        for stem, err in failures[:20]:
            summary_lines.append(f'    - {stem}: {err}')
        if len(failures) > 20:
            summary_lines.append(f'    ... and {len(failures)-20} more')

    # Seg failures
    seg_fails = [r['filename'] for r in all_rows
                 if r.get('seg_iou', 1.0) < 0.3]
    if seg_fails:
        summary_lines.append(f'\n  SEGMENTATION FAILURES (seg_iou < 0.3): {len(seg_fails)}')
        for s in seg_fails[:20]:
            summary_lines.append(f'    - {s}')
    else:
        summary_lines.append(f'\n  No segmentation failures (all seg_iou >= 0.3)')

    summary_text = '\n'.join(summary_lines)
    print('\n' + summary_text)

    summary_path = os.path.join(args.outdir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write(summary_text + '\n')
    print(f'\nSummary saved: {summary_path}')

    # ====================== AGGREGATE PLOTS ======================
    _plot_aggregate(all_rows, args.outdir)
    print(f'\nDone. All results in {args.outdir}/')


def _plot_aggregate(rows, outdir):
    """Generate aggregate distribution and comparison plots."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. GT IoU histogram
    gt_ious = [r['gt_iou'] for r in rows if 'gt_iou' in r]
    if gt_ious:
        axes[0, 0].hist(gt_ious, bins=50, color='steelblue', edgecolor='white')
        axes[0, 0].axvline(np.mean(gt_ious), color='red', ls='--',
                           label=f'mean={np.mean(gt_ious):.3f}')
        axes[0, 0].set_title('GT IoU Distribution')
        axes[0, 0].set_xlabel('GT IoU')
        axes[0, 0].legend()

    # 2. FG-SSIM histogram
    fg_ssims = [r['ng_fg_ssim'] for r in rows if 'ng_fg_ssim' in r]
    if fg_ssims:
        axes[0, 1].hist(fg_ssims, bins=50, color='forestgreen', edgecolor='white')
        axes[0, 1].axvline(np.mean(fg_ssims), color='red', ls='--',
                           label=f'mean={np.mean(fg_ssims):.4f}')
        axes[0, 1].set_title('FG-SSIM Distribution')
        axes[0, 1].set_xlabel('FG-SSIM')
        axes[0, 1].legend()

    # 3. Seg F1 histogram
    seg_f1s = [r['seg_f1'] for r in rows if 'seg_f1' in r]
    if seg_f1s:
        axes[0, 2].hist(seg_f1s, bins=50, color='darkorange', edgecolor='white')
        axes[0, 2].axvline(np.mean(seg_f1s), color='red', ls='--',
                           label=f'mean={np.mean(seg_f1s):.3f}')
        axes[0, 2].set_title('Segmentation F1 Distribution')
        axes[0, 2].set_xlabel('F1')
        axes[0, 2].legend()

    # 4. Bytes distribution
    ng_bytes = [r['ng_bytes'] for r in rows if 'ng_bytes' in r]
    if ng_bytes:
        axes[1, 0].hist(ng_bytes, bins=50, color='purple', edgecolor='white')
        axes[1, 0].axvline(np.mean(ng_bytes), color='red', ls='--',
                           label=f'mean={np.mean(ng_bytes):.0f}B')
        axes[1, 0].set_title('Compressed Size Distribution')
        axes[1, 0].set_xlabel('Bytes')
        axes[1, 0].legend()

    # 5. Nanograph vs JPEG: delta GT-IoU scatter
    delta_ious = [r.get('delta_gt_iou', 0) for r in rows]
    ng_bs = [r['ng_bytes'] for r in rows if 'ng_bytes' in r]
    if delta_ious and ng_bs and len(delta_ious) == len(ng_bs):
        colors_scatter = ['green' if d > 0 else 'red' for d in delta_ious]
        axes[1, 1].scatter(ng_bs, delta_ious, c=colors_scatter, s=8, alpha=0.5)
        axes[1, 1].axhline(0, color='black', ls='-', lw=0.5)
        wins = sum(1 for d in delta_ious if d > 0)
        axes[1, 1].set_title(f'ΔGT-IoU vs JPEG  (wins: {wins}/{len(delta_ious)})')
        axes[1, 1].set_xlabel('Nanograph Bytes')
        axes[1, 1].set_ylabel('GT-IoU (Nanograph - JPEG)')

    # 6. Graph complexity distribution
    complexities = [r.get('graph_complexity', 0) for r in rows]
    if complexities:
        axes[1, 2].hist(complexities, bins=50, color='teal', edgecolor='white')
        axes[1, 2].axvline(np.mean(complexities), color='red', ls='--',
                           label=f'mean={np.mean(complexities):.2f}')
        axes[1, 2].set_title('Graph Complexity (cycles/component)')
        axes[1, 2].set_xlabel('Complexity')
        axes[1, 2].legend()

    plt.suptitle(f'Nanograph v5 — Dataset Evaluation ({len(rows)} images)', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'aggregate_plots.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Aggregate plots saved: {os.path.join(outdir, "aggregate_plots.png")}')


if __name__ == '__main__':
    main()
