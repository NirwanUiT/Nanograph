#!/usr/bin/env python3
"""
Nanograph v5 — Test / batch-evaluation script.

Single-image mode (sparse + dense pair):
    python test_v4.py --sparse /path/to/sparse.png --dense /path/to/dense.jpg

Batch mode with ground-truth masks:
    python test_v4.py --batch /path/to/images/ --masks /path/to/masks/ --outdir ./batch_results/

Optional SAM model:
    ... --sam /path/to/vit_b_lm.pt

Rate-distortion curves:
    python test_v4.py --sparse ... --dense ... --rd-curves
"""

import argparse
import sys
import os
import time
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt

# Add grandparent dir to path so nanograph_v4 can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanograph_v4 import (nanograph_encode, nanograph_decode, NanographConfig,
                           nanograph_morphometry, print_comparison,
                           rate_distortion_curve,
                           compute_betti_numbers, topology_preservation_score)
from nanograph_v4.evaluate import (evaluate_result, evaluate_result_with_gt,
                                    batch_evaluate_with_gt)


def load_sam_model(checkpoint_path, device='cuda'):
    """Load micro-SAM model if checkpoint exists."""
    import torch
    from segment_anything import sam_model_registry

    if not os.path.exists(checkpoint_path):
        print(f'WARNING: SAM checkpoint not found at {checkpoint_path}')
        return None

    sam = sam_model_registry['vit_b'](checkpoint=None)
    state = torch.load(checkpoint_path, map_location=device)
    if 'model_state' in state:
        ms = state['model_state']
        ms = {(k[4:] if k.startswith('sam.') else k): v for k, v in ms.items()}
        sam.load_state_dict(ms)
    else:
        sam.load_state_dict(state)
    sam.to(device).eval()
    print(f'micro-SAM loaded on {device} ({checkpoint_path})')
    return sam


def plot_results(result, img_raw, title, save_path):
    """Create a comprehensive visualization of the result."""
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))

    # Row 1: Pipeline stages
    axes[0, 0].set_title(f'Original ({result.shape[0]}×{result.shape[1]})')
    axes[0, 0].imshow(img_raw, cmap='gray')
    axes[0, 0].axis('off')

    axes[0, 1].set_title(f'Segmentation ({result.segmenter})\nType: {result.image_type}')
    axes[0, 1].imshow(result.mask, cmap='gray')
    axes[0, 1].axis('off')

    skel_ov = cv2.cvtColor(img_raw, cv2.COLOR_GRAY2RGB)
    skel_ov[result.skeleton > 0] = [0, 255, 0]
    axes[0, 2].set_title(f'Skeleton ({np.count_nonzero(result.skeleton):,} px)')
    axes[0, 2].imshow(skel_ov)
    axes[0, 2].axis('off')

    axes[0, 3].set_title(f'Reconstruction\nPSNR={result.psnr_full:.2f} SSIM={result.ssim_full:.4f}')
    axes[0, 3].imshow(result.reconstruction, cmap='gray')
    axes[0, 3].axis('off')

    # Row 2: Analysis
    err = np.abs(img_raw.astype(float) / 255.0 - result.reconstruction)
    axes[1, 0].set_title(f'Error Map\nFG-PSNR={result.psnr_fg:.2f} FG-SSIM={result.ssim_fg:.4f}')
    axes[1, 0].imshow(err, cmap='hot', vmin=0, vmax=0.5)
    axes[1, 0].axis('off')

    # Nanograph points colored by type
    pt_img = np.zeros((*img_raw.shape, 3), dtype=np.uint8)
    pt_img[:, :, 0] = img_raw // 3
    pt_img[:, :, 1] = img_raw // 3
    pt_img[:, :, 2] = img_raw // 3
    colors = {'endpoint': [255, 0, 0], 'junction': [0, 0, 255], 'sampled': [0, 255, 0]}
    for pt, t in zip(result.points.astype(int), result.types):
        y, x = pt
        if 0 <= y < img_raw.shape[0] and 0 <= x < img_raw.shape[1]:
            c = colors.get(str(t), [255, 255, 255])
            r = 2
            pt_img[max(0, y-r):y+r+1, max(0, x-r):x+r+1] = c
    axes[1, 1].set_title(f'Nanograph Points ({result.n_points:,})\n'
                          f'R=EP, B=JN, G=Sampled')
    axes[1, 1].imshow(pt_img)
    axes[1, 1].axis('off')

    # Orientation visualization — quiver arrows on original image
    axes[1, 2].imshow(img_raw, cmap='gray')
    if result.orientations is not None:
        pts_int = result.points.astype(int)
        valid = (
            (pts_int[:, 0] >= 0) & (pts_int[:, 0] < img_raw.shape[0]) &
            (pts_int[:, 1] >= 0) & (pts_int[:, 1] < img_raw.shape[1])
        )
        ys = pts_int[valid, 0]
        xs = pts_int[valid, 1]
        oris = result.orientations[valid]
        u = np.cos(oris)
        v = -np.sin(oris)
        axes[1, 2].quiver(xs, ys, u, v, color='yellow',
                          scale=30, headwidth=2, headlength=2,
                          width=0.003, alpha=0.85)
        axes[1, 2].set_title(f'Orientations ({valid.sum():,} pts)')
    else:
        axes[1, 2].set_title('Orientations (none)')
    axes[1, 2].axis('off')

    # Graph visualization
    if result.graph is not None:
        graph_img = cv2.cvtColor(img_raw, cv2.COLOR_GRAY2RGB)
        for edge in result.graph.edges:
            src = result.graph.nodes[edge.source].position
            tgt = result.graph.nodes[edge.target].position
            cv2.line(graph_img, (src[1], src[0]), (tgt[1], tgt[0]),
                     (0, 200, 200), 1)
        for node in result.graph.nodes:
            y, x = node.position
            color = {'endpoint': (255, 0, 0), 'junction': (0, 0, 255),
                     'sampled': (0, 180, 0)}.get(node.node_type, (255, 255, 255))
            cv2.circle(graph_img, (x, y), 2, color, -1)
        gs = result.graph.summary()
        axes[1, 3].set_title(f'Graph ({gs["n_nodes"]}N, {gs["n_edges"]}E, '
                              f'{gs["n_components"]}C)')
        axes[1, 3].imshow(graph_img)
        axes[1, 3].axis('off')
    else:
        axes[1, 3].axis('off')

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Visualization saved: {save_path}')


def plot_result_with_gt(result, img_raw, gt_mask, title, save_path):
    """
    Visualization with an extra panel comparing pipeline segmentation to GT.

    Panel layout (3 rows × 3 cols):
      Row 1: Original | Segmentation | GT mask
      Row 2: Reconstruction | Error map | Seg vs GT overlay
      Row 3: Skeleton | Nanograph points | Graph
    """
    fig, axes = plt.subplots(3, 3, figsize=(21, 21))

    # ---- Row 1 ----
    axes[0, 0].set_title(f'Original ({result.shape[0]}×{result.shape[1]})')
    axes[0, 0].imshow(img_raw, cmap='gray')
    axes[0, 0].axis('off')

    axes[0, 1].set_title(f'Segmentation ({result.segmenter})\nType: {result.image_type}')
    axes[0, 1].imshow(result.mask, cmap='gray')
    axes[0, 1].axis('off')

    if gt_mask is not None:
        axes[0, 2].set_title('Ground Truth Mask')
        axes[0, 2].imshow(gt_mask, cmap='gray')
    else:
        axes[0, 2].set_title('Ground Truth (not provided)')
    axes[0, 2].axis('off')

    # ---- Row 2 ----
    axes[1, 0].set_title(f'Reconstruction\nPSNR={result.psnr_full:.2f} '
                          f'SSIM={result.ssim_full:.4f}')
    axes[1, 0].imshow(result.reconstruction, cmap='gray')
    axes[1, 0].axis('off')

    err = np.abs(img_raw.astype(float) / 255.0 - result.reconstruction)
    axes[1, 1].set_title(f'Error Map\nFG-PSNR={result.psnr_fg:.2f} '
                          f'FG-SSIM={result.ssim_fg:.4f}')
    axes[1, 1].imshow(err, cmap='hot', vmin=0, vmax=0.5)
    axes[1, 1].axis('off')

    # Seg vs GT overlay: green=TP, red=FN, blue=FP
    if gt_mask is not None:
        gt_bin = (gt_mask > 0)
        seg_bin = (result.mask > 0)
        # Resize GT if needed
        if gt_bin.shape != seg_bin.shape:
            gt_resized = cv2.resize(gt_mask.astype(np.uint8),
                                    (seg_bin.shape[1], seg_bin.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)
            gt_bin = gt_resized > 0
        overlay = np.zeros((*img_raw.shape, 3), dtype=np.uint8)
        overlay[:, :, 0] = img_raw // 3
        overlay[:, :, 1] = img_raw // 3
        overlay[:, :, 2] = img_raw // 3
        tp = seg_bin & gt_bin
        fn = (~seg_bin) & gt_bin
        fp = seg_bin & (~gt_bin)
        overlay[tp] = [0, 200, 0]    # green = true positive
        overlay[fn] = [200, 0, 0]    # red   = false negative
        overlay[fp] = [0, 0, 200]    # blue  = false positive
        tp_n = int(tp.sum())
        fn_n = int(fn.sum())
        fp_n = int(fp.sum())
        prec = tp_n / max(tp_n + fp_n, 1)
        rec  = tp_n / max(tp_n + fn_n, 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-9)
        axes[1, 2].set_title(f'Seg vs GT\nGreen=TP  Red=FN  Blue=FP\n'
                              f'P={prec:.3f} R={rec:.3f} F1={f1:.3f}')
        axes[1, 2].imshow(overlay)
    else:
        axes[1, 2].set_title('Seg vs GT (no GT)')
    axes[1, 2].axis('off')

    # ---- Row 3 ----
    skel_ov = cv2.cvtColor(img_raw, cv2.COLOR_GRAY2RGB)
    skel_ov[result.skeleton > 0] = [0, 255, 0]
    axes[2, 0].set_title(f'Skeleton ({np.count_nonzero(result.skeleton):,} px)')
    axes[2, 0].imshow(skel_ov)
    axes[2, 0].axis('off')

    pt_img = np.zeros((*img_raw.shape, 3), dtype=np.uint8)
    pt_img[:, :, 0] = img_raw // 3
    pt_img[:, :, 1] = img_raw // 3
    pt_img[:, :, 2] = img_raw // 3
    colors = {'endpoint': [255, 0, 0], 'junction': [0, 0, 255], 'sampled': [0, 255, 0]}
    for pt, t in zip(result.points.astype(int), result.types):
        y, x = pt
        if 0 <= y < img_raw.shape[0] and 0 <= x < img_raw.shape[1]:
            c = colors.get(str(t), [255, 255, 255])
            r = 2
            pt_img[max(0, y-r):y+r+1, max(0, x-r):x+r+1] = c
    axes[2, 1].set_title(f'Nanograph Points ({result.n_points:,})\n'
                          f'R=EP, B=JN, G=Sampled')
    axes[2, 1].imshow(pt_img)
    axes[2, 1].axis('off')

    if result.graph is not None:
        graph_img = cv2.cvtColor(img_raw, cv2.COLOR_GRAY2RGB)
        for edge in result.graph.edges:
            src = result.graph.nodes[edge.source].position
            tgt = result.graph.nodes[edge.target].position
            cv2.line(graph_img, (src[1], src[0]), (tgt[1], tgt[0]),
                     (0, 200, 200), 1)
        for node in result.graph.nodes:
            y, x = node.position
            color = {'endpoint': (255, 0, 0), 'junction': (0, 0, 255),
                     'sampled': (0, 180, 0)}.get(node.node_type, (255, 255, 255))
            cv2.circle(graph_img, (x, y), 2, color, -1)
        gs = result.graph.summary()
        axes[2, 2].set_title(f'Graph ({gs["n_nodes"]}N, {gs["n_edges"]}E, '
                              f'{gs["n_components"]}C)')
        axes[2, 2].imshow(graph_img)
        axes[2, 2].axis('off')
    else:
        axes[2, 2].axis('off')

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'  Visualization saved: {save_path}')


def main():
    parser = argparse.ArgumentParser(description='Nanograph v5 test / batch eval')
    # Single-image mode
    parser.add_argument('--sparse', default=None, help='Path to sparse organelle image')
    parser.add_argument('--dense', default=None, help='Path to dense network image')
    # Batch mode
    parser.add_argument('--batch', default=None,
                        help='Directory of images for batch evaluation')
    parser.add_argument('--masks', default=None,
                        help='Directory of ground-truth masks (batch mode)')
    # Common
    parser.add_argument('--sam', default=None, help='Path to SAM checkpoint (optional)')
    parser.add_argument('--outdir', default='./v5_results', help='Output directory')
    parser.add_argument('--no-graph', action='store_true', help='Skip graph construction')
    parser.add_argument('--rd-curves', action='store_true',
                        help='Generate rate-distortion curves (slower)')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load SAM model if provided
    sam_model = None
    if args.sam:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        sam_model = load_sam_model(args.sam, device)

    # ----------------------------------------------------------------
    # Batch mode
    # ----------------------------------------------------------------
    if args.batch:
        if args.masks:
            print(f'Batch mode: images={args.batch}, masks={args.masks}')
        else:
            print(f'Batch mode: images={args.batch} (no GT masks)')
        batch_evaluate_with_gt(
            image_dir=args.batch,
            mask_dir=args.masks,
            sam_model=sam_model,
            output_dir=args.outdir,
            verbose=True,
            plot_fn=plot_result_with_gt,
        )
        return

    # ----------------------------------------------------------------
    # Single-image mode (requires --sparse and --dense)
    # ----------------------------------------------------------------
    if not args.sparse or not args.dense:
        parser.error('Provide --sparse and --dense for single-image mode, '
                     'or --batch for batch mode.')

    cfg = NanographConfig()
    print(f'Nanograph v5 — {cfg.param_count()} configurable parameters')
    print(f'Features: oriented_psf={cfg.recon.use_oriented_psf}, '
          f'cascade={cfg.segment.cascade_enable}, '
          f'bg_grid={cfg.recon.bg_grid_size}x{cfg.recon.bg_grid_size}, '
          f'fg_residual={cfg.recon.use_fg_residual}, '
          f'graph_predictive={cfg.compress.graph_predictive}')

    # --- Test 1: Sparse ---
    print(f'\n{"="*60}')
    print(f'  TEST 1: Sparse organelle image')
    print(f'{"="*60}')
    img_sp = cv2.imread(args.sparse, cv2.IMREAD_GRAYSCALE)
    if img_sp is None:
        print(f'ERROR: Cannot read {args.sparse}')
        return

    t0 = time.time()
    result_sparse = nanograph_encode(img_sp, sam_model=sam_model,
                                      build_graph=not args.no_graph,
                                      verbose=True, config=cfg)
    t_sparse = time.time() - t0

    row_sp = print_comparison(result_sparse, img_sp, label='Sparse')
    row_sp['filename'] = os.path.basename(args.sparse)
    plot_results(result_sparse, img_sp, 'Nanograph v5 — Sparse Organelles',
                 os.path.join(args.outdir, 'sparse_result.png'))

    # Graph morphometry
    if result_sparse.graph:
        morph = nanograph_morphometry(result_sparse.graph)
        print(f'\n  Morphometry:')
        for k, v in morph.items():
            if isinstance(v, float):
                print(f'    {k}: {v:.3f}')
            elif isinstance(v, dict):
                print(f'    {k}: {v}')
            else:
                print(f'    {k}: {v}')

    # Round-trip test
    recon_rt, pts_rt, _, _, _, oris_rt, shape_rt = nanograph_decode(
        result_sparse.compressed, cfg=cfg)
    print(f'\n  Round-trip: {len(pts_rt)} pts, shape={shape_rt}, '
          f'has_orientations={np.any(oris_rt > 0)}')

    # --- Test 2: Dense ---
    print(f'\n{"="*60}')
    print(f'  TEST 2: Dense filamentous network')
    print(f'{"="*60}')
    img_dn = cv2.imread(args.dense, cv2.IMREAD_GRAYSCALE)
    if img_dn is None:
        print(f'ERROR: Cannot read {args.dense}')
        return

    t0 = time.time()
    result_dense = nanograph_encode(img_dn, sam_model=sam_model,
                                     build_graph=not args.no_graph,
                                     verbose=True, config=cfg)
    t_dense = time.time() - t0

    row_dn = print_comparison(result_dense, img_dn, label='Dense')
    row_dn['filename'] = os.path.basename(args.dense)
    plot_results(result_dense, img_dn, 'Nanograph v5 — Dense Network',
                 os.path.join(args.outdir, 'dense_result.png'))

    if result_dense.graph:
        morph = nanograph_morphometry(result_dense.graph)
        print(f'\n  Morphometry:')
        for k, v in morph.items():
            if isinstance(v, float):
                print(f'    {k}: {v:.3f}')
            elif isinstance(v, dict):
                print(f'    {k}: {v}')
            else:
                print(f'    {k}: {v}')

    # --- Side-by-side summary ---
    print(f'\n{"="*60}')
    print(f'  v5 COMPARISON SUMMARY')
    print(f'{"="*60}')
    print(f'  {"Metric":<30} {"Sparse":>12} {"Dense":>12}')
    print(f'  {"-"*55}')
    summary_keys = [
        'image_type', 'segmenter', 'n_points', 'ng_bytes',
        'ng_psnr', 'ng_ssim', 'ng_fg_psnr', 'ng_fg_ssim',
        'ng_iou', 'ng_topo_q',
        # v5 graph topology
        'ng_n_components', 'ng_n_cycles', 'ng_n_junctions', 'ng_n_endpoints',
        'ng_graph_complexity', 'ng_seg_beta_0', 'ng_seg_beta_1',
        # multi-codec
        'webp_psnr', 'webp_ssim', 'webp_fg_ssim',
        'jp2_psnr', 'jp2_ssim', 'jp2_fg_ssim',
        # delta
        'delta_fg_ssim', 'delta_fg_psnr', 'delta_iou',
        'total_time_ms',
    ]
    for key in summary_keys:
        v1 = row_sp.get(key, 'N/A')
        v2 = row_dn.get(key, 'N/A')
        if isinstance(v1, float):
            print(f'  {key:<30} {v1:>12.4f} {v2:>12.4f}')
        elif isinstance(v1, int):
            print(f'  {key:<30} {v1:>12,} {v2:>12,}')
        else:
            print(f'  {key:<30} {str(v1):>12} {str(v2):>12}')

    if result_sparse.graph and result_dense.graph:
        print(f'\n  {"Graph Metric":<25} {"Sparse":>12} {"Dense":>12}')
        print(f'  {"-"*50}')
        gs1 = result_sparse.graph.summary()
        gs2 = result_dense.graph.summary()
        for key in ['n_nodes', 'n_edges', 'n_components', 'mean_degree',
                     'mean_edge_length', 'total_edge_length', 'mean_curvature']:
            v1 = gs1.get(key, 0)
            v2 = gs2.get(key, 0)
            if isinstance(v1, float):
                print(f'  {key:<25} {v1:>12.2f} {v2:>12.2f}')
            else:
                print(f'  {key:<25} {v1:>12} {v2:>12}')

    # --- Helper: transposed table (metric per row, image per column) ---
    def _fmt(v):
        if isinstance(v, float):
            return f'{v:.4f}' if abs(v) < 100 else f'{v:.2f}'
        return str(v)

    def _write_transposed(path, rows, col_keys, col_labels):
        fmt = {k: [_fmt(r.get(k, '')) for r in rows] for k in col_keys}
        label_w = max(len(str(k)) for k in col_keys) + 2
        val_ws = [max(max((len(fmt[k][i]) for k in col_keys), default=0),
                      len(col_labels[i])) + 2
                  for i in range(len(rows))]
        with open(path, 'w') as f:
            f.write('Metric'.ljust(label_w))
            for lbl, w in zip(col_labels, val_ws):
                f.write(lbl.rjust(w))
            f.write('\n' + '-' * (label_w + sum(val_ws)) + '\n')
            for k in col_keys:
                f.write(str(k).ljust(label_w))
                for i, w in enumerate(val_ws):
                    f.write(fmt[k][i].rjust(w))
                f.write('\n')

    # --- Save per-image metrics (transposed table) ---
    metrics_rows = [row_sp, row_dn]
    metrics_path = os.path.join(args.outdir, 'metrics.txt')
    all_keys = list(metrics_rows[0].keys())
    for k in metrics_rows[1].keys():
        if k not in all_keys:
            all_keys.append(k)
    primary_cols = ['filename', 'ng_fg_ssim', 'ng_fg_psnr', 'ng_iou', 'ng_topo_q']
    ordered_keys = [k for k in primary_cols if k in all_keys]
    ordered_keys += [k for k in all_keys if k not in ordered_keys]
    _write_transposed(metrics_path, metrics_rows, ordered_keys,
                      ['Sparse', 'Dense'])

    # --- Save per-image graph_summary (transposed table) ---
    graph_rows = []
    graph_labels = []
    for result, fname in [(result_sparse, os.path.basename(args.sparse)),
                           (result_dense, os.path.basename(args.dense))]:
        if result.graph:
            gs = result.graph.summary()
            gs['filename'] = fname
            graph_rows.append(gs)
            graph_labels.append(result.image_type.capitalize())

    graph_path = None
    if graph_rows:
        graph_path = os.path.join(args.outdir, 'graph_summary.txt')
        gs_keys = ['filename'] + [
            k for k in graph_rows[0].keys() if k != 'filename']
        _write_transposed(graph_path, graph_rows, gs_keys, graph_labels)

    print(f'\nMetrics table:        {metrics_path}')
    if graph_path:
        print(f'Graph summary table:  {graph_path}')

    # --- v5: Rate-distortion curves ---
    if args.rd_curves:
        print(f'\n{"="*60}')
        print(f'  RATE-DISTORTION CURVES')
        print(f'{"="*60}')
        for label, img, fname in [('Sparse', img_sp, args.sparse),
                                    ('Dense', img_dn, args.dense)]:
            print(f'\n  Generating R-D curve for {label}...')
            rd_pts = rate_distortion_curve(img, sam_model=sam_model,
                                            n_points=cfg.eval.rate_distortion_points,
                                            cfg=cfg, verbose=True)
            if rd_pts:
                rd_path = os.path.join(args.outdir, f'rd_curve_{label.lower()}.png')
                _plot_rd_curve(rd_pts, label, rd_path)

    print(f'\nDone. Results saved to {args.outdir}/')


def _plot_rd_curve(rd_points, label, save_path):
    """Plot rate-distortion curve comparing Nanograph vs baselines."""
    if not rd_points:
        return

    ng_bpp = [p['ng_bpp'] for p in rd_points]
    ng_psnr = [p['ng_psnr'] for p in rd_points]
    ng_ssim = [p['ng_ssim'] for p in rd_points]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # PSNR vs BPP
    ax1.plot(ng_bpp, ng_psnr, 'o-', color='#2196F3', linewidth=2,
             markersize=6, label='Nanograph', zorder=3)
    jpeg_psnr = [p.get('jpeg_psnr', 0) for p in rd_points]
    if any(v > 0 for v in jpeg_psnr):
        ax1.plot(ng_bpp, jpeg_psnr, 's--', color='#FF5722', linewidth=1.5,
                 markersize=5, label='JPEG (byte-matched)', alpha=0.8)
    webp_psnr = [p.get('webp_psnr', 0) for p in rd_points]
    if any(v > 0 for v in webp_psnr):
        ax1.plot(ng_bpp, webp_psnr, '^--', color='#4CAF50', linewidth=1.5,
                 markersize=5, label='WebP (byte-matched)', alpha=0.8)
    ax1.set_xlabel('Bits per pixel (bpp)')
    ax1.set_ylabel('PSNR (dB)')
    ax1.set_title(f'{label} — Rate-Distortion (PSNR)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # SSIM vs BPP
    ax2.plot(ng_bpp, ng_ssim, 'o-', color='#2196F3', linewidth=2,
             markersize=6, label='Nanograph', zorder=3)
    jpeg_ssim = [p.get('jpeg_ssim', 0) for p in rd_points]
    if any(v > 0 for v in jpeg_ssim):
        ax2.plot(ng_bpp, jpeg_ssim, 's--', color='#FF5722', linewidth=1.5,
                 markersize=5, label='JPEG (byte-matched)', alpha=0.8)
    webp_ssim = [p.get('webp_ssim', 0) for p in rd_points]
    if any(v > 0 for v in webp_ssim):
        ax2.plot(ng_bpp, webp_ssim, '^--', color='#4CAF50', linewidth=1.5,
                 markersize=5, label='WebP (byte-matched)', alpha=0.8)
    ax2.set_xlabel('Bits per pixel (bpp)')
    ax2.set_ylabel('SSIM')
    ax2.set_title(f'{label} — Rate-Distortion (SSIM)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  R-D curve saved: {save_path}')


# ---------------------------------------------------------------------------
# pytest acceptance tests (T1: stored edges / decode_graph round trip)
# ---------------------------------------------------------------------------
_ORG_IMAGES = '/mnt/nas1/nba055-2/idea_1/nmi_data/org'
_CROSS_IMAGES = {
    'cells3d_membrane': '/mnt/nas1/nba055-2/idea_1/datasets/cells3d_membrane/images',
    'cells3d_nuclei': '/mnt/nas1/nba055-2/idea_1/datasets/cells3d_nuclei/images',
    'retina': '/mnt/nas1/nba055-2/idea_1/datasets/retina/images',
    'cell': '/mnt/nas1/nba055-2/idea_1/datasets/cell/images',
}


def _beta0(n_nodes, adjacency):
    seen = set()
    comps = 0
    for start in range(n_nodes):
        if start in seen:
            continue
        comps += 1
        stack = [start]
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            stack.extend(adjacency.get(u, []))
    return comps


def _graph_invariants(n_nodes, edge_pairs, positions):
    adj = {}
    for u, v in edge_pairs:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)
    b0 = _beta0(n_nodes, adj)
    cycle_rank = len(edge_pairs) - n_nodes + b0
    total_len = sum(float(np.hypot(*(positions[u] - positions[v])))
                    for u, v in edge_pairs)
    return b0, cycle_rank, total_len


def _roundtrip_one(img_path):
    from nanograph_v4 import nanograph_encode, decode_graph, NanographConfig
    from collections import Counter
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    assert img is not None, img_path
    r = nanograph_encode(img, sam_model=None, verbose=False, optimize=False,
                         config=NanographConfig())
    if r.graph is None or r.graph.n_nodes == 0:
        return None
    dec = decode_graph(r.compressed)

    # Map encoder edges (original node ids) into decoded (row-sorted) indices.
    rows = r.points[:, 0].astype(np.int32)
    order = np.argsort(rows)
    inv = np.empty(len(rows), dtype=np.int64)
    inv[order] = np.arange(len(rows))
    enc_pairs = Counter(tuple(sorted((int(inv[e.source]), int(inv[e.target]))))
                        for e in r.graph.edges)
    dec_pairs = Counter(tuple(sorted((u, v))) for u, v in
                        [(e.source, e.target) for e in dec.edges])
    assert enc_pairs == dec_pairs, f'edge set mismatch on {img_path}'

    enc_pos = r.points[order].astype(float)
    dec_pos = dec.node_positions.astype(float)
    assert np.array_equal(enc_pos, dec_pos), f'position mismatch on {img_path}'

    e_inv = _graph_invariants(len(rows), list(enc_pairs.elements()), enc_pos)
    d_inv = _graph_invariants(dec.n_nodes,
                              [(e.source, e.target) for e in dec.edges], dec_pos)
    assert e_inv[0] == d_inv[0], 'beta0 mismatch'
    assert e_inv[1] == d_inv[1], 'cycle rank mismatch'
    assert abs(e_inv[2] - d_inv[2]) <= 1e-6 * max(1.0, e_inv[2]), 'edge length mismatch'
    return True


def test_edge_roundtrip_organelle():
    import glob
    paths = sorted(glob.glob(os.path.join(_ORG_IMAGES, '*.png')))[:20]
    assert len(paths) == 20
    for p in paths:
        _roundtrip_one(p)


def test_edge_roundtrip_cross_modality():
    import glob
    for name, root in _CROSS_IMAGES.items():
        paths = sorted(glob.glob(os.path.join(root, '*.png')))[:5]
        assert paths, f'no images for {name}'
        for p in paths:
            _roundtrip_one(p)


if __name__ == '__main__':
    main()
