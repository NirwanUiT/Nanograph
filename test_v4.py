#!/usr/bin/env python3
"""
Nanograph v4 — Test script for sparse + dense demo images.

Usage:
    python test_v4.py --sparse /path/to/sparse.png --dense /path/to/dense.jpg
    python test_v4.py --sparse /path/to/sparse.png --dense /path/to/dense.jpg --sam /path/to/vit_b_lm.pt

If --sam is not provided, only cheap segmenters are used (which is actually
the expected behavior with cascade early-exit — SAM is the fallback).
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

# Add parent dir to path so nanograph_v4 can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nanograph_v4 import (nanograph_encode, nanograph_decode, NanographConfig,
                           nanograph_morphometry, print_comparison)
from nanograph_v4.evaluate import evaluate_result


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

    # Orientation visualization
    if result.orientations is not None:
        ori_img = np.zeros(img_raw.shape, dtype=np.float64)
        for pt, ori in zip(result.points.astype(int), result.orientations):
            y, x = pt
            if 0 <= y < img_raw.shape[0] and 0 <= x < img_raw.shape[1]:
                ori_img[y, x] = ori
        axes[1, 2].set_title('Point Orientations')
        axes[1, 2].imshow(ori_img, cmap='hsv', vmin=0, vmax=np.pi)
        axes[1, 2].axis('off')
    else:
        axes[1, 2].axis('off')

    # Graph visualization
    if result.graph is not None:
        graph_img = cv2.cvtColor(img_raw, cv2.COLOR_GRAY2RGB)
        # Draw edges
        for edge in result.graph.edges:
            src = result.graph.nodes[edge.source].position
            tgt = result.graph.nodes[edge.target].position
            cv2.line(graph_img, (src[1], src[0]), (tgt[1], tgt[0]),
                     (0, 200, 200), 1)
        # Draw nodes
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


def main():
    parser = argparse.ArgumentParser(description='Nanograph v4 test')
    parser.add_argument('--sparse', required=True, help='Path to sparse organelle image')
    parser.add_argument('--dense', required=True, help='Path to dense network image')
    parser.add_argument('--sam', default=None, help='Path to SAM checkpoint (optional)')
    parser.add_argument('--outdir', default='./v4_results', help='Output directory')
    parser.add_argument('--no-graph', action='store_true', help='Skip graph construction')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load SAM model if provided
    sam_model = None
    if args.sam:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        sam_model = load_sam_model(args.sam, device)

    # Configure v4
    cfg = NanographConfig()
    print(f'Nanograph v4 — {cfg.param_count()} configurable parameters')
    print(f'Features: oriented_psf={cfg.recon.use_oriented_psf}, '
          f'bg_grid={cfg.recon.bg_grid_size}x{cfg.recon.bg_grid_size}, '
          f'cascade={cfg.segment.cascade_enable}')

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
    plot_results(result_sparse, img_sp, 'Nanograph v4 — Sparse Organelles',
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
    plot_results(result_dense, img_dn, 'Nanograph v4 — Dense Network',
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
    print(f'  v4 COMPARISON SUMMARY')
    print(f'{"="*60}')
    print(f'  {"Metric":<25} {"Sparse":>12} {"Dense":>12}')
    print(f'  {"-"*50}')
    for key in ['image_type', 'segmenter', 'n_points', 'ng_bytes',
                 'ng_psnr', 'ng_ssim', 'ng_fg_psnr', 'ng_fg_ssim',
                 'ng_iou', 'ng_topo_q', 'total_time_ms']:
        v1 = row_sp.get(key, 'N/A')
        v2 = row_dn.get(key, 'N/A')
        if isinstance(v1, float):
            print(f'  {key:<25} {v1:>12.4f} {v2:>12.4f}')
        elif isinstance(v1, int):
            print(f'  {key:<25} {v1:>12,} {v2:>12,}')
        else:
            print(f'  {key:<25} {str(v1):>12} {str(v2):>12}')

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

    print(f'\nDone. Results saved to {args.outdir}/')


if __name__ == '__main__':
    main()
