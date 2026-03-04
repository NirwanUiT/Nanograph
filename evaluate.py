"""
Nanograph v4 — Evaluation, comparison, and batch benchmarking.
"""

import numpy as np
import cv2
import csv
import os
import time
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

from .config import NanographConfig, DEFAULT_CONFIG
from .utils import iou, topology_quality_score


def jpeg_for_budget(image, budget):
    """Find highest JPEG quality that fits within byte budget."""
    lo_q, hi_q, best_q, best_buf = 1, 100, 1, None
    while lo_q <= hi_q:
        mid = (lo_q + hi_q) // 2
        _, buf = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, mid])
        if len(buf.tobytes()) <= budget:
            best_q = mid
            best_buf = buf.tobytes()
            lo_q = mid + 1
        else:
            hi_q = mid - 1
    return best_q, best_buf


def evaluate_result(result, img_raw, cfg=None):
    """
    Comprehensive evaluation of a NanographResult vs baselines.
    Returns a dict with all metrics for CSV export.
    """
    if cfg is None:
        cfg = result.config if result.config is not None else DEFAULT_CONFIG

    orig_n = img_raw.astype(float) / 255.0
    fg_mask = (result.mask > 0).astype(float)

    # PNG baseline
    _, png_buf = cv2.imencode('.png', img_raw,
                               [cv2.IMWRITE_PNG_COMPRESSION, cfg.eval.png_compression])
    png_bytes = len(png_buf.tobytes())

    # Nanograph metrics (already in result)
    ng_bytes = result.compressed_bytes

    # Byte-matched JPEG
    jpeg_q, jpeg_buf = jpeg_for_budget(img_raw, ng_bytes)
    jpeg_metrics = {}
    if jpeg_buf:
        jpeg_dec = cv2.imdecode(np.frombuffer(jpeg_buf, np.uint8), cv2.IMREAD_GRAYSCALE)
        jpeg_n = jpeg_dec.astype(float) / 255.0
        jpeg_metrics = {
            'jpeg_quality': jpeg_q,
            'jpeg_bytes': len(jpeg_buf),
            'jpeg_psnr': psnr(orig_n, jpeg_n),
            'jpeg_ssim': ssim(orig_n, jpeg_n, data_range=1.0),
        }
        try:
            jpeg_metrics['jpeg_fg_psnr'] = psnr(orig_n[result.mask > 0],
                                                  jpeg_n[result.mask > 0])
        except:
            jpeg_metrics['jpeg_fg_psnr'] = jpeg_metrics['jpeg_psnr']
        jpeg_metrics['jpeg_fg_ssim'] = ssim(orig_n * fg_mask, jpeg_n * fg_mask,
                                             data_range=1.0)
        _, jpeg_otsu = cv2.threshold(jpeg_dec, 0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        jpeg_metrics['jpeg_iou'] = iou(result.mask, jpeg_otsu)
        jpeg_metrics['jpeg_topo_q'] = topology_quality_score(jpeg_otsu)

    # Nanograph IoU and topology
    recon_u8 = (result.reconstruction * 255).astype(np.uint8) if result.reconstruction is not None else np.zeros_like(img_raw)
    _, ng_otsu = cv2.threshold(recon_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ng_iou = iou(result.mask, ng_otsu)
    ng_topo_q = topology_quality_score(ng_otsu)

    # Graph metrics
    graph_metrics = {}
    if result.graph is not None:
        gs = result.graph.summary()
        graph_metrics = {f'graph_{k}': v for k, v in gs.items()}

    # Timing
    total_time = sum(result.timing.values()) if result.timing else 0

    row = {
        'image_shape': f'{img_raw.shape[0]}x{img_raw.shape[1]}',
        'image_pixels': img_raw.size,
        'image_type': result.image_type,
        'segmenter': result.segmenter,
        'n_points': result.n_points,
        'n_structures': len(result.structures) if result.structures else 0,
        'ng_bytes': ng_bytes,
        'png_bytes': png_bytes,
        'ng_vs_png': png_bytes / max(ng_bytes, 1),
        'ng_vs_raw': img_raw.size / max(ng_bytes, 1),
        'ng_psnr': result.psnr_full,
        'ng_ssim': result.ssim_full,
        'ng_fg_psnr': result.psnr_fg,
        'ng_fg_ssim': result.ssim_fg,
        'ng_iou': ng_iou,
        'ng_topo_q': ng_topo_q,
        'total_time_ms': total_time * 1000,
        **jpeg_metrics,
        **graph_metrics,
    }

    # Delta columns (nanograph - jpeg)
    if jpeg_metrics:
        row['delta_psnr'] = result.psnr_full - jpeg_metrics.get('jpeg_psnr', 0)
        row['delta_ssim'] = result.ssim_full - jpeg_metrics.get('jpeg_ssim', 0)
        row['delta_fg_psnr'] = result.psnr_fg - jpeg_metrics.get('jpeg_fg_psnr', 0)
        row['delta_fg_ssim'] = result.ssim_fg - jpeg_metrics.get('jpeg_fg_ssim', 0)

    return row


def print_comparison(result, img_raw, label='', cfg=None):
    """Print a formatted comparison table."""
    row = evaluate_result(result, img_raw, cfg)

    print(f"\n{'='*65}")
    print(f"  EVALUATION: {label}")
    print(f"{'='*65}")
    print(f"  Image: {row['image_shape']}  Type: {row['image_type']}  "
          f"Segmenter: {row['segmenter']}")
    print(f"  Points: {row['n_points']:,}  Structures: {row['n_structures']}")
    print(f"  {'─'*55}")
    print(f"  {'Metric':<22} {'Nanograph':>12} {'JPEG':>12} {'Delta':>10}")
    print(f"  {'-'*56}")
    
    metrics = [
        ('Bytes',        row['ng_bytes'],     row.get('jpeg_bytes', 0), False),
        ('Full PSNR',    row['ng_psnr'],      row.get('jpeg_psnr', 0), True),
        ('Full SSIM',    row['ng_ssim'],      row.get('jpeg_ssim', 0), True),
        ('FG-PSNR',      row['ng_fg_psnr'],   row.get('jpeg_fg_psnr', 0), True),
        ('FG-SSIM',      row['ng_fg_ssim'],   row.get('jpeg_fg_ssim', 0), True),
        ('IoU',          row['ng_iou'],        row.get('jpeg_iou', 0), True),
        ('Topology Q',   row['ng_topo_q'],     row.get('jpeg_topo_q', 0), True),
    ]
    
    for name, ng_val, j_val, higher_better in metrics:
        delta = ng_val - j_val
        sign = '+' if delta >= 0 else ''
        nf = f'{ng_val:,.4f}' if isinstance(ng_val, float) and ng_val < 1000 else f'{ng_val:,}'
        jf = f'{j_val:,.4f}' if isinstance(j_val, float) and j_val < 1000 else f'{j_val:,}'
        print(f"  {name:<22} {nf:>12} {jf:>12} {sign}{delta:>9.4f}")

    if result.graph:
        gs = result.graph.summary()
        print(f"  {'─'*55}")
        print(f"  Graph: {gs['n_nodes']} nodes, {gs['n_edges']} edges, "
              f"{gs['n_components']} components")
        print(f"  Mean degree: {gs['mean_degree']:.1f}  "
              f"Mean edge length: {gs['mean_edge_length']:.1f}px  "
              f"Mean curvature: {gs['mean_curvature']:.3f}")

    print(f"  {'─'*55}")
    print(f"  Time: {row['total_time_ms']:.0f} ms")
    print(f"{'='*65}")

    return row


def batch_evaluate(image_paths, sam_model=None, output_csv=None,
                   verbose=True, config=None):
    """
    Run nanograph_encode on a batch of images and collect metrics.
    
    Args:
        image_paths: list of file paths to grayscale microscopy images
        sam_model: optional SAM model (will use cascade early-exit)
        output_csv: path to save results CSV
        verbose: print per-image results
        config: NanographConfig override
    
    Returns: list of metric dicts
    """
    from .api import nanograph_encode

    results = []
    for i, path in enumerate(image_paths):
        if verbose:
            print(f'\n[{i+1}/{len(image_paths)}] {os.path.basename(path)}')

        try:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f'  ERROR: Cannot read {path}')
                continue

            result = nanograph_encode(img, sam_model=sam_model,
                                       verbose=verbose, config=config)
            row = evaluate_result(result, img)
            row['filename'] = os.path.basename(path)
            row['filepath'] = path
            results.append(row)

            if verbose:
                print(f'  -> {row["ng_bytes"]:,}B, PSNR={row["ng_psnr"]:.2f}, '
                      f'FG-SSIM={row["ng_fg_ssim"]:.4f}, '
                      f'{row["total_time_ms"]:.0f}ms')

        except Exception as e:
            print(f'  ERROR: {e}')
            continue

    # Save CSV
    if output_csv and results:
        keys = results[0].keys()
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
        if verbose:
            print(f'\nResults saved to {output_csv}')

    # Summary
    if results and verbose:
        print(f'\n{"="*60}')
        print(f'  BATCH SUMMARY ({len(results)} images)')
        print(f'{"="*60}')
        for key in ['ng_psnr', 'ng_ssim', 'ng_fg_psnr', 'ng_fg_ssim',
                     'ng_bytes', 'total_time_ms']:
            vals = [r[key] for r in results if key in r]
            if vals:
                print(f'  {key:<20}: mean={np.mean(vals):.4f}  '
                      f'std={np.std(vals):.4f}  '
                      f'min={np.min(vals):.4f}  max={np.max(vals):.4f}')

    return results
