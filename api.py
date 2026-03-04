"""
Nanograph v4 — Main encode/decode API.

Single-call interface:
    result = nanograph_encode('image.png', sam_model=sam)
    recon  = nanograph_decode(result.compressed)
"""

import time
import numpy as np
import cv2
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional, Tuple
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

from .config import NanographConfig, DEFAULT_CONFIG
from .detect import detect_image_type
from .preprocess import preprocess, upsample_bg_grid
from .segment import auto_segment
from .skeleton import (skeletonize_and_classify, extract_nanograph_points,
                       compute_skeleton_orientations)
from .graph import build_nanograph, Nanograph, nanograph_morphometry
from .reconstruct import (reconstruct_width_aware, reconstruct_with_background,
                          topology_optimizer)
from .compress import compress_nanograph, decompress_nanograph
from .classify import classify_structures
from .utils import safe_normalize


@dataclass
class NanographResult:
    """Complete output of nanograph_encode()."""
    compressed: bytes
    shape: Tuple[int, int]
    points: np.ndarray
    intensities: np.ndarray
    widths: np.ndarray
    orientations: np.ndarray       # v4: per-point orientation angle
    types: np.ndarray
    image_type: str
    segmenter: str
    n_points: int
    raw_bytes: int
    compressed_bytes: int
    compression_ratio: float
    bytes_per_point: float
    psnr_full: float
    ssim_full: float
    psnr_fg: float
    ssim_fg: float
    graph: Optional[Nanograph] = field(default=None, repr=False)  # v4: formal graph
    reconstruction: Optional[np.ndarray] = field(default=None, repr=False)
    mask: Optional[np.ndarray] = field(default=None, repr=False)
    skeleton: Optional[np.ndarray] = field(default=None, repr=False)
    structures: Optional[list] = field(default=None, repr=False)
    timing: Optional[dict] = field(default=None, repr=False)
    seg_candidates: Optional[dict] = field(default=None, repr=False)
    compression_stats: Optional[dict] = field(default=None, repr=False)
    opt_history: Optional[dict] = field(default=None, repr=False)
    bg_model: Optional[np.ndarray] = field(default=None, repr=False)
    bg_grid: Optional[np.ndarray] = field(default=None, repr=False)  # v4
    config: Optional[object] = field(default=None, repr=False)


def nanograph_encode(image_path_or_array, sam_model=None,
                     sigma_scale=None, optimize=True,
                     max_iter=None, pts_per_iter=None,
                     build_graph=True,
                     verbose=True, config=None):
    """
    Full adaptive nanograph encoding pipeline (v4).
    
    v4 additions:
      - Oriented PSF reconstruction
      - Low-rank background grid
      - Cascading segmentation (early-exit)
      - Formal graph construction
    
    Accepts a file path (str) or a grayscale numpy array (uint8).
    Returns a NanographResult with compressed bytes, metrics, graph, and metadata.
    """
    cfg = config if config is not None else NanographConfig()
    timing = {}

    # Load image
    t0 = time.time()
    if isinstance(image_path_or_array, str):
        img_raw = cv2.imread(image_path_or_array, cv2.IMREAD_GRAYSCALE)
        if img_raw is None:
            raise FileNotFoundError(f'Cannot read: {image_path_or_array}')
    else:
        img_raw = image_path_or_array.copy()
    timing['load'] = time.time() - t0

    # Auto-detect image type
    t0 = time.time()
    det = detect_image_type(img_raw, cfg=cfg)
    timing['detect'] = time.time() - t0

    cfg = cfg.for_image_type(det['type'])

    if verbose:
        print(f'Image: {img_raw.shape}, type={det["type"]} '
              f'(FG={det["fg_pct"]:.1f}%, n_cc={det["n_components"]}, '
              f'mean_fg_int={det["mean_fg_int"]:.2f})')
        print(f'  bg_kernel={det["bg_kernel_size"]}, spacing={det["spacing"]}, '
              f'width_cap={det["width_cap"]}')

    # Preprocess (v4: returns bg_grid)
    t0 = time.time()
    img_enhanced, img_bg_sub, reflect_img, bg_model, bg_grid = preprocess(
        img_raw, bg_kernel_size=det['bg_kernel_size'], cfg=cfg)
    timing['preprocess'] = time.time() - t0
    if verbose:
        print(f'Preprocessing: {timing["preprocess"]*1000:.0f} ms '
              f'(bg_grid: {bg_grid.shape[0]}x{bg_grid.shape[1]})')

    # Segment (v4: cascading early-exit)
    t0 = time.time()
    clean, seg_name, seg_candidates = auto_segment(
        img_raw, img_bg_sub, img_enhanced,
        sam_model=sam_model, image_type=det['type'], verbose=verbose, cfg=cfg)
    timing['segment'] = time.time() - t0

    # Skeleton + distance transform
    t0 = time.time()
    skel, ep_mask, jn_mask = skeletonize_and_classify(clean)
    dist_transform = cv2.distanceTransform(
        (clean > 0).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    timing['skeleton'] = time.time() - t0

    # v4: Compute orientations along skeleton
    t0 = time.time()
    orientation_map = compute_skeleton_orientations(
        skel, smooth_sigma=cfg.recon.orientation_smooth_sigma)
    timing['orientation'] = time.time() - t0

    # Extract nanograph points (v4: with orientations)
    t0 = time.time()
    pts, ints, widths, wts, types, orientations = extract_nanograph_points(
        skel, ep_mask, jn_mask, img_enhanced, dist_transform, reflect_img,
        orientation_map=orientation_map,
        spacing=det['spacing'], width_cap=det['width_cap'])
    timing['extract'] = time.time() - t0
    if verbose:
        print(f'Nanograph: {len(pts)} points (EP={int(np.sum(types=="endpoint"))}, '
              f'JN={int(np.sum(types=="junction"))}, '
              f'S={int(np.sum(types=="sampled"))})')

    # Topology optimisation
    _sigma_scale = sigma_scale if sigma_scale is not None else cfg.recon.sigma_scale

    if optimize and len(pts) > 0:
        n_before = len(types)
        t0 = time.time()
        pts, ints, widths, orientations, opt_hist, recon_opt = topology_optimizer(
            pts, ints, widths, orientations,
            img_enhanced, clean, dist_transform,
            sigma_scale=_sigma_scale, width_cap=det['width_cap'],
            max_iter=max_iter, pts_per_iter=pts_per_iter,
            bg_model=bg_model, image_type=det['type'], cfg=cfg)
        timing['optimize'] = time.time() - t0

        if len(pts) > n_before:
            extra = np.array(['sampled'] * (len(pts) - n_before))
            types = np.concatenate([types, extra])
            # orientations already extended in optimizer
    else:
        opt_hist = None

    # v4: Build formal graph
    graph = None
    if build_graph and len(pts) > 0:
        t0 = time.time()
        graph = build_nanograph(
            pts, ints, widths, orientations, types,
            skel, dist_transform, img_enhanced, img_raw.shape,
            image_type=det['type'], cfg=cfg)
        timing['graph'] = time.time() - t0
        if verbose:
            gs = graph.summary()
            print(f'Graph: {gs["n_nodes"]} nodes, {gs["n_edges"]} edges, '
                  f'{gs["n_components"]} components')

    # Reconstruct (v4: with oriented PSF + bg grid)
    t0 = time.time()
    fg_recon = reconstruct_width_aware(
        pts, ints, widths, img_raw.shape, _sigma_scale,
        orientations=orientations, cfg=cfg)
    recon = reconstruct_with_background(fg_recon, bg_model, clean, cfg=cfg)
    timing['reconstruct'] = time.time() - t0

    # Metrics
    orig_n = img_enhanced.astype(float) / 255.0
    fg_mask = (clean > 0).astype(float)
    p_full = psnr(orig_n, recon)
    s_full = ssim(orig_n, recon, data_range=1.0)
    try:
        p_fg = psnr(orig_n[clean > 0], recon[clean > 0])
    except Exception:
        p_fg = p_full
    s_fg = ssim(orig_n * fg_mask, recon * fg_mask, data_range=1.0)

    # Compress (v4: with orientations + bg_grid)
    t0 = time.time()
    compressed_data, comp_stats = compress_nanograph(
        pts, ints, widths, img_raw.shape, types,
        orientations=orientations, bg_grid=bg_grid,
        bg_model=bg_model, cfg=cfg)
    timing['compress'] = time.time() - t0

    raw_bytes = len(pts) * cfg.compress.raw_bytes_per_point
    compressed_bytes = len(compressed_data)
    image_bytes = img_raw.size

    # Classify structures
    t0 = time.time()
    structures = classify_structures(clean, skel, ep_mask, jn_mask,
                                      dist_transform, cfg=cfg)
    timing['classify'] = time.time() - t0

    if verbose:
        print(f'\n{"="*60}')
        print(f'  RESULT: {len(pts)} pts, {compressed_bytes:,} bytes '
              f'(+{comp_stats.get("bg_grid_bytes", 0)} bg grid)')
        _, png_buf = cv2.imencode('.png', img_raw,
                                   [cv2.IMWRITE_PNG_COMPRESSION, cfg.eval.png_compression])
        png_bytes = len(png_buf.tobytes())
        print(f'  vs raw pixels: {image_bytes/max(compressed_bytes,1):.0f}x  '
              f'vs PNG: {png_bytes/max(compressed_bytes,1):.1f}x')
        print(f'  PSNR={p_full:.2f}  SSIM={s_full:.4f}  '
              f'FG-PSNR={p_fg:.2f}  FG-SSIM={s_fg:.4f}')
        print(f'  Structures: {len(structures)} -- '
              f'{dict(Counter(s["shape"] for s in structures))}')
        total_ms = sum(timing.values()) * 1000
        print(f'  Total time: {total_ms:.0f} ms')
        print(f'{"="*60}')

    return NanographResult(
        compressed=compressed_data,
        shape=img_raw.shape,
        points=pts, intensities=ints, widths=widths,
        orientations=orientations, types=types,
        image_type=det['type'], segmenter=seg_name,
        n_points=len(pts),
        raw_bytes=raw_bytes,
        compressed_bytes=compressed_bytes,
        compression_ratio=image_bytes / max(compressed_bytes, 1),
        bytes_per_point=comp_stats['bytes_per_point_effective'],
        psnr_full=p_full, ssim_full=s_full,
        psnr_fg=p_fg, ssim_fg=s_fg,
        graph=graph,
        reconstruction=recon, mask=clean, skeleton=skel,
        structures=structures, timing=timing,
        seg_candidates=seg_candidates,
        compression_stats=comp_stats,
        opt_history=opt_hist,
        bg_model=bg_model,
        bg_grid=bg_grid,
        config=cfg,
    )


def nanograph_decode(compressed_data, sigma_scale=None, cfg=None):
    """
    Decode compressed nanograph bytes back to a reconstruction.
    v4: Uses orientations and bg_grid for better reconstruction.
    """
    if cfg is None:
        cfg = DEFAULT_CONFIG
    rc = cfg.recon if isinstance(cfg, NanographConfig) else DEFAULT_CONFIG.recon
    if sigma_scale is None:
        sigma_scale = rc.sigma_scale

    (points, intensities, widths, types, orientations,
     shape, mean_bg, bg_grid) = decompress_nanograph(compressed_data, cfg=cfg)

    fg_recon = reconstruct_width_aware(
        points, intensities, widths, shape, sigma_scale,
        orientations=orientations, cfg=cfg)
    fg_norm = safe_normalize(fg_recon)

    # v4: Use bg_grid if available, else fall back to mean_bg
    if bg_grid is not None:
        bg_upsampled = upsample_bg_grid(bg_grid, shape, sigma=rc.bg_grid_sigma)
    else:
        bg_upsampled = np.full(shape, mean_bg, dtype=np.float64)

    fg_presence = (fg_norm > rc.fg_presence_thresh).astype(np.float32)
    fg_presence = cv2.GaussianBlur(fg_presence, (0, 0), sigmaX=rc.blend_sigma)
    fg_presence = np.clip(fg_presence, 0, 1).astype(np.float64)
    recon = fg_norm * fg_presence + bg_upsampled * (1.0 - fg_presence)
    recon = np.clip(recon, 0, 1)

    return recon, points, intensities, widths, types, orientations, shape
