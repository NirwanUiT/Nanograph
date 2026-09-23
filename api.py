"""
Nanograph v5 — Main encode/decode API.

v5 additions:
  - Spatial background grid (16×16 bilinearly interpolated)
  - Foreground residual coding (DCT-based)
  - Graph-predictive compression
  - Persistent homology metrics

Single-call interface:
    result = nanograph_encode('image.png', sam_model=sam)
    recon  = nanograph_decode(result.compressed)
"""

import time
import numpy as np
import cv2
import warnings
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

from .config import NanographConfig, DEFAULT_CONFIG
from .detect import detect_image_type, detect_polarity
from .preprocess import preprocess
from .segment import auto_segment
from .skeleton import (skeletonize_and_classify, extract_nanograph_points,
                       compute_skeleton_orientations)
from .graph import build_nanograph, Nanograph, nanograph_morphometry
from .reconstruct import (reconstruct_width_aware, reconstruct_with_background,
                          topology_optimizer)
from .compress import compress_nanograph, decompress_nanograph
from .classify import classify_structures


# Cache for the learned U-Net segmenter (keyed by checkpoint path) so it is
# loaded once and reused across frames in a dataset run.
_LEARNED_MODEL_CACHE: Dict[str, object] = {}


def _get_learned_model(ckpt_path):
    if ckpt_path not in _LEARNED_MODEL_CACHE:
        try:
            import torch
            from .unet_seg import load_unet
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            _LEARNED_MODEL_CACHE[ckpt_path] = load_unet(ckpt_path, device=device)
        except Exception as exc:  # torch missing / weight unreadable -> classical fallback
            warnings.warn(f'learned segmenter unavailable ({exc}); '
                          f'falling back to classical cascade')
            _LEARNED_MODEL_CACHE[ckpt_path] = None
    return _LEARNED_MODEL_CACHE[ckpt_path]


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
    pre_psnr_full: float = 0.0   # debug: metrics on the pre-compression render
    pre_ssim_full: float = 0.0
    pre_psnr_fg: float = 0.0
    pre_ssim_fg: float = 0.0
    graph: Optional[Nanograph] = field(default=None, repr=False)  # v4: formal graph
    reconstruction: Optional[np.ndarray] = field(default=None, repr=False)
    pre_reconstruction: Optional[np.ndarray] = field(default=None, repr=False)  # render before compression
    mask: Optional[np.ndarray] = field(default=None, repr=False)
    skeleton: Optional[np.ndarray] = field(default=None, repr=False)
    structures: Optional[list] = field(default=None, repr=False)
    timing: Optional[dict] = field(default=None, repr=False)
    seg_candidates: Optional[dict] = field(default=None, repr=False)
    compression_stats: Optional[dict] = field(default=None, repr=False)
    opt_history: Optional[dict] = field(default=None, repr=False)
    bg_model: Optional[np.ndarray] = field(default=None, repr=False)
    config: Optional[object] = field(default=None, repr=False)
    # v5 additions
    bg_grid: Optional[np.ndarray] = field(default=None, repr=False)
    fg_residual_bytes: Optional[bytes] = field(default=None, repr=False)
    fg_residual_shape_info: Optional[tuple] = field(default=None, repr=False)
    topology_metrics: Optional[Dict] = field(default=None, repr=False)


def nanograph_encode(image_path_or_array, sam_model=None,
                     sigma_scale=None, optimize=True,
                     max_iter=None, pts_per_iter=None,
                     build_graph=True,
                     verbose=True, config=None, learned_model=None):
    """
    Full adaptive nanograph encoding pipeline (v4).
    
    v4 additions:
      - Oriented PSF reconstruction
      - Flat background fill (mean_bg byte)
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

    # Polarity: invert dark-on-light modalities (brightfield/absorption, e.g.
    # retinal vessels, EM) so the structures of interest are the bright
    # foreground the rest of the pipeline expects.
    inverted = detect_polarity(img_raw, cfg=cfg)
    if inverted:
        img_raw = 255 - img_raw
        if verbose:
            print('Polarity: dark-on-light detected -> image inverted')

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

    # Preprocess
    t0 = time.time()
    img_enhanced, img_bg_sub, reflect_img, bg_model = preprocess(
        img_raw, bg_kernel_size=det['bg_kernel_size'], cfg=cfg)
    timing['preprocess'] = time.time() - t0
    if verbose:
        print(f'Preprocessing: {timing["preprocess"]*1000:.0f} ms')

    # Segment (v4: cascading early-exit)
    t0 = time.time()
    # Lazy-load the learned U-Net segmenter from config if requested (cached).
    if (learned_model is None and getattr(cfg.segment, 'use_learned', False)
            and getattr(cfg.segment, 'learned_ckpt', '')):
        learned_model = _get_learned_model(cfg.segment.learned_ckpt)
    clean, seg_name, seg_candidates = auto_segment(
        img_raw, img_bg_sub, img_enhanced,
        sam_model=sam_model, image_type=det['type'], verbose=verbose, cfg=cfg,
        learned_model=learned_model)
    timing['segment'] = time.time() - t0

    # Skeleton + distance transform
    t0 = time.time()
    skel, ep_mask, jn_mask = skeletonize_and_classify(
        clean, prune_len=cfg.graph.spur_min_length)
    dist_transform = cv2.distanceTransform(
        (clean > 0).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    timing['skeleton'] = time.time() - t0

    # v4: Compute orientations along skeleton
    t0 = time.time()
    orientation_map = compute_skeleton_orientations(
        skel, smooth_sigma=cfg.recon.orientation_smooth_sigma)
    timing['orientation'] = time.time() - t0

    # Extract nanograph points (intensities from img_enhanced for better PSF modeling)
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

    # v5: Spatial background grid (replaces flat mean_bg)
    # Computed from img_raw (not img_enhanced) — CLAHE noise would defeat grid smoothing
    from .utils import compute_bg_grid, interpolate_bg_grid
    t0 = time.time()
    img_raw_f = img_raw.astype(np.float64) / 255.0
    bg_grid = compute_bg_grid(img_raw_f, clean, grid_size=cfg.recon.bg_grid_size)
    bg_fill = interpolate_bg_grid(bg_grid, img_raw.shape)
    timing['bg_grid'] = time.time() - t0
    if verbose:
        print(f'Background grid: {cfg.recon.bg_grid_size}×{cfg.recon.bg_grid_size} '
              f'({timing["bg_grid"]*1000:.0f} ms)')

    # v4: Build formal graph from skeleton-extracted points BEFORE optimizer.
    # These points are all on the skeleton so branch assignment is 100%.
    graph = None
    if build_graph and len(pts) > 0:
        t0 = time.time()
        graph = build_nanograph(
            pts, ints, widths, orientations, types,
            skel, dist_transform, img_enhanced, img_raw.shape,
            image_type=det['type'], cfg=cfg)
        # Reconnect filament fragments split by skeleton breaks (collinear
        # endpoint bridging) BEFORE dropping small components, so pieces of a
        # real structure are rejoined rather than discarded.
        if getattr(cfg.graph, 'bridge_gaps', False):
            graph = graph.bridge_gaps(
                max_gap=getattr(cfg.graph, 'bridge_max_gap', 12.0),
                min_align=getattr(cfg.graph, 'bridge_min_align', 0.6))
        # Drop tiny components (vesicle/noise blobs) that inflate the component
        # count without representing real structural topology.
        _min_comp = getattr(cfg.graph, 'min_component_nodes', 0)
        if _min_comp and _min_comp > 1:
            graph = graph.remove_small_components(_min_comp)
        timing['graph'] = time.time() - t0
        if verbose:
            gs = graph.summary()
            print(f'Graph: {gs["n_nodes"]} nodes, {gs["n_edges"]} edges, '
                  f'{gs["n_components"]} components')

    if optimize and len(pts) > 0:
        n_before = len(types)
        t0 = time.time()
        pts, ints, widths, orientations, opt_hist, recon_opt = topology_optimizer(
            pts, ints, widths, orientations,
            img_enhanced, clean, dist_transform,
            sigma_scale=_sigma_scale, width_cap=det['width_cap'],
            max_iter=max_iter, pts_per_iter=pts_per_iter,
            bg_model=bg_fill, image_type=det['type'], cfg=cfg)
        timing['optimize'] = time.time() - t0

        if len(pts) > n_before:
            extra = np.array(['sampled'] * (len(pts) - n_before))
            types = np.concatenate([types, extra])
            # orientations already extended in optimizer

            # Attach optimizer-added points to graph as leaf nodes.
            # These points sit at high PSF-error locations (typically structure
            # edges, off the medial axis). Attaching them improves pixel
            # reconstruction accounting but degrades the structural fidelity of
            # the graph. By default we keep the graph a clean skeleton network
            # and use the extra points only for PSF reconstruction.
            # Build a KDTree over the n_before initial node positions (IDs 0..n_before-1).
            if (graph is not None and n_before > 0
                    and getattr(cfg.graph, 'attach_optimizer_points', False)):
                from scipy.spatial import KDTree
                init_positions = np.array(
                    [graph.nodes[i].position for i in range(n_before)], dtype=float)
                tree = KDTree(init_positions)
                for i in range(n_before, len(pts)):
                    pos = (int(pts[i][0]), int(pts[i][1]))
                    dist, idx = tree.query(pos)
                    graph.add_leaf_node(
                        position=pos,
                        width=float(widths[i]),
                        intensity=float(ints[i]),
                        orientation=float(orientations[i]),
                        node_type='sampled',
                        nearest_node_id=int(idx),
                        edge_length=float(dist),
                    )
    else:
        opt_hist = None

    # Reconstruct (v5: with oriented PSF + spatial background grid)
    # Intensity-match to img_raw (the reconstruction target)
    t0 = time.time()
    fg_recon = reconstruct_width_aware(
        pts, ints, widths, img_raw.shape, _sigma_scale,
        orientations=orientations, cfg=cfg)
    recon = reconstruct_with_background(
        fg_recon, bg_fill, clean, original_img=img_raw, cfg=cfg)
    timing['reconstruct'] = time.time() - t0

    # v5: Foreground residual coding
    from .utils import encode_fg_residual, decode_fg_residual
    fg_residual_data = None
    fg_residual_shape = None
    if cfg.recon.use_fg_residual and len(pts) > 0:
        t0 = time.time()
        orig_n_for_resid = img_raw.astype(np.float64) / 255.0
        fg_residual_data, fg_residual_shape = encode_fg_residual(
            orig_n_for_resid, recon, clean,
            quality=cfg.recon.residual_quality,
            block_size=cfg.recon.residual_block_size)
        # Decode and apply residual to get improved reconstruction
        if fg_residual_data and len(fg_residual_data) > 0:
            decoded_residual = decode_fg_residual(
                fg_residual_data, fg_residual_shape, img_raw.shape,
                quality=cfg.recon.residual_quality,
                block_size=cfg.recon.residual_block_size)
            recon = np.clip(recon + decoded_residual * (clean > 0).astype(np.float64), 0, 1)
        timing['residual'] = time.time() - t0
        if verbose and fg_residual_data and len(fg_residual_data) > 0:
            print(f'FG residual: {len(fg_residual_data):,} bytes '
                  f'({timing["residual"]*1000:.0f} ms)')

    # Metrics (computed against img_raw — the actual reconstruction target)
    orig_n = img_raw.astype(float) / 255.0
    fg_mask = (clean > 0).astype(float)
    p_full = psnr(orig_n, recon)
    s_full = ssim(orig_n, recon, data_range=1.0)
    try:
        p_fg = psnr(orig_n[clean > 0], recon[clean > 0])
    except Exception:
        p_fg = p_full
    s_fg = ssim(orig_n * fg_mask, recon * fg_mask, data_range=1.0)

    # Compress (v5: with bg grid, graph-predictive, residual)
    t0 = time.time()
    compressed_data, comp_stats = compress_nanograph(
        pts, ints, widths, img_raw.shape, types,
        orientations=orientations,
        bg_model=bg_grid,     # v5: pass the grid, not full-res bg
        graph=graph,          # v5: for graph-predictive encoding
        fg_residual=fg_residual_data if (fg_residual_data and len(fg_residual_data) > 0) else None,
        fg_residual_shape=fg_residual_shape,
        cfg=cfg)
    timing['compress'] = time.time() - t0

    # T11.1: the stored edge list must decode to exactly the encoder graph's
    # edges, checked by node coordinate (independent of how ids were mapped).
    if comp_stats.get('has_edges'):
        dec = decompress_nanograph(compressed_data, cfg=cfg)
        dec_pts, dec_edges = dec[0], dec[7]
        assert all(0 <= u < len(dec_pts) and 0 <= v < len(dec_pts)
                   for u, v in dec_edges), 'stored edge index out of range'
        pos = {nd.id: (int(nd.position[0]), int(nd.position[1])) for nd in graph.nodes}
        enc_set = Counter(tuple(sorted((pos[e.source], pos[e.target]))) for e in graph.edges)
        dec_set = Counter(tuple(sorted(((int(dec_pts[u][0]), int(dec_pts[u][1])),
                                        (int(dec_pts[v][0]), int(dec_pts[v][1])))))
                          for u, v in dec_edges)
        assert enc_set == dec_set, 'decoded edges differ from encoder graph'

    raw_bytes = len(pts) * cfg.compress.raw_bytes_per_point
    compressed_bytes = len(compressed_data)
    image_bytes = img_raw.size

    # T3: every reported fidelity metric is computed on the render decoded
    # from the actual payload; the pre-compression values become debug fields.
    t0 = time.time()
    recon_dec = nanograph_decode(compressed_data, original_img=img_raw,
                                 mask=clean, cfg=cfg)[0]
    timing['decode_render'] = time.time() - t0
    p_full_pre, s_full_pre, p_fg_pre, s_fg_pre = p_full, s_full, p_fg, s_fg
    p_full = psnr(orig_n, recon_dec)
    s_full = ssim(orig_n, recon_dec, data_range=1.0)
    try:
        p_fg = psnr(orig_n[clean > 0], recon_dec[clean > 0])
    except Exception:
        p_fg = p_full
    s_fg = ssim(orig_n * fg_mask, recon_dec * fg_mask, data_range=1.0)
    pre_recon = recon              # pre-compression render (A2 reporting)
    recon = recon_dec

    # Classify structures
    t0 = time.time()
    structures = classify_structures(clean, skel, ep_mask, jn_mask,
                                      dist_transform, cfg=cfg)
    timing['classify'] = time.time() - t0

    # v5: Compute topology metrics from graph (not from Otsu-thresholded recon)
    from .utils import graph_topology_score, compute_betti_numbers
    topo_metrics = None
    t0 = time.time()
    graph_topo = graph_topology_score(graph)
    # Also compute Betti numbers of the segmentation mask for reference
    seg_b0, seg_b1 = compute_betti_numbers(clean)
    topo_metrics = {
        **graph_topo,
        'seg_beta_0': seg_b0,
        'seg_beta_1': seg_b1,
    }
    timing['topology'] = time.time() - t0

    if verbose:
        print(f'\n{"="*60}')
        print(f'  RESULT (v5): {len(pts)} pts, {compressed_bytes:,} bytes')
        _, png_buf = cv2.imencode('.png', img_raw,
                                   [cv2.IMWRITE_PNG_COMPRESSION, cfg.eval.png_compression])
        png_bytes = len(png_buf.tobytes())
        print(f'  vs raw pixels: {image_bytes/max(compressed_bytes,1):.0f}x  '
              f'vs PNG: {png_bytes/max(compressed_bytes,1):.1f}x')
        print(f'  PSNR={p_full:.2f}  SSIM={s_full:.4f}  '
              f'FG-PSNR={p_fg:.2f}  FG-SSIM={s_fg:.4f}')
        if topo_metrics:
            print(f'  Graph topology: {graph_topo["n_components"]} components, '
                  f'{graph_topo["n_cycles"]} cycles, '
                  f'{graph_topo["n_junctions"]} junctions, '
                  f'{graph_topo["n_endpoints"]} endpoints')
            print(f'  Segmentation: β₀={seg_b0}, β₁={seg_b1}')
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
        pre_psnr_full=p_full_pre, pre_ssim_full=s_full_pre,
        pre_psnr_fg=p_fg_pre, pre_ssim_fg=s_fg_pre,
        graph=graph,
        reconstruction=recon, pre_reconstruction=pre_recon, mask=clean, skeleton=skel,
        structures=structures, timing=timing,
        seg_candidates=seg_candidates,
        compression_stats=comp_stats,
        opt_history=opt_hist,
        bg_model=bg_model,
        config=cfg,
        bg_grid=bg_grid,
        fg_residual_bytes=fg_residual_data,
        fg_residual_shape_info=fg_residual_shape,
        topology_metrics=topo_metrics,
    )


def nanograph_decode(compressed_data, sigma_scale=None,
                     original_img=None, mask=None, cfg=None):
    """
    Decode compressed nanograph bytes back to a reconstruction.

    v5: Supports spatial background grid + foreground residual.
    Auto-detects v4 vs v5 format.

    If *original_img* (uint8 grayscale) and *mask* (binary) are provided,
    the foreground is intensity-matched to the original.  Otherwise
    percentile normalization is used (decode-only path).
    """
    if cfg is None:
        cfg = DEFAULT_CONFIG
    rc = cfg.recon if isinstance(cfg, NanographConfig) else DEFAULT_CONFIG.recon
    if sigma_scale is None:
        sigma_scale = rc.sigma_scale

    (points, intensities, widths, types, orientations,
     shape, bg_info, _edges, _adjacency) = decompress_nanograph(compressed_data, cfg=cfg)

    fg_recon = reconstruct_width_aware(
        points, intensities, widths, shape, sigma_scale,
        orientations=orientations, cfg=cfg)

    # v5: Use bg grid if available, otherwise flat fill
    if isinstance(bg_info, dict):
        bg_grid = bg_info.get('bg_grid')
        if bg_grid is not None:
            from .utils import interpolate_bg_grid
            bg_fill = interpolate_bg_grid(bg_grid, shape)
        else:
            bg_fill = np.full(shape, bg_info.get('mean_bg', 0.0), dtype=np.float64)
        fg_residual_data = bg_info.get('fg_residual_data')
    else:
        # v4 backward compat: bg_info is a float (mean_bg)
        bg_fill = np.full(shape, float(bg_info), dtype=np.float64)
        fg_residual_data = None

    # Decode fg residual if present (v6 stores shape_info in the stream, so
    # the payload is self-contained).
    fg_residual = None
    if fg_residual_data is not None and isinstance(bg_info, dict):
        shape_info = bg_info.get('fg_residual_shape')
        if shape_info is not None and any(shape_info):
            from .utils import decode_fg_residual
            fg_residual = decode_fg_residual(
                fg_residual_data, shape_info, shape,
                quality=rc.residual_quality,
                block_size=rc.residual_block_size)

    recon = reconstruct_with_background(
        fg_recon, bg_fill, mask, original_img=original_img,
        fg_residual=None, cfg=rc)

    # Apply the residual exactly as the encoder does: inside the mask if one
    # is available, else wherever the residual is nonzero.
    if fg_residual is not None:
        if mask is not None:
            recon = np.clip(recon + fg_residual * (mask > 0).astype(np.float64), 0, 1)
        else:
            recon = np.clip(recon + fg_residual, 0, 1)

    return recon, points, intensities, widths, types, orientations, shape


def decode_graph(compressed_data, cfg=None):
    """Rebuild the Nanograph object from a v6 payload (stored nodes + edges).

    Edge attributes that are functions of node attributes (length as the
    Euclidean distance between endpoints, mean width/intensity as the endpoint
    means) are recomputed from the decoded nodes; curvature is a path
    property and is set to 0 (not recoverable from two endpoints).
    """
    from .graph import Nanograph, GraphNode, GraphEdge

    if cfg is None:
        cfg = DEFAULT_CONFIG
    (points, intensities, widths, types, orientations,
     shape, bg_info, edges, adjacency) = decompress_nanograph(compressed_data, cfg=cfg)

    nodes = []
    for i in range(len(points)):
        nodes.append(GraphNode(
            id=i, position=(int(points[i, 0]), int(points[i, 1])),
            width=float(widths[i]), intensity=float(intensities[i]),
            orientation=float(orientations[i]), node_type=str(types[i]),
            degree=len(adjacency.get(i, []))))

    graph_edges = []
    for eid, (u, v) in enumerate(edges):
        du = np.array(nodes[u].position, dtype=float)
        dv = np.array(nodes[v].position, dtype=float)
        length = float(np.hypot(*(du - dv)))
        graph_edges.append(GraphEdge(
            id=eid, source=u, target=v, length=length,
            pixel_count=int(round(length)) + 1,
            mean_width=(nodes[u].width + nodes[v].width) / 2.0,
            mean_intensity=(nodes[u].intensity + nodes[v].intensity) / 2.0,
            curvature=0.0))

    return Nanograph(nodes=nodes, edges=graph_edges,
                     adjacency={k: list(v) for k, v in adjacency.items()},
                     shape=shape, image_type='decoded')
