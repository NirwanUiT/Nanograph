"""
Nanograph v4 — Multi-segmenter with cascading early-exit.

v4 improvement: Run cheap methods first (Otsu, Frangi, Meijering).
Only invoke SAM if none of the cheap methods exceed a confidence
threshold. This cuts runtime from ~6-12s to <200ms on most images.
"""

import time
import numpy as np
import cv2
from skimage.morphology import skeletonize
from skimage.filters import frangi, meijering

from .config import DEFAULT_CONFIG, NanographConfig, SegmentConfig


def segmentation_quality_score(mask, image_gray, expected_fg_range=None,
                                method_name='unknown', image_type='sparse',
                                cfg=None):
    """
    Score a binary segmentation mask by 6 criteria.
    Components: contrast, coverage, coherence, skeleton, sharpness, shape_match.
    """
    sc = cfg if isinstance(cfg, SegmentConfig) else (
         cfg.segment if isinstance(cfg, NanographConfig) else DEFAULT_CONFIG.segment)

    fg_area = np.count_nonzero(mask)
    if fg_area == 0:
        return 0.0

    img_f = image_gray.astype(np.float64) / 255.0
    fg_pct = fg_area / mask.size * 100.0

    if expected_fg_range is not None:
        lo, hi = expected_fg_range
    else:
        lo, hi = sc.expected_fg_lo, sc.expected_fg_hi

    # 1. Intensity contrast
    mean_fg = img_f[mask > 0].mean()
    mean_bg = img_f[mask == 0].mean() if np.any(mask == 0) else 0.0
    contrast = np.clip(mean_fg - mean_bg, 0, 1)

    # 2. Coverage plausibility
    if fg_pct < lo * 0.5:
        coverage = max(0.0, fg_pct / (lo * 0.5))
    elif fg_pct < lo:
        coverage = 0.7 + 0.3 * (fg_pct - lo * 0.5) / (lo * 0.5)
    elif fg_pct > hi:
        coverage = max(0.0, 1.0 - (fg_pct - hi) / (100.0 - hi))
    else:
        coverage = 1.0

    # 3. Structural coherence
    n_cc, cc_labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), 8)
    if n_cc > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest = areas.max()
        raw_coherence = largest / fg_area
        coherence = np.clip(sc.coherence_floor +
                            (1.0 - sc.coherence_floor) * raw_coherence, 0, 1)
    else:
        coherence = 1.0

    # 4. Skeleton quality
    s = skeletonize(mask > 0).astype(np.uint8)
    skel_len = np.count_nonzero(s)
    skel_density = min(skel_len / fg_area, 1.0)

    # 5. Boundary sharpness
    ks = sc.morph_kernel_size
    k_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    boundary = mask.astype(np.uint8) - cv2.erode(mask.astype(np.uint8), k_erode)
    if np.any(boundary > 0):
        grad_x = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=sc.boundary_sobel_ksize)
        grad_y = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=sc.boundary_sobel_ksize)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        grad_mag_n = grad_mag / (grad_mag.max() + 1e-15)
        sharpness = float(np.mean(grad_mag_n[boundary > 0]))
    else:
        sharpness = 0.0

    # 6. Shape match
    blob_ratio = fg_area / max(skel_len ** 2, 1)
    if method_name in ('Frangi', 'Meijering'):
        if image_type == 'sparse' and blob_ratio > 0.5:
            shape_match = max(0.0, 1.0 - blob_ratio)
        elif skel_len > 0 and (skel_len / fg_area) < 0.05:
            shape_match = 0.3
        else:
            shape_match = 1.0
    elif method_name == 'Otsu':
        shape_match = 0.8
    else:
        shape_match = 1.0

    score = (sc.w_contrast * contrast +
             sc.w_coverage * coverage +
             sc.w_coherence * coherence +
             sc.w_skeleton * skel_density +
             sc.w_sharpness * sharpness +
             sc.w_shape_match * shape_match)
    return float(score)


def microsam_segment(model, image_gray, cfg=None):
    """micro-SAM segmentation using finetuned vit_b_lm weights."""
    from segment_anything import SamAutomaticMaskGenerator
    
    sc = cfg if isinstance(cfg, SegmentConfig) else (
         cfg.segment if isinstance(cfg, NanographConfig) else DEFAULT_CONFIG.segment)

    gen = SamAutomaticMaskGenerator(
        model=model,
        points_per_side=sc.sam_points_per_side,
        pred_iou_thresh=sc.sam_pred_iou_thresh,
        stability_score_thresh=sc.sam_stability_thresh,
        crop_n_layers=sc.sam_crop_n_layers,
        crop_n_points_downscale_factor=sc.sam_crop_downscale,
        min_mask_region_area=sc.sam_min_region_area,
    )

    img_rgb = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)
    masks = gen.generate(img_rgb)

    h, w = image_gray.shape[:2]
    total = h * w
    combined = np.zeros((h, w), dtype=np.uint8)
    kept = []
    for m in sorted(masks, key=lambda x: x['predicted_iou'], reverse=True):
        af = m['area'] / total
        if sc.sam_min_area_frac < af < sc.sam_max_area_frac:
            combined = np.maximum(combined, m['segmentation'].astype(np.uint8) * 255)
            kept.append(m)

    if np.count_nonzero(combined) > 0:
        k_bridge = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (sc.sam_bridge_kernel, sc.sam_bridge_kernel))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k_bridge, iterations=1)

    return combined, kept


def otsu_segment(image):
    """Otsu thresholding + morphological cleaning."""
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k, iterations=1)
    return binary


def frangi_segment(image, cfg=None):
    """Frangi vesselness filter."""
    sc = cfg if isinstance(cfg, SegmentConfig) else (
         cfg.segment if isinstance(cfg, NanographConfig) else DEFAULT_CONFIG.segment)
    img_f = image.astype(np.float64) / 255.0
    vessel = frangi(img_f, sigmas=sc.frangi_sigmas, beta=sc.frangi_beta,
                    gamma=sc.frangi_gamma, black_ridges=False)
    vessel = (vessel - vessel.min()) / (vessel.max() - vessel.min() + 1e-15)
    v_u8 = (vessel * 255).astype(np.uint8)
    _, mask = cv2.threshold(v_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                   (sc.morph_kernel_size, sc.morph_kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    return mask


def meijering_segment(image, cfg=None):
    """Meijering neuriteness filter."""
    sc = cfg if isinstance(cfg, SegmentConfig) else (
         cfg.segment if isinstance(cfg, NanographConfig) else DEFAULT_CONFIG.segment)
    img_f = image.astype(np.float64) / 255.0
    neuri = meijering(img_f, sigmas=sc.meijering_sigmas, black_ridges=False)
    neuri = (neuri - neuri.min()) / (neuri.max() - neuri.min() + 1e-15)
    n_u8 = (neuri * 255).astype(np.uint8)
    _, mask = cv2.threshold(n_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                   (sc.morph_kernel_size, sc.morph_kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    return mask


def morphological_clean(mask, min_area_frac=None, cfg=None):
    """Close, open, remove small components."""
    sc = cfg if isinstance(cfg, SegmentConfig) else (
         cfg.segment if isinstance(cfg, NanographConfig) else DEFAULT_CONFIG.segment)
    if min_area_frac is None:
        min_area_frac = sc.morph_min_area_frac
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                   (sc.morph_kernel_size, sc.morph_kernel_size))
    out = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k, iterations=1)
    min_area = int(min_area_frac * mask.size)
    n_cc, labels, stats, _ = cv2.connectedComponentsWithStats(out, 8)
    for i in range(1, n_cc):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            out[labels == i] = 0
    return out


def auto_segment(image_gray, img_bg_sub, img_enhanced, sam_model=None,
                 image_type='sparse', verbose=True, cfg=None):
    """
    v4: Cascading segmentation with early-exit.
    
    1. Run cheap methods (Otsu, Frangi, Meijering) first
    2. If best cheap score > cascade_confidence_thresh, skip SAM entirely
    3. Only invoke SAM (expensive) when cheap methods are uncertain
    
    This reduces typical runtime from 6-12s to <300ms.
    """
    sc = cfg if isinstance(cfg, SegmentConfig) else (
         cfg.segment if isinstance(cfg, NanographConfig) else DEFAULT_CONFIG.segment)

    candidates = {}
    expected_fg = (sc.expected_fg_lo, sc.expected_fg_hi)

    # --- Phase 1: Cheap methods ---
    t0 = time.time()
    otsu_mask = morphological_clean(otsu_segment(img_bg_sub), cfg=sc)
    t_otsu = time.time() - t0
    q_otsu = segmentation_quality_score(otsu_mask, image_gray, expected_fg,
                                         method_name='Otsu',
                                         image_type=image_type, cfg=sc)
    candidates['Otsu'] = {'mask': otsu_mask, 'time': t_otsu, 'score': q_otsu}

    t0 = time.time()
    frangi_mask = morphological_clean(frangi_segment(img_bg_sub, cfg=sc), cfg=sc)
    t_frangi = time.time() - t0
    q_frangi = segmentation_quality_score(frangi_mask, image_gray, expected_fg,
                                           method_name='Frangi',
                                           image_type=image_type, cfg=sc)
    candidates['Frangi'] = {'mask': frangi_mask, 'time': t_frangi, 'score': q_frangi}

    t0 = time.time()
    meij_mask = morphological_clean(meijering_segment(img_bg_sub, cfg=sc), cfg=sc)
    t_meij = time.time() - t0
    q_meij = segmentation_quality_score(meij_mask, image_gray, expected_fg,
                                         method_name='Meijering',
                                         image_type=image_type, cfg=sc)
    candidates['Meijering'] = {'mask': meij_mask, 'time': t_meij, 'score': q_meij}

    # --- Phase 2: Early-exit check ---
    best_cheap = max(candidates.values(), key=lambda c: c['score'])
    best_cheap_score = best_cheap['score']
    sam_skipped = False

    if (sc.cascade_enable and 
        best_cheap_score >= sc.cascade_confidence_thresh):
        # Cheap method is good enough — skip SAM
        sam_skipped = True
        if verbose:
            print(f'  [CASCADE] Best cheap score={best_cheap_score:.4f} >= '
                  f'{sc.cascade_confidence_thresh:.2f}, skipping SAM')
    elif sam_model is not None:
        # Need SAM — run on 3 inputs with union merge
        t0 = time.time()
        seg_enhanced, _ = microsam_segment(sam_model, img_enhanced, cfg=sc)
        seg_bgsub, _ = microsam_segment(sam_model, img_bg_sub, cfg=sc)
        seg_raw, _ = microsam_segment(sam_model, image_gray, cfg=sc)
        merged = np.maximum(np.maximum(seg_enhanced, seg_bgsub), seg_raw)
        sam_mask = morphological_clean(merged, cfg=sc)
        t_sam = time.time() - t0
        q_sam = segmentation_quality_score(sam_mask, image_gray, expected_fg,
                                            method_name='microSAM',
                                            image_type=image_type, cfg=sc)
        candidates['microSAM'] = {'mask': sam_mask, 'time': t_sam, 'score': q_sam}

    # Select best
    best_name = max(candidates, key=lambda k: candidates[k]['score'])

    if verbose:
        print(f'{"Method":<12} {"FG%":>6} {"Score":>8} {"Time":>8}')
        print('-' * 45)
        for name, c in candidates.items():
            fg = np.count_nonzero(c['mask']) / c['mask'].size * 100
            marker = ' <<<' if name == best_name else ''
            print(f'{name:<12} {fg:>5.1f}% {c["score"]:>8.4f} '
                  f'{c["time"]*1000:>7.0f}ms{marker}')
        if sam_skipped:
            print(f'  (microSAM skipped by cascade early-exit)')
        print(f'\nSelected: {best_name}')

    return candidates[best_name]['mask'], best_name, candidates
