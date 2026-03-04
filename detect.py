"""
Nanograph v4 — Image type auto-detection (sparse vs dense).
"""

import numpy as np
import cv2

from .config import DEFAULT_CONFIG, NanographConfig, DetectConfig


def detect_image_type(image_gray, cfg=None):
    """
    Classify a microscopy image as 'sparse' or 'dense'.
    
    Uses multiple features:
      1. Adaptive threshold FG%
      2. Number of connected components
      3. Median component size vs image size
      4. Mean intensity of FG pixels
    """
    if cfg is None:
        dc = DEFAULT_CONFIG.detect
    elif isinstance(cfg, DetectConfig):
        dc = cfg
    elif isinstance(cfg, NanographConfig):
        dc = cfg.detect
    else:
        dc = DEFAULT_CONFIG.detect

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image_gray)

    adapt = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY,
                                   dc.adapt_block_size, dc.adapt_C)
    _, otsu_mask = cv2.threshold(enhanced, 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    fg_pct_adapt = np.count_nonzero(adapt) / adapt.size * 100
    fg_pct_otsu = np.count_nonzero(otsu_mask) / otsu_mask.size * 100

    if fg_pct_adapt < fg_pct_otsu:
        work_mask = adapt
        fg_pct = fg_pct_adapt
    else:
        work_mask = otsu_mask
        fg_pct = fg_pct_otsu

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean = cv2.morphologyEx(work_mask, cv2.MORPH_OPEN, k, iterations=1)
    n_cc, cc_labels, stats, _ = cv2.connectedComponentsWithStats(clean, 8)

    min_cc_area = max(dc.min_cc_area_abs, int(dc.min_cc_area_frac * image_gray.size))
    significant_ccs = 0
    total_significant_area = 0
    areas = []
    for i in range(1, n_cc):
        a = stats[i, cv2.CC_STAT_AREA]
        if a >= min_cc_area:
            significant_ccs += 1
            total_significant_area += a
            areas.append(a)

    refined_fg_pct = total_significant_area / image_gray.size * 100

    if total_significant_area > 0:
        mean_fg_intensity = enhanced[clean > 0].mean() / 255.0
    else:
        mean_fg_intensity = 0.0

    if areas:
        largest_cc_ratio = max(areas) / total_significant_area
    else:
        largest_cc_ratio = 0.0

    is_dense = (
        refined_fg_pct > dc.dense_fg_pct_thresh and
        mean_fg_intensity > dc.dense_mean_fg_thresh and
        (largest_cc_ratio > dc.dense_largest_cc_ratio or
         significant_ccs < dc.dense_max_n_components)
    )

    if is_dense:
        img_type = 'dense'
        spacing = dc.dense_spacing
        width_cap = dc.dense_width_cap
        bg_kernel_size = dc.dense_bg_kernel
    else:
        img_type = 'sparse'
        spacing = dc.sparse_spacing
        width_cap = dc.sparse_width_cap
        bg_kernel_size = dc.sparse_bg_kernel

    return {
        'type':           img_type,
        'fg_pct':         refined_fg_pct,
        'fg_pct_raw':     fg_pct,
        'n_components':   significant_ccs,
        'mean_fg_int':    mean_fg_intensity,
        'largest_cc_ratio': largest_cc_ratio,
        'bg_kernel_size': bg_kernel_size,
        'spacing':        spacing,
        'width_cap':      width_cap,
    }
