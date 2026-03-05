"""
Nanograph v4 — Shared utility functions.
"""

import numpy as np
import cv2


def safe_normalize(arr):
    """Normalize array to [0, 1], returning zeros if range is degenerate."""
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-9:
        return np.zeros_like(arr, dtype=np.float64)
    return (arr - mn) / (mx - mn)


def create_psf_kernel(sigma, cfg=None):
    """Create a 2D isotropic Gaussian PSF kernel."""
    max_radius = cfg.psf_max_radius if cfg is not None else 30
    radius = min(int(np.ceil(3 * sigma)), max_radius)
    size = 2 * radius + 1
    y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    k = np.exp(-0.5 * (x ** 2 + y ** 2) / sigma ** 2)
    k /= k.sum()
    return k


def create_oriented_psf_kernel(sigma, orientation, aspect_ratio=2.5, cfg=None):
    """
    Create a 2D oriented elliptical Gaussian PSF kernel.

    sigma        : minor-axis std (pixels)
    orientation  : angle in radians (skeleton tangent direction)
    aspect_ratio : major / minor axis ratio
    """
    max_radius = cfg.psf_max_radius if cfg is not None else 30
    sigma_major = sigma * aspect_ratio
    radius = min(int(np.ceil(3 * sigma_major)), max_radius)
    size = 2 * radius + 1
    y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    cos_a, sin_a = np.cos(orientation), np.sin(orientation)
    x_rot = x * cos_a + y * sin_a
    y_rot = -x * sin_a + y * cos_a
    k = np.exp(-0.5 * (x_rot ** 2 / sigma_major ** 2 + y_rot ** 2 / sigma ** 2))
    k /= k.sum()
    return k


def iou(mask1, mask2):
    """Intersection-over-union for two binary masks."""
    m1 = mask1 > 0
    m2 = mask2 > 0
    intersection = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    if union == 0:
        return 1.0
    return float(intersection) / float(union)


def topology_quality_score(binary_mask):
    """
    Estimate topology quality from a binary mask via skeleton analysis.

    Returns a score in [0, 1] where higher means fewer, longer connected
    branches (better preserved topology).
    """
    from skimage.morphology import skeletonize

    if not np.any(binary_mask > 0):
        return 0.0

    bm = (binary_mask > 0).astype(np.uint8)
    skel = skeletonize(bm).astype(np.uint8)
    skel_px = int(np.count_nonzero(skel))
    if skel_px == 0:
        return 0.0

    n_components, _ = cv2.connectedComponents(skel)
    n_components -= 1  # subtract background label
    if n_components == 0:
        return 0.0

    avg_branch_len = skel_px / n_components
    score = 1.0 - np.exp(-avg_branch_len / 50.0)
    return float(np.clip(score, 0.0, 1.0))
