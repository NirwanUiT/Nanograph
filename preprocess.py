"""
Nanograph v4 — Preprocessing with low-rank background grid.

v4 improvement: Instead of storing just mean background (1 byte),
we compute a low-rank NxN grid (default 16x16 = 256 bytes) that
captures spatial variation in the background. This dramatically
improves full-image PSNR at negligible cost.
"""

import numpy as np
import cv2

from .config import DEFAULT_CONFIG, NanographConfig, PreprocessConfig


def preprocess(image_gray, bg_kernel_size=None, cfg=None):
    """
    CLAHE, BG subtraction, fast reflectivity estimation.
    Returns: img_enhanced, img_bg_sub, reflect_img, background_model, bg_grid
    
    v4: Also returns bg_grid (NxN downsampled background) for compression.
    """
    pc = cfg if isinstance(cfg, PreprocessConfig) else (
         cfg.preprocess if isinstance(cfg, NanographConfig) else DEFAULT_CONFIG.preprocess)
    rc = cfg.recon if isinstance(cfg, NanographConfig) else DEFAULT_CONFIG.recon

    if bg_kernel_size is None:
        bg_kernel_size = pc.bg_kernel_size

    clahe = cv2.createCLAHE(clipLimit=pc.clahe_clip_limit,
                             tileGridSize=(pc.clahe_tile_size, pc.clahe_tile_size))
    img = clahe.apply(image_gray)

    bgk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                     (bg_kernel_size, bg_kernel_size))
    background = cv2.morphologyEx(img, cv2.MORPH_OPEN, bgk)
    img_bg_sub = cv2.subtract(img, background)

    # --- Fast reflectivity via convolution ---
    img_f = img_bg_sub.astype(np.float64) / 255.0

    density = cv2.GaussianBlur(img_f, (0, 0), sigmaX=pc.density_sigma)
    density_n = (density - density.min()) / (density.max() - density.min() + 1e-15)
    inv_density = 1.0 - density_n

    local_mean = cv2.GaussianBlur(img_f, (0, 0), sigmaX=pc.mean_sigma)
    local_mean_n = (local_mean - local_mean.min()) / (local_mean.max() - local_mean.min() + 1e-15)

    grad_x = cv2.Sobel(img_bg_sub, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_bg_sub, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    grad_n = (grad_mag - grad_mag.min()) / (grad_mag.max() - grad_mag.min() + 1e-15)

    refl_raw = inv_density * local_mean_n * (1.0 + pc.edge_boost * grad_n)
    refl_u8 = (refl_raw / (refl_raw.max() + 1e-15) * 255).astype(np.uint8)
    refl_bilateral = cv2.bilateralFilter(refl_u8, d=pc.bilateral_d,
                                          sigmaColor=pc.bilateral_sigma_color,
                                          sigmaSpace=pc.bilateral_sigma_space)
    reflect_img = refl_bilateral.astype(np.float64) / 255.0

    # Full background model (float, for reconstruction)
    bg_model = background.astype(np.float64) / 255.0

    # v4: Low-rank background grid
    grid_size = rc.bg_grid_size
    if grid_size > 0:
        bg_grid = cv2.resize(bg_model, (grid_size, grid_size),
                              interpolation=cv2.INTER_AREA)
    else:
        bg_grid = np.array([[np.mean(bg_model)]])  # fallback: single value

    return img, img_bg_sub, reflect_img, bg_model, bg_grid


def upsample_bg_grid(bg_grid, target_shape, sigma=2.0):
    """
    v4: Upsample a low-rank background grid to full image size.
    Uses bicubic interpolation + Gaussian smoothing for smooth transitions.
    """
    upsampled = cv2.resize(bg_grid.astype(np.float64), 
                            (target_shape[1], target_shape[0]),
                            interpolation=cv2.INTER_CUBIC)
    if sigma > 0:
        upsampled = cv2.GaussianBlur(upsampled, (0, 0), sigmaX=sigma)
    return np.clip(upsampled, 0, 1)
