"""
Nanograph v4 — Preprocessing with background subtraction and reflectivity.

Computes CLAHE enhancement, morphological-opening background model,
background-subtracted image, and a fast reflectivity estimate.
The full-resolution bg_model is used during encoding for intensity-matched
reconstruction; only a single mean_bg byte is stored in the compressed output.
"""

import numpy as np
import cv2

from .config import DEFAULT_CONFIG, NanographConfig, PreprocessConfig


def preprocess(image_gray, bg_kernel_size=None, cfg=None):
    """
    CLAHE, BG subtraction, fast reflectivity estimation.
    Returns: img_enhanced, img_bg_sub, reflect_img, bg_model
    """
    pc = cfg if isinstance(cfg, PreprocessConfig) else (
         cfg.preprocess if isinstance(cfg, NanographConfig) else DEFAULT_CONFIG.preprocess)

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

    return img, img_bg_sub, reflect_img, bg_model
