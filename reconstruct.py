"""Nanograph v5 — Reconstruction with oriented PSF, spatial background grid, and residual coding.

v5 improvements over v4:
1. Spatial background grid: 16×16 bilinearly interpolated grid instead of flat mean_bg.
   Captures non-uniform illumination, dramatically improving full-image PSNR/SSIM.
2. Foreground residual coding: optional DCT-encoded residual captures high-frequency
   detail the PSF model misses.
3. Oriented elliptical PSF: (inherited from v4) filament-aligned anisotropic kernels.
4. Intensity-matched foreground scaling.
"""

import numpy as np
import cv2
from scipy.signal import fftconvolve
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

from .config import DEFAULT_CONFIG, NanographConfig, ReconConfig, OptimizerConfig
from .utils import safe_normalize, create_psf_kernel, create_oriented_psf_kernel


def reconstruct_width_aware(points, intensities, widths, shape,
                             sigma_scale=None, orientations=None, cfg=None):
    """
    Width-aware PSF reconstruction.
    
    v4: Uses oriented elliptical PSFs when orientations are provided,
    giving much better reconstruction of filamentous structures.
    """
    rc = cfg if isinstance(cfg, ReconConfig) else (
         cfg.recon if isinstance(cfg, NanographConfig) else DEFAULT_CONFIG.recon)
    if sigma_scale is None:
        sigma_scale = rc.sigma_scale

    result = np.zeros(shape, dtype=np.float64)
    if len(points) == 0:
        return result

    sigmas = np.clip(widths * sigma_scale, rc.min_sigma, rc.max_sigma)
    sigma_bins = np.round(sigmas / rc.sigma_quantize) * rc.sigma_quantize

    use_oriented = (rc.use_oriented_psf and orientations is not None 
                    and len(orientations) == len(points))

    if use_oriented:
        # v4: Oriented PSF reconstruction
        # Group by (sigma_bin, orientation_bin) for efficiency
        ori_bins = np.round(orientations / (np.pi / 12)) * (np.pi / 12)  # 15-degree bins
        
        for s in np.unique(sigma_bins):
            for o in np.unique(ori_bins[sigma_bins == s]):
                mask_so = (sigma_bins == s) & (ori_bins == o)
                if not np.any(mask_so):
                    continue
                
                delta = np.zeros(shape, dtype=np.float64)
                for (y, x), w in zip(points[mask_so].astype(int), intensities[mask_so]):
                    if 0 <= y < shape[0] and 0 <= x < shape[1]:
                        delta[int(y), int(x)] += w
                
                psf = create_oriented_psf_kernel(
                    s, o, aspect_ratio=rc.orientation_aspect_ratio, cfg=rc)
                result += fftconvolve(delta, psf, mode='same')
    else:
        # Isotropic fallback (same as v3)
        for s in np.unique(sigma_bins):
            mask_s = sigma_bins == s
            delta = np.zeros(shape, dtype=np.float64)
            for (y, x), w in zip(points[mask_s].astype(int), intensities[mask_s]):
                if 0 <= y < shape[0] and 0 <= x < shape[1]:
                    delta[int(y), int(x)] += w
            psf = create_psf_kernel(s, cfg=rc)
            result += fftconvolve(delta, psf, mode='same')

    return result


def reconstruct_with_background(fg_recon, bg_model, mask, original_img=None,
                                 blend_sigma=None, fg_presence_thresh=None,
                                 fg_residual=None, cfg=None):
    """
    Composite foreground PSF reconstruction with the background.

    v5: bg_model can be either:
      - A full-resolution float64 array (same as v4)
      - A small grid (e.g. 16×16) that will be bilinearly interpolated

    fg_residual: optional float64 residual image to add to the foreground
                 (from DCT residual decoding).

    original_img: uint8 grayscale — used to match the fg mean intensity so the
                  reconstruction isn't dim due to PSF energy spreading.
    mask:         binary segmentation mask — used directly for blending so that
                  PSF tails outside the mask don't bleed into the background.
    """
    from .utils import interpolate_bg_grid

    rc = cfg if isinstance(cfg, ReconConfig) else (
         cfg.recon if isinstance(cfg, NanographConfig) else DEFAULT_CONFIG.recon)
    if blend_sigma is None:
        blend_sigma = rc.blend_sigma
    if fg_presence_thresh is None:
        fg_presence_thresh = rc.fg_presence_thresh

    target_shape = fg_recon.shape

    # v5: Handle bg_model as grid or full-resolution
    if bg_model is not None and bg_model.ndim == 2:
        if bg_model.shape[0] < target_shape[0] or bg_model.shape[1] < target_shape[1]:
            # It's a grid — interpolate to full resolution
            bg_full = interpolate_bg_grid(bg_model, target_shape)
        else:
            bg_full = bg_model
    elif bg_model is not None:
        bg_full = bg_model
    else:
        bg_full = np.zeros(target_shape, dtype=np.float64)

    # --- Intensity-matched scaling ---
    if original_img is not None and mask is not None and np.any(mask > 0):
        orig_f = original_img.astype(np.float64) / 255.0
        orig_fg_mean = float(np.mean(orig_f[mask > 0]))
        recon_fg_vals = fg_recon[mask > 0]
        recon_fg_mean = float(np.mean(recon_fg_vals)) if recon_fg_vals.size > 0 else 0.0
        if recon_fg_mean > 1e-9:
            fg_scaled = np.clip(fg_recon * (orig_fg_mean / recon_fg_mean), 0, 1)
        else:
            fg_scaled = safe_normalize(fg_recon)
    else:
        # Fallback: percentile normalization
        pos = fg_recon[fg_recon > 1e-9]
        if len(pos) > 0:
            scale = float(np.percentile(pos, 99.5))
            fg_scaled = np.clip(fg_recon / max(scale, 1e-9), 0, 1)
        else:
            fg_scaled = np.zeros_like(fg_recon)

    # v5: Add foreground residual if provided
    if fg_residual is not None:
        fg_scaled = np.clip(fg_scaled + fg_residual, 0, 1)

    # --- Mask-based fg presence ---
    if mask is not None:
        fg_presence = (mask > 0).astype(np.float32)
    else:
        fg_presence = (fg_scaled > fg_presence_thresh).astype(np.float32)
    fg_presence = cv2.GaussianBlur(fg_presence, (0, 0), sigmaX=blend_sigma)
    fg_presence = np.clip(fg_presence, 0, 1).astype(np.float64)

    combined = fg_scaled * fg_presence + bg_full * (1.0 - fg_presence)
    return np.clip(combined, 0, 1)


def topology_optimizer(points, intensities, widths, orientations,
                       original, mask, dist_transform, 
                       sigma_scale=None, width_cap=30.0,
                       max_iter=None, pts_per_iter=None, min_error=None,
                       bg_model=None, image_type='sparse', cfg=None):
    """
    Iterative topology optimisation using width-aware PSF.
    v4: Uses oriented PSF and error-guided placement.
    """
    oc = cfg if isinstance(cfg, OptimizerConfig) else (
         cfg.optimizer if isinstance(cfg, NanographConfig) else DEFAULT_CONFIG.optimizer)
    rc = cfg.recon if isinstance(cfg, NanographConfig) else DEFAULT_CONFIG.recon

    if sigma_scale is None:
        sigma_scale = rc.sigma_scale
    if max_iter is None:
        max_iter = oc.max_iter
    if pts_per_iter is None:
        pts_per_iter = oc.pts_per_iter
    if min_error is None:
        min_error = oc.min_error

    orig = original.astype(float) / 255.0
    fg = (mask > 0).astype(float)
    cur_pts = [list(map(int, p)) for p in points]
    cur_int = list(intensities)
    cur_w = list(widths)
    cur_ori = list(orientations) if orientations is not None else [0.0] * len(points)
    history = {'n': [], 'psnr': [], 'ssim': [], 'fg_psnr': [], 'fg_ssim': []}

    if image_type == 'dense':
        pts_per_iter = max(pts_per_iter, oc.dense_pts_per_iter)
        min_error = min(min_error, oc.dense_min_error)
        max_iter = max(max_iter, oc.dense_max_iter)

    prev_fg_ssim = 0.0

    for it in range(max_iter):
        pts_arr = np.array(cur_pts, dtype=float)
        int_arr = np.array(cur_int)
        w_arr = np.array(cur_w)
        ori_arr = np.array(cur_ori)

        fg_recon = reconstruct_width_aware(pts_arr, int_arr, w_arr,
                                            original.shape, sigma_scale,
                                            orientations=ori_arr, cfg=rc)
        if bg_model is not None:
            recon_n = reconstruct_with_background(
                fg_recon, bg_model, mask, original_img=original, cfg=rc)
        else:
            recon_n = safe_normalize(fg_recon)

        p_val = psnr(orig, recon_n)
        s_val = ssim(orig, recon_n, data_range=1.0)
        try:
            fp = psnr(orig[mask > 0], recon_n[mask > 0])
        except Exception:
            fp = p_val
        fs = ssim(orig * fg, recon_n * fg, data_range=1.0)

        history['n'].append(len(cur_pts))
        history['psnr'].append(p_val)
        history['ssim'].append(s_val)
        history['fg_psnr'].append(fp)
        history['fg_ssim'].append(fs)

        if it > oc.plateau_min_iter and abs(fs - prev_fg_ssim) < oc.plateau_thresh:
            break
        prev_fg_ssim = fs

        # Error map on foreground
        error = np.abs(orig - recon_n) * fg
        blur_ks = oc.error_blur_ksize
        error_sm = cv2.GaussianBlur(error.astype(np.float32),
                                     (blur_ks, blur_ks), 0)

        pts_set = set(tuple(p) for p in cur_pts)
        candidates = np.argwhere((mask > 0) & (error_sm > min_error))
        if len(candidates) == 0:
            break

        cand_err = error_sm[candidates[:, 0], candidates[:, 1]]
        candidates = candidates[np.argsort(-cand_err)]

        added, batch = 0, []
        for y, x in candidates:
            key = (int(y), int(x))
            if key in pts_set:
                continue
            if batch:
                d = np.sqrt(np.sum((np.array(batch) - [y, x]) ** 2, axis=1))
                if d.min() < oc.min_point_spacing:
                    continue
            batch.append([int(y), int(x)])
            cur_pts.append([int(y), int(x)])
            cur_int.append(float(orig[int(y), int(x)]))
            cur_w.append(float(min(max(dist_transform[int(y), int(x)], 1.0),
                                   width_cap)))
            cur_ori.append(0.0)  # orientation for optimizer-added points
            pts_set.add(key)
            added += 1
            if added >= pts_per_iter:
                break

        if added == 0:
            break

    # Final arrays
    pts_arr = np.array(cur_pts, dtype=float)
    int_arr = np.array(cur_int)
    w_arr = np.array(cur_w)
    ori_arr = np.array(cur_ori)

    fg_recon = reconstruct_width_aware(pts_arr, int_arr, w_arr,
                                        original.shape, sigma_scale,
                                        orientations=ori_arr, cfg=rc)
    if bg_model is not None:
        recon_n = reconstruct_with_background(
            fg_recon, bg_model, mask, original_img=original, cfg=rc)
    else:
        recon_n = safe_normalize(fg_recon)

    history['n'].append(len(cur_pts))
    history['psnr'].append(psnr(orig, recon_n))
    history['ssim'].append(ssim(orig, recon_n, data_range=1.0))
    try:
        history['fg_psnr'].append(psnr(orig[mask > 0], recon_n[mask > 0]))
    except Exception:
        history['fg_psnr'].append(history['psnr'][-1])
    history['fg_ssim'].append(ssim(orig * fg, recon_n * fg, data_range=1.0))

    return pts_arr, int_arr, w_arr, ori_arr, history, recon_n
