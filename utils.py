"""
Nanograph v5 — Shared utility functions.

v5 additions:
  - Spatial background grid (compute_bg_grid / interpolate_bg_grid)
  - Persistent homology metrics (compute_betti_numbers, betti_error)
  - Enhanced topology preservation (euler_characteristic, topology_preservation_score)
  - Foreground residual coding helpers (encode_fg_residual / decode_fg_residual)
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


# ============================================================
#  v5: Spatial background grid
# ============================================================

def compute_bg_grid(image_float, mask, grid_size=16):
    """
    Compute a downsampled background intensity grid from non-foreground pixels.

    Divides the image into grid_size × grid_size tiles and computes the
    mean intensity of background pixels (mask==0) in each tile.
    For tiles with no background pixels, uses overall tile mean.

    Parameters:
        image_float: float64 image normalized to [0, 1]
        mask: binary foreground mask (>0 = foreground)
        grid_size: grid resolution (default 16)

    Returns: (grid_size, grid_size) float64 array in [0, 1]
    """
    h, w = image_float.shape
    tile_h = h / grid_size
    tile_w = w / grid_size
    grid = np.zeros((grid_size, grid_size), dtype=np.float64)
    bg_mask = (mask == 0)

    for gy in range(grid_size):
        for gx in range(grid_size):
            y0 = int(round(gy * tile_h))
            y1 = min(int(round((gy + 1) * tile_h)), h)
            x0 = int(round(gx * tile_w))
            x1 = min(int(round((gx + 1) * tile_w)), w)

            tile_bg = bg_mask[y0:y1, x0:x1]
            tile_img = image_float[y0:y1, x0:x1]
            if np.any(tile_bg):
                grid[gy, gx] = np.mean(tile_img[tile_bg])
            else:
                grid[gy, gx] = np.mean(tile_img)

    return grid


def interpolate_bg_grid(grid, target_shape):
    """Bilinearly interpolate a background grid to full image resolution."""
    return cv2.resize(grid.astype(np.float64),
                      (target_shape[1], target_shape[0]),
                      interpolation=cv2.INTER_LINEAR)


def quantize_bg_grid(grid):
    """Quantize a float64 [0,1] grid to uint8."""
    return np.clip(np.round(grid * 255), 0, 255).astype(np.uint8)


def dequantize_bg_grid(grid_u8):
    """Dequantize a uint8 grid back to float64 [0,1]."""
    return grid_u8.astype(np.float64) / 255.0


# ============================================================
#  v5: Foreground residual coding (DCT-based)
# ============================================================

def encode_fg_residual(original_float, reconstruction_float, mask,
                       quality=20, block_size=8):
    """
    Compute and compress the foreground residual using block-DCT.

    The residual (original - reconstruction) within the foreground mask is
    encoded using 8×8 DCT blocks with aggressive quantization, then
    compressed with zlib.  This captures high-frequency detail the PSF
    model misses while adding minimal bytes.

    Parameters:
        original_float: float64 [0,1] original image
        reconstruction_float: float64 [0,1] PSF reconstruction
        mask: binary foreground mask
        quality: quantization quality 1-100 (lower = more compression)
        block_size: DCT block size

    Returns: (compressed_bytes, residual_shape_info) tuple
    """
    import zlib

    residual = (original_float - reconstruction_float) * (mask > 0).astype(np.float64)

    # Compute bounding box of foreground for efficiency
    fg_coords = np.argwhere(mask > 0)
    if len(fg_coords) == 0:
        return b'', (0, 0, 0, 0, 0, 0)

    y0, x0 = fg_coords.min(axis=0)
    y1, x1 = fg_coords.max(axis=0) + 1

    # Extract ROI
    roi = residual[y0:y1, x0:x1]
    roi_mask = (mask[y0:y1, x0:x1] > 0).astype(np.float64)
    roi = roi * roi_mask

    # Pad to multiple of block_size
    rh, rw = roi.shape
    ph = (block_size - rh % block_size) % block_size
    pw = (block_size - rw % block_size) % block_size
    roi_padded = np.pad(roi, ((0, ph), (0, pw)), mode='constant')

    # Quantization matrix (similar to JPEG but tunable)
    q_scale = max(1, 101 - quality)
    base_quant = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68,109,103, 77],
        [24, 35, 55, 64, 81,104,113, 92],
        [49, 64, 78, 87,103,121,120,101],
        [72, 92, 95, 98,112,100,103, 99],
    ], dtype=np.float64)
    quant_table = np.clip(base_quant * q_scale / 50.0, 1, 255)

    # DCT transform and quantize each block (skip blocks with no foreground)
    bh = roi_padded.shape[0] // block_size
    bw = roi_padded.shape[1] // block_size

    # Also pad the mask to check FG coverage per block
    mask_roi_padded = np.pad(roi_mask, ((0, ph), (0, pw)), mode='constant')
    quantized_blocks = np.zeros((bh, bw, block_size, block_size), dtype=np.int8)

    for by in range(bh):
        for bx in range(bw):
            # Skip blocks with no foreground pixels
            mask_block = mask_roi_padded[by*block_size:(by+1)*block_size,
                                         bx*block_size:(bx+1)*block_size]
            if not np.any(mask_block > 0):
                continue
            block = roi_padded[by*block_size:(by+1)*block_size,
                               bx*block_size:(bx+1)*block_size]
            # Scale residual to [-128, 127] range for DCT
            block_scaled = block * 255.0
            dct_block = cv2.dct(block_scaled.astype(np.float32))
            quantized = np.round(dct_block / quant_table).astype(np.int8)
            quantized_blocks[by, bx] = quantized

    # Compress
    raw_data = quantized_blocks.tobytes()
    compressed = zlib.compress(raw_data, 9)

    shape_info = (y0, x0, rh, rw, bh, bw)
    return compressed, shape_info


def decode_fg_residual(compressed_bytes, shape_info, image_shape,
                       quality=20, block_size=8):
    """
    Decode a compressed foreground residual back to a full-resolution residual image.

    Returns: float64 residual image (same shape as original)
    """
    import zlib

    y0, x0, rh, rw, bh, bw = shape_info
    if bh == 0 or bw == 0:
        return np.zeros(image_shape, dtype=np.float64)

    q_scale = max(1, 101 - quality)
    base_quant = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68,109,103, 77],
        [24, 35, 55, 64, 81,104,113, 92],
        [49, 64, 78, 87,103,121,120,101],
        [72, 92, 95, 98,112,100,103, 99],
    ], dtype=np.float64)
    quant_table = np.clip(base_quant * q_scale / 50.0, 1, 255)

    raw_data = zlib.decompress(compressed_bytes)
    quantized_blocks = np.frombuffer(raw_data, dtype=np.int8).reshape(bh, bw,
                                                                       block_size,
                                                                       block_size)

    # Inverse DCT
    ph = bh * block_size - rh
    pw = bw * block_size - rw
    padded_h = bh * block_size
    padded_w = bw * block_size
    roi_recon = np.zeros((padded_h, padded_w), dtype=np.float64)

    for by in range(bh):
        for bx in range(bw):
            dct_block = quantized_blocks[by, bx].astype(np.float32) * quant_table.astype(np.float32)
            block = cv2.idct(dct_block)
            roi_recon[by*block_size:(by+1)*block_size,
                      bx*block_size:(bx+1)*block_size] = block / 255.0

    # Unpad and place into full image
    roi_recon = roi_recon[:rh, :rw]
    result = np.zeros(image_shape, dtype=np.float64)
    result[y0:y0+rh, x0:x0+rw] = roi_recon
    return result


# ============================================================
#  v5: Persistent homology & topological metrics
# ============================================================

def compute_betti_numbers(binary_mask):
    """
    Compute Betti numbers (β₀, β₁) from a binary mask.

    β₀ = number of connected components (foreground objects)
    β₁ = number of 1-cycles (holes enclosed by foreground)

    Uses the discrete Euler relation for 2D binary images.
    """
    mask_u8 = (binary_mask > 0).astype(np.uint8)

    if not np.any(mask_u8):
        return 0, 0

    # β₀: foreground connected components
    n_fg, _ = cv2.connectedComponents(mask_u8, connectivity=8)
    beta_0 = n_fg - 1  # subtract background label

    # β₁: holes = background components fully enclosed by foreground
    inv_mask = 1 - mask_u8
    n_bg, labels_bg = cv2.connectedComponents(inv_mask, connectivity=4)

    # Find which background components touch the image border
    h, w = binary_mask.shape
    border_labels = set()
    border_labels.update(labels_bg[0, :].tolist())
    border_labels.update(labels_bg[h-1, :].tolist())
    border_labels.update(labels_bg[:, 0].tolist())
    border_labels.update(labels_bg[:, w-1].tolist())
    border_labels.discard(0)

    # Interior holes = total bg components - border components - label 0
    n_bg_total = n_bg - 1
    n_border_bg = len(border_labels)
    beta_1 = max(0, n_bg_total - n_border_bg)

    return beta_0, beta_1


def euler_characteristic(binary_mask):
    """
    Compute the Euler characteristic χ = β₀ - β₁ of a binary mask.
    A topologically faithful reconstruction should preserve χ.
    """
    b0, b1 = compute_betti_numbers(binary_mask)
    return b0 - b1


def betti_error(mask_orig, mask_recon):
    """
    Compute detailed Betti number comparison between original and reconstruction.

    Returns a dict with individual Betti numbers, relative errors,
    and a composite betti_preservation score in [0, 1].
    """
    b0_o, b1_o = compute_betti_numbers(mask_orig)
    b0_r, b1_r = compute_betti_numbers(mask_recon)

    err_0 = abs(b0_o - b0_r) / max(b0_o, 1)
    err_1 = abs(b1_o - b1_r) / max(b1_o, 1)

    chi_o = b0_o - b1_o
    chi_r = b0_r - b1_r
    chi_err = abs(chi_o - chi_r) / max(abs(chi_o), 1)

    return {
        'beta_0_orig': b0_o, 'beta_0_recon': b0_r,
        'beta_1_orig': b1_o, 'beta_1_recon': b1_r,
        'beta_0_error': err_0, 'beta_1_error': err_1,
        'euler_orig': chi_o, 'euler_recon': chi_r,
        'euler_error': chi_err,
        'betti_preservation': 1.0 - min(1.0, (err_0 + err_1) / 2),
    }


def sublevel_persistence(image_float, n_thresholds=50):
    """
    Compute a simplified persistence diagram via sublevel set filtration.

    Sweeps thresholds from min to max intensity and tracks connected
    component births/deaths. Returns list of (birth, death) pairs.

    This is a lightweight approximation of cubical persistent homology
    that doesn't require external libraries (gudhi/ripser).
    """
    mn, mx = float(image_float.min()), float(image_float.max())
    if mx - mn < 1e-9:
        return []

    thresholds = np.linspace(mn, mx, n_thresholds)
    prev_labels = None
    prev_n = 0
    births = {}
    diagram = []

    for t in thresholds:
        binary = (image_float >= t).astype(np.uint8)
        n_cc, labels = cv2.connectedComponents(binary, connectivity=8)
        n_cc -= 1  # subtract bg

        if prev_labels is not None:
            # Components that disappeared (merged or vanished)
            if n_cc < prev_n:
                n_died = prev_n - n_cc
                # Approximate: oldest components die
                sorted_births = sorted(births.items(), key=lambda x: x[1])
                for i in range(min(n_died, len(sorted_births))):
                    bid, birth_t = sorted_births[i]
                    if t - birth_t > (mx - mn) / n_thresholds:
                        diagram.append((birth_t, t))
                    del births[sorted_births[i][0]]
            # New components born
            if n_cc > prev_n:
                for new_id in range(prev_n, n_cc):
                    births[len(births) + new_id] = t

        else:
            for cid in range(n_cc):
                births[cid] = t

        prev_labels = labels
        prev_n = n_cc

    # Remaining births (never die)
    for bid, birth_t in births.items():
        diagram.append((birth_t, mx))

    return diagram


def wasserstein_persistence(dgm1, dgm2, p=1):
    """
    Approximate Wasserstein-p distance between two persistence diagrams.

    Uses a greedy matching for efficiency. For exact computation,
    install gudhi or persim.
    """
    if not dgm1 and not dgm2:
        return 0.0
    if not dgm1 or not dgm2:
        # Distance is sum of all persistences in the non-empty diagram
        non_empty = dgm1 if dgm1 else dgm2
        return sum(abs(d - b) ** p for b, d in non_empty) ** (1.0 / p)

    pts1 = np.array(dgm1)
    pts2 = np.array(dgm2)

    # Greedy nearest-neighbor matching
    from scipy.spatial.distance import cdist
    cost = cdist(pts1, pts2, metric='minkowski', p=p)

    total = 0.0
    used_j = set()
    for i in range(len(pts1)):
        best_j = None
        best_c = float('inf')
        for j in range(len(pts2)):
            if j not in used_j and cost[i, j] < best_c:
                best_c = cost[i, j]
                best_j = j
        if best_j is not None:
            total += best_c ** p
            used_j.add(best_j)
        else:
            # Unmatched: distance to diagonal
            total += abs(pts1[i, 1] - pts1[i, 0]) ** p

    # Unmatched points in dgm2
    for j in range(len(pts2)):
        if j not in used_j:
            total += abs(pts2[j, 1] - pts2[j, 0]) ** p

    return total ** (1.0 / p)


def topology_preservation_score(mask_orig, mask_recon):
    """
    Comprehensive topology preservation score combining:
    1. Betti number preservation (β₀, β₁)
    2. Euler characteristic preservation
    3. Persistence diagram similarity (Wasserstein)

    For masks that are very different (e.g. smooth reconstruction vs crisp
    segmentation), uses relative error with saturation at 1.0 per component.

    Returns a dict with all sub-scores and a composite 'topology_score' in [0, 1].
    """
    betti = betti_error(mask_orig, mask_recon)

    # Persistence diagram comparison on the masks themselves
    orig_float = mask_orig.astype(np.float64) / max(mask_orig.max(), 1)
    recon_float = mask_recon.astype(np.float64) / max(mask_recon.max(), 1)
    dgm_orig = sublevel_persistence(orig_float, n_thresholds=30)
    dgm_recon = sublevel_persistence(recon_float, n_thresholds=30)
    w_dist = wasserstein_persistence(dgm_orig, dgm_recon)
    # Normalize by max possible persistence
    w_norm = w_dist / max(1.0, max(
        max((d - b for b, d in dgm_orig), default=0),
        max((d - b for b, d in dgm_recon), default=0),
    ))
    persistence_sim = max(0.0, 1.0 - w_norm)

    # Composite score
    topology_score = (
        0.35 * betti['betti_preservation'] +
        0.30 * (1.0 - min(1.0, betti['euler_error'])) +
        0.35 * persistence_sim
    )

    return {
        **betti,
        'persistence_wasserstein': w_dist,
        'persistence_similarity': persistence_sim,
        'topology_score': float(np.clip(topology_score, 0, 1)),
    }


def graph_topology_score(graph):
    """
    Compute topology metrics directly from the Nanograph graph structure.

    This is more meaningful than comparing Betti numbers of binarized images,
    because the graph IS the structural representation and its topology
    is exactly preserved through encode/decode.

    Returns a dict with:
      - n_components: connected components (graph β₀)
      - n_cycles: independent cycles (graph β₁ = edges - nodes + components)
      - euler_characteristic: χ = nodes - edges
      - n_junctions: number of junction nodes (branch points)
      - n_endpoints: number of endpoint nodes (degree 1)
      - mean_degree: average node degree
      - graph_complexity: n_cycles / max(n_components, 1)
    """
    if graph is None:
        return {
            'n_components': 0, 'n_cycles': 0, 'euler_characteristic': 0,
            'n_junctions': 0, 'n_endpoints': 0, 'mean_degree': 0.0,
            'graph_complexity': 0.0,
        }

    gs = graph.summary()
    n_nodes = gs['n_nodes']
    n_edges = gs['n_edges']
    n_components = gs['n_components']
    # For a graph: β₁ = E - V + C  (number of independent cycles)
    n_cycles = max(0, n_edges - n_nodes + n_components)

    return {
        'n_components': n_components,
        'n_cycles': n_cycles,
        'euler_characteristic': n_nodes - n_edges,
        'n_junctions': gs.get('n_junctions', 0),
        'n_endpoints': gs.get('n_endpoints', 0),
        'mean_degree': gs.get('mean_degree', 0.0),
        'graph_complexity': n_cycles / max(n_components, 1),
    }
