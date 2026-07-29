"""
Nanograph v4 — Skeleton extraction and point sampling.
"""

import numpy as np
import cv2
from scipy.spatial import KDTree
from skimage.morphology import skeletonize


def _degree_map(skel):
    conn_kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    return cv2.filter2D(skel.astype(np.uint8), -1, conn_kernel) * (skel > 0)


def prune_skeleton_spurs(skel, min_length=5, max_iter=10):
    """Remove short spur branches from a skeleton.

    Skeletonising thick, rounded structures (e.g. mitochondria) produces short
    barb branches hanging off the medial axis near blobby regions and tips.
    These spurs spawn spurious endpoint nodes and edges that splay off the true
    centreline. This routine walks inward from every endpoint; if the branch
    reaches a junction within ``min_length`` pixels it is deleted. Isolated short
    components (endpoint-to-endpoint, no junction) are preserved.

    Args:
        skel: binary skeleton (uint8/bool).
        min_length: branches shorter than this that terminate at a junction are
            pruned. Set <= 0 to disable.
        max_iter: pruning passes (pruning can expose new spurs).

    Returns:
        cleaned binary skeleton (uint8).
    """
    skel = (np.asarray(skel) > 0).astype(np.uint8).copy()
    if min_length is None or min_length <= 0:
        return skel
    H, W = skel.shape

    def neighbours(p, exclude):
        y, x = p
        out = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W and skel[ny, nx] and (ny, nx) not in exclude:
                    out.append((ny, nx))
        return out

    for _ in range(max_iter):
        deg = _degree_map(skel)
        endpoints = [tuple(p) for p in np.argwhere((deg == 1) & (skel > 0))]
        if not endpoints:
            break
        to_remove = []
        for ep in endpoints:
            path = [ep]
            prev = None
            cur = ep
            is_spur = False
            while len(path) < min_length:
                nbrs = neighbours(cur, exclude={prev} if prev is not None else set())
                nbrs = [n for n in nbrs if n not in path]
                if not nbrs:
                    break  # isolated short segment — keep it
                jn = [n for n in nbrs if deg[n] >= 3]
                if jn or len(nbrs) > 1:
                    is_spur = True  # branch meets a junction => spur off main axis
                    break
                prev, cur = cur, nbrs[0]
                path.append(cur)
            if is_spur and len(path) < min_length:
                to_remove.extend(path)
        if not to_remove:
            break
        for (y, x) in to_remove:
            skel[y, x] = 0
    return skel


def skeletonize_and_classify(mask, prune_len=0):
    """Skeleton + classify pixels by 8-connected degree.

    Args:
        mask: binary foreground mask.
        prune_len: if > 0, remove spur branches shorter than this many pixels
            before classification (reduces off-centreline noise).
    """
    skel = skeletonize(mask > 0).astype(np.uint8)
    if prune_len and prune_len > 0:
        skel = prune_skeleton_spurs(skel, min_length=prune_len)
    deg = _degree_map(skel)
    ep_mask = (deg == 1) & (skel > 0)
    jn_mask = (deg >= 3) & (skel > 0)
    return skel, ep_mask, jn_mask


def compute_skeleton_orientations(skeleton, smooth_sigma=3.0):
    """
    v4: Compute local tangent orientation at each skeleton pixel.
    
    Uses gradient of the distance-from-boundary along the skeleton
    to estimate the local direction. Falls back to structure tensor
    for more robust estimation.
    
    Returns: orientation map (same shape as skeleton), angles in radians [0, pi)
    """
    skel_f = skeleton.astype(np.float32)
    
    # Structure tensor approach: compute gradients of the skeleton itself
    # Smooth the skeleton slightly to get meaningful gradients
    skel_smooth = cv2.GaussianBlur(skel_f, (0, 0), sigmaX=smooth_sigma)
    
    gx = cv2.Sobel(skel_smooth, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(skel_smooth, cv2.CV_64F, 0, 1, ksize=3)
    
    # The gradient is PERPENDICULAR to the skeleton direction
    # So the tangent is rotated 90 degrees
    # tangent_angle = atan2(gy, gx) + pi/2
    # But we want the angle of the skeleton direction, so:
    orientation = np.arctan2(-gx, gy) % np.pi  # [0, pi)
    
    # Only valid at skeleton pixels
    orientation_map = orientation * (skeleton > 0).astype(np.float64)
    
    return orientation_map


def extract_nanograph_points(skeleton, ep_mask, jn_mask, original_img,
                              dist_transform, reflectivity, orientation_map=None,
                              spacing=3, width_cap=30.0, kdtree_rebuild_batch=50):
    """
    Extract nanograph nodes from skeleton.
    Each node: (row, col, width, intensity, [orientation])
    
    v4: Also extracts orientation angle per point if orientation_map provided.
    """
    pts, types = [], []

    # 1. Critical points (endpoints + junctions)
    for y, x in np.argwhere(ep_mask):
        pts.append([int(y), int(x)])
        types.append('endpoint')
    for y, x in np.argwhere(jn_mask):
        pts.append([int(y), int(x)])
        types.append('junction')
    selected = set(map(tuple, pts))

    # 2. Remaining skeleton pixels sorted by reflectivity
    remaining = [(int(y), int(x)) for y, x in np.argwhere(skeleton > 0)
                 if (int(y), int(x)) not in selected]
    remaining.sort(key=lambda p: -reflectivity[p[0], p[1]])

    # 3. Greedy add with minimum spacing
    if len(pts) > 0 and len(remaining) > 0:
        tree = KDTree(np.array(pts))
        ctr = 0
        for y, x in remaining:
            dist, _ = tree.query([[y, x]])
            if dist[0] >= spacing:
                pts.append([y, x])
                types.append('sampled')
                selected.add((y, x))
                ctr += 1
                if ctr >= kdtree_rebuild_batch:
                    tree = KDTree(np.array(pts))
                    ctr = 0
    elif len(pts) == 0:
        for y, x in remaining:
            if not pts:
                pts.append([y, x])
                types.append('sampled')
                continue
            dists = np.sqrt(np.sum((np.array(pts) - [y, x]) ** 2, axis=1))
            if dists.min() >= spacing:
                pts.append([y, x])
                types.append('sampled')

    if len(pts) == 0:
        return (np.zeros((0, 2)), np.array([]), np.array([]),
                np.array([]), np.array([]), np.array([]))

    points = np.array(pts)
    intensities = np.array([original_img[y, x] / 255.0 for y, x in points])
    widths = np.array([min(max(float(dist_transform[y, x]), 1.0), width_cap)
                       for y, x in points])
    weights = np.array([reflectivity[y, x] for y, x in points])
    types = np.array(types)

    # v4: Extract orientations
    if orientation_map is not None:
        orientations = np.array([orientation_map[y, x] for y, x in points])
    else:
        orientations = np.zeros(len(points))

    return points, intensities, widths, weights, types, orientations
