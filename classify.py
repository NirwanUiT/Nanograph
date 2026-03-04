"""
Nanograph v4 — Structure shape classification.
"""

import numpy as np
import cv2

from .config import DEFAULT_CONFIG, NanographConfig, ClassifyConfig


def classify_structures(clean_mask, skeleton, ep_mask, jn_mask, dist_transform,
                        cfg=None):
    """Classify each connected component by topology + geometry."""
    cl = cfg if isinstance(cfg, ClassifyConfig) else (
         cfg.classify if isinstance(cfg, NanographConfig) else DEFAULT_CONFIG.classify)

    n_cc, cc_labels = cv2.connectedComponents(clean_mask, connectivity=8)
    structures = []

    for i in range(1, n_cc):
        component = (cc_labels == i).astype(np.uint8)
        area = int(np.count_nonzero(component))
        if area < cl.min_area:
            continue

        comp_skel = skeleton * component
        skel_len = int(np.count_nonzero(comp_skel))
        n_ep_i = int(np.count_nonzero(ep_mask & (component > 0)))
        n_jn_i = int(np.count_nonzero(jn_mask & (component > 0)))

        ys, xs = np.where(component > 0)
        bb_h = ys.max() - ys.min() + 1
        bb_w = xs.max() - xs.min() + 1
        aspect = max(bb_h, bb_w) / max(min(bb_h, bb_w), 1)

        contours, hierarchy = cv2.findContours(
            component, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        n_holes = 0
        if hierarchy is not None:
            for h in hierarchy[0]:
                if h[3] >= 0:
                    n_holes += 1

        perimeter = cv2.arcLength(contours[0], True) if contours else 1.0
        compactness = 4 * np.pi * area / max(perimeter ** 2, 1e-10)

        skel_px = comp_skel > 0
        mean_width = float(dist_transform[skel_px].mean()) if np.any(skel_px) else 0.0

        if area < cl.punctate_max_area and compactness > cl.punctate_min_compactness:
            shape_type = 'punctate'
        elif n_holes > 0:
            shape_type = 'ring'
        elif n_jn_i > 0:
            shape_type = 'branched'
        elif aspect > cl.curved_min_aspect and skel_len > cl.curved_min_skel_len:
            shape_type = 'curved'
        elif aspect > cl.linear_min_aspect:
            shape_type = 'linear'
        else:
            shape_type = 'punctate'

        structures.append({
            'id': i, 'area': area, 'skel_length': skel_len,
            'endpoints': n_ep_i, 'junctions': n_jn_i,
            'holes': n_holes, 'aspect_ratio': aspect,
            'compactness': compactness, 'mean_width': mean_width,
            'shape': shape_type,
            'centroid': (int(np.mean(ys)), int(np.mean(xs))),
        })
    return structures
