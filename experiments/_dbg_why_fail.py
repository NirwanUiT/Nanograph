"""Diagnose WHY segmentation fails on the external (dark-on-bright) datasets.

For one representative image per dataset, faithfully reproduce the pipeline's
front-end (polarity -> type detect -> preprocess) and then, for each cheap
segmenter, report:
  - fg%, IoU-vs-GT, and the cascade quality score   (as the pipeline sees it)
  - the SAME on the polarity-INVERTED image          (what SHOULD happen)
  - an ORACLE ceiling: best IoU over all thresholds of a dark-ridge vesselness
    map (Frangi black_ridges=True) = is a good vessel mask even achievable?

This separates a SELECTION failure (good mask exists, wrong pick) from a
REPRESENTATION failure (no good mask is produced at all).
"""
import os, sys, glob
import numpy as np
import cv2
from skimage.filters import frangi, meijering

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nanograph_v4 import NanographConfig
from nanograph_v4.detect import detect_image_type, detect_polarity
from nanograph_v4.preprocess import preprocess
from nanograph_v4.segment import (otsu_segment, frangi_segment, meijering_segment,
                                  segmentation_quality_score)

P = "/mnt/nas1/nba055-2/idea_1/ext_datasets/prepared"
REP = {"stare": "im0001", "drive": "22", "epfl_mito": "slice000",
       "microtubules": "irm_in_vitro_microtubules_0"}


def iou(m, g):
    m, g = m > 0, g > 0
    u = np.logical_or(m, g).sum()
    return float(np.logical_and(m, g).sum()) / u if u else 0.0


def oracle_darkridge_iou(img_gray, gt):
    """Best achievable IoU: dark-ridge Frangi map, swept over thresholds."""
    f = img_gray.astype(np.float64) / 255.0
    v = frangi(f, sigmas=range(1, 6), black_ridges=True)   # DARK ridges
    v = (v - v.min()) / (v.max() - v.min() + 1e-15)
    best = 0.0; best_t = 0
    for t in np.linspace(0.02, 0.4, 20):
        best_iou = iou((v > t).astype(np.uint8), gt)
        if best_iou > best:
            best, best_t = best_iou, t
    return best, best_t


def analyse(ds):
    idd = REP[ds]
    im = cv2.imread(f"{P}/{ds}/images/{idd}.png", 0)
    gt = cv2.imread(f"{P}/{ds}/labels/{idd}.png", 0)
    cfg = NanographConfig()
    pol = detect_polarity(im, cfg=cfg)
    det = detect_image_type(im, cfg=cfg)
    cfgt = cfg.for_image_type(det["type"])
    efg = (cfgt.segment.expected_fg_lo, cfgt.segment.expected_fg_hi)

    print(f"\n===== {ds}/{idd}  shape={im.shape}  GT_fg={100*(gt>0).mean():.1f}% =====")
    print(f"  detect_polarity -> {pol} (True=would invert)   "
          f"type={det['type']}  expected_fg={efg[0]:.0f}-{efg[1]:.0f}%")

    for tag, x in [("as-is (pipeline)", im), ("INVERTED (correct polarity)", 255 - im)]:
        _, bgs, _, _ = preprocess(x, bg_kernel_size=det["bg_kernel_size"], cfg=cfgt)
        print(f"  -- {tag} --")
        for name, fn in [("Otsu", lambda i: otsu_segment(i)),
                         ("Frangi", lambda i: frangi_segment(i, cfg=cfgt)),
                         ("Meijering", lambda i: meijering_segment(i, cfg=cfgt))]:
            m = fn(bgs)
            q = segmentation_quality_score(m, x, expected_fg_range=efg, cfg=cfgt)
            print(f"     {name:10s} fg%={100*(m>0).mean():5.1f}  "
                  f"IoU={iou(m,gt):.3f}  qscore={q:.3f}")

    orc, t = oracle_darkridge_iou(im, gt)
    print(f"  ORACLE dark-ridge Frangi best IoU = {orc:.3f} (thr={t:.2f}) "
          f"<- is a good mask even achievable?")


if __name__ == "__main__":
    for ds in ["stare", "drive", "epfl_mito", "microtubules"]:
        analyse(ds)
