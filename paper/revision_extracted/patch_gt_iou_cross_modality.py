"""
Item 10 -- compute GT-IoU for the four cross-modality datasets.

WHY THIS IS THE FIRST THING TO RUN
  cross_dataset_results/*/metrics.csv contains ng_iou and jpeg_iou but no
  gt_iou column. So the 82-100% win rates the paper reports for the
  cross-modality datasets have no ground-truth-referenced counterpart -- and
  on the organelle set, where GT-IoU *was* computed, the ground-truth version
  of that same comparison turns out to be parity (49%, sign test p = 0.53).
  Until this runs, Sec. cross rests entirely on the self-referenced metric.

WHY IT COULD NOT BE RUN IN-SESSION
  The ground-truth masks are not in the committed snapshot. They are reachable
  on your machine: experiments/eval_ext_datasets.py reads them from
  f"{P}/{ds}/labels/{idd}.png" and its own docstring notes that all four
  datasets carry ground-truth masks. numpy/OpenCV were also unavailable
  (package index down).

WHERE TO PUT IT
  In whichever driver emitted cross_dataset_results/ (the one producing
  ng_iou/jpeg_iou), call gt_iou_columns() per image and add both keys to the
  CSV row. Definitions mirror the organelle set's exactly, so the new numbers
  are directly comparable to Table tab:nmi.
"""
import cv2
import numpy as np


def otsu_mask(img_u8):
    _, m = cv2.threshold(img_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return m > 0


def iou(a, b):
    a, b = a > 0, b > 0
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def gt_iou_columns(recon_u8, jpeg_recon_u8, gt_mask_u8):
    """Otsu of the reconstruction vs the INDEPENDENT ground-truth mask.

    Mirrors the organelle set's gt_iou / jpeg_gt_iou definition. Note the
    contrast with Recon-IoU, which thresholds the same reconstruction but
    compares it against the pipeline's own selected mask.
    """
    gt = gt_mask_u8 > 0
    return {
        "gt_iou":      iou(otsu_mask(recon_u8), gt),
        "jpeg_gt_iou": iou(otsu_mask(jpeg_recon_u8), gt),
    }


# Worth adding in the same pass, while the masks are already loaded:
#   jpeg_beta_0 / jpeg_beta_1 / jpeg_betti_preservation exist for the organelle
#   set but for no other codec and no cross-modality dataset. Table tab:topo in
#   the revised manuscript therefore covers JPEG on organelles only, and says so.
