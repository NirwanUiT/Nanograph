"""Sample comparison panels for the external curvilinear datasets.

Columns: Original | Ground Truth | Classical | Default (learned+gate)
Title per seg column = selected segmenter + IoU vs GT.
Rows = several images from the chosen dataset.
"""
import os, sys, glob
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nanograph_v4 import nanograph_encode, NanographConfig

P = "/mnt/nas1/nba055-2/idea_1/ext_datasets/prepared"
DS = os.environ.get("DS", "stare")
N_SHOW = int(os.environ.get("N_SHOW", "5"))
OUT = os.path.join(os.path.dirname(__file__), f"ext_samples_{DS}.png")


def iou(p, g):
    p, g = p > 0, g > 0
    u = np.logical_or(p, g).sum()
    return float(np.logical_and(p, g).sum()) / u if u else 0.0


def seg(im, use_learned):
    c = NanographConfig()
    c.segment.use_learned = use_learned
    r = nanograph_encode(im, sam_model=None, verbose=False, optimize=False, config=c)
    return (r.mask > 0).astype(np.uint8), r.segmenter, r.compression_ratio, r.psnr_full


def main():
    imgs = sorted(glob.glob(f"{P}/{DS}/images/*.png"))[:N_SHOW]
    rows = []
    for ip in imgs:
        idd = os.path.splitext(os.path.basename(ip))[0]
        im = cv2.imread(ip, 0)
        gt = cv2.imread(f"{P}/{DS}/labels/{idd}.png", 0)
        mc, sc, rc, pc = seg(im, False)
        md, sd, rd, pd = seg(im, True)
        rows.append((idd, im, gt, mc, sc, iou(mc, gt), md, sd, iou(md, gt)))
        print(f"{DS}/{idd}: classical={sc} IoU={iou(mc,gt):.3f}  "
              f"default={sd} IoU={iou(md,gt):.3f}")

    n = len(rows)
    fig, axes = plt.subplots(n, 4, figsize=(12, 3 * n))
    if n == 1:
        axes = axes[None, :]
    titles = ["Original", "Ground Truth", "Classical", "Default"]
    for r, (idd, im, gt, mc, sc, ic, md, sd, idd_i) in enumerate(rows):
        panels = [(im, "gray", idd), (gt, "gray", "GT"),
                  (mc, "magma", f"{sc}\nIoU {ic:.3f}"),
                  (md, "viridis", f"{sd}\nIoU {idd_i:.3f}")]
        for c, (m, cm, ttl) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(m, cmap=cm)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"{titles[c]}\n{ttl}" if c >= 2 else titles[c], fontsize=10)
            elif c >= 1:
                ax.set_title(ttl, fontsize=9)
            if c == 0 and r != 0:
                ax.set_title(idd, fontsize=9)
    fig.suptitle(f"Nanograph segmentation on {DS}", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(OUT, dpi=110, bbox_inches="tight")
    print("saved:", OUT)


if __name__ == "__main__":
    main()
