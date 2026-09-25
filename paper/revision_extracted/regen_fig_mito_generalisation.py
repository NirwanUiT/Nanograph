"""
Generates figures/fig_mito_generalisation.png: pipeline stages (input, mask,
skeleton, reconstruction, graph overlay) on one representative image from each
additional mitochondrial acquisition evaluated in the generalisation study:
the Aaron temporal clip (same acquisition as the main set), the public STED
TOM20 dataset (learned-replace preset), and the public MITO two-microscope
dataset (max-projection tiles).

Run from the Nanograph repo root:
    python paper/revision_extracted/regen_fig_mito_generalisation.py
"""
import os
import sys

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from nanograph_v4 import nanograph_encode, NanographConfig

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "figures", "fig_mito_generalisation.png")

def replace_cfg():
    c = NanographConfig()
    c.segment.learned_mode = "replace"
    return c

ROWS = [
    ("Temporal clip\n(same acquisition)",
     "/mnt/nas1/nba055-2/idea_1/mito_aaron/images/1_22.png", NanographConfig),
    ("STED TOM20\n(learned-replace)",
     "/mnt/nas1/nba055-2/idea_1/public_mito/sted_prep/images/d2_d0002_p0000_t0000.png", replace_cfg),
    ("MITO two-microscope\n(max-projection tile)",
     "/mnt/nas1/nba055-2/idea_1/public_mito/mito_mip_tiles/images/M1_008_y768x512.png", NanographConfig),
]
COLS = ["input", "selected mask", "skeleton", "reconstruction", "graph"]

fig, axes = plt.subplots(len(ROWS), 5, figsize=(16, 3.2 * len(ROWS)))
for ri, (label, path, cfg_fn) in enumerate(ROWS):
    img = cv2.imread(path, 0)
    r = nanograph_encode(img, sam_model=None, verbose=False, optimize=False,
                         config=cfg_fn())
    panels = [img, r.mask, r.skeleton, r.reconstruction, None]
    for ci, (ax, p) in enumerate(zip(axes[ri], panels)):
        if p is None:
            ax.imshow(img, cmap="gray")
            pts = r.graph.node_positions
            ax.scatter(pts[:, 1], pts[:, 0], s=1.5, c="red", linewidths=0)
        else:
            ax.imshow(p, cmap="gray")
        ax.axis("off")
        if ri == 0:
            ax.set_title(COLS[ci], fontsize=11)
    axes[ri][0].set_ylabel(label, fontsize=10)
    axes[ri][0].axis("on")
    axes[ri][0].set_xticks([]); axes[ri][0].set_yticks([])
    axes[ri][4].set_title(
        (COLS[4] if ri == 0 else "") +
        f"\n{r.graph.n_nodes}n/{r.graph.n_edges}e, {r.compressed_bytes} B, "
        f"FG-PSNR {r.psnr_fg:.1f} dB", fontsize=8)

fig.tight_layout()
fig.savefig(OUT, dpi=250)
print("wrote", OUT)
