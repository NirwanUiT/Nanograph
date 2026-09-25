"""
Regenerates figures/fig_seg_bottleneck.png. Panels (a)/(b) reproduce the
original from dataset_results/metrics.csv. Panel (c) replaces the standalone
U-Net line with the actual in-pipeline selected-mask Seg-IoU distribution from
the learned_mode='replace' run (dataset_results_learned_replace/metrics.csv),
per item N7.

Run from the Nanograph repo root:
    python paper/revision_extracted/regen_fig_seg_bottleneck.py
"""
import csv
import os

import numpy as np
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "figures", "fig_seg_bottleneck.png")
old = list(csv.DictReader(open("dataset_results/metrics.csv")))
new = list(csv.DictReader(open("dataset_results_learned_replace/metrics.csv")))
col = lambda rows, c: np.array([float(r[c]) for r in rows])

fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

# (a) precision vs recall of the classical cascade
ax = axes[0]
prec, rec = col(old, "seg_precision"), col(old, "seg_recall")
ax.scatter(rec, prec, s=6, color="#4878a8", alpha=0.55)
ax.axhline(prec.mean(), ls="--", color="#c0392b",
           label=f"mean precision = {prec.mean():.3f}")
ax.axvline(rec.mean(), ls="--", color="#2e7d32",
           label=f"mean recall = {rec.mean():.3f}")
ax.set_xlabel("Segmentation recall")
ax.set_ylabel("Segmentation precision")
ax.set_title("(a) Classical cascade over-predicts")
ax.legend(fontsize=9, loc="lower left")

# (b) GT-IoU of reconstruction vs raw Seg-IoU
ax = axes[1]
si, gi = col(old, "seg_iou"), col(old, "gt_iou")
above = (gi > si).mean() * 100
ax.scatter(si, gi, s=6, color="#4878a8", alpha=0.55)
lim = [min(si.min(), gi.min()) - 0.02, max(si.max(), gi.max()) + 0.02]
ax.plot(lim, lim, "k--", lw=1.2, label="no change")
ax.text(0.03, 0.97, f"above diagonal:\n{above:.0f}% of images\n"
                    f"(+{(gi - si).mean():.3f} mean)",
        transform=ax.transAxes, va="top", fontsize=9,
        bbox=dict(boxstyle="round", fc="white", ec="0.7"))
ax.set_xlabel("Segmentation IoU (mask vs. GT)")
ax.set_ylabel("GT-IoU (reconstruction vs. GT)")
ax.set_title("(b) Reconstruction sharpens localisation")
ax.legend(fontsize=9, loc="lower right")

# (c) in-pipeline selected-mask Seg-IoU: default cascade vs learned-replace run
ax = axes[2]
si_new = col(new, "seg_iou")
bins = np.linspace(0.15, 1.0, 55)
ax.hist(si, bins=bins, color="#4878a8", alpha=0.85,
        label=f"default cascade (mean {si.mean():.3f})")
ax.hist(si_new, bins=bins, color="#2e7d32", alpha=0.75,
        label=f"learned_mode='replace' (mean {si_new.mean():.3f})")
ax.axvline(si.mean(), ls="--", color="#c0392b", lw=1.2)
ax.axvline(si_new.mean(), ls="--", color="#1b5e20", lw=1.2)
ax.set_xlabel("In-pipeline selected-mask Seg-IoU")
ax.set_ylabel("Images")
ax.set_title("(c) Selected-mask IoU: cascade vs. learned replace")
ax.legend(fontsize=9, loc="upper left")

for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
fig.savefig(OUT, dpi=300)
print("wrote", OUT)
print(f"(c) default mean {si.mean():.4f}  replace mean {si_new.mean():.4f}")
