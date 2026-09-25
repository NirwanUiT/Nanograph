"""
Generates figures/fig_seg_perturb.png from seg_perturbation_ablation.csv
(review item C): per-perturbation curves of relative error in beta0, beta1
(cycle rank) and mean width against the degraded mask's Seg-IoU.

Run from the Nanograph repo root:
    python paper/revision_extracted/regen_fig_seg_perturb.py
"""
import csv
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "figures", "fig_seg_perturb.png")
rows = list(csv.DictReader(open("seg_perturbation_ablation.csv")))
base = {r["filename"]: r for r in rows if r["kind"] == "baseline"}

agg = defaultdict(list)
for r in rows:
    if r["kind"] == "baseline":
        continue
    b = base[r["filename"]]
    b0 = float(b["beta0"])
    if b0 == 0:
        continue
    e0 = abs(float(r["beta0"]) - b0) / b0
    e1 = abs(float(r["beta1"]) - float(b["beta1"])) / max(1.0, float(b["beta1"]))
    ew = abs(float(r["mean_width"]) - float(b["mean_width"])) / float(b["mean_width"])
    agg[(r["kind"], float(r["level"]))].append((float(r["seg_iou"]), e0, e1, ew))

KINDS = [("dilate", "Dilation (1–5 px)"),
         ("erode", "Erosion (1–5 px)"),
         ("drop_components", "Component removal (5–40%)"),
         ("boundary_noise", "Boundary noise (1–3 px band)")]
COLORS = {"beta0": "#1565c0", "beta1": "#c62828", "width": "#2e7d32"}
base_iou = np.mean([float(b["seg_iou"]) for b in base.values()])

fig, axes = plt.subplots(1, 4, figsize=(18, 4.2), sharey=True)
for ax, (kind, title) in zip(axes, KINDS):
    levels = sorted(l for k, l in agg if k == kind)
    x = [np.mean([v[0] for v in agg[(kind, l)]]) for l in levels]
    for j, (key, lbl) in enumerate([("beta0", r"$\beta_0$"),
                                    ("beta1", r"$\beta_1$ (cycle rank)"),
                                    ("width", "mean width")]):
        y = [100 * np.median([v[1 + j] for v in agg[(kind, l)]]) for l in levels]
        ax.plot(x, y, "o-", color=COLORS[key], label=lbl)
    ax.axhline(10, ls="--", lw=1.0, color="0.4")
    ax.annotate("10%", xy=(0.02, 10), xytext=(0, 3), textcoords="offset points",
                fontsize=8, color="0.4")
    ax.axvline(base_iou, ls=":", lw=1.0, color="0.55")
    ax.axvline(0.489, ls=":", lw=1.0, color="#8e6d1a")
    ax.set_yscale("symlog", linthresh=10)
    ax.set_xlabel("Seg-IoU of degraded mask (vs. GT)")
    ax.set_title(title, fontsize=10)
    ax.invert_xaxis()
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("median relative error (%)")
axes[0].legend(fontsize=9, loc="upper left")
axes[0].annotate("unperturbed\n(learned mask)", xy=(base_iou, 300),
                 fontsize=7, color="0.45", ha="center")
axes[0].annotate("classical\ncascade", xy=(0.489, 300),
                 fontsize=7, color="#8e6d1a", ha="center")

fig.tight_layout()
fig.savefig(OUT, dpi=300)
print("wrote", OUT)
