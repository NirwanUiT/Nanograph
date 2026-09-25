"""
Regenerates figures/cross_dataset_comparison.png from the cross-dataset and
organelle metrics CSVs. Replaces the original interactive figure, which had
"Nanograph v5" baked into the title and a "Recon IoU" axis label (renamed
Self-IoU in the revision).

Run from the Nanograph repo root:
    python paper/revision_extracted/regen_fig_cross_comparison.py
"""
import csv
import os

import numpy as np
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "figures", "cross_dataset_comparison.png")
SETS = [("Organelles", "dataset_results/metrics.csv", "#1f77b4"),
        ("Cells3D\nMembrane", "cross_dataset_results/cells3d_membrane/metrics.csv", "#2ca02c"),
        ("Cells3D\nNuclei", "cross_dataset_results/cells3d_nuclei/metrics.csv", "#ffbf00"),
        ("Retina\nVessels", "cross_dataset_results/retina/metrics.csv", "#ff5722"),
        ("Cell\n(fluorescence)", "cross_dataset_results/cell/metrics.csv", "#9c27b0")]
PANELS = [("FG-SSIM", "ng_fg_ssim", "{:.2f}"),
          ("Full PSNR (dB)", "ng_psnr", "{:.1f}"),
          ("Payload (bytes)", "ng_bytes", "{:.0f}"),
          ("Self-IoU", "ng_iou", "{:.2f}"),
          ("Topology-Q (heuristic)", "ng_topo_q", "{:.2f}"),
          ("Skeleton-graph cycles", "ng_n_cycles", "{:.0f}")]

data = {}
for name, path, _ in SETS:
    rows = list(csv.DictReader(open(path)))
    data[name] = {c: np.array([float(r[c]) for r in rows]) for _, c, _ in PANELS}

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
x = np.arange(len(SETS))
for ax, (label, col, fmt) in zip(axes.flat, PANELS):
    means = [data[n][col].mean() for n, _, _ in SETS]
    stds = [data[n][col].std(ddof=1) for n, _, _ in SETS]
    ax.bar(x, means, yerr=stds, capsize=4,
           color=[c for _, _, c in SETS], width=0.65)
    for xi, m in zip(x, means):
        ax.annotate(fmt.format(m), xy=(xi, m), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=8)
    ax.set_ylabel(label)
    ax.set_xticks(x)
    ax.set_xticklabels([n for n, _, _ in SETS], fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
fig.savefig(OUT, dpi=300)
print("wrote", OUT)
