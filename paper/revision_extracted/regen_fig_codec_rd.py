"""
Regenerates figures/fig_codec_rd.png (all three panels) from
dataset_results/metrics.csv, adding the GT-IoU bar to panel (c) that the
revised caption describes. Extends regen_fig_codec_panel_c.py, which only
produced a standalone panel; the manuscript includes one composite PNG.
Panel labels use "Self-IoU" (renamed from "Recon-IoU" in the revision).

Run from the Nanograph repo root:
    python paper/revision_extracted/regen_fig_codec_rd.py
"""
import csv
import os

import numpy as np
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "figures", "fig_codec_rd.png")
rows = list(csv.DictReader(open("dataset_results/metrics.csv")))
col = lambda c: np.array([float(r[c]) for r in rows])

fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

# (a) storage reduction histograms
ax = axes[0]
vs_raw, vs_png = col("ng_vs_raw"), col("ng_vs_png")
ax.hist(vs_raw, bins=50, color="#2f5d8a", alpha=0.9,
        label=f"vs. raw  ({vs_raw.mean():.1f}$\\times$ mean)")
ax.hist(vs_png, bins=50, color="#7bbf7b", alpha=0.75,
        label=f"vs. PNG  ({vs_png.mean():.1f}$\\times$ mean)")
ax.set_xlabel("Compression ratio ($\\times$)")
ax.set_ylabel("Images")
ax.set_title("(a) Storage reduction")
ax.legend(fontsize=9)

# (b) per-image self-IoU margin vs JPEG
ax = axes[1]
d = col("ng_iou") - col("jpeg_iou")
b = col("ng_bytes")
win = d > 0
ax.scatter(b[win], d[win], s=6, color="#7bbf7b", alpha=0.7,
           label=f"Nanograph wins ({win.sum()})")
ax.scatter(b[~win], d[~win], s=6, color="#c26d6d", alpha=0.7,
           label=f"JPEG wins ({(~win).sum()})")
ax.axhline(0, color="k", lw=1.0)
ax.set_xlabel("Payload (bytes)")
ax.set_ylabel("$\\Delta$ self-IoU")
ax.set_title("(b) Per-image self-IoU margin vs. JPEG")
ax.legend(fontsize=9)

# (c) win rates: self-referenced (green), GT-referenced (blue), full SSIM (grey)
ax = axes[2]

def win_rate(a, c):
    A, B = col(a), col(c)
    return 100.0 * (A > B).sum() / len(A)

SELF, GT, FULL = "#2e7d32", "#1565c0", "#9e9e9e"
bars = [
    ("Self-IoU\nvs JPEG", win_rate("ng_iou", "jpeg_iou"), SELF),
    ("Self-IoU\nvs WebP", win_rate("ng_iou", "webp_iou"), SELF),
    ("Self-IoU\nvs JP2",  win_rate("ng_iou", "jp2_iou"),  SELF),
    ("GT-IoU\nvs JPEG",   win_rate("gt_iou", "jpeg_gt_iou"), GT),
    ("Full SSIM\nvs JPEG", win_rate("ng_ssim", "jpeg_ssim"), FULL),
]
x = np.arange(len(bars))
ax.bar(x, [v for _, v, _ in bars], color=[c for _, _, c in bars], width=0.68)
ax.axhline(50, ls="--", lw=1.0, color="0.35", zorder=0)
ax.annotate("parity", xy=(len(bars) - 0.4, 50), xytext=(0, 4),
            textcoords="offset points", ha="right", fontsize=8, color="0.35")
for xi, (_, v, _) in zip(x, bars):
    ax.annotate(f"{v:.0f}%", xy=(xi, v), xytext=(0, 3),
                textcoords="offset points", ha="center", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels([lbl for lbl, _, _ in bars], fontsize=8)
ax.set_ylabel("Images where Nanograph wins (%)")
ax.set_ylim(0, 100)
ax.set_title("(c) Win-rate at matched bytes")

for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
fig.savefig(OUT, dpi=300)
print("wrote", OUT)
for lbl, v, _ in bars:
    print(f"  {lbl.replace(chr(10), ' '):20s} {v:5.1f}%")
