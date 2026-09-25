"""
Regenerates panel (c) of figures/fig_codec_rd.png with the GT-IoU bar that the
revised caption describes.

Requires numpy + matplotlib. Both were unavailable in the session that produced
the revision (the package index returned 502 at the proxy throughout), so this
panel is the one item the revised manuscript describes but does not yet ship.
Run from the Nanograph repo root; nothing else in the revision depends on it.
"""
import csv
import numpy as np
import matplotlib.pyplot as plt

rows = list(csv.DictReader(open("dataset_results/metrics.csv")))

def win_rate(a, b):
    A = [float(r[a]) for r in rows]
    B = [float(r[b]) for r in rows]
    return 100.0 * sum(1 for x, y in zip(A, B) if x > y) / len(A)

SELF, GT, FULL = "#2e7d32", "#1565c0", "#9e9e9e"
bars = [
    ("Recon-IoU\nvs JPEG", win_rate("ng_iou", "jpeg_iou"), SELF),
    ("Recon-IoU\nvs WebP", win_rate("ng_iou", "webp_iou"), SELF),
    ("Recon-IoU\nvs JP2",  win_rate("ng_iou", "jp2_iou"),  SELF),
    ("GT-IoU\nvs JPEG",    win_rate("gt_iou", "jpeg_gt_iou"), GT),
    ("Full SSIM\nvs JPEG", win_rate("ng_ssim", "jpeg_ssim"), FULL),
]

fig, ax = plt.subplots(figsize=(6.6, 3.5))
x = np.arange(len(bars))
ax.bar(x, [b[1] for b in bars], color=[b[2] for b in bars], width=0.68)
ax.axhline(50, ls="--", lw=1.0, color="0.35", zorder=0)
ax.annotate("parity", xy=(len(bars) - 0.4, 50), xytext=(0, 4),
            textcoords="offset points", ha="right", fontsize=8, color="0.35")
for xi, (_, v, _) in zip(x, bars):
    ax.annotate(f"{v:.0f}%", xy=(xi, v), xytext=(0, 3),
                textcoords="offset points", ha="center", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels([b[0] for b in bars], fontsize=8)
ax.set_ylabel("images won by Nanograph (%)")
ax.set_ylim(0, 100)
ax.spines[["top", "right"]].set_visible(False)
ax.set_title("Self-referenced (green) vs ground-truth-referenced (blue) win rates",
             fontsize=9)
fig.tight_layout()
fig.savefig("figures/fig_codec_rd_panel_c.png", dpi=300)
print("wrote figures/fig_codec_rd_panel_c.png")
for lbl, v, _ in bars:
    print(f"  {lbl.replace(chr(10), ' '):22s} {v:5.1f}%")
