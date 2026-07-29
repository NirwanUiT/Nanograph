"""Paper-style figure: the complete Nanograph pipeline on ONE image.

Renders each stage of nanograph_encode as an annotated panel, in order, with a
short caption explaining what happens — like a methods figure in a paper.

Stages:
  1. Input microscopy image
  2. Preprocessing (background subtraction + CLAHE)
  3. Segmentation (learned clDice U-Net mask)
  4. Skeletonization + distance transform (medial axis, endpoints/junctions)
  5. Graph construction (nodes + edges = topological representation)
  6. Point sampling (PSF parameters: position, width, orientation)
  7. Reconstruction (decoded from the compact code)
  8. Fidelity / compression summary (residual + numbers)
"""
import os, sys
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nanograph_v4 import nanograph_encode, NanographConfig
from nanograph_v4.detect import detect_image_type, detect_polarity
from nanograph_v4.preprocess import preprocess

# Default = NMI organelles; override ORG/SEG dirs (and IMG_ID) for other datasets.
ORG = os.environ.get("ORG_DIR", "/mnt/nas1/nba055-2/idea_1/nmi_data/org")
SEG = os.environ.get("SEG_DIR", "/mnt/nas1/nba055-2/idea_1/nmi_data/seg")
IMG_ID = os.environ.get("IMG_ID", "7378")
_TAG = os.environ.get("TAG", "")
OUT = os.path.join(os.path.dirname(__file__),
                   f"pipeline_diagram{('_' + _TAG) if _TAG else ''}.png")


def iou(pred, gt):
    p, g = pred > 0, gt > 0
    u = np.logical_or(p, g).sum()
    return float(np.logical_and(p, g).sum()) / u if u else 0.0


def main():
    img = cv2.imread(os.path.join(ORG, IMG_ID + ".png"), cv2.IMREAD_GRAYSCALE)
    gt = cv2.imread(os.path.join(SEG, IMG_ID + ".png"), cv2.IMREAD_GRAYSCALE)
    assert img is not None, f"missing {IMG_ID}"

    cfg = NanographConfig()
    r = nanograph_encode(img, sam_model=None, verbose=False,
                         optimize=(os.environ.get("OPTIMIZE", "1") == "1"), config=cfg)

    # --- reproduce the preprocessing intermediate for display ---
    raw = img.copy()
    if detect_polarity(raw, cfg=cfg):
        raw = 255 - raw
    det = detect_image_type(raw, cfg=cfg)
    cfg_t = cfg.for_image_type(det["type"])
    img_enh, img_bgsub, reflect_img, bg_model = preprocess(
        raw, bg_kernel_size=det["bg_kernel_size"], cfg=cfg_t)

    mask = (r.mask > 0).astype(np.uint8)
    skel = (r.skeleton > 0).astype(np.uint8)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    recon = (np.clip(r.reconstruction, 0, 1) * 255).astype(np.uint8)
    g = r.graph

    seg_iou = iou(mask, gt) if gt is not None else float("nan")

    # ---------- figure ----------
    fig, axes = plt.subplots(2, 4, figsize=(18, 11))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.06,
                        wspace=0.10, hspace=0.42)
    axes = axes.ravel()
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])

    def cap(ax, title, text):
        ax.set_title(title, fontsize=12, fontweight="bold", pad=6)
        ax.text(0.5, -0.03, text, transform=ax.transAxes, ha="center",
                va="top", fontsize=8.5, wrap=True, color="#222")

    # 1 — input
    axes[0].imshow(raw, cmap="gray")
    cap(axes[0], "1. Input image",
        f"{det['type']} microscopy, {raw.shape[0]}×{raw.shape[1]} px\n"
        f"FG={det['fg_pct']:.1f}%  raw={raw.size:,} bytes")

    # 2 — preprocessing
    axes[1].imshow(img_bgsub, cmap="gray")
    cap(axes[1], "2. Preprocessing",
        "Rolling-ball background subtraction\n+ CLAHE contrast enhancement")

    # 3 — segmentation (overlay)
    ov = np.dstack([raw, raw, raw]).astype(np.float32)
    ov[mask > 0] = 0.55 * ov[mask > 0] + 0.45 * np.array([40, 220, 90])
    axes[2].imshow(ov.astype(np.uint8))
    cap(axes[2], "3. Segmentation",
        f"Learned clDice U-Net ({r.segmenter})\nmask IoU vs GT = {seg_iou:.3f}")

    # 4 — skeleton + distance transform + EP/JN
    d_disp = (dist / (dist.max() + 1e-9))
    axes[3].imshow(d_disp, cmap="magma")
    ys, xs = np.where(skel > 0)
    axes[3].scatter(xs, ys, s=0.4, c="cyan", marker=".", linewidths=0)
    if g is not None and g.nodes:
        ep = np.array([n.position for n in g.nodes if n.node_type == "endpoint"])
        jn = np.array([n.position for n in g.nodes if n.node_type == "junction"])
        if len(ep):
            axes[3].scatter(ep[:, 1], ep[:, 0], s=22, c="lime",
                            edgecolors="k", linewidths=0.4, label="endpoint")
        if len(jn):
            axes[3].scatter(jn[:, 1], jn[:, 0], s=26, c="red", marker="^",
                            edgecolors="k", linewidths=0.4, label="junction")
    cap(axes[3], "4. Skeleton + distance",
        "Medial axis (cyan) over distance transform\n"
        "green=endpoints  red=junctions")

    # 5 — graph
    axes[4].imshow(np.zeros_like(raw), cmap="gray", vmin=0, vmax=1)
    if g is not None:
        for e in g.edges:
            pp = e.path_pixels
            if pp is not None and len(pp) > 1:
                axes[4].plot(pp[:, 1], pp[:, 0], "-", color="#4da6ff", lw=1.0)
            else:
                a = g.nodes[e.source].position; b = g.nodes[e.target].position
                axes[4].plot([a[1], b[1]], [a[0], b[0]], "-",
                             color="#4da6ff", lw=1.0)
        pos = np.array([n.position for n in g.nodes])
        axes[4].scatter(pos[:, 1], pos[:, 0], s=6, c="orange", linewidths=0)
    gs = g.summary() if g is not None else {"n_nodes": 0, "n_edges": 0, "n_components": 0}
    cap(axes[4], "5. Graph construction",
        f"{gs['n_nodes']} nodes, {gs['n_edges']} edges, "
        f"{gs['n_components']} components\ntopological structure of the specimen")

    # 6 — point sampling (width + orientation)
    axes[5].imshow(raw, cmap="gray", alpha=0.35)
    pts = r.points; w = r.widths; ori = r.orientations
    if len(pts):
        sc = axes[5].scatter(pts[:, 1], pts[:, 0], c=w, cmap="viridis",
                             s=10, linewidths=0)
        # orientation ticks (subsample for clarity)
        step = max(1, len(pts) // 120)
        L = 3.0
        for i in range(0, len(pts), step):
            dy = L * np.sin(ori[i]); dx = L * np.cos(ori[i])
            axes[5].plot([pts[i, 1] - dx, pts[i, 1] + dx],
                         [pts[i, 0] - dy, pts[i, 0] + dy], "-",
                         color="red", lw=0.6, alpha=0.8)
    cap(axes[5], "6. Point sampling (PSF)",
        f"{r.n_points} points — position, width (color),\n"
        "orientation (red ticks) = the stored parameters")

    # 7 — reconstruction
    axes[6].imshow(recon, cmap="gray")
    cap(axes[6], "7. Reconstruction",
        f"Oriented-PSF render from the code\n"
        f"PSNR={r.psnr_full:.1f} dB  SSIM={r.ssim_full:.3f}")

    # 8 — residual + compression summary
    resid = np.abs(raw.astype(np.float32) / 255 - np.clip(r.reconstruction, 0, 1))
    axes[7].imshow(resid, cmap="inferno", vmin=0, vmax=0.5)
    _, png = cv2.imencode(".png", raw, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    cap(axes[7], "8. Fidelity & compression",
        f"|orig − recon| residual\n"
        f"{r.compressed_bytes:,} B  ·  {r.compression_ratio:.0f}× vs raw  ·  "
        f"{len(png)/max(r.compressed_bytes,1):.1f}× vs PNG\n"
        f"FG-PSNR={r.psnr_fg:.1f} dB  FG-SSIM={r.ssim_fg:.3f}")

    fig.suptitle(
        f"Nanograph pipeline — image {IMG_ID}  "
        f"(raw {raw.size:,} B  →  {r.compressed_bytes:,} B, "
        f"{r.compression_ratio:.0f}× compression)",
        fontsize=15, fontweight="bold", y=0.965)

    # draw arrows between panels (left->right, wrapping)
    fig.canvas.draw()
    for i in range(7):
        a0 = axes[i].get_position(); a1 = axes[i + 1].get_position()
        if i == 3:  # wrap from top-right to bottom-left
            continue
        x0 = a0.x1; y0 = (a0.y0 + a0.y1) / 2
        x1 = a1.x0; y1 = (a1.y0 + a1.y1) / 2
        fig.add_artist(FancyArrowPatch((x0 + 0.002, y0), (x1 - 0.002, y1),
                       transform=fig.transFigure, arrowstyle="-|>",
                       mutation_scale=14, color="#444", lw=1.2))

    plt.savefig(OUT, dpi=120, bbox_inches="tight")
    print("saved:", OUT)
    print(f"{IMG_ID}: seg={r.segmenter} segIoU={seg_iou:.3f} "
          f"pts={r.n_points} bytes={r.compressed_bytes} "
          f"ratio={r.compression_ratio:.0f}x PSNR={r.psnr_full:.1f}")


if __name__ == "__main__":
    main()
