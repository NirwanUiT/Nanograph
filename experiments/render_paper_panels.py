"""Regenerate the three qualitative paper panels from the current code:

  fig_qualitative_organelle.png  3x3 montage of one organelle frame through the
                                 pipeline (input / GT / recon-mask, reconstruction /
                                 error / seg-vs-GT, skeleton / points / graph).
  fig_ood_failure.png            the 8-panel pipeline diagram on an out-of-distribution
                                 EM slice (delegates to render_pipeline_diagram.py).
  fig_mito_generalisation.png    3 rows (temporal clip / STED / MITO MIP tile) x
                                 5 cols (input / mask / skeleton / reconstruction / graph).

The MITO row uses the MEDIAN tile by annotated foreground fraction
(M2_019_y256x256, fg=0.073) instead of the near-empty tile shown before.

Usage:
    python experiments/render_paper_panels.py --out paper/figures
"""
import argparse
import os
import subprocess
import sys

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nanograph_v4 import nanograph_encode, NanographConfig

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = "/mnt/nas1/nba055-2/idea_1"

ORG_DIR = os.path.join(DATA, "nmi_data/org")
SEG_DIR = os.path.join(DATA, "nmi_data/seg")

# median tile by annotated foreground fraction (see --pick-mito-tile)
MITO_MIP_TILE = "M2_019_y256x256"

MITO_ROWS = [
    ("Temporal clip", os.path.join(DATA, "mito_aaron/images"),
     os.path.join(DATA, "mito_aaron/masks"), "d1_d0001_p0004_t0000"),
    ("STED TOM20", os.path.join(DATA, "public_mito/sted_prep/images"), None, None),
    ("MITO (MIP tile)", os.path.join(DATA, "public_mito/mito_mip_tiles/images"),
     os.path.join(DATA, "public_mito/mito_mip_tiles/masks"), MITO_MIP_TILE),
]


def _iou_pr(pred, gt):
    p, g = pred > 0, gt > 0
    tp = int(np.logical_and(p, g).sum())
    fp = int(np.logical_and(p, ~g).sum())
    fn = int(np.logical_and(~p, g).sum())
    u = int(np.logical_or(p, g).sum())
    iou = tp / u if u else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return iou, prec, rec, f1


def _encode(img):
    cfg = NanographConfig()
    r = nanograph_encode(img, sam_model=None, verbose=False, optimize=True, config=cfg)
    return r


def _first_existing(d, stem=None):
    if stem is not None:
        for ext in (".png", ".tif", ".tiff", ".jpg"):
            p = os.path.join(d, stem + ext)
            if os.path.exists(p):
                return p
    files = sorted(f for f in os.listdir(d) if f.lower().endswith((".png", ".tif", ".tiff", ".jpg")))
    return os.path.join(d, files[0]) if files else None


def _seg_overlay(raw, mask, gt):
    """green=TP, red=FN, blue=FP over the grayscale input."""
    ov = np.dstack([raw, raw, raw]).astype(np.float32)
    p, g = mask > 0, (gt > 0 if gt is not None else np.zeros_like(mask, bool))
    tp = p & g
    fn = (~p) & g
    fp = p & (~g)
    ov[tp] = 0.4 * ov[tp] + 0.6 * np.array([40, 220, 90])
    ov[fn] = 0.4 * ov[fn] + 0.6 * np.array([230, 60, 60])
    ov[fp] = 0.4 * ov[fp] + 0.6 * np.array([60, 120, 240])
    return ov.astype(np.uint8)


def qualitative_organelle(out, img_id):
    ip = os.path.join(ORG_DIR, img_id + ".png")
    mp = os.path.join(SEG_DIR, img_id + ".png")
    raw = cv2.imread(ip, cv2.IMREAD_GRAYSCALE)
    gt = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
    r = _encode(raw)
    mask = (r.mask > 0).astype(np.uint8)
    skel = (r.skeleton > 0).astype(np.uint8)
    recon = (np.clip(r.reconstruction, 0, 1) * 255).astype(np.uint8)
    err = np.abs(raw.astype(np.float32) / 255 - np.clip(r.reconstruction, 0, 1))
    iou, prec, rec, f1 = _iou_pr(mask, gt)
    g = r.graph
    gs = g.summary() if g is not None else {"n_nodes": 0, "n_edges": 0, "n_components": 0}

    fig, ax = plt.subplots(3, 3, figsize=(12, 12))
    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])

    ax[0, 0].imshow(raw, cmap="gray"); ax[0, 0].set_title("Input image")
    ax[0, 1].imshow(gt, cmap="gray"); ax[0, 1].set_title("GT mask")
    ax[0, 2].imshow(mask * 255, cmap="gray"); ax[0, 2].set_title("Nanograph mask")

    ax[1, 0].imshow(recon, cmap="gray")
    ax[1, 0].set_title(f"Reconstruction (PSNR={r.psnr_full:.2f} SSIM={r.ssim_full:.4f})")
    ax[1, 1].imshow(err, cmap="inferno", vmin=0, vmax=0.5)
    ax[1, 1].set_title(f"Error map (FG-PSNR={r.psnr_fg:.2f} FG-SSIM={r.ssim_fg:.4f})")
    ax[1, 2].imshow(_seg_overlay(raw, mask, gt))
    ax[1, 2].set_title(f"Seg vs GT (G=TP R=FN B=FP)  P={prec:.3f} R={rec:.3f} F1={f1:.3f}")

    ax[2, 0].imshow(skel * 255, cmap="gray")
    ax[2, 0].set_title(f"Skeleton ({int(skel.sum())} px)")
    ax[2, 1].imshow(raw, cmap="gray", alpha=0.35)
    if len(r.points):
        ax[2, 1].scatter(r.points[:, 1], r.points[:, 0], c=r.widths, cmap="viridis", s=8, linewidths=0)
    ax[2, 1].set_title(f"Nanograph points ({r.n_points})")
    ax[2, 2].imshow(np.zeros_like(raw), cmap="gray", vmin=0, vmax=1)
    if g is not None:
        for e in g.edges:
            pp = e.path_pixels
            if pp is not None and len(pp) > 1:
                ax[2, 2].plot(pp[:, 1], pp[:, 0], "-", color="#4da6ff", lw=0.9)
            else:
                a0 = g.nodes[e.source].position; b0 = g.nodes[e.target].position
                ax[2, 2].plot([a0[1], b0[1]], [a0[0], b0[0]], "-", color="#4da6ff", lw=0.9)
        pos = np.array([n.position for n in g.nodes])
        ax[2, 2].scatter(pos[:, 1], pos[:, 0], s=5, c="orange", linewidths=0)
    ax[2, 2].set_title(f"Graph ({gs['n_nodes']}N, {gs['n_edges']}E, {gs['n_components']}C)")

    fig.suptitle(f"Nanograph on organelle {img_id}  "
                 f"({r.compressed_bytes:,} B, {r.compression_ratio:.0f}x)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}  (organelle {img_id}: segIoU={iou:.3f} pts={r.n_points} "
          f"bytes={r.compressed_bytes})")


def mito_generalisation(out):
    fig, ax = plt.subplots(len(MITO_ROWS), 5, figsize=(18, 3.6 * len(MITO_ROWS)))
    cols = ["Input", "Selected mask", "Skeleton", "Reconstruction", "Graph"]
    for j, c in enumerate(cols):
        ax[0, j].set_title(c, fontsize=12, fontweight="bold")
    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])

    for i, (label, idir, mdir, stem) in enumerate(MITO_ROWS):
        ip = _first_existing(idir, stem)
        raw = cv2.imread(ip, cv2.IMREAD_GRAYSCALE)
        r = _encode(raw)
        mask = (r.mask > 0).astype(np.uint8)
        skel = (r.skeleton > 0).astype(np.uint8)
        recon = (np.clip(r.reconstruction, 0, 1) * 255).astype(np.uint8)
        g = r.graph
        gs = g.summary() if g is not None else {"n_nodes": 0, "n_edges": 0, "n_components": 0}

        ax[i, 0].imshow(raw, cmap="gray")
        ax[i, 0].set_ylabel(label, fontsize=12, fontweight="bold")
        ax[i, 0].set_yticks([])
        ax[i, 1].imshow(mask * 255, cmap="gray")
        ax[i, 2].imshow(skel * 255, cmap="gray")
        ax[i, 3].imshow(recon, cmap="gray")
        ax[i, 3].set_xlabel(f"PSNR={r.psnr_full:.1f} dB", fontsize=9)
        ax[i, 4].imshow(np.zeros_like(raw), cmap="gray", vmin=0, vmax=1)
        if g is not None:
            for e in g.edges:
                pp = e.path_pixels
                if pp is not None and len(pp) > 1:
                    ax[i, 4].plot(pp[:, 1], pp[:, 0], "-", color="#4da6ff", lw=0.7)
            pos = np.array([n.position for n in g.nodes]) if g.nodes else np.zeros((0, 2))
            if len(pos):
                ax[i, 4].scatter(pos[:, 1], pos[:, 0], s=4, c="orange", linewidths=0)
        ax[i, 4].set_xlabel(f"{gs['n_nodes']}N/{gs['n_edges']}E, "
                            f"{r.compressed_bytes:,} B, FG-PSNR={r.psnr_fg:.1f} dB", fontsize=9)
        print(f"  {label}: {os.path.basename(ip)} segmenter={r.segmenter} "
              f"nodes={gs['n_nodes']} bytes={r.compressed_bytes}")

    fig.suptitle("Nanograph generalisation to unseen mitochondria modalities",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def ood_failure(out, em_slice):
    """8-panel pipeline diagram on an OOD EM slice (reuses render_pipeline_diagram.py)."""
    env = dict(os.environ)
    env["ORG_DIR"] = os.path.join(DATA, "ext_datasets/prepared/epfl_mito/images")
    env["SEG_DIR"] = os.path.join(DATA, "ext_datasets/prepared/epfl_mito/labels")
    env["IMG_ID"] = em_slice
    env["OUT_PATH"] = out
    subprocess.run([sys.executable, os.path.join(HERE, "render_pipeline_diagram.py")],
                   env=env, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper/figures")
    ap.add_argument("--organelle-id", default="7378")
    ap.add_argument("--em-slice", default="slice036")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    qualitative_organelle(os.path.join(args.out, "fig_qualitative_organelle.png"), args.organelle_id)
    mito_generalisation(os.path.join(args.out, "fig_mito_generalisation.png"))
    ood_failure(os.path.join(args.out, "fig_ood_failure.png"), args.em_slice)


if __name__ == "__main__":
    main()
