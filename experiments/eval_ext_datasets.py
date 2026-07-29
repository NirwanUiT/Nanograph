"""Cross-domain evaluation of the Nanograph pipeline on curvilinear datasets.

For each dataset (stare, drive, epfl_mito, microtubules) run the FULL pipeline
under two configs:
  CLASSICAL : use_learned=False  (Otsu/Frangi/Meijering cascade)
  DEFAULT   : out-of-box config  (learned U-Net candidate + gate)

Report, per image and averaged: segmentation IoU/precision/recall vs GT,
reconstruction PSNR/SSIM (full + FG), compression ratio, points, bytes.

All datasets carry ground-truth masks, so segmentation IoU is an honest,
leakage-free measure of cross-domain generalisation (none of these images were
in the U-Net training set, which was fluorescent organelles only).
"""
import os, sys, glob, csv
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nanograph_v4 import nanograph_encode, NanographConfig

P = "/mnt/nas1/nba055-2/idea_1/ext_datasets/prepared"
OUT = os.path.join(os.path.dirname(__file__), "ext_eval")
os.makedirs(OUT, exist_ok=True)
N = int(os.environ.get("N", "10"))
OPTIMIZE = os.environ.get("OPTIMIZE", "0") == "1"  # point-optimizer off = fast; seg IoU unaffected
DATASETS = os.environ.get("DATASETS", "stare,drive,epfl_mito,microtubules").split(",")


def seg_metrics(pred, gt):
    p, g = pred > 0, gt > 0
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    iou = inter / union if union else 0.0
    prec = inter / p.sum() if p.sum() else 0.0
    rec = inter / g.sum() if g.sum() else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return iou, prec, rec, f1


def run(ds, cfg_factory):
    imgs = sorted(glob.glob(f"{P}/{ds}/images/*.png"))[:N]
    rows = []
    for ip in imgs:
        idd = os.path.splitext(os.path.basename(ip))[0]
        im = cv2.imread(ip, 0)
        gt = cv2.imread(f"{P}/{ds}/labels/{idd}.png", 0)
        if im is None or gt is None:
            continue
        r = nanograph_encode(im, sam_model=None, verbose=False,
                              optimize=OPTIMIZE, config=cfg_factory())
        m = (r.mask > 0).astype(np.uint8)
        iou, prec, rec, f1 = seg_metrics(m, gt)
        rows.append(dict(id=idd, seg=r.segmenter, iou=iou, prec=prec, rec=rec,
                         f1=f1, psnr=r.psnr_full, ssim=r.ssim_full,
                         fg_psnr=r.psnr_fg, ratio=r.compression_ratio,
                         pts=r.n_points, bytes=r.compressed_bytes))
        print(f"    {ds}/{idd} seg={r.segmenter} IoU={iou:.3f}", flush=True)
    return rows


def agg(rows, key):
    v = [r[key] for r in rows]
    return float(np.mean(v)) if v else 0.0


def main():
    def classical():
        c = NanographConfig(); c.segment.use_learned = False; return c
    def default():
        return NanographConfig()

    summary = []
    for ds in DATASETS:
        for cname, cfac in [("CLASSICAL", classical), ("DEFAULT", default)]:
            rows = run(ds, cfac)
            with open(f"{OUT}/{ds}_{cname}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader(); w.writerows(rows)
            rec = dict(dataset=ds, config=cname, n=len(rows),
                       segIoU=agg(rows, "iou"), P=agg(rows, "prec"),
                       R=agg(rows, "rec"), F1=agg(rows, "f1"),
                       PSNR=agg(rows, "psnr"), SSIM=agg(rows, "ssim"),
                       FGPSNR=agg(rows, "fg_psnr"), ratio=agg(rows, "ratio"),
                       pts=agg(rows, "pts"))
            summary.append(rec)
            print(f"{ds:13s} {cname:9s} n={rec['n']:2d} "
                  f"segIoU={rec['segIoU']:.3f} P={rec['P']:.3f} R={rec['R']:.3f} "
                  f"F1={rec['F1']:.3f} PSNR={rec['PSNR']:.1f} "
                  f"FGPSNR={rec['FGPSNR']:.1f} ratio={rec['ratio']:.0f}x "
                  f"pts={rec['pts']:.0f}")

    with open(f"{OUT}/summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader(); w.writerows(summary)
    print("\nsaved ->", OUT)


if __name__ == "__main__":
    main()
