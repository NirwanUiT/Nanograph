"""Prepare external curvilinear datasets into a uniform layout:
    prepared/<dataset>/images/<id>.png   (grayscale uint8)
    prepared/<dataset>/labels/<id>.png   (binary 0/255 foreground = structure)

Datasets:
  stare        - retinal vessels (.ppm image + .ah.ppm vessel label)   [green ch]
  drive        - retinal vessels (.tif image + .png label)             [green ch]
  epfl_mito    - EM mitochondria (165-slice tif + gt tif)  -> sample slices
  microtubules - IRM microtubule filaments (HF parquet, image + instance masks)
"""
import os, io, glob
import numpy as np
import cv2

ROOT = "/mnt/nas1/nba055-2/idea_1/ext_datasets"
OUT = os.path.join(ROOT, "prepared")


def _save(ds, idd, img, lab):
    os.makedirs(os.path.join(OUT, ds, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUT, ds, "labels"), exist_ok=True)
    cv2.imwrite(os.path.join(OUT, ds, "images", idd + ".png"), img)
    cv2.imwrite(os.path.join(OUT, ds, "labels", idd + ".png"), lab)


def prep_stare():
    imgs = sorted(glob.glob(os.path.join(ROOT, "stare/images/*.ppm")))
    n = 0
    for p in imgs:
        idd = os.path.splitext(os.path.basename(p))[0]
        lp = os.path.join(ROOT, "stare/labels", idd + ".ah.ppm")
        if not os.path.isfile(lp):
            continue
        im = cv2.imread(p, cv2.IMREAD_COLOR)          # BGR
        g = im[:, :, 1]                                # green channel
        lab = cv2.imread(lp, cv2.IMREAD_GRAYSCALE)
        lab = ((lab > 0).astype(np.uint8)) * 255
        _save("stare", idd, g, lab)
        n += 1
    print("stare:", n)


def prep_drive():
    imgs = sorted(glob.glob(os.path.join(ROOT, "drive/images/*.tif")))
    n = 0
    for p in imgs:
        idd = os.path.splitext(os.path.basename(p))[0]
        lp = os.path.join(ROOT, "drive/labels", idd + ".png")
        if not os.path.isfile(lp):
            continue
        im = cv2.imread(p, cv2.IMREAD_COLOR)
        g = im[:, :, 1]
        lab = cv2.imread(lp, cv2.IMREAD_GRAYSCALE)
        lab = ((lab > 127).astype(np.uint8)) * 255
        _save("drive", idd, g, lab)
        n += 1
    print("drive:", n)


def prep_epfl(n_slices=10):
    import tifffile as tf
    vol = tf.imread(os.path.join(ROOT, "epfl_mito/training.tif"))
    gt = tf.imread(os.path.join(ROOT, "epfl_mito/training_groundtruth.tif"))
    idxs = np.linspace(0, vol.shape[0] - 1, n_slices).astype(int)
    for k in idxs:
        img = vol[k]
        lab = ((gt[k] > 0).astype(np.uint8)) * 255
        _save("epfl_mito", f"slice{k:03d}", img, lab)
    print("epfl_mito:", len(idxs))


def prep_microtubules():
    import pyarrow.parquet as pq
    t = pq.read_table(os.path.join(ROOT, "microtubules/train.parquet"))
    rows = t.to_pylist()
    n = 0
    for r in rows:
        idd = str(r["id"])
        img = cv2.imdecode(np.frombuffer(r["image"]["bytes"], np.uint8),
                           cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        lab = np.zeros(img.shape, np.uint8)
        for inst in (r["mask"] or []):
            m = cv2.imdecode(np.frombuffer(inst["bytes"], np.uint8),
                             cv2.IMREAD_GRAYSCALE)
            if m is not None and m.shape == img.shape:
                lab[m > 0] = 255
        _save("microtubules", idd, img, lab)
        n += 1
    print("microtubules:", n)


if __name__ == "__main__":
    prep_stare()
    prep_drive()
    prep_epfl()
    prep_microtubules()
    print("done ->", OUT)
