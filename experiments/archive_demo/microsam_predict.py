#!/usr/bin/env python3
"""
micro-SAM (Archit et al., Nat Methods 2025) automatic instance segmentation
(AIS) over a folder of PNGs; the union of the instances is the mask, written
as <out>/<stem>.png (0/255), optionally restricted to <cell-dir>/<stem>.png.
Seconds per image are appended to --timing.

  zero-shot:  --model vit_b_lm                 (light-microscopy generalist)
  fine-tuned: --model vit_b --checkpoint PATH  (from microsam_train.py)

Runs in the micro-SAM environment:
    MICROSAM_CACHEDIR=/mnt/nas1/nba055-2/.microsam_cache \
    /home/nba055/micromamba/envs/microsam/bin/python experiments/archive_demo/microsam_predict.py ...
"""
import argparse
import contextlib
import glob
import io
import os
import time

import numpy as np


def run(args):
    paths, a = args
    import cv2
    import torch
    from micro_sam.automatic_segmentation import automatic_instance_segmentation, get_predictor_and_segmenter
    torch.set_num_threads(a.threads)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        predictor, segmenter = get_predictor_and_segmenter(model_type=a.model, checkpoint=a.checkpoint,
                                                           device=a.device, segmentation_mode='ais')
    out = []
    for p in paths:
        stem = os.path.basename(p)[:-4]
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        t0 = time.perf_counter()
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                seg = automatic_instance_segmentation(predictor=predictor, segmenter=segmenter, input_path=img,
                                                      ndim=2, verbose=False)
            m = np.asarray(seg) > 0
            err = ''
        except Exception as ex:
            m, err = np.zeros(img.shape, bool), f'{type(ex).__name__}: {ex}'
        dt = time.perf_counter() - t0
        if a.cell_dir:
            m &= cv2.imread(os.path.join(a.cell_dir, f'{stem}.png'), cv2.IMREAD_GRAYSCALE) > 0
        cv2.imwrite(os.path.join(a.out, f'{stem}.png'), m.astype(np.uint8) * 255)
        out.append((stem, dt, err))
    return out


def main():
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument('--img-dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--cell-dir', default=None)
    ap.add_argument('--model', default='vit_b_lm')
    ap.add_argument('--checkpoint', default=None)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--timing', default=None)
    ap.add_argument('--label', default='X')
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(a.img_dir, '*.png')))
    if a.limit:
        paths = paths[:a.limit]
    chunks = [paths[i::a.workers] for i in range(a.workers)]
    with mp.get_context('spawn').Pool(a.workers) as pool:
        res = [r for rs in pool.map(run, [(c, a) for c in chunks if c]) for r in rs]
    if a.timing:
        new = not os.path.exists(a.timing)
        with open(a.timing, 'a') as f:
            if new:
                f.write('dataset,id,seconds\n')
            for s, dt, _ in res:
                f.write(f'{a.label},{s},{dt}\n')
    errs = [f'{s}: {e}' for s, _, e in res if e]
    print(f'{len(res) - len(errs)}/{len(res)} images; errors {len(errs)}; median s/image '
          f'{np.median([d for _, d, _ in res]):.2f}')
    for e in errs[:5]:
        print(' ', e)


if __name__ == '__main__':
    main()
