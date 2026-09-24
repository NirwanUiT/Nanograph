#!/usr/bin/env python3
"""
Regenerate the simulated organelle data WITH its ground-truth geometry (T13).

The 726 organelle images are 2x2 mosaics of 128x128 patches from the
physics-based simulator of Sekh et al. (2021)
(/mnt/nas1/nba055-2/skarifahmed-physeg-3875361, "2. Mitochondria Simulation
Airy"): per patch 1-2 unbranched tubes (quadratic spline centreline through 3-4
random control points, width 200-600 nm, z range per batch config), emitters
on the tube surface, Gibson-Lanni PSF at 42 nm pixels, the mask = emitter xy
positions on the pixel grid ("physics_gt"); mosaics get Poisson noise with
signal range [nlow, nhigh] (data_generator.py + add_poisson_noise_image.m).

This module calls the simulator's own geometry (helper_generator_v5) and PSF
(microscPSFmod) code, re-implements the two steps that need MATLAB or file
round-trips (brightness trace, Poisson noise) exactly as written there, and
additionally SAVES the geometry the original discarded: per tube its 2-D
centreline in pixel coordinates and its diameter in pixels.

Usage (from repo root):
    python experiments/sim_mito.py --n 300 --workers 12 --out /mnt/nas1/nba055-2/idea_1/sim_truth
Writes images/<i>.png (uint8), masks/<i>.png (simulator mask), truth/<i>.json.
"""
import argparse
import contextlib
import glob
import io
import json
import os
import sys

import numpy as np

# The organelle set matches the AIRYSCAN variant: its structures stay within
# ~29 px of each quadrant centre (Airy max_xy 1344 nm = 32 px; Epi: 2000 nm) and
# its median diameter is 235 nm (Airy widths 20-300 nm). The Airy variant differs
# from Epi only in its PSF module (63x, squared PSF) and batch configs.
SIM = '/mnt/nas1/nba055-2/skarifahmed-physeg-3875361/src/2. Mitochondria Simulation Airy/simulator'
PX_NM = 42.0          # pixel size (step_size_xy = 0.042 um)
SIZE = 128            # patch size
OFFSET_NM = 2688.0    # save_physics_gt: + 2 * 1344 nm


def batch_configs():
    import pandas as pd
    files = sorted(glob.glob(os.path.join(SIM, 'batch', 'b*.csv')))
    keep = os.environ.get('SIM_BATCHES', 'b1,b4')     # calibrated default: in-focus batches
    if keep:
        files = [f for f in files if os.path.splitext(os.path.basename(f))[0] in keep.split(',')]
    return [pd.read_csv(p).iloc[0].to_dict() for p in files]


def brightness_trace(t_off, t_on, rate, frames):
    """generator_batch_parallel.brightness_trace, verbatim logic."""
    T = np.zeros((2, frames))
    T[0, :] = np.random.exponential(scale=t_off, size=(1, frames))
    T[1, :] = np.random.exponential(scale=t_on, size=(1, frames))
    B = np.zeros((2, frames))
    B[1, :] = rate * T[1, :]
    T_cum = np.cumsum(T.ravel(order='F'))
    B_cum = np.cumsum(B.ravel(order='F'))
    start = np.random.randint(0, 10)
    return np.diff(np.interp(np.arange(start, start + frames + 1), T_cum, B_cum))


# Calibration knobs (the organelle set's exact generation settings were not
# kept); defaults are the values calibrated against it, see --calibrate.
# Calibrated (T13) against the organelle set: foreground 0.023 vs 0.023,
# median skeleton diameter 5.6 vs 5.6 px, 95th-pct extent from the quadrant
# centre 28 vs 27 px, single-component quadrants 0.81 vs 0.85, shipped-style
# U-Net IoU 0.85 vs 0.88 (held-out originals). In-focus batches only (b1: z
# 600-800 nm, b4: 800-1000 nm); defocused batches drop that IoU to 0.51-0.71.
CAL = {'max_xy': float(os.environ.get('SIM_MAX_XY', 1100)),
       'p_one': float(os.environ.get('SIM_P_ONE', 0.5)),
       'nlow': tuple(int(v) for v in os.environ.get('SIM_NLOW', '40,70').split(',')),
       'nhigh': tuple(int(v) for v in os.environ.get('SIM_NHIGH', '200,240').split(','))}


def make_patch(seed, cfgs):
    """One 128x128 patch: (uint16 image scaled to max 255, bool mask, tubes)."""
    sys.path.insert(0, SIM)
    import helper_generator_v5 as H
    import microscPSFmod as msPSF
    np.random.seed(seed)
    cfg = cfgs[np.random.randint(len(cfgs))]
    cp = np.random.randint(int(cfg['control_points_low']), int(cfg['control_points_up']) + 1)
    n_mito = 1 if np.random.random() < CAL['p_one'] else int(cfg['no_of_mito'])
    tubes, emitters = [], []
    for k in range(n_mito):
        width = np.random.randint(int(cfg['wlow']), int(cfg['whigh']))
        with contextlib.redirect_stdout(io.StringIO()):
            x, y, z, _, dist, _, _ = H.get_mitochondria_2D_points(
                int(cfg['zhigh']), int(cfg['zlow']), int(CAL['max_xy']), int(cfg['max_length']), cp)
            data, _ = H.get_mitochondria_3D_points(x, y, z, dist, width, int(cfg['density']),
                                                   cfg['emmiters_percentage'])
        emitters.append(np.array(data, float).reshape(-1, 3) * 0.001)      # um
        tubes.append({'row': ((np.asarray(y) + OFFSET_NM) / PX_NM).tolist(),
                      'col': ((np.asarray(x) + OFFSET_NM) / PX_NM).tolist(),
                      'diameter_px': width / PX_NM, 'width_nm': int(width)})
    P = np.concatenate(emitters)
    br = np.array([brightness_trace(0, 1, 10, 1)[0] for _ in range(len(P))])
    img = np.zeros(SIZE * SIZE)
    with contextlib.redirect_stdout(io.StringIO()):
        for s in range(0, len(P), 1000):
            c = P[s:s + 1000]
            psf = msPSF.gLXYZParticleScan(msPSF.m_params, PX_NM / 1000, SIZE, c[:, 2], zv=-1,
                                          wvl=0.510, normalize=False, px=c[:, 0], py=c[:, 1])
            img += (psf * br[s:s + 1000, None]).sum(0)
    img = img.reshape(SIZE, SIZE)
    d = (img / img.max() * 255).astype(np.uint16)
    # physics_gt: emitter xy on a 1-nm grid, max-pooled to 42-nm pixels
    rc = np.floor((P[:, :2] * 1000 + OFFSET_NM) / PX_NM).astype(int)
    ok = (rc >= 0).all(1) & (rc < SIZE).all(1)
    mask = np.zeros((SIZE, SIZE), bool)
    mask[rc[ok, 1], rc[ok, 0]] = True
    return d, mask, tubes


def make_image(job):
    i, out = job
    import cv2
    cfgs = batch_configs()
    if os.path.exists(os.path.join(out, 'truth', f'{i}.json')):
        return i
    rng = np.random.default_rng(10_000 + i)
    mosaic = np.zeros((256, 256), np.float64)
    mask = np.zeros((256, 256), bool)
    tubes = []
    for r in range(2):
        for c in range(2):
            d, m, t = make_patch(int(rng.integers(1 << 30)), cfgs)
            # data_generator.py: img.paste(im, (r*128, c*128)) -> column offset r*128, row offset c*128
            mosaic[c * 128:(c + 1) * 128, r * 128:(r + 1) * 128] = d
            mask[c * 128:(c + 1) * 128, r * 128:(r + 1) * 128] = m
            for tb in t:
                tb['row'] = [v + c * 128 for v in tb['row']]
                tb['col'] = [v + r * 128 for v in tb['col']]
                tb['quadrant'] = [c, r]
                tubes.append(tb)
    nhigh, nlow = int(rng.integers(*CAL['nhigh'])), int(rng.integers(*CAL['nlow']))
    sig = np.floor((nhigh - nlow) * mosaic / mosaic.max() + nlow)       # uint16(...) in MATLAB
    noisy = np.clip(np.random.default_rng(20_000 + i).poisson(sig), 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(out, 'images', f'{i}.png'), noisy)
    cv2.imwrite(os.path.join(out, 'masks', f'{i}.png'), mask.astype(np.uint8) * 255)
    with open(os.path.join(out, 'truth', f'{i}.json'), 'w') as f:
        json.dump({'tubes': tubes, 'nhigh': nhigh, 'nlow': nlow, 'calibration': CAL}, f)
    return i


def main():
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=300)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--out', default='/mnt/nas1/nba055-2/idea_1/sim_truth')
    a = ap.parse_args()
    for d in ('images', 'masks', 'truth'):
        os.makedirs(os.path.join(a.out, d), exist_ok=True)
    with mp.get_context('spawn').Pool(a.workers) as pool:
        for k, i in enumerate(pool.imap_unordered(make_image, [(i, a.out) for i in range(a.n)])):
            if k % 25 == 0:
                print(f'[{k + 1}/{a.n}]', flush=True)


if __name__ == '__main__':
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    main()
