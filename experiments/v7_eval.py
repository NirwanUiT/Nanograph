#!/usr/bin/env python3
"""
Evaluate the v7 (experimental) structure layer against the paper-v2 arms.

Stages
  encode  (parallel, GPU ok) run the default encoder for the mask and the
          appearance data; build the v7 structure layer from the mask's
          UNPRUNED skeleton at several Douglas-Peucker tolerances (DT widths)
          and once with profile-fitted widths; serialise; cache.
  measure (parallel, CPU) branch tables for REF (annotation), SEG (the
          encoder's own mask), JPEG at the payload budget (same segmenter),
          JPEG at fixed qualities (rate-distortion), GRAPH6 (paper-v2
          payload), PRE7 (encoder-side v7 graph, pixel-path lengths) and
          GRAPH7 (decoded v7 structure layers); descriptors at L = 0/2/5/10
          and junction/endpoint F1 against REF.
  stats   summaries (see results/.../v7/*.csv).
  width   width robustness under mask dilation/erosion (DT vs profile).

Usage (from repo root):
    python experiments/v7_eval.py --stage encode --workers 6
    python experiments/v7_eval.py --stage measure --workers 12
    python experiments/v7_eval.py --stage stats
    python experiments/v7_eval.py --stage width --workers 12
"""
import argparse
import contextlib
import io
import json
import os
import sys
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

IMAGES = '/mnt/nas1/nba055-2/idea_1/nmi_data/org'
MASKS = '/mnt/nas1/nba055-2/idea_1/nmi_data/seg'
V6_CACHE = '/mnt/nas1/nba055-2/idea_1/Nanograph/Nanograph/results/paper/downstream/cache'
EPS = [0.5, 0.75, 1.5, 3.0]          # Douglas-Peucker tolerances (px)
EPS_MAIN = 0.75
JPEG_Q = [5, 10, 20, 30, 50, 75]
LS = [0, 2, 5, 10]
TOLS = [3, 5]                         # junction / endpoint matching tolerance (px)


def _skel_dt(mask):
    import cv2
    from skimage.morphology import skeletonize
    m = (mask > 0).astype(np.uint8)
    return skeletonize(m > 0), cv2.distanceTransform(m, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)


# ---------------------------------------------------------------------------
def _encode_one(job):
    stem, out = job
    import cv2
    from nanograph_v4 import nanograph_encode, NanographConfig
    from nanograph_v4.compress import compress_nanograph
    from nanograph_v4 import graph_branch as gb
    f = os.path.join(out, 'cache', stem + '.npz')
    if os.path.exists(f):
        return stem
    img = cv2.imread(os.path.join(IMAGES, stem + '.png'), cv2.IMREAD_GRAYSCALE)
    with contextlib.redirect_stdout(io.StringIO()):
        r = nanograph_encode(img, verbose=False, config=NanographConfig())
    appearance, _ = compress_nanograph(
        r.points, r.intensities, r.widths, r.shape, r.types, orientations=r.orientations,
        bg_model=r.bg_grid, graph=None,
        fg_residual=r.fg_residual_bytes if r.fg_residual_bytes else None,
        fg_residual_shape=r.fg_residual_shape_info, cfg=r.config)
    skel, dt = _skel_dt(r.mask)
    blobs = {}
    for e in EPS:
        st = gb.branch_structure(skel, dt, eps=e)
        blobs[f'struct_dt_{e}'] = np.frombuffer(gb.encode_structure(st), np.uint8)
        if e == EPS_MAIN:
            pre = st
    stp = gb.branch_structure(skel, dt, eps=EPS_MAIN, width_mode='profile', img=img)
    blobs['struct_profile'] = np.frombuffer(gb.encode_structure(stp), np.uint8)
    pos, rad, edges, elen = gb.structure_to_arrays(pre, path_lengths=True)
    np.savez_compressed(f, mask=np.packbits(r.mask > 0), shape=np.array(r.mask.shape),
                        segmenter=np.array(r.segmenter), v6_bytes=len(r.compressed),
                        appearance=np.frombuffer(appearance, np.uint8), fg_psnr=r.psnr_fg,
                        pre_pos=pos, pre_rad=rad, pre_edges=edges, pre_elen=elen, **blobs)
    return stem


def stage_encode(a):
    import glob
    import multiprocessing as mp
    os.makedirs(os.path.join(a.out, 'cache'), exist_ok=True)
    stems = sorted(os.path.splitext(os.path.basename(p))[0]
                   for p in glob.glob(os.path.join(IMAGES, '*.png')))
    if a.limit:
        stems = stems[:a.limit]
    with mp.get_context('spawn').Pool(a.workers) as pool:
        for i, s in enumerate(pool.imap_unordered(_encode_one, [(s, a.out) for s in stems])):
            if i % 50 == 0:
                print(f'[encode {i + 1}/{len(stems)}] {s}', flush=True)


# ---------------------------------------------------------------------------
def _mask_of(c):
    return np.unpackbits(c['mask'])[:int(np.prod(c['shape']))].reshape(c['shape']).astype(np.uint8)


def _measure_one(job):
    stem, out = job
    import cv2
    import torch
    import downstream_morphometry as dm
    from nanograph_v4 import graph_branch as gb
    from nanograph_v4.config import DEFAULT_CONFIG
    from nanograph_v4.evaluate import jpeg_for_budget
    from nanograph_v4.unet_seg import load_unet
    torch.set_num_threads(1)
    global _MODEL
    if '_MODEL' not in globals():
        _MODEL = load_unet(DEFAULT_CONFIG.segment.learned_ckpt, device='cpu')
    c = np.load(os.path.join(out, 'cache', stem + '.npz'))
    img = cv2.imread(os.path.join(IMAGES, stem + '.png'), cv2.IMREAD_GRAYSCALE)
    gt = cv2.imread(os.path.join(MASKS, stem + '.png'), cv2.IMREAD_GRAYSCALE)
    seg = str(c['segmenter'])
    mask = _mask_of(c)
    tabs, nbytes = {}, {}
    tabs['REF'] = dm.pixel_arm_table(gt)
    tabs['SEG'] = dm.pixel_arm_table(mask)
    nbytes['SEG'] = len(zlib.compress(np.packbits(mask > 0).tobytes(), 9))    # lossless mask
    nbytes['SEG_png'] = len(cv2.imencode('.png', mask * 255, [cv2.IMWRITE_PNG_COMPRESSION, 9])[1])
    v6 = np.load(os.path.join(V6_CACHE, stem + '.npz'))['payload_tagged'].tobytes()
    tabs['GRAPH6'] = dm.graph_arm_table(*dm._decoded_graph_arrays(v6))
    nbytes['GRAPH6'] = len(v6)
    _, jb = jpeg_for_budget(img, len(v6))
    if jb:
        dec = cv2.imdecode(np.frombuffer(jb, np.uint8), cv2.IMREAD_GRAYSCALE)
        tabs['JPEG'] = dm.pixel_arm_table(dm.segment_like_pipeline(dec, seg, _MODEL))
        nbytes['JPEG'] = len(jb)
    for q in JPEG_Q:
        _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, q])
        dec = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        tabs[f'JPEGq{q}'] = dm.pixel_arm_table(dm.segment_like_pipeline(dec, seg, _MODEL))
        nbytes[f'JPEGq{q}'] = len(buf)
    tabs['PRE7'] = dm.graph_arm_table(c['pre_pos'], c['pre_rad'], c['pre_edges'], c['pre_elen'])
    app = len(c['appearance'])
    for k in [f'struct_dt_{e}' for e in EPS] + ['struct_profile']:
        st = gb.decode_structure(c[k].tobytes())
        arm = 'GRAPH7' if k == f'struct_dt_{EPS_MAIN}' else ('GRAPH7p' if k == 'struct_profile'
                                                           else f'GRAPH7_eps{k.split("_")[-1]}')
        tabs[arm] = dm.graph_arm_table(*gb.structure_to_arrays(st, path_lengths=True))
        nbytes[arm] = len(c[k])
    nbytes['APPEARANCE'] = app
    with open(os.path.join(out, 'tables', stem + '.json'), 'w') as f:
        json.dump({'tables': tabs, 'bytes': nbytes, 'fg_psnr': float(c['fg_psnr']),
                   'v6_bytes': int(c['v6_bytes'])}, f)
    rows = []
    for arm, t in tabs.items():
        for L in LS:
            d = dm.descriptors_at(t, L)[0]
            r = {'stem': stem, 'arm': arm, 'L': L, 'bytes': nbytes.get(arm, np.nan), **d}
            jr, er = dm.vertex_positions(tabs['REF'], L)
            ja, ea = dm.vertex_positions(t, L)
            for tol in TOLS:
                r[f'junc_p{tol}'], r[f'junc_r{tol}'], r[f'junc_f1_{tol}'] = dm.point_f1(ja, jr, tol)
                r[f'end_p{tol}'], r[f'end_r{tol}'], r[f'end_f1_{tol}'] = dm.point_f1(ea, er, tol)
            rows.append(r)
    return rows


def stage_measure(a):
    import multiprocessing as mp
    import pandas as pd
    os.makedirs(os.path.join(a.out, 'tables'), exist_ok=True)
    stems = sorted(f[:-4] for f in os.listdir(os.path.join(a.out, 'cache')) if f.endswith('.npz'))
    if a.limit:
        stems = stems[:a.limit]
    rows = []
    with mp.get_context('spawn').Pool(a.workers) as pool:
        for i, rs in enumerate(pool.imap_unordered(_measure_one, [(s, a.out) for s in stems])):
            rows += rs
            if i % 50 == 0:
                print(f'[measure {i + 1}/{len(stems)}]', flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, 'per_image.csv'), index=False)


# ---------------------------------------------------------------------------
def stage_stats(a):
    import pandas as pd
    import downstream_morphometry as dm
    P = pd.read_csv(os.path.join(a.out, 'per_image.csv'), dtype={'stem': str})
    held = set(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 'results', 'paper', 'heldout_organelle.txt')).read().split())
    core = ['REF', 'SEG', 'JPEG', 'GRAPH6', 'GRAPH7', 'PRE7']
    stems_all = sorted(set.intersection(*(set(P[P.arm == x].stem) for x in core)))
    out = []
    for subset, stems in (('all', stems_all), ('heldout', [s for s in stems_all if s in held])):
        for L in LS:
            W = {x: g.set_index('stem').loc[stems] for x, g in P[P.L == L].groupby('arm')}
            y = {d: W['REF'][d].to_numpy(float) for d in dm.DESCRIPTORS}
            for arm in [x for x in W if x != 'REF']:
                r = {'subset': subset, 'L': L, 'arm': arm, 'n': len(stems),
                     'bytes_mean': W[arm].bytes.mean()}
                for d in dm.DESCRIPTORS:
                    x = W[arm][d].to_numpy(float)
                    ok = np.isfinite(x) & np.isfinite(y[d])
                    ag = dm.agreement(x[ok], y[d][ok])
                    r[f'{d}_bias_pct'], r[f'{d}_ccc'], r[f'{d}_mdape'] = ag['bias_pct'], ag['ccc'], ag['mdape']
                    if arm != 'JPEG' and 'JPEG' in W:
                        j = W['JPEG'][d].to_numpy(float)
                        ok3 = ok & np.isfinite(j)
                        t = dm.paired_error_test(x[ok3], j[ok3], y[d][ok3])
                        r[f'{d}_wins_vs_jpeg'] = f"{t['wins_graph']}/{t['wins_jpeg']}"
                        r[f'{d}_p_vs_jpeg'] = t['wilcoxon_p']
                for tol in TOLS:
                    for k in ('junc', 'end'):
                        r[f'{k}_f1_{tol}'] = W[arm][f'{k}_f1_{tol}'].mean()
                        r[f'{k}_p{tol}'] = W[arm][f'{k}_p{tol}'].mean()
                        r[f'{k}_r{tol}'] = W[arm][f'{k}_r{tol}'].mean()
                out.append(r)
    S = pd.DataFrame(out)
    S.to_csv(os.path.join(a.out, 'summary.csv'), index=False)
    cols = ['arm', 'bytes_mean', 'n_branches_bias_pct', 'n_branches_ccc', 'n_junctions_bias_pct',
            'n_junctions_ccc', 'total_length_px_bias_pct', 'mean_width_px_bias_pct', 'cycle_rank_bias_pct',
            'junc_f1_3', 'end_f1_3']
    for subset in ('all', 'heldout'):
        for L in (0, 5):
            print(f'\n== {subset} L={L}')
            print(S[(S.subset == subset) & (S.L == L)][cols].round(3).to_string(index=False))


# ---------------------------------------------------------------------------
def _width_one(job):
    stem, out = job
    import cv2
    import downstream_morphometry as dm
    from nanograph_v4 import graph_branch as gb
    c = np.load(os.path.join(out, 'cache', stem + '.npz'))
    img = cv2.imread(os.path.join(IMAGES, stem + '.png'), cv2.IMREAD_GRAYSCALE)
    gt = cv2.imread(os.path.join(MASKS, stem + '.png'), cv2.IMREAD_GRAYSCALE)
    mask = _mask_of(c)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    ops = {'none': mask, 'dilate1': cv2.dilate(mask, k, iterations=1),
           'dilate2': cv2.dilate(mask, k, iterations=2), 'erode1': cv2.erode(mask, k, iterations=1)}
    ref = dm.descriptors_at(dm.pixel_arm_table(gt), 5)[0]['mean_width_px']
    row = {'stem': stem, 'ref_width': ref}
    for name, m in ops.items():
        skel, dt = _skel_dt(m)
        for mode in ('dt', 'profile'):
            st = gb.branch_structure(skel, dt, eps=EPS_MAIN, width_mode=mode, img=img)
            t = dm.graph_arm_table(*gb.structure_to_arrays(st, path_lengths=True))
            row[f'{mode}_{name}'] = dm.descriptors_at(t, 5)[0]['mean_width_px']
    return row


def stage_width(a):
    import multiprocessing as mp
    import pandas as pd
    import downstream_morphometry as dm
    stems = sorted(f[:-4] for f in os.listdir(os.path.join(a.out, 'cache')) if f.endswith('.npz'))
    stems = sorted(np.random.default_rng(0).choice(stems, min(150, len(stems)), replace=False))
    with mp.get_context('spawn').Pool(a.workers) as pool:
        W = pd.DataFrame(pool.map(_width_one, [(s, a.out) for s in stems]))
    W.to_csv(os.path.join(a.out, 'width_robustness.csv'), index=False)
    rows = []
    for mode in ('dt', 'profile'):
        base = W[f'{mode}_none']
        for op in ('none', 'dilate1', 'dilate2', 'erode1'):
            x = W[f'{mode}_{op}']
            ok = x.notna() & W.ref_width.notna()
            rows.append({'width': mode, 'mask_op': op,
                         'change_vs_unperturbed_pct': 100 * np.median(x / base - 1),
                         'ccc_vs_ref': dm.ccc(x[ok].to_numpy(), W.ref_width[ok].to_numpy()),
                         'r_vs_ref': np.corrcoef(x[ok], W.ref_width[ok])[0, 1],
                         'bias_vs_ref_pct': 100 * (x[ok].mean() / W.ref_width[ok].mean() - 1)})
    S = pd.DataFrame(rows)
    S.to_csv(os.path.join(a.out, 'width_summary.csv'), index=False)
    print(S.round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', required=True, choices=['encode', 'measure', 'stats', 'width'])
    ap.add_argument('--out', default='results/v7')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    {'encode': stage_encode, 'measure': stage_measure, 'stats': stage_stats,
     'width': stage_width}[a.stage](a)


if __name__ == '__main__':
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    main()
