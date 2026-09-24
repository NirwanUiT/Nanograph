#!/usr/bin/env python3
"""
Accuracy against the simulator's exact geometry (T13).

TRUE: every tube centreline (experiments/sim_mito.py) drawn 8-connected on the
256x256 grid; crossings in projection become junctions exactly as any 2-D
analysis must see them; width = the tube's true diameter. Descriptors use the
same branch decomposition as every other arm (skan pixel graph ->
branch_table), so TRUE differs from the arms only in where its centreline and
widths come from. Object level: the number of simulated tubes.

Arms per image:
  REF     skeleton of the simulator's mask (the reference used so far)
  SEG     skeleton of the pipeline's own mask
  GRAPH7  decoded v7 structure layer (default pipeline)
  GRAPH6  decoded v6 graph (graph.builder = 'pixel')
  JPEG    byte-matched JPEG (budget = v7 payload), same segmenter
Descriptors at L = 0/2/5/10 (degree-based junctions); junction / endpoint F1
against TRUE (3 and 5 px).

Usage (from repo root):
    python experiments/sim_truth_eval.py --workers 12
"""
import argparse
import contextlib
import io
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA = '/mnt/nas1/nba055-2/idea_1/sim_truth'
CLIP = '/mnt/nas1/nba055-2/idea_1/Aaron_data/4mitos_withTIFF_285px_73frames'
CLIP_IMG = '/mnt/nas1/nba055-2/idea_1/mito_aaron'
LS = [0, 2, 5, 10]
# (label, L, min_len): fixed pruning lengths (T11.2 protocol) and the
# one-diameter rule declared in T13 before this evaluation was run.
SETTINGS = [(str(L), L, 0.0) for L in LS] + [('auto', 'auto', 'auto')]


def _read_ply(p):
    b = open(p, 'rb').read()
    h = b.index(b'end_header\n') + len(b'end_header\n')
    hdr = b[:h].decode()
    n = int([ln for ln in hdr.split('\n') if ln.startswith('element vertex')][0].split()[-1])
    k = sum(1 for ln in hdr.split('\n') if ln.startswith('property'))
    return np.frombuffer(b[h:h + n * 8 * k], '<f8').reshape(n, k)


def clip_truth(frame):
    """Temporal clip (CODS simulator, 4 unbranched mitochondria, 285 px over
    12000 nm): per mitochondrion its skeleton polyline (ply/mito_k/skeleton)
    in pixels (row = y, col = x, verified 100 % inside the mask) and diameter =
    2 x median surface-to-skeleton distance (ply/mito_k/surface)."""
    from scipy.spatial import cKDTree
    px = 12000.0 / 285
    tubes = []
    for k in range(4):
        sk = _read_ply(os.path.join(CLIP, 'ply', f'mito_{k}', 'skeleton', f'{frame}.ply'))
        sf = _read_ply(os.path.join(CLIP, 'ply', f'mito_{k}', 'surface', f'{frame}.ply'))
        d, _ = cKDTree(sk).query(sf[:, :3])
        tubes.append({'row': ((sk[:, 1] + 6000) / px).tolist(), 'col': ((sk[:, 0] + 6000) / px).tolist(),
                      'diameter_px': 2 * float(np.median(d)) / px})
    return tubes


def load_item(i):
    """(image, mask, tubes, shape) for a sim_truth id (int) or a clip frame ('clip:<f>')."""
    import cv2
    if isinstance(i, str) and i.startswith('clip:'):
        f = i.split(':')[1]
        img = cv2.imread(os.path.join(CLIP_IMG, 'images', f'1_{f}.png'), cv2.IMREAD_GRAYSCALE)
        msk = cv2.imread(os.path.join(CLIP_IMG, 'masks', f'1_{f}.png'), cv2.IMREAD_GRAYSCALE)
        return img, msk, clip_truth(f)
    img = cv2.imread(os.path.join(DATA, 'images', f'{i}.png'), cv2.IMREAD_GRAYSCALE)
    msk = cv2.imread(os.path.join(DATA, 'masks', f'{i}.png'), cv2.IMREAD_GRAYSCALE)
    return img, msk, json.load(open(os.path.join(DATA, 'truth', f'{i}.json')))['tubes']


def truth_table(tubes, shape):
    """Branch table of the true centrelines, widths = true diameters."""
    import cv2
    import skan
    import downstream_morphometry as dm
    from skimage.morphology import skeletonize
    line = np.zeros(shape, np.uint8)
    wmap = np.zeros(shape, np.float32)
    for tb in tubes:
        pts = np.stack([np.round(tb['col']), np.round(tb['row'])], 1).astype(np.int32)
        cv2.polylines(line, [pts], False, 1, 1, lineType=cv2.LINE_8)
        cv2.polylines(wmap, [pts], False, float(tb['diameter_px']), 3, lineType=cv2.LINE_8)
    skel = skeletonize(line > 0)
    S = skan.Skeleton(skel)
    g = S.graph.tocoo()
    sel = g.row < g.col
    coords = np.asarray(S.coordinates).astype(int)
    width = wmap[coords[:, 0], coords[:, 1]]
    return dm.branch_table(coords, width, np.stack([g.row[sel], g.col[sel]], 1), g.data[sel])


def endcorr_length(table, L):
    """Total length after pruning at L, plus the branch's mean radius for every
    free end (a skeleton stops about one radius short of each tube end)."""
    import downstream_morphometry as dm
    br, inc = dm._prune_merge([list(b) for b in table['branches']], L)
    deg = {v: len(ks) for v, ks in inc.items()}
    tot = 0.0
    for bb in br:
        a, b, ln, w = bb[0], bb[1], bb[2], bb[3]
        rad = (w / ln) / 2 if ln > 0 else 0.0            # w = length-weighted diameter sum
        tot += ln + rad * ((deg.get(a, 0) == 1) + (deg.get(b, 0) == 1 and a != b))
    return tot


def _one(i):
    import cv2
    import torch
    import downstream_morphometry as dm
    from nanograph_v4 import nanograph_encode, NanographConfig
    from nanograph_v4.evaluate import jpeg_for_budget
    torch.set_num_threads(1)
    img, msk, tubes = load_item(i)
    with contextlib.redirect_stdout(io.StringIO()):
        r7 = nanograph_encode(img, verbose=False, config=NanographConfig())
        c6 = NanographConfig()
        c6.graph.builder = 'pixel'
        r6 = nanograph_encode(img, verbose=False, config=c6)
    tabs = {'TRUE': truth_table(tubes, img.shape), 'REF': dm.pixel_arm_table(msk), 'SEG': dm.pixel_arm_table(r7.mask),
            'GRAPH7': dm.graph_arm_table(*dm._decoded_graph_arrays(r7.compressed)),
            'GRAPH6': dm.graph_arm_table(*dm._decoded_graph_arrays(r6.compressed))}
    from nanograph_v4 import graph_branch as gb
    from skimage.morphology import skeletonize
    dt7 = cv2.distanceTransform((r7.mask > 0).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    stp = gb.branch_structure(skeletonize(r7.mask > 0), dt7, width_mode='profile', img=img)
    tabs['GRAPH7p'] = dm.graph_arm_table(*gb.structure_to_arrays(stp, path_lengths=True))
    nbytes = {'GRAPH7': len(r7.compressed), 'GRAPH6': len(r6.compressed),
              'GRAPH7_structure': r7.compression_stats.get('structure_bytes', np.nan)}
    _, jb = jpeg_for_budget(img, len(r7.compressed))
    if jb:
        from nanograph_v4.unet_seg import load_unet
        from nanograph_v4.config import DEFAULT_CONFIG
        global _M
        if '_M' not in globals():
            _M = load_unet(DEFAULT_CONFIG.segment.learned_ckpt, device='cpu')
        dec = cv2.imdecode(np.frombuffer(jb, np.uint8), cv2.IMREAD_GRAYSCALE)
        tabs['JPEG'] = dm.pixel_arm_table(dm.segment_like_pipeline(dec, r7.segmenter, _M))
        nbytes['JPEG'] = len(jb)
    g = msk > 0
    p = r7.mask > 0
    rows = []
    for arm, t in tabs.items():
        for L, Lv, mv in SETTINGS:
            d = dm.descriptors_at(t, Lv, min_len=mv)[0]
            r = {'img': i, 'arm': arm, 'L': L, 'n_tubes': len(tubes), 'bytes': nbytes.get(arm, np.nan),
                 'length_endcorr': (endcorr_length(t, Lv if Lv != 'auto' else 0) if arm != 'TRUE'
                                    else d['total_length_px']),
                 'structure_bytes': nbytes['GRAPH7_structure'], 'seg_iou': (p & g).sum() / max((p | g).sum(), 1),
                 **d}
            jt, et = dm.vertex_positions(tabs['TRUE'], Lv if Lv != 'auto' else 0)
            ja, ea = dm.vertex_positions(t, Lv, min_len=mv)
            for tol in (3, 5):
                r[f'junc_f1_{tol}'] = dm.point_f1(ja, jt, tol)[2]
                r[f'end_f1_{tol}'] = dm.point_f1(ea, et, tol)[2]
            rows.append(r)
    return rows


def main():
    import multiprocessing as mp
    import pandas as pd
    import downstream_morphometry as dm
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--dataset', default='sim', choices=['sim', 'clip'])
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    a.out = a.out or f'results/real/{a.dataset}_truth'
    os.makedirs(a.out, exist_ok=True)
    if a.dataset == 'sim':
        ids = sorted(int(f[:-5]) for f in os.listdir(os.path.join(DATA, 'truth')))
    else:
        ids = sorted((f'clip:{os.path.splitext(f)[0].split("_")[1]}'
                      for f in os.listdir(os.path.join(CLIP_IMG, 'images'))), key=lambda s: int(s.split(':')[1]))
    with mp.get_context('spawn').Pool(a.workers) as pool:
        rows = [r for rs in pool.imap_unordered(_one, ids) for r in rs]
    P = pd.DataFrame(rows)
    P.to_csv(os.path.join(a.out, 'per_image.csv'), index=False)
    S = []
    common = sorted(set.intersection(*(set(P[P.arm == x].img) for x in ('TRUE', 'REF', 'SEG', 'GRAPH7',
                                                                           'GRAPH6', 'JPEG'))))
    for L, _, _ in SETTINGS:
        W = {x: g.set_index('img').loc[common] for x, g in P[P.L == L].groupby('arm')}
        for arm in ('REF', 'SEG', 'GRAPH7', 'GRAPH7p', 'GRAPH6', 'JPEG'):
            r = {'L': L, 'arm': arm, 'n': len(common), 'bytes': W[arm].bytes.mean()}
            for d in dm.DESCRIPTORS:
                x, y = W[arm][d].to_numpy(float), W['TRUE'][d].to_numpy(float)
                ok = np.isfinite(x) & np.isfinite(y)
                ag = dm.agreement(x[ok], y[ok])
                r[f'{d}_bias_pct'], r[f'{d}_ccc'], r[f'{d}_mdape'] = ag['bias_pct'], ag['ccc'], ag['mdape']
                if arm != 'JPEG':
                    j = W['JPEG'][d].to_numpy(float)
                    ok3 = ok & np.isfinite(j)
                    t = dm.paired_error_test(x[ok3], j[ok3], y[ok3])
                    r[f'{d}_vs_jpeg'] = f"{t['wins_graph']}/{t['wins_jpeg']} p={t['wilcoxon_p']:.1e}"
            x = W[arm].length_endcorr.to_numpy(float)
            y = W['TRUE'].total_length_px.to_numpy(float)
            ag = dm.agreement(x, y)
            r['length_endcorr_bias_pct'], r['length_endcorr_ccc'] = ag['bias_pct'], ag['ccc']
            for k in ('junc_f1_3', 'end_f1_3', 'junc_f1_5', 'end_f1_5'):
                r[k] = W[arm][k].mean()
            # object level: components vs number of simulated tubes
            r['components_eq_tubes'] = float((W[arm].n_components == W[arm].n_tubes).mean())
            S.append(r)
    S = pd.DataFrame(S)
    S.to_csv(os.path.join(a.out, 'summary.csv'), index=False)
    W0 = P[(P.arm == 'TRUE') & (P.L == '0')]
    print(f"TRUE per image: tubes {W0.n_tubes.mean():.2f}, components {W0.n_components.mean():.2f}, "
          f"branches {W0.n_branches.mean():.2f}, junctions {W0.n_junctions.mean():.2f}, "
          f"cycle rank {W0.cycle_rank.mean():.2f}; pipeline Seg-IoU {P.seg_iou.mean():.3f}")
    cols = ['arm', 'bytes', 'n_components_bias_pct', 'total_length_px_bias_pct', 'length_endcorr_bias_pct',
            'mean_width_px_bias_pct',
            'n_branches_bias_pct', 'n_junctions_bias_pct', 'cycle_rank_bias_pct', 'n_branches_ccc',
            'junc_f1_3', 'end_f1_3']
    pd.set_option('display.width', 250)
    for L in ('0', '5', 'auto'):
        print(f'\n== L = {L}')
        print(S[S.L == L][cols].round(3).to_string(index=False))


if __name__ == '__main__':
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    main()
