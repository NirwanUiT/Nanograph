#!/usr/bin/env python3
"""
T18 step 1 — Allen Cell hiPSC single-cell dataset (Viana et al., Nature 2023):
download manifest and resolution check for the TOMM20 (mitochondria) line.

Data: s3://allencell/aics/hipsc_single_cell_image_dataset (public, HTTPS).
Licence: Allen Institute for Cell Science terms of use
(https://www.allencell.org/terms-of-use.html): research / non-commercial use,
derivative works and academic publication allowed with citation; no
commercial redistribution.

Per cell, `crop_raw` is a ZCYX uint16 stack (channels dna, membrane,
structure; 0.108 um isotropic voxels) and `crop_seg` holds the cell mask
(membrane_segmentation) and Allen's own 3-D structure segmentation
(struct_segmentation).

2-D input for Nanograph (declared here, before the check is run): the
structure channel max-projected over a slab of +-1 um around the plane with
the largest in-cell mitochondrial signal ("slab"). A whole-cell MIP of these
~10 um-thick cells stacks several mitochondrial layers into one dense field;
on a pilot cell the slab matched Allen's segmentation far better
(Dice 0.67 vs 0.47). Intensities are stretched to uint8 with the (0.5, 99.8)
percentiles inside the projected cell mask, as for the real training data.

Resolution check (thresholds fixed before running): the real-mito
segmenter's mask inside the cell is NON-TRIVIAL if its foreground fraction is
in [0.02, 0.60] and its skeleton is >= 50 px long. The go/no-go rule of the
task list: stop and report if fewer than 80 % of the sampled cells pass.
Reported alongside, as a quality indicator only: Dice with Allen's
segmentation projected over the same slab (itself a model output, not
ground truth).

Writes <out>/manifest_tomm20.csv (every TOMM20 cell, with URLs),
<out>/check/per_cell.csv, <out>/check/summary.txt, <out>/check/panels.png,
and the S3 ETag of every downloaded file (md5 for single-part uploads) in
<out>/check/etags.csv.

Usage (from repo root):
    CUDA_VISIBLE_DEVICES= python experiments/archive_demo/allen_check.py \
        --out /mnt/nas1/nba055-2/idea_1/archive_demo/allen --n 100
"""
import argparse
import contextlib
import io
import os
import sys
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

BASE = 'https://allencell.s3.amazonaws.com/aics/hipsc_single_cell_image_dataset'
COLS = ['CellId', 'FOVId', 'PlateId', 'WellId', 'structure_name', 'cell_stage', 'crop_raw', 'crop_seg',
        'name_dict', 'scale_micron', 'roi', 'edge_flag', 'outlier', 'fov_path', 'struct_seg_path']
SLAB_UM = 1.0
FG_RANGE = (0.02, 0.60)
MIN_SKEL = 50
PASS_FRAC = 0.80


def manifest(meta, out):
    import pandas as pd
    p = os.path.join(out, 'manifest_tomm20.csv')
    if os.path.exists(p):
        return pd.read_csv(p)
    it = pd.read_csv(meta, usecols=COLS, chunksize=20000)
    M = pd.concat(c[c.structure_name == 'TOMM20'] for c in it)
    for c in ('crop_raw', 'crop_seg', 'fov_path', 'struct_seg_path'):
        M[c + '_url'] = BASE + '/' + M[c]
    M.to_csv(p, index=False)
    return M


def fetch(url, dst):
    if not os.path.exists(dst):
        tmp = dst + '.part'
        with urllib.request.urlopen(url, timeout=300) as r, open(tmp, 'wb') as f:
            etag = r.headers.get('ETag', '').strip('"')
            while True:
                b = r.read(1 << 20)
                if not b:
                    break
                f.write(b)
        os.replace(tmp, dst)
        open(dst + '.etag', 'w').write(etag)
    return open(dst + '.etag').read() if os.path.exists(dst + '.etag') else ''


def slab_projection(raw, seg, names, scale_um):
    """-> (uint8 image, cell mask, Allen slab mask, z0)."""
    ch_raw, ch_seg = names['crop_raw'], names['crop_seg']
    mito = raw[:, ch_raw.index('structure')].astype(np.float64)
    cell = seg[:, ch_seg.index('membrane_segmentation')] > 0
    aseg = seg[:, ch_seg.index('struct_segmentation')] > 0
    z0 = int(np.argmax(np.where(cell, mito, 0).sum((1, 2))))
    h = int(round(SLAB_UM / scale_um))
    sl = slice(max(z0 - h, 0), z0 + h + 1)
    pr, c2 = mito[sl].max(0), cell[sl].max(0)
    ar = aseg[sl].max(0) & c2
    v = pr[c2] if c2.any() else pr.ravel()
    lo, hi = np.percentile(v, 0.5), np.percentile(v, 99.8)
    im = (np.clip((pr - lo) / max(hi - lo, 1e-9), 0, 1) * 255).astype(np.uint8)
    return im, c2, ar, z0


def check_cell(args):
    row, cache = args
    import ast
    import cv2
    import tifffile
    import torch
    from skimage.morphology import skeletonize
    from nanograph_v4 import NanographConfig, graph_branch as gb
    from nanograph_v4.detect import detect_polarity
    from nanograph_v4.segment import learned_segment
    from nanograph_v4.unet_seg import load_unet
    torch.set_num_threads(1)
    global _NET
    cfg = NanographConfig().for_real_mito()
    if '_NET' not in globals():
        _NET = load_unet(cfg.segment.learned_ckpt, 'cpu')
    rp = os.path.join(cache, f"{row['CellId']}_raw.ome.tif")
    sp = os.path.join(cache, f"{row['CellId']}_seg.ome.tif")
    try:
        e1 = fetch(row['crop_raw_url'], rp)
        e2 = fetch(row['crop_seg_url'], sp)
        raw, seg = tifffile.imread(rp), tifffile.imread(sp)
        names = ast.literal_eval(row['name_dict'])
        scale = ast.literal_eval(row['scale_micron'])[-1]
        im, cell, aref, z0 = slab_projection(raw, seg, names, scale)
        x = 255 - im if detect_polarity(im, cfg=cfg) else im
        with contextlib.redirect_stdout(io.StringIO()):
            p = (learned_segment(_NET, x, cfg=cfg, device='cpu') > 0) & cell
        fg = float(p[cell].mean()) if cell.any() else 0.0
        skel_len = int(skeletonize(p).sum())
        dice = 2 * (p & aref).sum() / max(p.sum() + aref.sum(), 1)
        dt = cv2.distanceTransform(p.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        sbytes = len(gb.encode_structure(gb.branch_structure(skeletonize(p), dt))) if skel_len >= 2 else 0
        ok = FG_RANGE[0] <= fg <= FG_RANGE[1] and skel_len >= MIN_SKEL
        cv2.imwrite(os.path.join(cache, f"{row['CellId']}_panel.png"),
                    np.hstack([im, (p * 255).astype(np.uint8), (aref * 255).astype(np.uint8)]))
        return {'CellId': row['CellId'], 'cell_stage': row['cell_stage'], 'PlateId': row['PlateId'],
                'z0': z0, 'shape': 'x'.join(map(str, im.shape)), 'fg_in_cell': fg,
                'allen_fg_in_cell': float(aref[cell].mean()) if cell.any() else np.nan,
                'skeleton_px': skel_len, 'dice_vs_allen': float(dice), 'structure_bytes': sbytes,
                'raw_bytes': os.path.getsize(rp), 'nontrivial': bool(ok), 'error': '',
                'etag_raw': e1, 'etag_seg': e2}
    except Exception as ex:  # report, never hide
        return {'CellId': row['CellId'], 'cell_stage': row['cell_stage'], 'nontrivial': False,
                'error': f'{type(ex).__name__}: {ex}'}


def main():
    import multiprocessing as mp
    import cv2
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/mnt/nas1/nba055-2/idea_1/archive_demo/allen')
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    M = manifest(os.path.join(a.out, 'metadata.csv'), a.out)
    print(f'TOMM20 cells: {len(M)}; FOVs {M.FOVId.nunique()}; plates {M.PlateId.nunique()}')
    print(M.cell_stage.value_counts().to_string())
    chk = os.path.join(a.out, 'check')
    cache = os.path.join(a.out, 'cells')
    os.makedirs(chk, exist_ok=True)
    os.makedirs(cache, exist_ok=True)
    S = M.sample(a.n, random_state=a.seed)
    with mp.get_context('spawn').Pool(a.workers) as pool:
        rows = pool.map(check_cell, [(r, cache) for r in S.to_dict('records')])
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(chk, 'per_cell.csv'), index=False)
    R[['CellId', 'etag_raw', 'etag_seg']].dropna().to_csv(os.path.join(chk, 'etags.csv'), index=False)
    ok = R[R.error == '']
    frac = R.nontrivial.mean()
    lines = [f'sampled {len(R)} TOMM20 cells (seed {a.seed}); errors {int((R.error != "").sum())}',
             f'non-trivial masks: {int(R.nontrivial.sum())}/{len(R)} = {100 * frac:.1f} % '
             f'(threshold {100 * PASS_FRAC:.0f} %) -> {"GO" if frac >= PASS_FRAC else "NO-GO"}',
             f'fg in cell: median {ok.fg_in_cell.median():.3f} (Allen {ok.allen_fg_in_cell.median():.3f})',
             f'Dice vs Allen segmentation (slab): median {ok.dice_vs_allen.median():.3f}, '
             f'IQR {ok.dice_vs_allen.quantile(.25):.3f}-{ok.dice_vs_allen.quantile(.75):.3f}',
             f'structure layer: median {ok.structure_bytes.median():.0f} B per cell; '
             f'raw crop median {ok.raw_bytes.median() / 1e6:.1f} MB',
             'by stage (fraction non-trivial):',
             R.groupby('cell_stage').nontrivial.agg(['mean', 'size']).round(2).to_string()]
    open(os.path.join(chk, 'summary.txt'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    pans = [cv2.imread(os.path.join(cache, f'{c}_panel.png'), 0) for c in ok.CellId[:12]]
    pans = [cv2.resize(p, (600, 200)) for p in pans if p is not None]
    if pans:
        cv2.imwrite(os.path.join(chk, 'panels.png'), np.vstack(pans))


if __name__ == '__main__':
    main()
