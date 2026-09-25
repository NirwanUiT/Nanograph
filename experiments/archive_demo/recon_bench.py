#!/usr/bin/env python3
"""
Reference-free segmenter check: does a better mask give a better Nanograph
reconstruction of the real image? (Suggested by the author; tested here,
reported next to the selection rule, not a ranking benchmark.)

For every real test tile and every candidate's mask (plus the EXPERT mask as
an upper-bound candidate), the full encoder runs with that mask in place of
its own segmentation (the package is untouched: nanograph_v4.api.auto_segment
is replaced inside this process only). Two variants:
  full       default encoder (foreground residual and error-guided refinement on)
  structure  residual and refinement off: the render comes from the stored
             points alone, so pixel corrections cannot hide mask errors
Fidelity is measured on the decoded render (full-image PSNR / SSIM) with the
payload bytes. Pitfall recorded in advance: a larger mask buys more points and
bytes, so fidelity is also reported per kB and compared at similar bytes.

Validation (does it track truth?): per dataset, Spearman correlation across
candidates between mean fidelity and mean clDice against the expert masks
(realbench eval), and per-tile Spearman pooled over candidates.

Writes <root>/recon/per_tile.csv, summary.csv, validation.csv.
"""
import argparse
import contextlib
import glob
import io
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
ROOT = '/mnt/nas1/nba055-2/idea_1/archive_demo/realbench'
SETS = ['UIT', 'CBMI', 'MITO', 'HUMAN']


def encode(args):
    img_p, mask_p, method, d, tid, variant = args
    import cv2
    import nanograph_v4.api as api
    from nanograph_v4 import NanographConfig
    img = cv2.imread(img_p, cv2.IMREAD_GRAYSCALE)
    m = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return {'method': method, 'dataset': d, 'id': tid, 'variant': variant, 'missing': True}
    mask = ((m > 0) * 255).astype(np.uint8)
    api.auto_segment = lambda *a, **k: (mask.copy(), 'external', {})
    cfg = NanographConfig().for_real_mito()
    if variant == 'structure':
        cfg.compress.store_fg_residual = False
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            r = api.nanograph_encode(img, config=cfg, verbose=False, optimize=(variant == 'full'))
    except Exception as ex:          # e.g. empty mask: the encoder writes an empty payload (known bug)
        return {'method': method, 'dataset': d, 'id': tid, 'variant': variant, 'missing': False,
                'error': f'{type(ex).__name__}: {ex}', 'fg_frac': float((mask > 0).mean())}
    return {'error': '', 'method': method, 'dataset': d, 'id': tid, 'variant': variant, 'missing': False,
            'psnr': float(r.psnr_full), 'ssim': float(r.ssim_full), 'bytes': int(r.compressed_bytes),
            'fg_frac': float((mask > 0).mean())}


def main():
    import multiprocessing as mp
    import pandas as pd
    from scipy.stats import spearmanr
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--per-dataset', type=int, default=0, help='tiles per dataset (seeded), 0 = all')
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--out', default=None)
    ap.add_argument('--validate-only', action='store_true', help='recompute summary/validation from per_tile.csv')
    a = ap.parse_args()
    out = a.out or os.path.join(a.root, 'recon')
    os.makedirs(out, exist_ok=True)
    if a.validate_only:
        return validate(pd.read_csv(f'{out}/per_tile.csv'), a.root, out)
    methods = sorted(os.path.basename(p) for p in glob.glob(f'{a.root}/preds/*') if os.path.isdir(p))
    rng = np.random.default_rng(0)
    jobs = []
    for d in SETS:
        ids = sorted(os.path.basename(p)[:-4] for p in glob.glob(f'{a.root}/{d}/ref/*.png'))
        if a.per_dataset:
            ids = sorted(rng.choice(ids, min(a.per_dataset, len(ids)), replace=False))
        for tid in ids:
            img = f'{a.root}/{d}/img_raw/{tid}.png'
            for v in ('full', 'structure'):
                jobs.append((img, f'{a.root}/{d}/ref/{tid}.png', 'EXPERT', d, tid, v))
                jobs += [(img, f'{a.root}/preds/{m}/{d}/{tid}.png', m, d, tid, v) for m in methods]
    with mp.get_context('spawn').Pool(a.workers) as pool:
        R = pd.DataFrame(pool.map(encode, jobs, chunksize=4))
    R.to_csv(f'{out}/per_tile.csv', index=False)
    validate(R, a.root, out)


def validate(R, root, out):
    import pandas as pd
    from scipy.stats import spearmanr
    err = R[R.error.fillna('') != '']
    print(f'{len(err)} encodes failed:', err.groupby(['method', 'variant']).size().to_dict(),
          '| empty masks among them:', int((err.fg_frac == 0).sum()))
    R = R[~R.missing & (R.error.fillna('') == '')]
    S = R.groupby(['variant', 'dataset', 'method']).agg(psnr=('psnr', 'mean'), ssim=('ssim', 'mean'),
                                                         kB=('bytes', lambda b: b.mean() / 1e3),
                                                         fg=('fg_frac', 'mean'), n=('id', 'size')).reset_index()
    S.to_csv(f'{out}/summary.csv', index=False)
    E = pd.read_csv(f'{root}/eval/summary.csv')
    P = pd.read_csv(f'{root}/eval/per_tile.csv')
    # candidates incomplete in the expert-mask evaluation are excluded from the correlations
    E = E[E.missing_frac <= 0.05].dropna(subset=['cldice'])
    P = P[P.method.isin(E.method)].dropna(subset=['cldice'])
    val = []
    for v in ('full', 'structure'):
        for d in SETS:
            g = S[(S.variant == v) & (S.dataset == d) & (S.method != 'EXPERT')].merge(
                E[E.dataset == d][['method', 'cldice']], on='method')
            t = R[(R.variant == v) & (R.dataset == d) & (R.method != 'EXPERT')].merge(
                P[P.dataset == d][['method', 'id', 'cldice']], on=['method', 'id'])
            ex = S[(S.variant == v) & (S.dataset == d) & (S.method == 'EXPERT')]
            best = g.sort_values('ssim', ascending=False).method.iloc[0] if len(g) else ''
            val.append({'variant': v, 'dataset': d, 'n_methods': len(g),
                        'spearman_methods_ssim_vs_cldice': spearmanr(g.ssim, g.cldice).correlation if len(g) > 2 else np.nan,
                        'spearman_methods_psnr_vs_cldice': spearmanr(g.psnr, g.cldice).correlation if len(g) > 2 else np.nan,
                        'spearman_tiles_ssim_vs_cldice': spearmanr(t.ssim, t.cldice).correlation if len(t) > 2 else np.nan,
                        'expert_ssim': ex.ssim.iloc[0] if len(ex) else np.nan,
                        'best_candidate_ssim': g.ssim.max() if len(g) else np.nan,
                        'best_by_ssim': best,
                        'best_by_cldice': g.sort_values('cldice', ascending=False).method.iloc[0] if len(g) else ''})
    V = pd.DataFrame(val)
    V.to_csv(f'{out}/validation.csv', index=False)
    pd.set_option('display.width', 220)
    print(V.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
