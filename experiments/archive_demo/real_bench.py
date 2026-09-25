#!/usr/bin/env python3
"""
Default-segmenter benchmarks 1-4 (results/allen/SELECTION_RULE.md): every
candidate on the real test sets with EXPERT masks (UiT-Rat, CBMI, MITO,
EP-UiT-Human; real_mito manifest, split == test).

Stages
  calibrate  physical pixel size per dataset for scale-aware methods (Nellie):
             chosen so that the median mitochondrial width (2 x distance
             transform on the skeleton) of the dataset's TRAINING masks equals
             that of Allen's training masks in um (0.108333 um / px). The
             sources record no pixel size. EP-UiT-Human has no training split
             and takes the UiT-Rat value (same lab and microscope type).
             -> <root>/pixel_sizes.json
  export     test tiles -> <root>/<D>/img_raw/<id>.png, <root>/<D>/ref/<id>.png
  predict    in-process U-Net candidates (real_mito, allen_ft [--ckpt]) ->
             <root>/preds/<name>/<D>/<id>.png, with seconds per tile
  eval       every <root>/preds/<name> against the expert masks:
             per-tile clDice, Dice, junction/endpoint F1 (3 px) on the v7
             structure layers (one-diameter rule); per dataset: means, and
             descriptor CCC / bias of the structure layers -> <root>/eval/

Allen-trained candidates saw only cell-masked Allen slabs in training; real
tiles are given to every candidate unmasked, as they come.
"""
import argparse
import contextlib
import glob
import io
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

REAL = '/mnt/nas1/nba055-2/idea_1/real_mito'
ALLEN_TILES = '/mnt/nas1/nba055-2/idea_1/archive_demo/allen/trainset'
ALLEN_UM = 0.108333
ROOT = '/mnt/nas1/nba055-2/idea_1/archive_demo/realbench'
SETS = ['UIT', 'CBMI', 'MITO', 'HUMAN']
DESC = ['n_components', 'n_branches', 'n_junctions', 'total_length_px', 'mean_width_px', 'cycle_rank']


def median_width(mask_paths):
    import cv2
    from skimage.morphology import skeletonize
    w = []
    for p in mask_paths:
        m = (cv2.imread(p, cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
        sk = skeletonize(m > 0)
        if sk.sum():
            w.append(2 * cv2.distanceTransform(m, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[sk])
    return float(np.median(np.concatenate(w)))


def calibrate(root):
    import pandas as pd
    rng = np.random.default_rng(0)
    A = pd.read_csv(f'{ALLEN_TILES}/manifest.csv')
    A = A[A.split == 'train']
    a_px = median_width([f'{ALLEN_TILES}/masks/{i}.png' for i in rng.choice(A.id, min(500, len(A)), replace=False)])
    a_um = a_px * ALLEN_UM
    M = pd.read_csv(f'{REAL}/manifest.csv')
    out = {'allen_median_width_px': a_px, 'allen_median_width_um': a_um, 'rule': __doc__.split('calibrate')[1].split('export')[0].strip()}
    for d in ('UIT', 'CBMI', 'MITO'):
        g = M[(M.dataset == d) & (M.split == 'train')]
        w = median_width([f'{REAL}/{d}/masks/{i}.png' for i in g.id])
        out[d] = {'train_median_width_px': w, 'um_per_px': a_um / w}
    out['HUMAN'] = dict(out['UIT'], note='no training split; UiT-Rat calibration')
    os.makedirs(root, exist_ok=True)
    json.dump(out, open(f'{root}/pixel_sizes.json', 'w'), indent=1)
    print(json.dumps(out, indent=1))


def export(root):
    import shutil
    import pandas as pd
    M = pd.read_csv(f'{REAL}/manifest.csv')
    for d in SETS:
        g = M[(M.dataset == d) & (M.split == 'test')]
        for sub in ('img_raw', 'ref'):
            os.makedirs(f'{root}/{d}/{sub}', exist_ok=True)
        for i in g.id:
            shutil.copyfile(f'{REAL}/{d}/images/{i}.png', f'{root}/{d}/img_raw/{i}.png')
            shutil.copyfile(f'{REAL}/{d}/masks/{i}.png', f'{root}/{d}/ref/{i}.png')
        print(d, len(g), 'test tiles')


def predict(root, name, ckpt):
    import cv2
    import pandas as pd
    import torch
    from nanograph_v4 import NanographConfig
    from nanograph_v4.detect import detect_polarity
    from nanograph_v4.segment import learned_segment
    from nanograph_v4.unet_seg import load_unet
    torch.set_num_threads(1)
    cfg = NanographConfig().for_real_mito()
    net = load_unet(ckpt or cfg.segment.learned_ckpt, 'cpu')
    rows = []
    for d in SETS:
        out = f'{root}/preds/{name}/{d}'
        os.makedirs(out, exist_ok=True)
        for p in sorted(glob.glob(f'{root}/{d}/img_raw/*.png')):
            x = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            t0 = time.perf_counter()
            x = 255 - x if detect_polarity(x, cfg=cfg) else x
            with contextlib.redirect_stdout(io.StringIO()):
                m = learned_segment(net, x, cfg=cfg, device='cpu') > 0
            rows.append({'dataset': d, 'id': os.path.basename(p)[:-4], 'seconds': time.perf_counter() - t0})
            cv2.imwrite(f'{out}/{os.path.basename(p)}', m.astype(np.uint8) * 255)
    pd.DataFrame(rows).to_csv(f'{root}/preds/{name}/timing.csv', index=False)
    print(name, len(rows), 'tiles, median s/tile', np.median([r['seconds'] for r in rows]))


def score_tile(args):
    root, name, d, tid = args
    import cv2
    import downstream_morphometry as dm
    from allen_eval_seg import cldice, structure_table
    ref = cv2.imread(f'{root}/{d}/ref/{tid}.png', cv2.IMREAD_GRAYSCALE) > 0
    pp = f'{root}/preds/{name}/{d}/{tid}.png'
    m = cv2.imread(pp, cv2.IMREAD_GRAYSCALE) > 0 if os.path.exists(pp) else None
    if m is None:
        return {'dataset': d, 'id': tid, 'method': name, 'missing': True}
    tr, _ = structure_table(ref)
    tm, b = structure_table(m)
    z = {k: 0.0 for k in DESC}
    dr = dm.descriptors_at(tr, 'auto', 'degree')[0] if tr is not None else z
    dd = dm.descriptors_at(tm, 'auto', 'degree')[0] if tm is not None else z
    jr, er = dm.vertex_positions(tr, 'auto') if tr is not None else ([], [])
    jm, em = dm.vertex_positions(tm, 'auto') if tm is not None else ([], [])
    return {'dataset': d, 'id': tid, 'method': name, 'missing': False,
            'cldice': cldice(m, ref), 'dice': 2 * (m & ref).sum() / max(m.sum() + ref.sum(), 1),
            'junc_f1': dm.point_f1(jm, jr, 3)[2], 'end_f1': dm.point_f1(em, er, 3)[2], 'structure_bytes': b,
            **{k: dd[k] for k in DESC}, **{k + '_ref': dr[k] for k in DESC}}


def evaluate(root, workers):
    import multiprocessing as mp
    import pandas as pd
    import downstream_morphometry as dm
    names = sorted(os.path.basename(p) for p in glob.glob(f'{root}/preds/*') if os.path.isdir(p))
    jobs = [(root, n, d, os.path.basename(p)[:-4]) for n in names for d in SETS
            for p in sorted(glob.glob(f'{root}/{d}/ref/*.png'))]
    with mp.get_context('spawn').Pool(workers) as pool:
        R = pd.DataFrame(pool.map(score_tile, jobs, chunksize=8))
    os.makedirs(f'{root}/eval', exist_ok=True)
    R.to_csv(f'{root}/eval/per_tile.csv', index=False)
    S = []
    for (n, d), g in R.groupby(['method', 'dataset']):
        miss = g.missing.mean()
        g = g[~g.missing]
        tf = f'{root}/preds/{n}/timing.csv'
        sec = pd.read_csv(tf).query('dataset == @d').seconds.median() if os.path.exists(tf) else np.nan
        r = {'method': n, 'dataset': d, 'n': len(g), 'missing_frac': miss, 'cldice': g.cldice.mean(),
             'dice': g.dice.mean(), 'junc_f1': g.junc_f1.mean(), 'end_f1': g.end_f1.mean(), 'sec_per_tile': sec}
        for k in DESC:
            ag = dm.agreement(g[k].to_numpy(float), g[k + '_ref'].to_numpy(float))
            r[f'{k}_ccc'], r[f'{k}_bias_pct'] = ag['ccc'], ag['bias_pct']
        S.append(r)
    S = pd.DataFrame(S)
    S.to_csv(f'{root}/eval/summary.csv', index=False)
    pd.set_option('display.width', 200)
    print(S.pivot(index='method', columns='dataset', values='cldice').round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['calibrate', 'export', 'predict', 'eval'])
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--name', default='real_mito')
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--workers', type=int, default=6)
    a = ap.parse_args()
    if a.stage == 'calibrate':
        calibrate(a.root)
    elif a.stage == 'export':
        export(a.root)
    elif a.stage == 'predict':
        predict(a.root, a.name, a.ckpt)
    else:
        evaluate(a.root, a.workers)


if __name__ == '__main__':
    main()
