#!/usr/bin/env python3
"""
Benchmark 6 material (SELECTION_RULE.md): 60 Allen held-out cells (seed 0,
showcase cells excluded). Per cell, one composite PNG: the slab image, then
four unlabelled masks A-D in a seeded random order (Allen's segmentation and
the top-3 candidates of benchmarks 1-5). The letter -> method key is written
separately (key.csv) and must not be shown to raters.

Usage:
    python experiments/archive_demo/preference_pack.py --top nellie,real_mito,nnunet
"""
import argparse
import os

import numpy as np

A = '/mnt/nas1/nba055-2/idea_1/archive_demo/allen'
SHOWCASE = f'{A}/showcase/cells.csv'


def main():
    import cv2
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument('--top', required=True, help='comma list of the 3 candidates to show')
    ap.add_argument('--n', type=int, default=60)
    ap.add_argument('--out', default='results/allen/preference_pack')
    a = ap.parse_args()
    top = a.top.split(',')
    assert len(top) == 3, 'benchmark 6 shows exactly the top-3 candidates plus Allen'
    C = pd.read_csv(f'{A}/testset/cells.csv')
    excl = set(pd.read_csv(SHOWCASE).CellId) if os.path.exists(SHOWCASE) else set()
    C = C[~C.CellId.isin(excl)]
    rng = np.random.default_rng(0)
    cells = sorted(rng.choice(C.CellId, a.n, replace=False))
    os.makedirs(os.path.join(a.out, 'panels'), exist_ok=True)
    key = []
    for k, cid in enumerate(cells):
        rd = lambda p: cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        img, cell = rd(f'{A}/testset/img_raw/{cid}.png'), rd(f'{A}/testset/cell/{cid}.png') > 0
        masks = {'allen': rd(f'{A}/testset/allen/{cid}.png') > 0}
        for m in top:
            p = rd(f'{A}/preds/{m}/{cid}.png')
            assert p is not None, f'missing prediction {m}/{cid}'
            masks[m] = (p > 0) & cell
        order = list(rng.permutation(list(masks)))
        ys, xs = np.nonzero(cell)
        y0, y1, x0, x1 = max(ys.min() - 4, 0), ys.max() + 5, max(xs.min() - 4, 0), xs.max() + 5
        crop = lambda z: z[y0:y1, x0:x1]
        base = cv2.cvtColor(crop(img), cv2.COLOR_GRAY2BGR)
        cs, _ = cv2.findContours(crop(cell).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(base, cs, -1, (214, 143, 90), 1)          # the cell being judged
        tiles = [base]
        for letter, m in zip('ABCD', order):
            t = np.zeros(crop(img).shape + (3,), np.uint8)
            t[crop(masks[m])] = 255
            cv2.putText(t, letter, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 200, 255), 2)
            tiles.append(t)
            key.append({'panel': k, 'CellId': cid, 'letter': letter, 'method': m})
        sep = np.full((tiles[0].shape[0], 6, 3), 40, np.uint8)
        row = np.hstack(sum([[t, sep] for t in tiles], [])[:-1])
        cv2.imwrite(os.path.join(a.out, 'panels', f'{k:02d}.png'), row)
    pd.DataFrame(key).to_csv(os.path.join(a.out, 'key.csv'), index=False)
    pd.DataFrame({'panel': range(len(cells)), 'CellId': cells}).to_csv(os.path.join(a.out, 'cells.csv'), index=False)
    print(f'{len(cells)} panels in {a.out}/panels; key in {a.out}/key.csv (do not show to raters)')


if __name__ == '__main__':
    main()
