#!/usr/bin/env python3
"""
T18 pilot figure: per cell-cycle stage, the cell whose fragmentation
(components per 100 um) is closest to its stage median. Columns: the +-1 um
slab image; the segmentation (ours vs Allen's own, same slab); the decoded structure layer drawn on it (one colour per separate
mitochondrion, junctions white, endpoints yellow); the graph alone with the
descriptors read from the stored bytes.

Writes <out>/pilot/fig_cells.png.
"""
import argparse
import ast
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from allen_check import slab_projection  # noqa: E402
from allen_pilot import STAGES  # noqa: E402

LABEL = {'M0': 'interphase (M0)', 'M1M2': 'prophase (M1–M2)', 'M3': 'metaphase (M3)',
         'M4M5': 'anaphase (M4–M5)', 'M6M7_single': 'telophase (M6–M7, one daughter)',
         'M6M7_complete': 'telophase (M6–M7, both daughters)'}


def components(pos, edges):
    n = len(pos)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for u, v in edges:
        parent[find(u)] = find(v)
    return np.array([find(i) for i in range(n)])


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    import tifffile
    import contextlib
    import io
    from nanograph_v4 import NanographConfig, graph_branch as gb
    from nanograph_v4.detect import detect_polarity
    from nanograph_v4.segment import learned_segment
    from nanograph_v4.unet_seg import load_unet
    cfg = NanographConfig().for_real_mito()
    net = load_unet(cfg.segment.learned_ckpt, 'cpu')
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/mnt/nas1/nba055-2/idea_1/archive_demo/allen')
    a = ap.parse_args()
    P = pd.read_csv(os.path.join(a.out, 'pilot', 'per_cell.csv'))
    M = pd.read_csv(os.path.join(a.out, 'manifest_tomm20.csv')).set_index('CellId')
    picks = []
    for s in STAGES:
        g = P[P.cell_stage == s]
        picks.append(g.loc[(g.components_per_100um - g.components_per_100um.median()).abs().idxmin()])
    fig, ax = plt.subplots(len(picks), 4, figsize=(14, 3.6 * len(picks)))
    cmap = plt.get_cmap('tab20')
    for i, r in enumerate(picks):
        cid = int(r.CellId)
        m = M.loc[cid]
        raw = tifffile.imread(os.path.join(a.out, 'cells', f'{cid}_raw.ome.tif'))
        seg = tifffile.imread(os.path.join(a.out, 'cells', f'{cid}_seg.ome.tif'))
        im, cell, aref, _ = slab_projection(raw, seg, ast.literal_eval(m.name_dict),
                                            ast.literal_eval(m.scale_micron)[-1])
        x = 255 - im if detect_polarity(im, cfg=cfg) else im
        with contextlib.redirect_stdout(io.StringIO()):
            ours = (learned_segment(net, x, cfg=cfg, device='cpu') > 0) & cell
        both = np.zeros(im.shape + (3,))
        both[..., 0] = ours & ~aref            # ours only: magenta
        both[..., 2] = ours & ~aref
        both[..., 1] = aref & ~ours            # Allen only: green
        both[ours & aref] = 1.0                # both: white
        dice = 2 * (ours & aref).sum() / max(ours.sum() + aref.sum(), 1)
        sb = open(os.path.join(a.out, 'ng_structure', f'{cid}.ngs'), 'rb').read()
        st = gb.decode_structure(sb)
        pos, rad, edges, _ = gb.structure_to_arrays(st)
        comp = components(pos, edges)
        cid_of = {c: k for k, c in enumerate(np.unique(comp))}
        ys, xs = np.nonzero(cell)
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        bar = 5.0 / r.um_per_px          # 5 um scale bar
        for j in range(4):
            A = ax[i, j]
            if j in (0, 2):
                A.imshow(im, cmap='gray', vmin=0, vmax=255)
                A.contour(cell, [0.5], colors=['#5a8fd6'], linewidths=0.6)
            elif j == 1:
                A.imshow(both)
                A.contour(cell, [0.5], colors=['#5a8fd6'], linewidths=0.6)
                A.set_xlabel(f'Dice {dice:.2f}', fontsize=8)
            else:
                A.set_facecolor('black')
            if j > 1:
                for u, v in edges:
                    c = cmap(cid_of[comp[u]] % 20)
                    A.plot([pos[u][1], pos[v][1]], [pos[u][0], pos[v][0]], color=c,
                           lw=1.3 if j == 2 else 1.8, solid_capstyle='round')
                kinds = st['vkind']
                vp = np.asarray(st['vpos'])
                for kind, col, mk in (('junction', 'white', 'o'), ('endpoint', '#ffd400', 's')):
                    sel = [k for k, t in enumerate(kinds) if t == kind]
                    if sel:
                        A.scatter(vp[sel, 1], vp[sel, 0], s=7, c=col, marker=mk, lw=0, zorder=3)
            A.set_xlim(x0 - 4, x1 + 4)
            A.set_ylim(y1 + 4, y0 - 4)
            A.set_xticks([])
            A.set_yticks([])
            A.set_aspect('equal')
        ax[i, 0].plot([x0 + 4, x0 + 4 + bar], [y1, y1], color='white', lw=2.5)
        ax[i, 0].set_ylabel(LABEL[r.cell_stage], fontsize=9)
        ax[i, 3].set_xlabel(f'{len(sb)} B stored | {int(r.n_components)} mitochondria, '
                            f'{int(r.n_junctions)} junctions\n{r.total_length_um:.0f} µm long | '
                            f'{r.components_per_100um:.1f} separate/100 µm | '
                            f'branch {r.mean_branch_um:.2f} µm', fontsize=7.5)
    for j, t in enumerate(['TOMM20, ±1 µm slab; 5 µm bar',
                           'mask: white both, magenta ours only,\ngreen Allen only',
                           'stored structure layer on the image',
                           'structure layer alone']):
        ax[0, j].set_title(t, fontsize=9, loc='left')
    fig.tight_layout()
    out = os.path.join(a.out, 'pilot', 'fig_cells.png')
    fig.savefig(out, dpi=130)
    print('wrote', out)


if __name__ == '__main__':
    main()
