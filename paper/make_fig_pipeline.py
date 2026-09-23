#!/usr/bin/env python3
"""Compose Figure 1: the Nanograph pipeline as one narrative diagram.

Panels are cropped from the stage montage produced by
experiments/render_pipeline_diagram.py (default: paper/figures/pipeline_overview.png)
and re-laid out as: encode path -> the stored object -> the two reads.

Usage:
    python paper/make_fig_pipeline.py \
        --stages paper/figures/pipeline_overview.png \
        --out paper/figures/fig_pipeline.png
Numbers in the labels come from --meta (JSON) if given, else from the defaults
below; make_numbers.py writes the same values as macros for the caption.
"""
import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

INK, ACCENT, GRAPH, STORE, MUTED = '#1a1a1a', '#1f5fa8', '#b8322a', '#0f6d5c', '#6b6b6b'
DEFAULT_META = dict(image='7378', raw_bytes=65536, payload=1957, nodes=144, edges=141,
                    components=4, points=264, psnr=30.9, fg_psnr=28.8, seg_iou=0.902)


def crop_panels(path):
    """Return the eight stage panels of the montage, in order."""
    im = Image.open(path).convert('RGB')
    a = np.asarray(im.convert('L'))
    dark = a < 235
    rows = np.where(dark.sum(1) > 0.25 * dark.shape[1])[0]
    cols = np.where(dark.sum(0) > 0.10 * dark.shape[0])[0]

    def bands(idx, gap=12, minlen=60):
        out, s, p = [], idx[0], idx[0]
        for i in idx[1:]:
            if i - p > gap:
                if p - s > minlen:
                    out.append((s, p))
                s = i
            p = i
        if p - s > minlen:
            out.append((s, p))
        return out

    rb, cb = bands(rows), bands(cols)
    panels = []
    for r0, r1 in rb[:2]:
        for c0, c1 in cb[:4]:
            sub = np.asarray(im.crop((c0, r0, c1, r1)).convert('L'))
            solid = np.where((sub < 235).mean(1) > 0.6)[0]   # the image block, not its title
            if len(solid) > 30:
                panels.append(im.crop((c0, r0 + solid[0], c1, r0 + solid[-1] + 1)))
            else:
                panels.append(im.crop((c0, r0, c1, r1)))
    return panels


FIGW, FIGH = 7.2, 7.4


def put(ax, img, x, y, w, edge=None):
    """Place an image with its left-top at (x, y); returns the bottom y."""
    ar = img.size[1] / img.size[0]
    h = w * ar * FIGW / FIGH
    box = ax.inset_axes([x, y - h, w, h], transform=ax.transAxes)
    box.imshow(img)
    box.set_xticks([]); box.set_yticks([])
    for s in box.spines.values():
        s.set_color(edge or '#cccccc'); s.set_linewidth(1.2 if edge else 0.6)
    return y - h


def arrow(ax, x0, y0, x1, y1, color=MUTED, style='-|>', lw=1.3, rad=0.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), transform=ax.transAxes,
                                 arrowstyle=style, mutation_scale=12, lw=lw,
                                 color=color, connectionstyle=f'arc3,rad={rad}',
                                 shrinkA=2, shrinkB=2, zorder=5))


def box(ax, x, y, w, h, color, alpha=.06, lw=1.0, ls='-'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, transform=ax.transAxes,
                                boxstyle='round,pad=0.008,rounding_size=0.012',
                                fc=color, ec=color, alpha=alpha, lw=0, zorder=0))
    ax.add_patch(FancyBboxPatch((x, y), w, h, transform=ax.transAxes,
                                boxstyle='round,pad=0.008,rounding_size=0.012',
                                fc='none', ec=color, lw=lw, ls=ls, zorder=1))


def txt(ax, x, y, s, size=8, color=INK, weight='normal', ha='left', va='bottom', style='normal'):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=size, color=color,
            fontweight=weight, ha=ha, va=va, style=style, zorder=6)


def node_record(ax, x, y, w, meta):
    """Schematic of one stored node record plus its edge."""
    fields = [('x, y', 'position'), ('w', 'width'), ('I', 'intensity'), (r'$\theta$', 'orientation')]
    cw = w / len(fields)
    for i, (sym, lab) in enumerate(fields):
        ax.add_patch(FancyBboxPatch((x + i * cw, y), cw * 0.94, 0.045, transform=ax.transAxes,
                                    boxstyle='round,pad=0.002,rounding_size=0.004',
                                    fc='white', ec=STORE, lw=1.0, zorder=4))
        txt(ax, x + i * cw + cw * 0.47, y + 0.0135, sym, 8.5, STORE, 'bold', ha='center')
        txt(ax, x + i * cw + cw * 0.47, y - 0.019, lab, 6.6, MUTED, ha='center')
    return y + 0.045


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stages', default='paper/figures/pipeline_overview.png')
    ap.add_argument('--meta', default=None)
    ap.add_argument('--out', default='paper/figures/fig_pipeline.png')
    a = ap.parse_args()
    meta = dict(DEFAULT_META)
    if a.meta:
        meta.update(json.load(open(a.meta)))
    p = crop_panels(a.stages)
    inp, seg, skel, graph, recon = p[0], p[2], p[3], p[4], p[6]

    fig = plt.figure(figsize=(7.2, 7.4))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis('off')

    # ---------------- encode path -------------------------------------------
    txt(ax, .035, .965, 'ENCODE   the image is reduced once', 8.5, MUTED, 'bold')
    xs, w, ytop = [.035, .275, .515, .755], .19, .945
    labels = [('Input image', f"sparse fluorescence\n{meta['raw_bytes']:,} bytes"),
              ('Foreground', f"learned clDice U-Net\nmask IoU {meta['seg_iou']:.2f}"),
              ('Medial axis', 'skeleton + distance\ntransform gives width'),
              ('Attributed graph', f"{meta['nodes']} nodes, {meta['edges']} edges,\n"
                                   f"{meta['components']} components")]
    ybot = ytop - .03
    for x, img, (t, sub) in zip(xs, [inp, seg, skel, graph], labels):
        ybot = put(ax, img, x, ytop - .03, w, edge=GRAPH if t == 'Attributed graph' else None)
        txt(ax, x, ytop - .022, t, 8.4, INK, 'bold')
        txt(ax, x, ybot - .042, sub, 6.9, MUTED)
    ymid = (ytop - .03 + ybot) / 2
    for x in xs[:-1]:
        arrow(ax, x + w + .006, ymid, x + .24 - .006, ymid)

    # ---------------- the stored object -------------------------------------
    box(ax, .035, .455, .93, .215, STORE, alpha=.05)  # stored-object band
    txt(ax, .055, .638, 'THE STORED OBJECT', 8.5, STORE, 'bold')
    txt(ax, .252, .6385, f"— the graph itself, {meta['payload']:,} bytes "
                        f"({meta['raw_bytes'] // meta['payload']}× smaller than the array)",
        8, MUTED)
    arrow(ax, .945, .700, .945, .676, GRAPH)

    txt(ax, .055, .592, 'per node', 7.2, INK, 'bold')
    node_record(ax, .055, .535, .42, meta)
    txt(ax, .055, .487, f"× {meta['nodes']} nodes, delta-coded along the graph", 6.9, MUTED)

    txt(ax, .525, .592, 'per edge', 7.2, INK, 'bold')
    ax.add_patch(FancyBboxPatch((.525, .535), .14, .045, transform=ax.transAxes,
                                boxstyle='round,pad=0.002,rounding_size=0.004',
                                fc='white', ec=STORE, lw=1.0, zorder=4))
    txt(ax, .595, .5485, 'i → j', 8.5, STORE, 'bold', ha='center')
    txt(ax, .595, .516, 'connectivity', 6.6, MUTED, ha='center')

    txt(ax, .70, .592, 'plus', 7.2, INK, 'bold')
    for i, (s, lab) in enumerate([('16×16', 'background field'), ('DCT', 'foreground residual')]):
        ax.add_patch(FancyBboxPatch((.70 + i * .135, .535), .12, .045, transform=ax.transAxes,
                                    boxstyle='round,pad=0.002,rounding_size=0.004',
                                    fc='white', ec=MUTED, lw=1.0, zorder=4))
        txt(ax, .76 + i * .135, .5485, s, 8, MUTED, 'bold', ha='center')
        txt(ax, .76 + i * .135, .516, lab, 6.6, MUTED, ha='center')
    txt(ax, .525, .487, 'no pixel array is kept; the decoder rebuilds the graph from this stream',
        6.9, MUTED, style='italic')

    # ---------------- the two reads -----------------------------------------
    arrow(ax, .30, .452, .22, .405, STORE, rad=.12)
    arrow(ax, .70, .452, .78, .405, ACCENT, rad=-.12)
    txt(ax, .035, .385, 'READ THE STRUCTURE   directly, no pixels touched', 8.5, STORE, 'bold')
    txt(ax, .60, .385, 'RENDER THE IMAGE   on demand', 8.5, ACCENT, 'bold')

    box(ax, .035, .085, .50, .285, STORE, alpha=.04, ls=':')
    rows = [(r'$\beta_0 = %d$' % meta['components'], 'connected components, counted on the graph'),
            (r'$\beta_1 = |E| - |V| + \beta_0$', 'cycle rank: path redundancy of the network'),
            ('mean width, edge length', 'per-edge morphometry in stored units'),
            ('junction degrees', 'branching pattern of the network')]
    for i, (sym, lab) in enumerate(rows):
        y = .330 - i * .055
        txt(ax, .06, y, sym, 8.2, STORE, 'bold')
        txt(ax, .06, y - .023, lab, 6.9, MUTED)
    txt(ax, .06, .092, 'exact properties of the stored graph: no re-segmentation,\n'
                       'no re-tracing, identical on every read', 6.9, INK, style='italic')

    rb = put(ax, recon, .625, .360, .17, edge=ACCENT)
    txt(ax, .625, rb - .035, 'elliptical PSFs placed at the nodes,', 6.9, MUTED)
    txt(ax, .625, rb - .060, 'each aligned to its stored orientation', 6.9, MUTED)
    txt(ax, .625, rb - .092, f"PSNR {meta['psnr']:.1f} dB · FG-PSNR {meta['fg_psnr']:.1f} dB",
        7.2, ACCENT, 'bold')

    # ---------------- contrast strip ----------------------------------------
    box(ax, .035, .012, .93, .055, MUTED, alpha=.05, lw=0.8, ls='-')
    txt(ax, .055, .045, 'pixel codec:', 7.2, MUTED, 'bold')
    txt(ax, .155, .045, 'bytes → decode → segment → trace → estimate descriptors  (re-derived, tool-dependent)',
        7.2, MUTED)
    txt(ax, .055, .022, r'$\bf{Nanograph}$:', 7.2, STORE)
    txt(ax, .155, .022, 'bytes → read descriptors  ·  bytes → render image   (stored once, then read)',
        7.2, STORE)

    fig.savefig(a.out, dpi=300)
    fig.savefig(a.out.replace('.png', '.pdf'))
    print('wrote', a.out)


if __name__ == '__main__':
    main()
