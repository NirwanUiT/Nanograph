#!/usr/bin/env python3
"""
T14 — strong lossless baselines for the structure layer.

Every format stores the SAME mask (the reference mask: simulator ground truth
on the organelle set, the expert mask on the real test sets), so the byte
counts compare encodings, not segmentations.

Formats (per image):
  mask_zlib           bit-packed mask, zlib level 9 (the number used so far)
  mask_png            1-bit PNG, optimize=True
  mask_jbig2          JBIG2 generic region, lossless (jbig2enc without -s),
                      standalone file; verified bit-exact with jbig2dec.
                      Falls back to JBIG1 (jbigkit pbmtojbg) if jbig2enc is
                      missing; the `codec` column says which was used.
  mask_webp_lossless  WebP lossless of the 0/255 mask
  skel_chain          the mask's skeleton as 8-direction Freeman chain codes
                      per skan path (+ isolated pixels) and the per-pixel
                      radius (1/4 px, 0-255) in raster order, zigzag-varint
                      deltas, zlib; decodes
                      to the identical skeleton, so skan's decomposition is
                      reproduced exactly (asserted)
  swc, swc_gz         SWC text (and gzip) of the v7 branch graph with radii;
                      SWC is a tree format, so each independent cycle needs
                      one duplicated node (`swc_dup_nodes`), which the reader
                      merges back
  v7_struct_e0.75     v7 structure layer, Douglas-Peucker eps 0.75 (default)
  v7_struct_e3        v7 structure layer, eps 3

Per format: bytes; single-thread decode + descriptor time (one-diameter rule;
external-process codecs include process start-up, flagged in `external`);
whether each of the six descriptors (L = 0, degree junctions) equals the one
from the original mask (`exact_<descriptor>`) and its relative error.

Writes <out>/per_image.csv and <out>/summary.csv (mean/median bytes per
dataset and format; ratio of mean v7 bytes to mean format bytes with a
bootstrap 95 % CI over images; median decode time; fraction exact).

Usage (from repo root):
    OMP_NUM_THREADS=1 python experiments/lossless_baselines.py --workers 8 \
        --out results/paper/lossless
"""
import argparse
import glob
import gzip
import io
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zlib

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

ORG = '/mnt/nas1/nba055-2/idea_1/nmi_data/seg'
REAL = '/mnt/nas1/nba055-2/idea_1/real_mito/test_sets'
DATASETS = [('ORGANELLE', ORG), ('UIT', f'{REAL}/UIT/masks'), ('CBMI', f'{REAL}/CBMI/masks'),
            ('MITO', f'{REAL}/MITO/masks'), ('HUMAN', f'{REAL}/HUMAN/masks')]
TOOLS = os.environ.get('NG_TOOLS', os.path.expanduser('~/anaconda3/envs/ngtools/bin'))
V7 = 'v7_struct_e0.75'
COUNTS = ('n_components', 'n_branches', 'n_junctions', 'cycle_rank')
RQ = 4.0                      # skel_chain radius quantisation (1/4 px)
# Freeman directions: code k -> (drow, dcol)
DIRS = [(0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1)]
DIR_CODE = {d: k for k, d in enumerate(DIRS)}


def _tool(name):
    p = os.path.join(TOOLS, name)
    return p if os.path.exists(p) else shutil.which(name)


# ---------------------------------------------------------------------------
# varints
def _uv(buf, u):
    while True:
        b = u & 0x7F
        u >>= 7
        buf.append(b | (0x80 if u else 0))
        if not u:
            return


def _sv(buf, v):
    _uv(buf, (v << 1) ^ (v >> 63))


def _ruv(d, o):
    u = s = 0
    while True:
        b = d[o]
        o += 1
        u |= (b & 0x7F) << s
        s += 7
        if not b & 0x80:
            return u, o


def _rsv(d, o):
    u, o = _ruv(d, o)
    return (u >> 1) ^ -(u & 1), o


# ---------------------------------------------------------------------------
# skeleton-level analysis shared by the mask and chain-code routes
def skeleton_of(mask):
    from skimage.morphology import skeletonize
    return skeletonize(mask > 0)


def dist_of(mask):
    return cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)


def skel_table(skel, radius_img):
    """As downstream_morphometry.pixel_arm_table, from the skeleton step on."""
    import downstream_morphometry as dm
    import skan
    if skel.sum() < 2:
        return dm.branch_table(np.zeros((0, 2)), [], np.zeros((0, 2)))
    S = skan.Skeleton(skel)
    g = S.graph.tocoo()
    sel = g.row < g.col
    coords = np.asarray(S.coordinates)
    width = 2.0 * radius_img[coords[:, 0].astype(int), coords[:, 1].astype(int)]
    return dm.branch_table(coords, width, np.stack([g.row[sel], g.col[sel]], 1), g.data[sel])


def mask_table(mask):
    return skel_table(skeleton_of(mask), dist_of(mask))


# ---------------------------------------------------------------------------
# pixel formats
def enc_mask_zlib(m):
    return zlib.compress(np.packbits(m > 0).tobytes(), 9)


def dec_mask_zlib(b, shape):
    return np.unpackbits(np.frombuffer(zlib.decompress(b), np.uint8))[:shape[0] * shape[1]].reshape(shape)


def enc_png(m):
    from PIL import Image
    f = io.BytesIO()
    Image.fromarray(((m > 0) * 255).astype(np.uint8)).convert('1').save(f, 'PNG', optimize=True)
    return f.getvalue()


def dec_pil(b, shape):
    from PIL import Image
    return (np.asarray(Image.open(io.BytesIO(b)).convert('L')) > 0).astype(np.uint8)


def enc_webp(m):
    from PIL import Image
    f = io.BytesIO()
    Image.fromarray(((m > 0) * 255).astype(np.uint8)).save(f, 'WEBP', lossless=True, quality=100, method=6)
    return f.getvalue()


def _pbm(m):
    """Raw PBM (P4); PBM 1 = black = foreground."""
    h, w = m.shape
    return b'P4\n%d %d\n' % (w, h) + np.packbits(m > 0, axis=1).tobytes()


def _read_pbm(b):
    parts, o = [], 0
    while len(parts) < 3:                       # magic, width, height
        while b[o:o + 1].isspace():
            o += 1
        if b[o:o + 1] == b'#':
            o = b.index(b'\n', o) + 1
            continue
        e = o
        while not b[e:e + 1].isspace():
            e += 1
        parts.append(b[o:e])
        o = e
    o += 1
    w, h = int(parts[1]), int(parts[2])
    rows = np.frombuffer(b[o:o + h * ((w + 7) // 8)], np.uint8).reshape(h, -1)
    return np.unpackbits(rows, axis=1)[:, :w]


def jbig_codec():
    if _tool('jbig2') and _tool('jbig2dec'):
        return 'jbig2'
    if _tool('pbmtojbg') and _tool('jbgtopbm'):
        return 'jbig1'
    return None


def enc_jbig(m, codec, tmp):
    src = os.path.join(tmp, 'm.pbm')
    open(src, 'wb').write(_pbm(m))
    if codec == 'jbig2':                        # generic region (no -s): lossless
        return subprocess.run([_tool('jbig2'), src], capture_output=True, check=True).stdout
    out = os.path.join(tmp, 'm.jbg')
    subprocess.run([_tool('pbmtojbg'), src, out], capture_output=True, check=True)
    return open(out, 'rb').read()


def dec_jbig(b, codec, tmp):
    src = os.path.join(tmp, 'd.jb')
    open(src, 'wb').write(b)
    out = os.path.join(tmp, 'd.pbm')
    if codec == 'jbig2':
        subprocess.run([_tool('jbig2dec'), '-t', 'pbm', '-o', out, src], capture_output=True, check=True)
    else:
        subprocess.run([_tool('jbgtopbm'), src, out], capture_output=True, check=True)
    return _read_pbm(open(out, 'rb').read())


# ---------------------------------------------------------------------------
# skeleton chain code
def enc_chain(m):
    import skan
    skel = skeleton_of(m)
    rad = dist_of(m)
    H, W = m.shape
    buf = bytearray()
    _uv(buf, H)
    _uv(buf, W)
    covered = np.zeros_like(skel)
    paths = []
    if skel.sum() >= 2:
        S = skan.Skeleton(skel)
        coords = np.asarray(S.coordinates).astype(np.int64)
        paths = [coords[np.asarray(S.path(i))] for i in range(S.n_paths)]
    for p in paths:
        covered[p[:, 0], p[:, 1]] = True
    isolated = np.argwhere(skel & ~covered)
    paths.sort(key=lambda p: (p[0][0], p[0][1]))
    _uv(buf, len(paths))
    prev = (0, 0)
    codes = []
    for p in paths:
        _sv(buf, int(p[0][0]) - prev[0])
        _sv(buf, int(p[0][1]) - prev[1])
        prev = (int(p[0][0]), int(p[0][1]))
        _uv(buf, len(p) - 1)
        codes += [DIR_CODE[(int(d[0]), int(d[1]))] for d in np.diff(p, axis=0)]
    _uv(buf, len(isolated))
    prev = (0, 0)
    for y, x in isolated:
        _sv(buf, int(y) - prev[0])
        _sv(buf, int(x) - prev[1])
        prev = (int(y), int(x))
    # 3-bit codes, packed
    bits = np.unpackbits(np.array(codes, np.uint8)[:, None], axis=1)[:, 5:].ravel() if codes else np.zeros(0, np.uint8)
    buf += np.packbits(bits).tobytes()
    # radii in raster order of the skeleton, 1/4 px, delta coded
    r = np.clip(np.round(rad[skel] * RQ), 0, 255).astype(np.int64)
    for v in np.diff(r, prepend=0):
        _sv(buf, int(v))
    return zlib.compress(bytes(buf), 9)


def dec_chain(b):
    d = zlib.decompress(b)
    o = 0
    H, o = _ruv(d, o)
    W, o = _ruv(d, o)
    n, o = _ruv(d, o)
    starts, lens, prev = [], [], (0, 0)
    for _ in range(n):
        dy, o = _rsv(d, o)
        dx, o = _rsv(d, o)
        prev = (prev[0] + dy, prev[1] + dx)
        L, o = _ruv(d, o)
        starts.append(prev)
        lens.append(L)
    ni, o = _ruv(d, o)
    iso, prev = [], (0, 0)
    for _ in range(ni):
        dy, o = _rsv(d, o)
        dx, o = _rsv(d, o)
        prev = (prev[0] + dy, prev[1] + dx)
        iso.append(prev)
    nc = sum(lens)
    nb = (3 * nc + 7) // 8
    codes = np.packbits(np.concatenate([np.zeros((nc, 5), np.uint8),
                                        np.unpackbits(np.frombuffer(d[o:o + nb], np.uint8))[:3 * nc].reshape(nc, 3)],
                                       1), axis=1).ravel() if nc else []
    o += nb
    skel = np.zeros((H, W), bool)
    k = 0
    for (y, x), L in zip(starts, lens):
        skel[y, x] = True
        for c in codes[k:k + L]:
            y, x = y + DIRS[c][0], x + DIRS[c][1]
            skel[y, x] = True
        k += L
    for y, x in iso:
        skel[y, x] = True
    r = []
    for _ in range(int(skel.sum())):
        v, o = _rsv(d, o)
        r.append(v)
    rad = np.zeros((H, W), np.float32)
    rad[skel] = np.cumsum(r) / RQ
    return skel, rad


# ---------------------------------------------------------------------------
# v7 structure layer and SWC
def v7_structure(m, eps):
    from nanograph_v4 import graph_branch as gb
    return gb.branch_structure(skeleton_of(m), dist_of(m), eps=eps)


def swc_from_structure(st):
    """SWC (n T x y z R parent) of the branch graph: one tree per component,
    BFS from its first vertex; every non-tree edge (u, v) adds a duplicate of
    v whose parent is u, listed in a '# DUP <new> <orig>' header line."""
    from nanograph_v4 import graph_branch as gb
    pos, rad, edges, _ = gb.structure_to_arrays(st)
    n = len(pos)
    adj = [[] for _ in range(n)]
    for e, (u, v) in enumerate(edges):       # edge ids: parallel edges and
        adj[u].append((v, e))                # self-loops are cycles too
        if u != v:
            adj[v].append((u, e))
    ids, lines, dups, seen_edge = {}, [], [], set()
    nxt = 1
    for root in range(n):
        if root in ids:
            continue
        ids[root] = nxt
        lines.append((nxt, root, -1))
        nxt += 1
        queue = [root]
        while queue:
            u = queue.pop(0)
            for v, e in adj[u]:
                if e in seen_edge:
                    continue
                seen_edge.add(e)
                if v not in ids:
                    ids[v] = nxt
                    lines.append((nxt, v, ids[u]))
                    nxt += 1
                    queue.append(v)
                else:                               # closes a cycle
                    lines.append((nxt, v, ids[u]))
                    dups.append((nxt, ids[v]))
                    nxt += 1
    out = ['# Nanograph branch graph; x = column, y = row (px)']
    out += [f'# DUP {a} {b}' for a, b in dups]
    for k, v, par in lines:
        out.append(f'{k} 3 {int(pos[v][1])} {int(pos[v][0])} 0 {rad[v]:.2f} {par}')
    return ('\n'.join(out) + '\n').encode(), len(dups)


def read_swc(b):
    """SWC -> (pos, radius, edges) with duplicated nodes merged."""
    alias, rows = {}, []
    for ln in b.decode().splitlines():
        if ln.startswith('# DUP'):
            _, _, a, o = ln.split()
            alias[int(a)] = int(o)
        elif ln and not ln.startswith('#'):
            k, _, x, y, _, r, p = ln.split()
            rows.append((int(k), float(y), float(x), float(r), int(p)))
    keep = [r for r in rows if r[0] not in alias]
    idx = {r[0]: i for i, r in enumerate(keep)}
    res = lambda k: idx[alias.get(k, k)]
    pos = np.array([(r[1], r[2]) for r in keep], float).reshape(-1, 2)
    rad = np.array([r[3] for r in keep], float)
    edges = np.array([(res(r[0]), res(r[4])) for r in rows if r[4] != -1], np.int64).reshape(-1, 2)
    return pos, rad, edges


# ---------------------------------------------------------------------------
def measure(args):
    dataset, path, codec = args
    import downstream_morphometry as dm
    from nanograph_v4 import graph_branch as gb
    m = (cv2.imread(path, cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
    shape = m.shape
    ref = dm.descriptors_at(mask_table(m), 0.0, 'degree')[0]
    rows = []

    def row(fmt, data, decode, lossless_mask=None, external=False, **extra):
        t0 = time.perf_counter()
        tab = decode(data)
        dm.descriptors_at(tab, 'auto', 'degree')
        dt = time.perf_counter() - t0
        d = dm.descriptors_at(tab, 0.0, 'degree')[0]
        r = {'dataset': dataset, 'id': os.path.splitext(os.path.basename(path))[0], 'format': fmt,
             'bytes': len(data), 'decode_descr_s': dt, 'lossless_mask': lossless_mask, 'external': external,
             'fg_px': int(m.sum()), **extra}
        for k in dm.DESCRIPTORS:
            a, b = d[k], ref[k]
            same = (a == b) or (a != a and b != b)
            r[f'exact_{k}'] = bool(same)
            r[f'relerr_{k}'] = 0.0 if same else (abs(a - b) / abs(b) if b else np.inf)
        rows.append(r)

    def via_mask(dec):
        def f(b):
            mm = dec(b)
            assert np.array_equal(mm > 0, m > 0), (dataset, path)
            return mask_table(mm)
        return f

    row('mask_zlib', enc_mask_zlib(m), via_mask(lambda b: dec_mask_zlib(b, shape)), True)
    row('mask_png', enc_png(m), via_mask(lambda b: dec_pil(b, shape)), True)
    row('mask_webp_lossless', enc_webp(m), via_mask(lambda b: dec_pil(b, shape)), True)
    if codec:
        with tempfile.TemporaryDirectory() as tmp:
            jb = enc_jbig(m, codec, tmp)
            row('mask_jbig2' if codec == 'jbig2' else 'mask_jbig1', jb,
                via_mask(lambda b: dec_jbig(b, codec, tmp)), True, external=True, codec=codec)
    # skeleton chain code: identical skeleton -> identical skan decomposition
    ch = enc_chain(m)
    skel0 = skeleton_of(m)

    def dec_chain_table(b):
        sk, rad = dec_chain(b)
        assert np.array_equal(sk, skel0), (dataset, path)
        return skel_table(sk, rad)
    row('skel_chain', ch, dec_chain_table, False)
    # v7 structure layers
    for eps, fmt in ((0.75, V7), (3.0, 'v7_struct_e3')):
        st = v7_structure(m, eps)
        sb = gb.encode_structure(st)
        row(fmt, sb, lambda b: dm.graph_arm_table(*gb.structure_to_arrays(gb.decode_structure(b), True)), False)
        if eps == 0.75:
            swc, ndup = swc_from_structure(st)
    row('swc', swc, lambda b: dm.graph_arm_table(*read_swc(b)), False, swc_dup_nodes=ndup)
    row('swc_gz', gzip.compress(swc, 9, mtime=0), lambda b: dm.graph_arm_table(*read_swc(gzip.decompress(b))),
        False, swc_dup_nodes=ndup)
    return rows


def summarise(df):
    import pandas as pd
    rng = np.random.default_rng(0)
    out = []
    for ds, g in df.groupby('dataset'):
        w = g.pivot_table(index='id', columns='format', values='bytes')
        for fmt in w.columns:
            x = g[g.format == fmt]
            v, b = w[V7].to_numpy(float), w[fmt].to_numpy(float)
            ok = np.isfinite(v) & np.isfinite(b)
            v, b = v[ok], b[ok]
            bs = []
            for _ in range(2000):
                i = rng.integers(0, len(v), len(v))
                bs.append(v[i].mean() / b[i].mean())
            r = {'dataset': ds, 'format': fmt, 'n': len(x), 'bytes_mean': x.bytes.mean(),
                 'bytes_median': x.bytes.median(), 'ratio_v7_over_format': v.mean() / b.mean(),
                 'ratio_ci_lo': np.percentile(bs, 2.5), 'ratio_ci_hi': np.percentile(bs, 97.5),
                 'decode_descr_ms_median': 1000 * x.decode_descr_s.median(),
                 'external_process': bool(x.external.any()),
                 'swc_dup_nodes_mean': x.swc_dup_nodes.mean() if 'swc_dup_nodes' in x else np.nan}
            for c in [c for c in x.columns if c.startswith('exact_')]:
                r[c] = x[c].mean()
            out.append(r)
    return pd.DataFrame(out)


def stamp_commit(out):
    import subprocess
    h = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True,
                       cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
    open(os.path.join(out, 'commit.txt'), 'w').write(h + '\n')


def main():
    import multiprocessing as mp
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results/paper/lossless')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--limit', type=int, default=0, help='images per dataset (0 = all)')
    a = ap.parse_args()
    codec = jbig_codec()
    print('bilevel codec:', codec or 'NONE (JBIG rows skipped)', flush=True)
    jobs = []
    for ds, d in DATASETS:
        ps = sorted(glob.glob(os.path.join(d, '*.png')))
        jobs += [(ds, p, codec) for p in (ps[:a.limit] if a.limit else ps)]
    with mp.get_context('spawn').Pool(a.workers) as pool:
        rows = [r for rs in pool.imap_unordered(measure, jobs, chunksize=4) for r in rs]
    df = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    df.to_csv(os.path.join(a.out, 'per_image.csv'), index=False)
    stamp_commit(a.out)
    S = summarise(df)
    S.to_csv(os.path.join(a.out, 'summary.csv'), index=False)
    pd.set_option('display.width', 200)
    print(S[['dataset', 'format', 'n', 'bytes_mean', 'ratio_v7_over_format', 'ratio_ci_lo', 'ratio_ci_hi',
             'decode_descr_ms_median']].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
