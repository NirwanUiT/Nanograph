#!/usr/bin/env python3
"""
T10 — downstream morphometry: descriptors read from the stored graph vs the
same descriptors re-derived from pixels.

Stage `encode` (parallel, GPU ok) re-encodes every image with the default
config and caches, per image:
  payload_tagged  the payload exactly as the tagged codec writes it
  payload_fixed   the same payload with graph edges mapped to the correct
                  point indices (see _remap_graph_to_points)
  PRE graph       encoder-side graph before compression (positions, radii,
                  edges, traced path lengths)
  segmenter, n_misaligned (graph nodes whose id != their point index)

Usage (from repo root):
    python experiments/downstream_morphometry.py \
        --images /mnt/nas1/nba055-2/idea_1/nmi_data/org \
        --masks  /mnt/nas1/nba055-2/idea_1/nmi_data/seg \
        --runs results/paper/org_default --out results/paper/downstream \
        --stage encode --workers 16
"""
import argparse
import contextlib
import copy
import glob
import io
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# ---------------------------------------------------------------------------
# descriptors: ONE definition, applied to every arm's node/edge graph
# ---------------------------------------------------------------------------
# Pixel arms (REF, JPEG): nodes = skeleton pixels of skan.Skeleton, edges =
# skan's 8-neighbour pixel graph. Graph arms (GRAPH, PRE): nodes/edges of the
# Nanograph. Every arm then goes through branch_descriptors():
#   - degree-0 nodes are ignored (single-pixel skeletons; in GRAPH, the
#     optimizer's PSF points, which carry no edges);
#   - edge length = Euclidean distance between its two node positions (on the
#     pixel graph this is skan's branch-distance; PRE uses the encoder's traced
#     path length, which is what the encoder-side graph holds);
#   - width at a node = distance-transform DIAMETER = 2 x (distance to the
#     nearest background pixel). The payload stores the radius (the encoder's
#     dist_transform value), so GRAPH/PRE widths are 2 x stored value;
#   - junction = a connected cluster of degree>=3 nodes (a branch point is
#     often 2-4 mutually adjacent degree-3 pixels); clusters are contracted to
#     one vertex and their internal links dropped, in every arm;
#   - branch = maximal chain between non-degree-2 vertices; a component with
#     no such vertex is one closed branch (skan type 3);
#   - branch type as skan: 0 end-end, 1 junction-end, 2 junction-junction,
#     3 cycle (starts and ends on the same vertex, or has no vertex); all
#     four types count towards length and branch totals;
#   - cycle_rank = n_branches - n_branch_vertices + n_components on the
#     contracted graph (an isolated cycle is 1 branch on 1 vertex).
def branch_descriptors(pos, width, edges, edge_len=None, contract=True):
    """Descriptors of an undirected graph.

    pos (n,2) float, width (n,) node diameters, edges (m,2) int,
    edge_len (m,) optional (default: Euclidean between endpoints).
    Returns (dict of scalars, list of branch lengths, list of branch types).
    """
    pos = np.asarray(pos, float).reshape(-1, 2)
    width = np.asarray(width, float)
    edges = np.asarray(edges, np.int64).reshape(-1, 2)
    edges = edges[edges[:, 0] != edges[:, 1]]
    if edge_len is None:
        edge_len = np.hypot(*(pos[edges[:, 0]] - pos[edges[:, 1]]).T)
    edge_len = np.asarray(edge_len, float).reshape(-1)
    n = len(pos)
    adj = [[] for _ in range(n)]            # (neighbour, edge index)
    for k, (u, v) in enumerate(edges):
        adj[u].append((v, k))
        adj[v].append((u, k))
    deg = np.array([len(a) for a in adj])

    # junction clusters -> vertex label; every non-degree-2 node is a vertex
    vlabel = -np.ones(n, np.int64)
    nv = 0
    for s in range(n):
        if deg[s] == 0 or deg[s] == 2 or vlabel[s] >= 0:
            continue
        vlabel[s] = nv
        if deg[s] >= 3 and contract:
            stack = [s]
            while stack:
                u = stack.pop()
                for w, _ in adj[u]:
                    if deg[w] >= 3 and vlabel[w] < 0:
                        vlabel[w] = nv
                        stack.append(w)
        nv += 1
    vdeg = np.zeros(nv, np.int64)            # degree of contracted vertex
    vjunc = np.zeros(nv, bool)
    for u in range(n):
        if vlabel[u] >= 0:
            vjunc[vlabel[u]] |= deg[u] >= 3
            vdeg[vlabel[u]] += sum(1 for w, _ in adj[u] if vlabel[w] != vlabel[u])

    used = np.zeros(len(edges), bool)
    for k, (u, v) in enumerate(edges):       # intra-cluster links
        if vlabel[u] >= 0 and vlabel[u] == vlabel[v]:
            used[k] = True
    lengths, types, wsum = [], [], 0.0
    ends = []                                # (vertex a, vertex b) per branch

    def walk(u, w, k):
        """Walk from vertex-node u through edge k to w until the next vertex."""
        L, W = edge_len[k], edge_len[k] * (width[u] + width[w]) / 2
        used[k] = True
        prev, cur = u, w
        while vlabel[cur] < 0:               # degree-2 node: continue
            nxt = [(x, kk) for x, kk in adj[cur] if not used[kk]]
            if not nxt:                      # closed back onto start
                break
            x, kk = nxt[0]
            used[kk] = True
            L += edge_len[kk]
            W += edge_len[kk] * (width[cur] + width[x]) / 2
            prev, cur = cur, x
        return cur, L, W

    for u in range(n):
        if vlabel[u] < 0:
            continue
        for w, k in adj[u]:
            if used[k]:
                continue
            end, L, W = walk(u, w, k)
            a, b = vlabel[u], vlabel[end]
            ja, jb = vjunc[a], vjunc[b]
            types.append(3 if a == b else 2 if (ja and jb) else 1 if (ja or jb) else 0)
            lengths.append(L)
            wsum += W
            ends.append((a, b))
    # isolated cycles: remaining edges lie on components with no vertex
    n_cyc_only = 0
    for k in range(len(edges)):
        if used[k]:
            continue
        u, w = edges[k]
        start = u
        L, W = edge_len[k], edge_len[k] * (width[u] + width[w]) / 2
        used[k] = True
        cur = w
        while cur != start:
            nxt = [(x, kk) for x, kk in adj[cur] if not used[kk]]
            if not nxt:
                break
            x, kk = nxt[0]
            used[kk] = True
            L += edge_len[kk]
            W += edge_len[kk] * (width[cur] + width[x]) / 2
            cur = x
        lengths.append(L)
        types.append(3)
        wsum += W
        n_cyc_only += 1

    # components over nodes with degree >= 1
    comp = -np.ones(n, np.int64)
    n_comp = 0
    for s in range(n):
        if deg[s] == 0 or comp[s] >= 0:
            continue
        comp[s] = n_comp
        stack = [s]
        while stack:
            u = stack.pop()
            for w, _ in adj[u]:
                if comp[w] < 0:
                    comp[w] = n_comp
                    stack.append(w)
        n_comp += 1

    total = float(np.sum(lengths)) if lengths else 0.0
    n_br = len(lengths)
    d = {
        'n_components': n_comp,
        'total_length_px': total,
        'mean_width_px': wsum / total if total > 0 else np.nan,
        'n_branches': n_br,
        'n_junctions': int(vjunc.sum()),
        'cycle_rank': int(n_br - (nv + n_cyc_only) + n_comp),
        # auxiliary (not among the seven)
        'n_junction_nodes': int((deg >= 3).sum()),
        'n_endpoints': int((deg == 1).sum()),
        'n_isolated_cycles': n_cyc_only,
    }
    return d, [float(x) for x in lengths], types


def pixel_arm_descriptors(mask, contract=True, timings=None):
    """Annotation/segmentation mask -> skeletonize -> skan pixel graph ->
    branch_descriptors. Widths = 2 x cv2 L2 distance transform (as encoder).
    If `timings` is a dict, the skeleton+skan and descriptor times go in it."""
    import time
    import cv2
    import skan
    from skimage.morphology import skeletonize
    t0 = time.perf_counter()
    fg = (mask > 0).astype(np.uint8)
    skel = skeletonize(fg > 0)
    if skel.sum() < 2:
        return branch_descriptors(np.zeros((0, 2)), [], np.zeros((0, 2)))
    dt = cv2.distanceTransform(fg, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    S = skan.Skeleton(skel)
    g = S.graph.tocoo()
    sel = g.row < g.col
    edges = np.stack([g.row[sel], g.col[sel]], 1)
    coords = np.asarray(S.coordinates)
    width = 2.0 * dt[coords[:, 0].astype(int), coords[:, 1].astype(int)]
    t1 = time.perf_counter()
    out = branch_descriptors(coords, width, edges, g.data[sel], contract=contract)
    if timings is not None:
        timings['skeleton_skan_s'] = t1 - t0
        timings['descriptors_s'] = time.perf_counter() - t1
    return out


def graph_arm_descriptors(nodes_pos, radius, edges, edge_len=None):
    """Nanograph nodes/edges (stored radius) -> branch_descriptors."""
    return branch_descriptors(nodes_pos, 2.0 * np.asarray(radius, float), edges, edge_len)


# ---------------------------------------------------------------------------
# encode stage
# ---------------------------------------------------------------------------
def _remap_graph_to_points(graph, points):
    """Return (graph copy whose node ids index `points`, n_misaligned).

    build_nanograph gives node i the position of points[i], but
    remove_small_components() re-indexes the surviving nodes 0..k-1, after
    which graph node ids no longer index the point array that
    compress_nanograph serialises. Skeleton points precede optimizer points in
    `points` and are unique pixels, so the first index with a node's position
    is that node's point index.
    """
    first = {}
    for i, (y, x) in enumerate(points.astype(int)):
        first.setdefault((int(y), int(x)), i)
    id_map = {n.id: first[tuple(int(c) for c in n.position)] for n in graph.nodes}
    n_mis = sum(1 for k, v in id_map.items() if k != v)
    g = copy.deepcopy(graph)
    for n in g.nodes:
        n.id = id_map[n.id]
    for e in g.edges:
        e.source, e.target = id_map[e.source], id_map[e.target]
    g.adjacency = {id_map[k]: [id_map[v] for v in vs] for k, vs in graph.adjacency.items()}
    return g, n_mis


def _encode_one(args):
    img_path, cache_dir = args
    import cv2
    from nanograph_v4 import nanograph_encode, NanographConfig
    from nanograph_v4.compress import compress_nanograph

    stem = os.path.splitext(os.path.basename(img_path))[0]
    out = os.path.join(cache_dir, stem + '.npz')
    if os.path.exists(out):
        return stem, 'cached'
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    with contextlib.redirect_stdout(io.StringIO()):
        r = nanograph_encode(img, verbose=False, config=NanographConfig())
    g = r.graph
    g_fixed, n_mis = _remap_graph_to_points(g, r.points)

    def _recompress(graph):
        data, _ = compress_nanograph(
            r.points, r.intensities, r.widths, r.shape, r.types,
            orientations=r.orientations, bg_model=r.bg_grid, graph=graph,
            fg_residual=r.fg_residual_bytes if r.fg_residual_bytes else None,
            fg_residual_shape=r.fg_residual_shape_info, cfg=r.config)
        return data

    # Sanity: recompressing with the encoder's own graph reproduces the payload.
    assert _recompress(g) == r.compressed, f'{stem}: recompress mismatch'
    payload_fixed = _recompress(g_fixed) if n_mis else r.compressed

    pos = np.array([n.position for n in g.nodes], dtype=np.int32).reshape(-1, 2)
    rad = np.array([n.width for n in g.nodes], dtype=np.float64)
    edges = np.array([(e.source, e.target) for e in g.edges], dtype=np.int32).reshape(-1, 2)
    elen = np.array([e.length for e in g.edges], dtype=np.float64)
    np.savez_compressed(
        out,
        payload_tagged=np.frombuffer(r.compressed, np.uint8),
        payload_fixed=np.frombuffer(payload_fixed, np.uint8),
        pre_pos=pos, pre_rad=rad, pre_edges=edges, pre_elen=elen,
        segmenter=np.array(r.segmenter), n_misaligned=np.int64(n_mis),
        n_points=np.int64(len(r.points)))
    return stem, 'ok'


def stage_encode(args):
    import multiprocessing as mp
    cache_dir = os.path.join(args.out, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(args.images, '*.png')))
    if args.limit:
        paths = paths[:args.limit]
    ctx = mp.get_context('spawn')
    with ctx.Pool(args.workers) as pool:
        for i, (stem, status) in enumerate(pool.imap_unordered(
                _encode_one, [(p, cache_dir) for p in paths])):
            if i % 50 == 0:
                print(f'[{i + 1}/{len(paths)}] {stem} {status}', flush=True)


# ---------------------------------------------------------------------------
# measure stage (sequential, single-threaded: this is where timing happens)
# ---------------------------------------------------------------------------
DESCRIPTORS = ['n_components', 'total_length_px', 'mean_width_px',
               'n_branches', 'n_junctions', 'cycle_rank']


def segment_like_pipeline(img, segmenter, model, device='cpu'):
    """Run the named cascade candidate exactly as auto_segment builds it
    (polarity -> image type -> [preprocess] -> segmenter -> its clean-up)."""
    from nanograph_v4 import NanographConfig
    from nanograph_v4.detect import detect_polarity, detect_image_type
    from nanograph_v4.preprocess import preprocess
    from nanograph_v4.segment import (learned_segment, frangi_segment, meijering_segment,
                                      otsu_segment, morphological_clean)
    cfg = NanographConfig()
    if detect_polarity(img, cfg=cfg):
        img = 255 - img
    det = detect_image_type(img, cfg=cfg)
    cfg = cfg.for_image_type(det['type'])
    if segmenter == 'Learned':
        return learned_segment(model, img, cfg=cfg, device=device)
    _, bg_sub, _, _ = preprocess(img, bg_kernel_size=det['bg_kernel_size'], cfg=cfg)
    raw = {'Frangi': lambda: frangi_segment(bg_sub, cfg=cfg),
           'Meijering': lambda: meijering_segment(bg_sub, cfg=cfg),
           'Otsu': lambda: otsu_segment(bg_sub)}[segmenter]()
    return morphological_clean(raw, cfg=cfg)


def _decoded_graph_arrays(payload):
    from nanograph_v4 import decode_graph
    g = decode_graph(payload)
    pos = np.array([n.position for n in g.nodes], float).reshape(-1, 2)
    rad = np.array([n.width for n in g.nodes], float)
    edges = np.array([(e.source, e.target) for e in g.edges], np.int64).reshape(-1, 2)
    return pos, rad, edges


def stage_measure(args):
    import json
    import time
    import cv2
    import pandas as pd
    import torch
    from nanograph_v4.config import DEFAULT_CONFIG
    from nanograph_v4.evaluate import jpeg_for_budget
    from nanograph_v4.unet_seg import load_unet

    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    model = load_unet(DEFAULT_CONFIG.segment.learned_ckpt, device='cpu')
    cache_dir = os.path.join(args.out, 'cache')
    bl_dir = os.path.join(args.out, 'branch_lengths')
    os.makedirs(bl_dir, exist_ok=True)
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(cache_dir) if f.endswith('.npz'))
    if args.limit:
        stems = stems[:args.limit]
    primary = 'payload_' + args.payload
    other = 'payload_tagged' if args.payload == 'fixed' else 'payload_fixed'

    def load(stem):
        c = np.load(os.path.join(cache_dir, stem + '.npz'))
        img = cv2.imread(os.path.join(args.images, stem + '.png'), cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(os.path.join(args.masks, stem + '.png'), cv2.IMREAD_GRAYSCALE)
        return c, img, gt

    def arm_graph(payload, timings=None):
        t0 = time.perf_counter()
        arrays = _decoded_graph_arrays(payload)
        t1 = time.perf_counter()
        out = graph_arm_descriptors(*arrays)
        if timings is not None:
            timings['decode_graph_s'] = t1 - t0
            timings['descriptors_s'] = time.perf_counter() - t1
        return out

    def arm_jpeg(buf, segmenter):
        dec = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_GRAYSCALE)
        return pixel_arm_descriptors(segment_like_pipeline(dec, segmenter, model))

    # warm-up (imports, allocator, U-Net first call) on the first image
    c, img, gt = load(stems[0])
    pixel_arm_descriptors(gt)
    arm_graph(c[primary].tobytes())
    _, jb = jpeg_for_budget(img, len(c[primary]))
    if jb:
        arm_jpeg(jb, str(c['segmenter']))

    rows, timing = [], []
    t_start = time.time()
    for i, stem in enumerate(stems):
        c, img, gt = load(stem)
        seg = str(c['segmenter'])
        payload = c[primary].tobytes()
        _, png = cv2.imencode('.png', img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        jq, jbuf = jpeg_for_budget(img, len(payload))
        common = dict(stem=stem, segmenter=seg, n_misaligned=int(c['n_misaligned']),
                      payload_bytes=len(payload), jpeg_bytes=len(jbuf) if jbuf else np.nan,
                      jpeg_quality=jq if jbuf else np.nan, raw_bytes=int(img.size),
                      png_bytes=len(png))
        res, tt, split = {}, {}, {}

        t0 = time.perf_counter()
        sp = {}
        res['REF'] = pixel_arm_descriptors(gt, timings=sp)
        tt['REF'] = time.perf_counter() - t0
        split.update({f'ref_{k}': v for k, v in sp.items()})

        t0 = time.perf_counter()
        sp = {}
        res['GRAPH'] = arm_graph(payload, timings=sp)
        tt['GRAPH'] = time.perf_counter() - t0
        split.update({f'graph_{k}': v for k, v in sp.items()})

        if jbuf:
            t0 = time.perf_counter()
            dec = cv2.imdecode(np.frombuffer(jbuf, np.uint8), cv2.IMREAD_GRAYSCALE)
            t1 = time.perf_counter()
            jmask = segment_like_pipeline(dec, seg, model)
            t2 = time.perf_counter()
            sp = {}
            res['JPEG'] = pixel_arm_descriptors(jmask, timings=sp)
            tt['JPEG'] = time.perf_counter() - t0
            split.update(jpeg_decode_s=t1 - t0, jpeg_segment_s=t2 - t1,
                         **{f'jpeg_{k}': v for k, v in sp.items()})

        # untimed arms
        res['PRE'] = graph_arm_descriptors(c['pre_pos'], c['pre_rad'], c['pre_edges'],
                                           c['pre_elen'])
        res['GRAPH_' + other.split('_')[1].upper()] = arm_graph(c[other].tobytes())
        res['SEG'] = pixel_arm_descriptors(segment_like_pipeline(img, seg, model))

        for arm, (d, L, T) in res.items():
            rows.append({**common, 'arm': arm, **d,
                         'time_s': tt.get(arm, np.nan)})
            with open(os.path.join(bl_dir, f'{stem}_{arm}.json'), 'w') as f:
                json.dump({'branch_lengths': L, 'branch_types': T}, f)
        timing.append({'stem': stem, 'segmenter': seg,
                       **{f'{a.lower()}_s': v for a, v in tt.items()}, **split})
        if i % 50 == 0:
            el = time.time() - t_start
            print(f'[{i + 1}/{len(stems)}] {stem} {el:.0f}s', flush=True)

    pd.DataFrame(rows).to_csv(os.path.join(args.out, 'per_image.csv'), index=False)
    pd.DataFrame(timing).to_csv(os.path.join(args.out, 'timing.csv'), index=False)


# ---------------------------------------------------------------------------
# stats stage
# ---------------------------------------------------------------------------
# (arm, reference arm). REF is the ground truth; GRAPH|PRE isolates storage.
COMPARISONS = [('GRAPH', 'REF'), ('JPEG', 'REF'), ('PRE', 'REF'), ('SEG', 'REF'),
               ('GRAPH_TAGGED', 'REF'), ('GRAPH', 'PRE'), ('SEG', 'PRE')]


def ccc(x, y):
    """Lin's concordance correlation coefficient."""
    mx, my = x.mean(), y.mean()
    return 2 * np.mean((x - mx) * (y - my)) / (x.var() + y.var() + (mx - my) ** 2)


def agreement(x, y):
    """Agreement of measurement x with reference y (paired arrays)."""
    from scipy.stats import pearsonr
    d = x - y
    bias, sd = d.mean(), d.std(ddof=1)
    ym = y.mean()
    nz = y != 0
    return {
        'n': len(x), 'ref_mean': ym, 'arm_mean': x.mean(),
        'ccc': ccc(x, y), 'r': pearsonr(x, y)[0] if x.std() > 0 and y.std() > 0 else np.nan,
        'bias': bias, 'loa_lo': bias - 1.96 * sd, 'loa_hi': bias + 1.96 * sd,
        'bias_pct': 100 * bias / ym, 'loa_lo_pct': 100 * (bias - 1.96 * sd) / ym,
        'loa_hi_pct': 100 * (bias + 1.96 * sd) / ym,
        'mdape': 100 * np.median(np.abs(d[nz]) / np.abs(y[nz])), 'n_mdape': int(nz.sum()),
    }


def paired_error_test(a, b, ref):
    """|a-ref| vs |b-ref|: Wilcoxon signed-rank p (ties dropped) and counts."""
    from scipy.stats import wilcoxon
    ea, eb = np.abs(a - ref), np.abs(b - ref)
    wins, losses = int((ea < eb).sum()), int((ea > eb).sum())
    p = wilcoxon(ea, eb).pvalue if wins + losses > 0 else np.nan
    return {'wilcoxon_p': p, 'wins_graph': wins, 'wins_jpeg': losses,
            'ties': int(len(ea) - wins - losses)}


def branch_distance_table(out, stems, arms):
    import json
    from scipy.stats import wasserstein_distance, ks_2samp
    rows = []
    for s in stems:
        L = {}
        for a in ['REF'] + arms:
            with open(os.path.join(out, 'branch_lengths', f'{s}_{a}.json')) as f:
                L[a] = json.load(f)['branch_lengths']
        r = {'stem': s}
        for a in arms:
            if L['REF'] and L[a]:
                r[f'w1_{a}'] = wasserstein_distance(L[a], L['REF'])
                r[f'ks_{a}'] = ks_2samp(L[a], L['REF']).statistic
            else:
                r[f'w1_{a}'] = r[f'ks_{a}'] = np.nan
        rows.append(r)
    return __import__('pandas').DataFrame(rows)


def stage_stats(args):
    import platform
    import pandas as pd
    df = pd.read_csv(os.path.join(args.out, 'per_image.csv'), dtype={'stem': str})
    wide = {a: g.set_index('stem') for a, g in df.groupby('arm')}
    # Common image set: every arm defined (JPEG needs a quality that fits).
    stems = sorted(set.intersection(*(set(g.index) for g in wide.values())))
    rows = []
    for desc in DESCRIPTORS:
        for arm, ref in COMPARISONS:
            x = wide[arm].loc[stems, desc].to_numpy(float)
            y = wide[ref].loc[stems, desc].to_numpy(float)
            ok = np.isfinite(x) & np.isfinite(y)
            r = {'descriptor': desc, 'arm': arm, 'ref': ref, **agreement(x[ok], y[ok])}
            if ref == 'REF' and arm in ('GRAPH', 'JPEG'):
                g = wide['GRAPH'].loc[stems, desc].to_numpy(float)
                j = wide['JPEG'].loc[stems, desc].to_numpy(float)
                yy = wide['REF'].loc[stems, desc].to_numpy(float)
                ok3 = np.isfinite(g) & np.isfinite(j) & np.isfinite(yy)
                r.update(paired_error_test(g[ok3], j[ok3], yy[ok3]))
            rows.append(r)

    arms = ['GRAPH', 'JPEG', 'PRE', 'SEG']
    bd = branch_distance_table(args.out, stems, arms)
    bd.to_csv(os.path.join(args.out, 'branch_distances.csv'), index=False)
    from scipy.stats import wilcoxon
    for a in arms:
        r = {'descriptor': 'branch_lengths', 'arm': a, 'ref': 'REF',
             'n': int(bd[f'w1_{a}'].notna().sum()),
             'w1_mean': bd[f'w1_{a}'].mean(), 'w1_median': bd[f'w1_{a}'].median(),
             'ks_mean': bd[f'ks_{a}'].mean(), 'ks_median': bd[f'ks_{a}'].median()}
        if a in ('GRAPH', 'JPEG'):
            ok = bd['w1_GRAPH'].notna() & bd['w1_JPEG'].notna()
            for m in ('w1', 'ks'):
                g, j = bd.loc[ok, f'{m}_GRAPH'], bd.loc[ok, f'{m}_JPEG']
                r[f'{m}_wilcoxon_p'] = wilcoxon(g, j).pvalue
                r[f'{m}_wins_graph'] = int((g < j).sum())
                r[f'{m}_wins_jpeg'] = int((g > j).sum())
        rows.append(r)
    pd.DataFrame(rows).to_csv(os.path.join(args.out, 'summary.csv'), index=False)

    # cost: bytes (one row per image, from the REF rows) and median times
    t = pd.read_csv(os.path.join(args.out, 'timing.csv'), dtype={'stem': str})
    ref = wide['REF']
    cost = {'n_images': len(ref), 'n_common': len(stems)}
    for c in ('payload_bytes', 'jpeg_bytes', 'raw_bytes', 'png_bytes'):
        cost[f'{c}_mean'] = ref[c].mean()
        cost[f'{c}_median'] = ref[c].median()
    for c in t.columns:
        if c.endswith('_s'):
            cost[f'{c}_median'] = t[c].median()
    cost['ratio_jpeg_over_graph'] = cost['jpeg_s_median'] / cost['graph_s_median']
    cost['ratio_ref_over_graph'] = cost['ref_s_median'] / cost['graph_s_median']
    cost['ratio_jpeg_nosegment_over_graph'] = (
        (t['jpeg_s'] - t['jpeg_segment_s']).median() / cost['graph_s_median'])
    cost['cpu'] = _cpu_name()
    cost['python'] = platform.python_version()
    cost['threads'] = 'OMP/MKL/OPENBLAS_NUM_THREADS=1, torch.set_num_threads(1), cv2.setNumThreads(1), CPU only'
    pd.Series(cost).to_csv(os.path.join(args.out, 'cost_summary.csv'), header=['value'])
    print(pd.DataFrame(rows)[['descriptor', 'arm', 'ref', 'n', 'ccc', 'mdape', 'bias_pct',
                              'wilcoxon_p', 'wins_graph', 'wins_jpeg']].to_string())
    print(pd.Series(cost).to_string())


def _cpu_name():
    try:
        for line in open('/proc/cpuinfo'):
            if line.startswith('model name'):
                return line.split(':', 1)[1].strip()
    except OSError:
        pass
    import platform
    return platform.processor()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--images', default='/mnt/nas1/nba055-2/idea_1/nmi_data/org')
    ap.add_argument('--masks', default='/mnt/nas1/nba055-2/idea_1/nmi_data/seg')
    ap.add_argument('--runs', default='results/paper/org_default')
    ap.add_argument('--out', default='results/paper/downstream')
    ap.add_argument('--stage', default='all', choices=['encode', 'measure', 'stats', 'all'])
    ap.add_argument('--payload', default='fixed', choices=['fixed', 'tagged'],
                    help='payload the GRAPH arm (and the JPEG byte budget) uses')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()
    if args.stage in ('encode', 'all'):
        stage_encode(args)
    if args.stage in ('measure', 'all'):
        stage_measure(args)
    if args.stage in ('stats', 'all'):
        stage_stats(args)


if __name__ == '__main__':
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    main()
