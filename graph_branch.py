"""
Nanograph v7 (experimental) — branch-level graph construction and a separate
structure-layer codec.

Graph construction works on the skeleton's branch decomposition (skan) rather
than on greedily sampled pixels:
  * every connected cluster of degree>=3 skeleton pixels becomes ONE junction
    node (placed at the cluster pixel nearest its centroid), so a junction is
    a node with >= 3 branches by construction;
  * each branch's pixel path is simplified with Douglas-Peucker (tolerance
    `eps` px), with an extra vertex whenever a chord would exceed `max_seg`
    px, so nodes concentrate where the centreline bends;
  * node width is the mean distance-transform radius over the path pixels the
    node represents (half of each adjacent segment), or a width fitted to the
    image's intensity profile across the centreline (`width_mode='profile'`).

The structure layer stores only what analysis needs: junction/endpoint
vertices, and per branch its two end vertices and interior points as small
coordinate deltas, plus a width per point. No intensities, orientations or
render points (those belong to the appearance layer).
"""

import struct
import zlib

import numpy as np

from .graph import Nanograph, GraphNode, GraphEdge


# ---------------------------------------------------------------------------
# branch extraction
# ---------------------------------------------------------------------------
def _douglas_peucker(poly, eps):
    """Indices of the vertices kept by Douglas-Peucker on an (n,2) polyline."""
    n = len(poly)
    if n <= 2:
        return list(range(n))
    keep = np.zeros(n, bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = poly[i].astype(float), poly[j].astype(float)
        seg = b - a
        L = np.hypot(*seg)
        pts = poly[i + 1:j].astype(float)
        if L < 1e-9:
            d = np.hypot(*(pts - a).T)
        else:
            d = np.abs(seg[0] * (pts[:, 1] - a[1]) - seg[1] * (pts[:, 0] - a[0])) / L
        k = int(np.argmax(d))
        if d[k] > eps:
            m = i + 1 + k
            keep[m] = True
            stack += [(i, m), (m, j)]
    return list(np.flatnonzero(keep))


def _split_long(idx, poly, max_seg):
    """Insert path vertices so no chord between kept vertices exceeds max_seg."""
    out = [idx[0]]
    for a, b in zip(idx[:-1], idx[1:]):
        while np.hypot(*(poly[b] - poly[a]).astype(float)) > max_seg and b - a > 1:
            m = a + max(1, int(round(max_seg)))
            m = min(m, b - 1)
            out.append(m)
            a = m
        out.append(b)
    return out


def extract_branches(skel, eps=0.75, max_seg=8.0):
    """Skeleton -> (vertices, branches).

    vertices: dict key -> (row, col, kind) with kind 'junction' | 'endpoint'
    branches: list of dicts {a, b: vertex keys, path: (n,2) pixel polyline
              from vertex a to vertex b, keep: indices into path kept as nodes}
    """
    import skan
    skel = np.asarray(skel) > 0
    if skel.sum() < 2:
        return {}, []
    S = skan.Skeleton(skel)
    coords = np.asarray(S.coordinates).astype(np.int64)
    deg = np.asarray(S.degrees)
    g = S.graph.tocsr()
    junc = deg >= 3
    cluster = -np.ones(len(coords), np.int64)
    reps = []
    for s in np.flatnonzero(junc):
        if cluster[s] >= 0:
            continue
        cid = len(reps)
        members, stack = [s], [s]
        cluster[s] = cid
        while stack:
            u = stack.pop()
            for w in g.indices[g.indptr[u]:g.indptr[u + 1]]:
                if junc[w] and cluster[w] < 0:
                    cluster[w] = cid
                    members.append(w)
                    stack.append(w)
        c = coords[members].mean(0)
        reps.append(members[int(np.argmin(np.hypot(*(coords[members] - c).T)))])

    vertices = {}

    def vkey(pix):
        if junc[pix]:
            cid = int(cluster[pix])
            vertices.setdefault(('J', cid), (*coords[reps[cid]], 'junction'))
            return ('J', cid)
        vertices.setdefault(('P', int(pix)), (*coords[pix], 'endpoint'))
        return ('P', int(pix))

    branches = []
    for i in range(S.n_paths):
        ids = np.asarray(S.path(i))
        if junc[ids].all() and len({int(cluster[k]) for k in ids}) == 1:
            continue                      # link inside one junction cluster
        a, b = vkey(ids[0]), vkey(ids[-1])
        path = coords[ids].copy()
        # branch length on the original pixel path (as skan measures it),
        # before its junction-pixel ends move to the cluster representative
        plen0 = float(np.hypot(*np.diff(path, axis=0).T.astype(float)).sum())
        # replace junction-pixel ends by the cluster representative
        if a[0] == 'J':
            path[0] = vertices[a][:2]
        if b[0] == 'J':
            path[-1] = vertices[b][:2]
        keep = _split_long(_douglas_peucker(path, eps), path, max_seg)
        if len(keep) == 2 and a[0] == 'J' and b[0] == 'J' and len(path) > 2:
            # two junction nodes must never be adjacent: a junction-to-junction
            # branch keeps its middle pixel, or it would read as one junction
            keep = [0, len(path) // 2, len(path) - 1]
        branches.append({'a': a, 'b': b, 'path': path, 'keep': keep, 'plen': plen0})
    # pure cycles have no end vertex of degree != 2: give them one
    for br in branches:
        for end in ('a', 'b'):
            k = br[end]
            if k[0] == 'P' and deg[k[1]] == 2:
                vertices[k] = (*vertices[k][:2], 'loop')
    return vertices, branches


# ---------------------------------------------------------------------------
# width estimators
# ---------------------------------------------------------------------------
def _node_widths_dt(path, keep, dist_transform):
    """Mean DT radius over the path pixels each kept vertex represents."""
    dt = dist_transform[path[:, 0], path[:, 1]].astype(float)
    out = []
    for n, k in enumerate(keep):
        lo = (keep[n - 1] + k) // 2 if n > 0 else k
        hi = (k + keep[n + 1]) // 2 if n + 1 < len(keep) else k
        out.append(float(dt[lo:hi + 1].mean()))
    return np.array(out)


def profile_width(img, y, x, tangent, r0, half=None):
    """FWHM/2 (a radius) of a Gaussian fitted to the intensity profile across
    the centreline at (y, x). `tangent` is a unit (dy, dx); r0 an initial
    radius guess. Returns np.nan if the fit fails."""
    from scipy.ndimage import map_coordinates
    from scipy.optimize import curve_fit
    if half is None:
        half = max(6.0, 3.0 * r0 + 3.0)
    ny, nx = -tangent[1], tangent[0]              # normal
    t = np.arange(-half, half + 0.5, 0.5)
    prof = map_coordinates(img.astype(float), [y + t * ny, x + t * nx], order=1, mode='nearest')

    def f(t, amp, mu, sig, off):
        return off + amp * np.exp(-0.5 * ((t - mu) / sig) ** 2)
    try:
        p0 = [prof.max() - prof.min(), 0.0, max(r0 / 1.1774, 0.7), prof.min()]
        p, _ = curve_fit(f, t, prof, p0=p0, maxfev=400,
                         bounds=([0, -2, 0.3, -np.inf], [np.inf, 2, half, np.inf]))
        return float(1.1774 * p[2])            # FWHM / 2 = sqrt(2 ln 2) * sigma
    except Exception:
        return np.nan


# ---------------------------------------------------------------------------
# graph object
# ---------------------------------------------------------------------------
def branch_structure(skel, dist_transform, eps=0.75, max_seg=8.0,
                     width_mode='dt', img=None):
    """Skeleton -> structure dict (vertex table + branch polylines + widths),
    the unit that the structure layer serialises."""
    vertices, branches = extract_branches(skel, eps=eps, max_seg=max_seg)
    vkeys = list(vertices)
    vidx = {k: i for i, k in enumerate(vkeys)}
    vpos = np.array([vertices[k][:2] for k in vkeys], np.int64).reshape(-1, 2)
    vrad = np.array([float(dist_transform[y, x]) for y, x in vpos])
    vkind = [vertices[k][2] for k in vkeys]
    out_br = []
    for br in branches:
        path, keep = br['path'], br['keep']
        w = _node_widths_dt(path, keep, dist_transform)
        if width_mode == 'profile' and img is not None:
            for n, k in enumerate(keep):
                a, b = path[max(k - 2, 0)], path[min(k + 2, len(path) - 1)]
                tg = (b - a).astype(float)
                nrm = np.hypot(*tg)
                if nrm > 0:
                    pw = profile_width(img, path[k][0], path[k][1], tg / nrm, w[n])
                    if np.isfinite(pw):
                        w[n] = pw
        seg = [float(np.hypot(*np.diff(path[i:j + 1], axis=0).T.astype(float)).sum())
               for i, j in zip(keep[:-1], keep[1:])]
        tot = sum(seg)
        seg = [x * br['plen'] / tot for x in seg] if tot > 0 else seg   # sum = original length
        out_br.append({'a': vidx[br['a']], 'b': vidx[br['b']],
                       'pts': path[keep], 'rad': w, 'plen': br['plen'], 'seg_plen': seg})
    # vertex width: mean of the adjacent branch-end widths (junction pixels'
    # own DT over-estimates at the crossing)
    acc = [[] for _ in vkeys]
    for br in out_br:
        acc[br['a']].append(br['rad'][0])
        acc[br['b']].append(br['rad'][-1])
    for i, a in enumerate(acc):
        if a:
            vrad[i] = float(np.mean(a))
    return {'shape': tuple(skel.shape), 'vpos': vpos, 'vrad': vrad, 'vkind': vkind,
            'branches': out_br}


def structure_to_arrays(st, path_lengths=False):
    """Structure dict -> (pos, radius, edges, edge_len or None): vertices first,
    then each branch's interior points; edges join consecutive points.
    path_lengths=True gives each edge its share of the branch's pixel-path
    length (exact per segment on the encoder side; the stored branch length
    split in proportion to the chords after decoding)."""
    pos = [p for p in st['vpos']]
    rad = list(st['vrad'])
    edges, elen = [], []
    for br in st['branches']:
        chain = [br['a']]
        for p, r in zip(br['pts'][1:-1], br['rad'][1:-1]):
            pos.append(p)
            rad.append(r)
            chain.append(len(pos) - 1)
        chain.append(br['b'])
        chords = [float(np.hypot(*(np.asarray(p, float) - q))) for p, q in
                  zip(br['pts'][:-1], np.asarray(br['pts'][1:], float))]
        for n, (u, v) in enumerate(zip(chain[:-1], chain[1:])):
            edges.append((u, v))
            if path_lengths:
                if 'seg_plen' in br:
                    elen.append(br['seg_plen'][n])
                else:
                    tot = sum(chords)
                    elen.append(br['plen'] * chords[n] / tot if tot > 0 else br['plen'] / len(chords))
    pos = np.array(pos, float).reshape(-1, 2)
    edges = np.array(edges, np.int64).reshape(-1, 2)
    return pos, np.array(rad, float), edges, (np.array(elen) if path_lengths else None)


def structure_to_nanograph(st, image_type='decoded'):
    """Nanograph object; edge lengths are pixel-path lengths (exact on the
    encoder side, the stored branch length shared by chord after decoding)."""
    pos, rad, edges, elen = structure_to_arrays(st, path_lengths=True)
    kinds = list(st['vkind']) + ['sampled'] * (len(pos) - len(st['vkind']))
    adj = {}
    for u, v in edges:
        adj.setdefault(int(u), []).append(int(v))
        adj.setdefault(int(v), []).append(int(u))
    nodes = [GraphNode(id=i, position=(int(p[0]), int(p[1])), width=float(r), intensity=0.0,
                       orientation=0.0, node_type=('junction' if k == 'junction' else
                                                   'endpoint' if k == 'endpoint' else 'sampled'),
                       degree=len(adj.get(i, [])))
             for i, (p, r, k) in enumerate(zip(pos, rad, kinds))]
    E = [GraphEdge(id=k, source=int(u), target=int(v),
                   length=float(elen[k]), pixel_count=0,
                   mean_width=(rad[u] + rad[v]) / 2, mean_intensity=0.0, curvature=0.0)
         for k, (u, v) in enumerate(edges)]
    return Nanograph(nodes=nodes, edges=E, adjacency=adj, shape=st['shape'], image_type=image_type)


# ---------------------------------------------------------------------------
# structure-layer codec
# ---------------------------------------------------------------------------
def _uv(buf, u):
    while u >= 0x80:
        buf.append((u & 0x7F) | 0x80)
        u >>= 7
    buf.append(u)


def _sv(buf, v):
    _uv(buf, (v << 1) ^ (v >> 63) if v < 0 else v << 1)


def _ruv(d, o):
    s = r = 0
    while True:
        b = d[o]
        o += 1
        r |= (b & 0x7F) << s
        if b < 0x80:
            return r, o
        s += 7


def _rsv(d, o):
    u, o = _ruv(d, o)
    return (u >> 1) ^ -(u & 1), o


WQ = 4.0     # width quantisation: 1/4 px on the radius
PLQ = 4.0    # branch path length quantisation: 1/4 px


def encode_structure(st):
    """Serialise a structure dict. Layout (all varints, then zlib):
    n_vertices; vertices sorted by row as (drow, dcol_signed) deltas; vertex
    kinds (2 bits packed); vertex widths; n_branches; per branch: a, b (vertex
    indices, a as delta from previous a), n_interior, interior points as
    signed (dy, dx) deltas along the polyline, interior widths as signed
    deltas from the previous width along the branch."""
    H, W = st['shape']
    order = np.lexsort((st['vpos'][:, 1], st['vpos'][:, 0]))
    inv = np.empty(len(order), np.int64)
    inv[order] = np.arange(len(order))
    vp, vr = st['vpos'][order], np.round(st['vrad'][order] * WQ).astype(int)
    kinds = {'endpoint': 0, 'junction': 1, 'loop': 2}
    vk = [kinds[st['vkind'][i]] for i in order]
    buf = bytearray()
    _uv(buf, len(vp))
    py = px = 0
    for (y, x) in vp:
        _uv(buf, int(y - py))
        _sv(buf, int(x - px))
        py, px = y, x
    for i in range(0, len(vk), 4):
        b = 0
        for j, k in enumerate(vk[i:i + 4]):
            b |= k << (2 * j)
        buf.append(b)
    for r in vr:
        _uv(buf, int(r))
    brs = sorted(st['branches'], key=lambda br: (inv[br['a']], inv[br['b']]))
    _uv(buf, len(brs))
    pa = 0
    for br in brs:
        a, b = int(inv[br['a']]), int(inv[br['b']])
        _uv(buf, a - pa)
        _sv(buf, b - a)
        pa = a
        pts, rad = br['pts'], np.round(br['rad'] * WQ).astype(int)
        _uv(buf, int(round(br['plen'] * PLQ)))
        _uv(buf, len(pts) - 2)
        prev, pr = pts[0], int(vr[a])    # widths predicted from the start vertex
        for p, r in zip(pts[1:-1], rad[1:-1]):
            _sv(buf, int(p[0] - prev[0]))
            _sv(buf, int(p[1] - prev[1]))
            _sv(buf, int(r - pr))
            prev, pr = p, r
    body = zlib.compress(bytes(buf), 9)
    return struct.pack('<BHH', 7, H, W) + body


def decode_structure(data):
    ver, H, W = struct.unpack('<BHH', data[:5])
    assert ver == 7
    d = zlib.decompress(data[5:])
    o = 0
    nv, o = _ruv(d, o)
    vpos = np.zeros((nv, 2), np.int64)
    py = px = 0
    for i in range(nv):
        dy, o = _ruv(d, o)
        dx, o = _rsv(d, o)
        py, px = py + dy, px + dx
        vpos[i] = (py, px)
    kinds = ['endpoint', 'junction', 'loop']
    vk = []
    for i in range(0, nv, 4):
        b = d[o]
        o += 1
        vk += [kinds[(b >> (2 * j)) & 3] for j in range(min(4, nv - i))]
    vrad = np.zeros(nv)
    for i in range(nv):
        r, o = _ruv(d, o)
        vrad[i] = r / WQ
    nb, o = _ruv(d, o)
    brs, pa = [], 0
    for _ in range(nb):
        da, o = _ruv(d, o)
        db, o = _rsv(d, o)
        a = pa + da
        b = a + db
        pa = a
        pl, o = _ruv(d, o)
        ni, o = _ruv(d, o)
        pts, rad = [vpos[a]], [vrad[a] * WQ]
        prev, pr = vpos[a], None
        for k in range(ni):
            dy, o = _rsv(d, o)
            dx, o = _rsv(d, o)
            dr, o = _rsv(d, o)
            p = np.array([prev[0] + dy, prev[1] + dx])
            r = (int(round(vrad[a] * WQ)) if pr is None else pr) + dr
            pts.append(p)
            rad.append(r)
            prev, pr = p, r
        pts.append(vpos[b])
        rad.append(vrad[b] * WQ)
        brs.append({'a': a, 'b': b, 'pts': np.array(pts), 'rad': np.array(rad, float) / WQ,
                    'plen': pl / PLQ})
    return {'shape': (H, W), 'vpos': vpos, 'vrad': vrad, 'vkind': vk, 'branches': brs}
