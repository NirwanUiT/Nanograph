"""
Nanograph v5 — 4D deformable centreline tracking (`track.py`).

Status: implemented but UNVALIDATED engineering. No quantitative evaluation of
this module is part of the repository; treat every design choice below as a
hypothesis, not an established result.

Motivating hypothesis:
  Single-frame skeletonisation is expected to be degenerate at sub-PSF
  self-contacts: when a tortuous filament folds back on itself within the
  point-spread function, the two arms fuse into one blob and the medial axis
  collapses to a single short line, roughly halving the measured geodesic.
  Because a fold that is unresolvable in one frame may be resolved in another,
  temporal propagation should in principle recover geometry that any
  single-frame method loses. Systematic measurement of how often this failure
  occurs and how often propagation rescues it remains future work.

Idea:
  Replace per-frame skeletonisation with a single arc-length-parameterised 1D
  curve per mitochondrion, *propagated* through every frame. A 1D curve has a
  fixed number of material points whose spacing is conserved by resampling, so
  it CAN self-touch and CANNOT collapse — exactly the property a medial axis
  lacks. We:
    1. seed the curve from the best-resolved frame (max graph extent, leakage-
       free),
    2. propagate it frame-to-frame by optical-flow advection + a light snake
       refinement onto the foreground evidence, conserving arc-length spacing,
    3. measure geodesics as arc length between material parameters on the
       propagated curve.

This module is self-contained (numpy + cv2 + scipy) and does not alter the
classical single-frame pipeline.
"""
from __future__ import annotations

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from scipy.ndimage import distance_transform_edt, gaussian_filter


# --------------------------------------------------------------------------- #
# curve geometry
# --------------------------------------------------------------------------- #
def curve_arclength(curve: np.ndarray) -> np.ndarray:
    """Cumulative arc length (px) along an ordered (M,2) rc polyline."""
    if len(curve) < 2:
        return np.zeros(len(curve))
    seg = np.hypot(np.diff(curve[:, 0]), np.diff(curve[:, 1]))
    return np.concatenate([[0.0], np.cumsum(seg)])


def resample_curve(curve: np.ndarray, n: int) -> np.ndarray:
    """Resample an ordered (M,2) polyline to n points at uniform arc-length
    spacing. Preserves total length and endpoints; redistributes points so they
    cannot bunch up (the property that prevents fold collapse)."""
    cum = curve_arclength(curve)
    total = cum[-1]
    if total < 1e-6:
        return np.repeat(curve[:1], n, axis=0)
    targets = np.linspace(0.0, total, n)
    r = np.interp(targets, cum, curve[:, 0])
    c = np.interp(targets, cum, curve[:, 1])
    return np.stack([r, c], axis=1)


def _laplacian(x: np.ndarray) -> np.ndarray:
    """Discrete second difference along the curve, Neumann (free) ends."""
    lap = np.zeros_like(x)
    lap[1:-1] = x[:-2] - 2 * x[1:-1] + x[2:]
    lap[0] = x[1] - x[0]
    lap[-1] = x[-2] - x[-1]
    return lap


# --------------------------------------------------------------------------- #
# evidence field per frame
# --------------------------------------------------------------------------- #
@dataclass
class FrameEvidence:
    """Pre-computed per-frame fields the snake reads."""
    image: np.ndarray                 # float32 grayscale, smoothed
    grad_r: np.ndarray                # d(attraction)/d row
    grad_c: np.ndarray                # d(attraction)/d col
    mask: np.ndarray                  # bool foreground
    inside_dt: np.ndarray             # distance transform inside mask (ridge)
    outside_push_r: np.ndarray        # force pushing exterior points back in
    outside_push_c: np.ndarray
    shape: Tuple[int, int]


def build_evidence(image: np.ndarray, mask: np.ndarray,
                   smooth: float = 1.0) -> FrameEvidence:
    """Build the attraction field for one frame.

    Attraction potential P = -(smoothed intensity) - w*inside_distance, so the
    curve is pulled toward the bright medial ridge of the foreground; outside
    the mask a distance-transform force pushes points back in (containment).
    """
    img = image.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    img = gaussian_filter(img, smooth)
    m = mask.astype(bool)

    # ridge potential: bright intensity + medial distance inside the mask
    inside_dt = distance_transform_edt(m).astype(np.float32)
    inside_norm = inside_dt / (inside_dt.max() + 1e-6)
    potential = img + 0.5 * inside_norm
    gr, gc = np.gradient(potential)           # ascend toward bright/medial

    # containment: outside the mask, push toward nearest foreground
    out_dt, (ir, ic) = distance_transform_edt(~m, return_indices=True)
    rr, cc = np.indices(m.shape)
    push_r = np.where(m, 0.0, (ir - rr)).astype(np.float32)
    push_c = np.where(m, 0.0, (ic - cc)).astype(np.float32)
    norm = np.hypot(push_r, push_c) + 1e-6
    push_r /= norm
    push_c /= norm

    return FrameEvidence(img, gr.astype(np.float32), gc.astype(np.float32),
                         m, inside_dt, push_r, push_c, m.shape)


def _sample(field: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Bilinear sample a 2D field at (M,2) rc points (clamped)."""
    H, W = field.shape
    r = np.clip(pts[:, 0], 0, H - 1)
    c = np.clip(pts[:, 1], 0, W - 1)
    r0 = np.floor(r).astype(int); c0 = np.floor(c).astype(int)
    r1 = np.minimum(r0 + 1, H - 1); c1 = np.minimum(c0 + 1, W - 1)
    fr = r - r0; fc = c - c0
    v00 = field[r0, c0]; v01 = field[r0, c1]
    v10 = field[r1, c0]; v11 = field[r1, c1]
    return (v00 * (1 - fr) * (1 - fc) + v01 * (1 - fr) * fc +
            v10 * fr * (1 - fc) + v11 * fr * fc)


# --------------------------------------------------------------------------- #
# snake refinement (explicit, arc-length conserving)
# --------------------------------------------------------------------------- #
def _project_inextensible(x: np.ndarray, rest_len: np.ndarray,
                          anchor: np.ndarray, n_iter: int = 8) -> np.ndarray:
    """Position-based-dynamics distance projection: adjust points so each
    segment length |x[k+1]-x[k]| matches the seed rest length rest_len[k].

    This is what prevents fold collapse: optical flow inside a featureless blob
    averages out (aperture problem) and pulls material points together, but a
    physical filament cannot compress. Enforcing the rest lengths keeps the
    folded curve at full arc length even when the image evidence has fused.
    The least-constrained end is softly pinned to the anchor to fix the gauge.
    """
    x = x.copy()
    M = len(x)
    for _ in range(n_iter):
        for k in range(M - 1):
            d = x[k + 1] - x[k]
            L = np.hypot(d[0], d[1])
            if L < 1e-6:
                d = np.array([1.0, 0.0]); L = 1.0
            corr = 0.5 * (L - rest_len[k]) * d / L
            x[k] += corr
            x[k + 1] -= corr
        # soft anchor pin (gauge): pull whole curve gently to anchor centroid
        x += 0.05 * (anchor - x)
    return x


def refine_snake(curve: np.ndarray, ev: FrameEvidence,
                 anchor: Optional[np.ndarray] = None,
                 rest_len: Optional[np.ndarray] = None,
                 n_iter: int = 40, beta: float = 0.05,
                 step: float = 0.5, anchor_w: float = 0.30,
                 contain_w: float = 2.0, max_move: float = 1.0) -> np.ndarray:
    """Evolve `curve` onto frame evidence `ev` WITHOUT collapsing folds.

    Key design (learned from the collapse bug): inside the foreground mask we do
    NOT attract points to the bright medial ridge — that attraction is exactly
    what pulls a folded curve onto a single short line. Instead the fold shape is
    supplied by the temporal `anchor` (the flow-advected curve from the resolved
    neighbour) plus a small rigidity term; the mask acts only as a *containment*
    boundary (points drifting outside are pushed back in). Crucially, an
    inextensibility projection enforces the seed segment lengths every step so
    the curve keeps its full folded arc length even when flow tries to compress
    it inside an unresolved blob.

      beta      = rigidity (smoothness, preserves the fold; NO shrinking tension)
      anchor_w  = pull toward the flow-advected curve (temporal prior)
      contain_w = push points that left the mask back inside
      rest_len  = per-segment target lengths (inextensibility); if None, taken
                  from the initial curve.
    """
    M = len(curve)
    x = curve.copy()
    if anchor is None:
        anchor = curve.copy()
    if rest_len is None:
        rest_len = np.hypot(np.diff(curve[:, 0]), np.diff(curve[:, 1]))
    for it in range(n_iter):
        # rigidity only (biharmonic) — smooths without shrinking
        bih = _laplacian(_laplacian(x))
        internal = -beta * bih
        # containment: only outside the mask, push back toward foreground
        inside = _sample(ev.inside_dt, x) > 0.5
        pr = _sample(ev.outside_push_r, x)
        pc = _sample(ev.outside_push_c, x)
        contain = np.stack([pr, pc], axis=1)
        contain[inside] = 0.0
        # temporal anchor: stay near the flow-advected prediction
        anc = anchor - x
        move = step * (internal + contain_w * contain + anchor_w * anc)
        mn = np.hypot(move[:, 0], move[:, 1])
        clip = np.minimum(1.0, max_move / (mn + 1e-6))
        x += move * clip[:, None]
        # enforce inextensibility (prevents fold collapse)
        x = _project_inextensible(x, rest_len, anchor)
    return x


# --------------------------------------------------------------------------- #
# track
# --------------------------------------------------------------------------- #
def seed_curve_from_graph(graph, node_ids: set) -> Optional[np.ndarray]:
    """Extract the longest path through a mito's recovered graph as a dense,
    ordered (K,2) rc polyline (the seed centreline). Uses the graph diameter
    (longest shortest path between two endpoints) and stitches edge path_pixels.

    Returns None if no usable path exists.
    """
    import networkx as nx
    pos = {nd.id: np.array(nd.position, float) for nd in graph.nodes
           if nd.id in node_ids}
    if len(pos) < 2:
        return None
    G = nx.Graph()
    edge_pix = {}
    for nd in graph.nodes:
        if nd.id in node_ids:
            G.add_node(nd.id)
    for e in graph.edges:
        if e.source in node_ids and e.target in node_ids:
            if G.has_edge(e.source, e.target):
                continue
            G.add_edge(e.source, e.target, weight=max(e.length, 1e-3))
            pix = e.path_pixels
            edge_pix[(e.source, e.target)] = (
                np.asarray(pix, float) if pix is not None else
                np.stack([pos[e.source], pos[e.target]]))
    if G.number_of_edges() == 0:
        return None
    # work on the largest connected component
    comp = max(nx.connected_components(G), key=len)
    H = G.subgraph(comp)
    # diameter path: farthest pair via double BFS on weighted shortest paths
    def farthest(src):
        d = nx.single_source_dijkstra_path_length(H, src, weight='weight')
        t = max(d, key=d.get)
        return t, d[t]
    a, _ = farthest(next(iter(comp)))
    b, _ = farthest(a)
    node_path = nx.shortest_path(H, a, b, weight='weight')

    # stitch path_pixels, orienting each edge to chain continuously
    segments = [np.atleast_2d(pos[node_path[0]])]
    tail = pos[node_path[0]]
    for u, v in zip(node_path[:-1], node_path[1:]):
        pix = edge_pix.get((u, v))
        if pix is None:
            pix = edge_pix.get((v, u))
        if pix is None:
            pix = np.stack([pos[u], pos[v]])
        pix = np.asarray(pix, float).reshape(-1, 2)
        # orient so the first sample is closest to the current tail point
        d_first = float(np.hypot(pix[0, 0] - tail[0], pix[0, 1] - tail[1]))
        d_last = float(np.hypot(pix[-1, 0] - tail[0], pix[-1, 1] - tail[1]))
        if d_first > d_last:
            pix = pix[::-1]
        segments.append(pix)
        tail = pix[-1]
    out = np.vstack(segments)
    return out


@dataclass
class CurveTrack:
    """An arc-length-parameterised centreline propagated across frames.

    curves[frame] is an (M,2) rc polyline; point k is the same MATERIAL point in
    every frame (material coordinate = k / (M-1)). geodesic() reads arc length
    between two material parameters.
    """
    seed_frame: int
    curves: Dict[int, np.ndarray] = field(default_factory=dict)
    # step_to_seed[f] = (next_frame_toward_seed, flow) where flow advects an
    # image point at frame f one step toward the seed frame. Populated during
    # propagate_curve so a query point can be carried back to the well-resolved
    # seed frame for unambiguous material registration.
    step_to_seed: Dict[int, tuple] = field(default_factory=dict)

    @property
    def M(self) -> int:
        return len(next(iter(self.curves.values())))

    def param_of_point(self, xy: np.ndarray, frame: int) -> float:
        """Material parameter s in [0,1] of the curve point nearest xy at
        `frame` (spatial nearest-neighbour; ambiguous at collapsed frames)."""
        c = self.curves[frame]
        d = np.hypot(c[:, 0] - xy[0], c[:, 1] - xy[1])
        k = int(np.argmin(d))
        return k / (len(c) - 1)

    def register_param(self, xy: np.ndarray, frame: int) -> float:
        """Material parameter s in [0,1] for an image point at `frame`, resolved
        by advecting it back through the stored optical-flow chain to the SEED
        frame and snapping there.

        At a collapsed (self-contacting) frame, spatial nearest-neighbour picks
        the wrong arm of the fold. The seed frame is the best-resolved frame, so
        carrying the point's identity along the flow field the tracker already
        computed yields an unambiguous material coordinate.
        """
        cur = np.asarray(xy, float).reshape(1, 2)
        f = int(frame)
        # walk toward the seed frame, advecting the point one step at a time
        for _ in range(len(self.curves) + 1):
            if f == self.seed_frame or f not in self.step_to_seed:
                break
            nxt, flow = self.step_to_seed[f]
            dc = _sample(flow[..., 0], cur)
            dr = _sample(flow[..., 1], cur)
            cur = cur + np.stack([dr, dc], axis=1)
            f = nxt
        c = self.curves[f]
        d = np.hypot(c[:, 0] - cur[0, 0], c[:, 1] - cur[0, 1])
        k = int(np.argmin(d))
        return k / (len(c) - 1)

    def geodesic(self, frame: int, s0: float, s1: float) -> float:
        """Arc length (px) along the propagated curve at `frame` between two
        material parameters s0, s1 in [0,1]."""
        c = self.curves[frame]
        cum = curve_arclength(c)
        M = len(c)
        k0 = int(round(s0 * (M - 1)))
        k1 = int(round(s1 * (M - 1)))
        return float(abs(cum[k0] - cum[k1]))


def propagate_curve(seed_curve: np.ndarray, seed_frame: int,
                    images: Dict[int, np.ndarray],
                    evidence: Dict[int, FrameEvidence],
                    n_points: int = 200,
                    flow_params: Optional[dict] = None,
                    snake_params: Optional[dict] = None) -> CurveTrack:
    """Propagate a seed centreline through every frame via optical-flow
    advection + snake refinement, conserving material parameterisation.

    images/evidence are keyed by frame index (contiguous ordering assumed by
    sorted keys). The seed is refined at its own frame first, then marched
    outward in both directions.
    """
    fp = dict(pyr_scale=0.5, levels=3, winsize=21, iterations=3,
              poly_n=7, poly_sigma=1.5, flags=0)
    if flow_params:
        fp.update(flow_params)
    sp = snake_params or {}

    frames = sorted(images)
    track = CurveTrack(seed_frame=seed_frame)

    seed = resample_curve(seed_curve, n_points)
    seed = refine_snake(seed, evidence[seed_frame], anchor=seed, **sp)
    track.curves[seed_frame] = seed
    # rest lengths from the seed = the filament's conserved material spacing
    rest_len = np.hypot(np.diff(seed[:, 0]), np.diff(seed[:, 1]))

    def march(order):
        prev_f = seed_frame
        cur = seed
        for f in order:
            ia = images[prev_f]; ib = images[f]
            flow = cv2.calcOpticalFlowFarneback(ia, ib, None, **fp)
            # flow[...,0]=dx(col), flow[...,1]=dy(row); advect rc points
            dc = _sample(flow[..., 0], cur)
            dr = _sample(flow[..., 1], cur)
            adv = cur + np.stack([dr, dc], axis=1)
            # the advected curve is the temporal anchor; inextensibility keeps it
            # at full folded length even where flow collapses inside the blob
            adv = refine_snake(adv, evidence[f], anchor=adv,
                               rest_len=rest_len, **sp)
            track.curves[f] = adv
            # registration flow: f -> prev_f (one step back toward the seed),
            # so a query point at f can be carried to the well-resolved seed
            flow_back = cv2.calcOpticalFlowFarneback(ib, ia, None, **fp)
            track.step_to_seed[f] = (prev_f, flow_back)
            cur = adv
            prev_f = f

    si = frames.index(seed_frame)
    march(frames[si + 1:])              # forward
    march(frames[si - 1::-1])           # backward
    return track


# --------------------------------------------------------------------------- #
# high-level production API
# --------------------------------------------------------------------------- #
def _graph_nodes_near(graph, centreline, tau=5.0):
    from scipy.spatial import cKDTree as _KD
    tree = _KD(np.asarray(centreline, float))
    return {nd.id for nd in graph.nodes
            if tree.query(np.array(nd.position, float))[0] <= tau}


def _skeleton_geodesic(graph, p0, p1, snap_tau=6.0):
    """Single-frame skeleton geodesic (weighted shortest path) between two image
    points. Returns (length, ok)."""
    import networkx as nx
    from scipy.spatial import cKDTree as _KD
    if graph is None or len(graph.edges) == 0:
        return np.nan, False
    G = nx.Graph()
    pos = {}
    for nd in graph.nodes:
        G.add_node(nd.id)
        pos[nd.id] = np.array(nd.position, float)
    for e in graph.edges:
        if G.has_edge(e.source, e.target):
            if e.length < G[e.source][e.target]['weight']:
                G[e.source][e.target]['weight'] = e.length
        else:
            G.add_edge(e.source, e.target, weight=e.length)
    ids = list(pos)
    xy = np.array([pos[i] for i in ids])
    tree = _KD(xy)
    da, ia = tree.query(np.asarray(p0, float))
    db, ib = tree.query(np.asarray(p1, float))
    if da > snap_tau or db > snap_tau:
        return np.nan, False
    try:
        return float(nx.shortest_path_length(G, ids[ia], ids[ib],
                                             weight='weight')), True
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return np.nan, False


@dataclass
class MitoTrack:
    """A tracked mitochondrion across a clip: its propagated 4D centreline plus
    the per-frame recovered graphs, with a hybrid geodesic that uses the single-
    frame skeleton by default and the propagated curve only where the skeleton
    has collapsed at a self-contact."""
    curve: CurveTrack
    graphs: Dict[int, object]                       # frame -> Nanograph

    def hybrid_geodesic(self, frame: int, p0, p1,
                        fold_track: float = 1.5, straight_skel: float = 1.3,
                        snap_tau: float = 6.0) -> float:
        """Geodesic (px) between two image points at `frame`.

        Leakage-free selector: trust the single-frame skeleton unless it has
        collapsed — i.e. the propagated curve reports strong folding
        (track/euclid >= fold_track) while the skeleton looks near-straight
        (skel/euclid < straight_skel). Only then use the curve. Falls back to
        whichever measurement is available.
        """
        eu = float(np.hypot(p0[0] - p1[0], p0[1] - p1[1]))
        b, ok_b = _skeleton_geodesic(self.graphs.get(frame), p0, p1, snap_tau)
        s0 = self.curve.register_param(np.asarray(p0, float), frame)
        s1 = self.curve.register_param(np.asarray(p1, float), frame)
        t = self.curve.geodesic(frame, s0, s1)
        ok_t = t > 0
        if not ok_t:
            return b
        if not ok_b:
            return t
        if (t / max(eu, 1e-6) >= fold_track and
                b / max(eu, 1e-6) < straight_skel):
            return t
        return b


def track_mitochondria(results: Dict[int, object], images: Dict[int, np.ndarray],
                       centrelines: Optional[Dict[int, Dict[int, np.ndarray]]] = None,
                       assign_tau: float = 5.0, n_points: int = 220,
                       snake_params: Optional[dict] = None) -> Dict[int, MitoTrack]:
    """Build per-mitochondrion 4D tracks from a clip of encoded frames.

    Parameters
    ----------
    results : {frame -> NanographResult}   encoded frames (need .graph, .mask).
    images  : {frame -> uint8 grayscale}   the source frames.
    centrelines : optional {frame -> {mito_id -> (N,2) rc}} association hint. If
        omitted, every frame's graph is treated as ONE mitochondrion (id 0); the
        whole recovered structure is tracked. Provide per-mito centrelines (e.g.
        from a detector or GT) to track multiple filaments separately.

    Returns
    -------
    {mito_id -> MitoTrack}. Each track is seeded from the frame whose recovered
    centreline is longest (leakage-free: most-resolved) and propagated through
    all frames with the inextensible deformable curve.
    """
    frames = sorted(f for f in results
                    if results[f] is not None and
                    getattr(results[f], 'graph', None) is not None and
                    getattr(results[f], 'mask', None) is not None)
    if not frames:
        return {}
    evidence = {f: build_evidence(images[f], results[f].mask.astype(bool))
                for f in frames}
    imgs = {f: images[f] for f in frames}

    # determine mito ids
    if centrelines:
        mids = sorted({m for f in frames if f in centrelines
                       for m in centrelines[f]})
    else:
        mids = [0]

    tracks: Dict[int, MitoTrack] = {}
    for m in mids:
        # find the longest recovered seed across frames (leakage-free)
        best_f, best_seed, best_len = None, None, -1.0
        graphs = {}
        for f in frames:
            g = results[f].graph
            graphs[f] = g
            if centrelines and f in centrelines and m in centrelines[f]:
                ids = _graph_nodes_near(g, centrelines[f][m], assign_tau)
            else:
                ids = {nd.id for nd in g.nodes}
            sc = seed_curve_from_graph(g, ids)
            if sc is None or len(sc) < 5:
                continue
            L = float(curve_arclength(sc)[-1])
            if L > best_len:
                best_f, best_seed, best_len = f, sc, L
        if best_seed is None:
            continue
        curve = propagate_curve(best_seed, best_f, imgs, evidence,
                                n_points=n_points, snake_params=snake_params)
        tracks[m] = MitoTrack(curve=curve, graphs=graphs)
    return tracks

