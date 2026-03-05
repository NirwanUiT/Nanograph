"""
Nanograph v4 — Formal graph structure.

This is the KEY v4 addition. Instead of treating nanograph as a point cloud
with types, we build an explicit graph:

  - Nodes: endpoints, junctions, sampled points (with attributes: position,
    width, intensity, orientation)
  - Edges: skeleton segments connecting nodes (with attributes: length,
    mean width, mean intensity, curvature, pixel path)
  - Adjacency: stored explicitly, enabling graph queries

This unlocks:
  1. Graph-based compression (delta-encode along edges)
  2. Downstream analysis (connectivity, shortest paths, GNNs)
  3. Temporal tracking (compare graphs across timepoints)
  4. Structural queries (find all branched networks with width > N)
"""

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from collections import defaultdict


@dataclass
class GraphNode:
    """A node in the nanograph."""
    id: int
    position: Tuple[int, int]      # (row, col)
    width: float
    intensity: float
    orientation: float             # radians [0, pi)
    node_type: str                 # 'endpoint', 'junction', 'sampled'
    degree: int = 0                # number of connected edges


@dataclass
class GraphEdge:
    """An edge in the nanograph (skeleton segment between two nodes)."""
    id: int
    source: int                    # node id
    target: int                    # node id
    length: float                  # Euclidean path length in pixels
    pixel_count: int               # number of skeleton pixels on this edge
    mean_width: float
    mean_intensity: float
    curvature: float               # total angular change / length
    path_pixels: Optional[np.ndarray] = field(default=None, repr=False)  # (K, 2) pixel coords


@dataclass
class Nanograph:
    """
    Formal graph representation of a microscopy image.

    This is the central data structure for v4. It stores both nodes and edges
    with rich attributes, enabling graph-based analysis and compression.
    """
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    adjacency: Dict[int, List[int]]   # node_id -> [neighbor_node_ids]
    shape: Tuple[int, int]            # (H, W) of the source image
    image_type: str                   # 'sparse' or 'dense'

    @property
    def n_nodes(self):
        return len(self.nodes)

    @property
    def n_edges(self):
        return len(self.edges)

    @property
    def node_positions(self) -> np.ndarray:
        """(N, 2) array of node positions."""
        if not self.nodes:
            return np.zeros((0, 2))
        return np.array([n.position for n in self.nodes])

    @property
    def node_widths(self) -> np.ndarray:
        return np.array([n.width for n in self.nodes]) if self.nodes else np.array([])

    @property
    def node_intensities(self) -> np.ndarray:
        return np.array([n.intensity for n in self.nodes]) if self.nodes else np.array([])

    @property
    def node_orientations(self) -> np.ndarray:
        return np.array([n.orientation for n in self.nodes]) if self.nodes else np.array([])

    @property
    def node_types(self) -> np.ndarray:
        return np.array([n.node_type for n in self.nodes]) if self.nodes else np.array([])

    def get_endpoints(self) -> List[GraphNode]:
        return [n for n in self.nodes if n.node_type == 'endpoint']

    def get_junctions(self) -> List[GraphNode]:
        return [n for n in self.nodes if n.node_type == 'junction']

    def get_connected_components(self) -> List[List[int]]:
        """Find connected components using BFS."""
        visited = set()
        components = []
        for node in self.nodes:
            if node.id not in visited:
                comp = []
                queue = [node.id]
                while queue:
                    nid = queue.pop(0)
                    if nid in visited:
                        continue
                    visited.add(nid)
                    comp.append(nid)
                    for neighbor in self.adjacency.get(nid, []):
                        if neighbor not in visited:
                            queue.append(neighbor)
                components.append(comp)
        return components

    def subgraph(self, node_ids: set) -> 'Nanograph':
        """Extract a subgraph containing only the specified nodes."""
        id_map = {old: new for new, old in enumerate(sorted(node_ids))}
        new_nodes = []
        for n in self.nodes:
            if n.id in node_ids:
                new_node = GraphNode(
                    id=id_map[n.id], position=n.position, width=n.width,
                    intensity=n.intensity, orientation=n.orientation,
                    node_type=n.node_type, degree=0
                )
                new_nodes.append(new_node)

        new_edges = []
        new_adj = defaultdict(list)
        eid = 0
        for e in self.edges:
            if e.source in node_ids and e.target in node_ids:
                new_e = GraphEdge(
                    id=eid, source=id_map[e.source], target=id_map[e.target],
                    length=e.length, pixel_count=e.pixel_count,
                    mean_width=e.mean_width, mean_intensity=e.mean_intensity,
                    curvature=e.curvature, path_pixels=e.path_pixels
                )
                new_edges.append(new_e)
                new_adj[new_e.source].append(new_e.target)
                new_adj[new_e.target].append(new_e.source)
                eid += 1

        for n in new_nodes:
            n.degree = len(new_adj.get(n.id, []))

        return Nanograph(
            nodes=new_nodes, edges=new_edges,
            adjacency=dict(new_adj), shape=self.shape,
            image_type=self.image_type
        )

    def summary(self) -> dict:
        """Summary statistics of the graph."""
        n_ep = sum(1 for n in self.nodes if n.node_type == 'endpoint')
        n_jn = sum(1 for n in self.nodes if n.node_type == 'junction')
        n_sp = sum(1 for n in self.nodes if n.node_type == 'sampled')
        components = self.get_connected_components()
        return {
            'n_nodes': self.n_nodes,
            'n_edges': self.n_edges,
            'n_endpoints': n_ep,
            'n_junctions': n_jn,
            'n_sampled': n_sp,
            'n_components': len(components),
            'mean_degree': np.mean([n.degree for n in self.nodes]) if self.nodes else 0,
            'mean_edge_length': np.mean([e.length for e in self.edges]) if self.edges else 0,
            'mean_width': float(np.mean(self.node_widths)) if self.nodes else 0,
            'total_edge_length': sum(e.length for e in self.edges),
            'mean_curvature': np.mean([e.curvature for e in self.edges]) if self.edges else 0,
        }


_NEIGHBORS_8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _skeleton_neighbor_count(skeleton, row, col):
    """Count 8-connected skeleton neighbors of pixel (row, col)."""
    height, width = skeleton.shape
    count = 0
    for dr, dc in _NEIGHBORS_8:
        nr, nc = row + dr, col + dc
        if 0 <= nr < height and 0 <= nc < width and skeleton[nr, nc] > 0:
            count += 1
    return count


def _skel_nb8(skeleton, row, col):
    """Return list of 8-connected skeleton neighbors of (row, col)."""
    height, width = skeleton.shape
    result = []
    for dr, dc in _NEIGHBORS_8:
        nr, nc = row + dr, col + dc
        if 0 <= nr < height and 0 <= nc < width and skeleton[nr, nc] > 0:
            result.append((nr, nc))
    return result


def trace_skeleton_branches(skeleton, ep_mask, jn_mask):
    """
    Trace all branches of the skeleton between critical points.

    A branch is a path along the skeleton from one critical point to another
    (or to a dead end).  Critical points = endpoints (ep_mask) + junctions
    (jn_mask).

    Algorithm
    ---------
    Phase 1 — walk from every critical pixel:
      * Build critical_pixels from ep_mask ∪ jn_mask (on-skeleton only).
      * Maintain a *visited* set for non-critical pixels already claimed by a
        branch.  Critical pixels are NEVER added to visited — they can be the
        start or end of multiple branches.
      * For each critical pixel cp and each skeleton neighbour nb:
          - Skip nb if it is in visited AND is not a critical pixel.
          - If nb IS a critical pixel → record a length-2 branch [cp, nb]
            (deduplicated via seen_direct).
          - Otherwise → start walking from cp through nb, marking non-critical
            pixels as visited.  Stop at a critical pixel or dead end.
    Phase 2 — isolated loops:
      * Any remaining unvisited non-critical skeleton pixels belong to loops
        with no endpoints/junctions.  Walk them into additional branches.
    Coverage check: warn if < 90 % of skeleton pixels are covered.

    Returns list of dicts with keys:
      start_cp  (y, x) — starting critical pixel (or loop seed)
      end_cp    (y, x) — ending critical pixel (or dead-end pixel)
      path      list of (y, x) from start_cp to end_cp inclusive;
                position index in this list is used for node ordering.
    """
    height, width = skeleton.shape

    # Build critical pixel set — on-skeleton pixels from ep/jn masks
    critical_pixels = set()
    for row, col in zip(*np.where(ep_mask > 0)):
        rr, cc = int(row), int(col)
        if skeleton[rr, cc] > 0:
            critical_pixels.add((rr, cc))
    for row, col in zip(*np.where(jn_mask > 0)):
        rr, cc = int(row), int(col)
        if skeleton[rr, cc] > 0:
            critical_pixels.add((rr, cc))

    visited = set()       # non-critical pixels already claimed
    branches = []
    seen_direct = set()   # frozensets for direct crit↔crit branches

    # Phase 1: walk from critical pixels
    for cp in critical_pixels:
        cy, cx = cp
        for nb in _skel_nb8(skeleton, cy, cx):
            # Skip if already claimed by another branch (unless critical)
            if nb in visited and nb not in critical_pixels:
                continue

            if nb in critical_pixels:
                # Direct critical-to-critical branch
                key = frozenset((cp, nb))
                if key not in seen_direct:
                    seen_direct.add(key)
                    branches.append({'start_cp': cp, 'end_cp': nb,
                                     'path': [cp, nb]})
                continue

            # nb is unclaimed non-critical — start a new branch
            path = [cp, nb]
            visited.add(nb)
            prev = cp
            curr = nb

            while True:
                # Critical neighbours terminate the branch (allowed even if visited)
                crit_nb = [n for n in _skel_nb8(skeleton, curr[0], curr[1])
                           if n in critical_pixels]
                # Free neighbours: non-critical, unvisited, not the step we came from
                free_nb = [n for n in _skel_nb8(skeleton, curr[0], curr[1])
                           if n not in critical_pixels
                           and n not in visited
                           and n != prev]

                if crit_nb:
                    path.append(crit_nb[0])
                    break
                elif not free_nb:
                    break   # dead end
                elif len(free_nb) == 1:
                    nxt = free_nb[0]
                    path.append(nxt)
                    visited.add(nxt)
                    prev, curr = curr, nxt
                else:
                    # Undetected junction: pick neighbour with most connections
                    nxt = max(free_nb,
                              key=lambda p: _skeleton_neighbor_count(
                                  skeleton, p[0], p[1]))
                    path.append(nxt)
                    visited.add(nxt)
                    prev, curr = curr, nxt

            branches.append({'start_cp': cp, 'end_cp': path[-1], 'path': path})

    # Phase 2: isolated loops — pixels not reachable from any critical pixel
    skel_rows, skel_cols = np.where(skeleton > 0)
    all_skel = set(zip(skel_rows.tolist(), skel_cols.tolist()))
    uncovered = all_skel - visited - critical_pixels

    while uncovered:
        seed = next(iter(uncovered))
        path = [seed]
        visited.add(seed)
        uncovered.discard(seed)
        prev = None
        curr = seed

        while True:
            free_nb = [n for n in _skel_nb8(skeleton, curr[0], curr[1])
                       if n not in visited and n not in critical_pixels
                       and n != prev]
            if not free_nb:
                break
            nxt = free_nb[0]
            path.append(nxt)
            visited.add(nxt)
            uncovered.discard(nxt)
            prev, curr = curr, nxt

        if len(path) >= 2:
            branches.append({'start_cp': path[0], 'end_cp': path[-1],
                             'path': path})

    # Coverage check
    n_skel = int(np.count_nonzero(skeleton))
    n_covered = len(visited | critical_pixels)
    if n_skel > 0 and n_covered < int(0.90 * n_skel):
        pct = 100.0 * n_covered / n_skel
        print(f'WARNING: skeleton coverage {n_covered}/{n_skel} ({pct:.1f}%). '
              f'Some branches may be missing.')

    return branches


def build_nanograph(points, intensities, widths, orientations, types,
                    skeleton, dist_transform, original_img, shape,
                    image_type='sparse', ep_mask=None, jn_mask=None, cfg=None):
    """
    Build a formal Nanograph from extracted points and skeleton.

    Three-phase algorithm:

    Phase 1 — Trace skeleton branches (independent of node positions):
      trace_skeleton_branches() returns every branch as an ordered list of
      pixel coordinates.  Non-critical pixels appear in exactly one branch;
      critical pixels (endpoints/junctions) are shared as branch endpoints.

    Phase 2 — Assign every node to branches:
      Build a pixel→branch-list lookup from branch paths.
      For each node check its pixel position directly, then search within
      max(node.width, 5) px for the nearest branch pixel.  Nodes that can't
      be assigned within tolerance are flagged as orphans.
      Junction nodes map to every branch that starts/ends at their pixel.

    Phase 3 — Build edges between consecutive nodes on each branch:
      Sort nodes by their position index along the branch path.
      Create edges between consecutive pairs using the path segment between
      them for length/width/curvature attributes.
      Orphan nodes are connected to their nearest non-orphan node.
    """
    from .config import DEFAULT_CONFIG, NanographConfig, GraphConfig

    gc = cfg if isinstance(cfg, GraphConfig) else (
         cfg.graph if isinstance(cfg, NanographConfig) else DEFAULT_CONFIG.graph)

    if len(points) == 0:
        return Nanograph(nodes=[], edges=[], adjacency={},
                         shape=shape, image_type=image_type)

    # ------------------------------------------------------------------
    # Phase 0: Create nodes
    # ------------------------------------------------------------------
    nodes = []
    for i, (pt, w_val, inten, ori, t) in enumerate(
            zip(points, widths, intensities, orientations, types)):
        nodes.append(GraphNode(
            id=i, position=(int(pt[0]), int(pt[1])),
            width=float(w_val), intensity=float(inten),
            orientation=float(ori), node_type=str(t)
        ))

    img_h, img_w = skeleton.shape
    img_f = original_img.astype(float) / 255.0

    # ------------------------------------------------------------------
    # Phase 1: Trace skeleton branches
    # ------------------------------------------------------------------
    if ep_mask is None or jn_mask is None:
        ep_loc = np.zeros_like(skeleton)
        jn_loc = np.zeros_like(skeleton)
        for row, col in zip(*np.where(skeleton > 0)):
            deg = _skeleton_neighbor_count(skeleton, row, col)
            if deg == 1:
                ep_loc[row, col] = 1
            elif deg >= 3:
                jn_loc[row, col] = 1
        branches = trace_skeleton_branches(skeleton, ep_loc, jn_loc)
    else:
        branches = trace_skeleton_branches(skeleton, ep_mask, jn_mask)

    # pixel → list of (branch_idx, pos_in_path)
    # Critical pixels appear at pos=0 or pos=last of multiple branches.
    px_to_branch_pos = defaultdict(list)
    for bi, branch in enumerate(branches):
        for pos, pyx in enumerate(branch['path']):
            px_to_branch_pos[pyx].append((bi, pos))

    # ------------------------------------------------------------------
    # Phase 2: Assign nodes to branches
    # node_branch_assignments[node_id] = list of (branch_idx, pos_in_path)
    # ------------------------------------------------------------------
    node_branch_assignments = defaultdict(list)
    orphan_ids = []

    for node in nodes:
        ny, nx = node.position
        key = (ny, nx)

        if key in px_to_branch_pos:
            # Exact hit — assign to ALL branches sharing this pixel
            node_branch_assignments[node.id].extend(px_to_branch_pos[key])
        else:
            # Search within max(node.width, 5) px radius
            tol = max(int(node.width), 5)
            best_key = None
            best_d2 = tol * tol + 1
            for drow in range(-tol, tol + 1):
                for dcol in range(-tol, tol + 1):
                    d2 = drow * drow + dcol * dcol
                    if d2 > tol * tol or d2 >= best_d2:
                        continue
                    k = (ny + drow, nx + dcol)
                    if k in px_to_branch_pos:
                        best_d2 = d2
                        best_key = k
            if best_key is not None:
                node_branch_assignments[node.id].extend(
                    px_to_branch_pos[best_key])
            else:
                orphan_ids.append(node.id)

    # branch_idx → [(pos_in_path, node_id)]  (one entry per node per branch)
    branch_nodes = defaultdict(list)
    for node_id, assignments in node_branch_assignments.items():
        seen_bi = set()
        for bi, pos in assignments:
            if bi not in seen_bi:
                branch_nodes[bi].append((pos, node_id))
                seen_bi.add(bi)

    # ------------------------------------------------------------------
    # Phase 3: Build edges
    # ------------------------------------------------------------------
    edges = []
    adjacency = defaultdict(list)
    edge_id = 0
    seen_pairs = set()

    def _make_edge(nid1, nid2, path_segment):
        nonlocal edge_id
        pair = (min(nid1, nid2), max(nid1, nid2))
        if pair in seen_pairs or len(path_segment) < gc.min_edge_length:
            return
        seen_pairs.add(pair)

        path_arr = np.array(path_segment)
        if len(path_arr) > 1:
            diffs = np.diff(path_arr, axis=0)
            seg_lengths = np.sqrt(np.sum(diffs ** 2, axis=1))
            edge_length = float(np.sum(seg_lengths))
        else:
            diffs = np.zeros((1, 2))
            edge_length = 0.0

        edge_widths = [float(dist_transform[pr, pc])
                       for pr, pc in path_segment
                       if 0 <= pr < img_h and 0 <= pc < img_w]
        mean_w = float(np.mean(edge_widths)) if edge_widths else 0.0

        edge_intens = [float(img_f[pr, pc])
                       for pr, pc in path_segment
                       if 0 <= pr < img_h and 0 <= pc < img_w]
        mean_i = float(np.mean(edge_intens)) if edge_intens else 0.0

        if len(path_arr) > 2:
            angles = np.arctan2(diffs[:, 0], diffs[:, 1])
            angle_diffs = np.abs(np.diff(angles))
            angle_diffs = np.minimum(angle_diffs, 2 * np.pi - angle_diffs)
            curvature = float(np.sum(angle_diffs)) / max(edge_length, 1e-6)
        else:
            curvature = 0.0

        edges.append(GraphEdge(
            id=edge_id, source=nid1, target=nid2,
            length=edge_length, pixel_count=len(path_segment),
            mean_width=mean_w, mean_intensity=mean_i,
            curvature=curvature,
            path_pixels=path_arr if gc.compute_edge_features else None
        ))
        adjacency[nid1].append(nid2)
        adjacency[nid2].append(nid1)
        edge_id += 1

    # Edges from consecutive nodes along each branch
    for bi, branch in enumerate(branches):
        sorted_branch_nodes = sorted(branch_nodes.get(bi, []))
        if len(sorted_branch_nodes) < 2:
            continue
        bpath = branch['path']
        for k in range(len(sorted_branch_nodes) - 1):
            pos1, nid1 = sorted_branch_nodes[k]
            pos2, nid2 = sorted_branch_nodes[k + 1]
            segment = bpath[pos1: pos2 + 1]
            if segment:
                _make_edge(nid1, nid2, segment)

    # Orphan nodes: connect each to its nearest non-orphan
    # (bypass min_edge_length — these are direct synthetic connections)
    orphan_id_set = set(orphan_ids)
    non_orphan_ids = [n.id for n in nodes if n.id not in orphan_id_set]
    if orphan_ids and non_orphan_ids:
        non_orphan_pos = np.array([nodes[nid].position for nid in non_orphan_ids],
                                   dtype=float)
        for oid in orphan_ids:
            ny, nx = nodes[oid].position
            dists = np.sqrt(np.sum((non_orphan_pos - [ny, nx]) ** 2, axis=1))
            nearest_nid = non_orphan_ids[int(np.argmin(dists))]
            pair = (min(oid, nearest_nid), max(oid, nearest_nid))
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                seg = np.array([nodes[oid].position, nodes[nearest_nid].position])
                diffs = np.diff(seg.astype(float), axis=0)
                elen = float(np.sqrt(np.sum(diffs ** 2)))
                edges.append(GraphEdge(
                    id=edge_id, source=oid, target=nearest_nid,
                    length=elen, pixel_count=2,
                    mean_width=0.0, mean_intensity=0.0, curvature=0.0,
                    path_pixels=seg
                ))
                adjacency[oid].append(nearest_nid)
                adjacency[nearest_nid].append(oid)
                edge_id += 1

    # Update node degrees
    for node in nodes:
        node.degree = len(adjacency.get(node.id, []))

    result = Nanograph(
        nodes=nodes, edges=edges, adjacency=dict(adjacency),
        shape=shape, image_type=image_type
    )

    # Connectivity check
    skel_n_comp, _ = cv2.connectedComponents(skeleton.astype(np.uint8))
    skel_n_comp = max(skel_n_comp - 1, 1)
    n_graph_comp = len(result.get_connected_components())
    print(f'Graph: {result.n_nodes}N, {result.n_edges}E, {n_graph_comp}C '
          f'(skeleton: {skel_n_comp} components)')
    if n_graph_comp > 2 * skel_n_comp:
        print(f'WARNING: graph has {n_graph_comp} components vs skeleton '
              f'{skel_n_comp}. Some connections may be missing.')

    return result


def nanograph_morphometry(graph: Nanograph) -> dict:
    """
    v4: Extract biological descriptors from the formal graph.
    Much richer than v3 since we have edge information.
    """
    if graph.n_nodes == 0:
        return {}

    summary = graph.summary()

    # Add distribution stats
    node_widths = graph.node_widths
    summary['mean_width_px'] = float(np.mean(node_widths))
    summary['std_width_px'] = float(np.std(node_widths))
    summary['max_width_px'] = float(np.max(node_widths))
    summary['min_width_px'] = float(np.min(node_widths))
    summary['mean_intensity'] = float(np.mean(graph.node_intensities))

    # Edge-based metrics (NEW in v4)
    if graph.edges:
        edge_lengths = [e.length for e in graph.edges]
        edge_curvatures = [e.curvature for e in graph.edges]
        summary['mean_edge_length'] = float(np.mean(edge_lengths))
        summary['std_edge_length'] = float(np.std(edge_lengths))
        summary['max_edge_length'] = float(np.max(edge_lengths))
        summary['mean_edge_curvature'] = float(np.mean(edge_curvatures))
        summary['max_edge_curvature'] = float(np.max(edge_curvatures))
        summary['network_length'] = float(np.sum(edge_lengths))

    # Branching analysis
    junctions = graph.get_junctions()
    if junctions:
        degrees = [n.degree for n in junctions]
        summary['mean_junction_degree'] = float(np.mean(degrees))
        summary['max_junction_degree'] = int(np.max(degrees))

    # Spatial coverage
    pts = graph.node_positions
    summary['spatial_extent_rows'] = float(np.ptp(pts[:, 0]))
    summary['spatial_extent_cols'] = float(np.ptp(pts[:, 1]))
    summary['bounding_box_area'] = (
        float(np.ptp(pts[:, 0])) * float(np.ptp(pts[:, 1]))
    )

    # Type distribution
    from collections import Counter
    summary['type_distribution'] = dict(Counter(graph.node_types))

    return summary
