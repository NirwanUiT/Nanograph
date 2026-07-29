"""
Nanograph v4 — Formal graph structure with skeleton-walk edge building.

Graph construction strategy (v4.1):
  Phase 1: Trace skeleton branches between critical points (endpoints/junctions)
  Phase 2: Assign every node to a branch by pixel proximity
  Phase 3: Build edges between consecutive nodes along each branch
"""

import numpy as np
import cv2
from scipy.spatial import KDTree
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
    path_pixels: Optional[np.ndarray] = field(default=None, repr=False)


@dataclass
class Nanograph:
    """Formal graph representation of a microscopy image."""
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    adjacency: Dict[int, List[int]]
    shape: Tuple[int, int]
    image_type: str

    @property
    def n_nodes(self):
        return len(self.nodes)

    @property
    def n_edges(self):
        return len(self.edges)

    @property
    def node_positions(self) -> np.ndarray:
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

    def remove_small_components(self, min_nodes: int) -> 'Nanograph':
        """Return a copy with connected components smaller than min_nodes removed.

        Tiny components are typically vesicle/noise blobs that are not part of
        the true structural network. Isolated single/few-node fragments inflate
        the component count without representing real topology.
        """
        if min_nodes <= 1:
            return self
        keep = set()
        for comp in self.get_connected_components():
            if len(comp) >= min_nodes:
                keep.update(comp)
        if len(keep) == self.n_nodes:
            return self
        return self.subgraph(keep)

    def bridge_gaps(self, max_gap=12.0, min_align=0.6, max_added=None):
        """Reconnect filament fragments split by skeleton breaks.

        Skeletonization frequently breaks a single continuous structure into
        several disconnected branches (a 1-pixel gap is enough). This links
        degree-1 endpoint nodes that lie in *different* connected components
        when (a) they are within ``max_gap`` pixels and (b) the gap continues
        the local filament direction at *both* endpoints (collinearity test
        with cosine >= ``min_align``). The collinearity requirement is what
        prevents merging two distinct structures that merely pass close by.

        Returns a new Nanograph with the bridging edges added. Operates on a
        copy; ``self`` is unchanged.
        """
        import copy as _copy

        # endpoint = degree-1 node; tangent points outward into the gap
        deg = {n.id: len(self.adjacency.get(n.id, [])) for n in self.nodes}
        node_by_id = {n.id: n for n in self.nodes}
        endpoints = [n.id for n in self.nodes if deg.get(n.id, 0) == 1]
        if len(endpoints) < 2:
            return self

        # component label per node
        comp_of = {}
        for ci, comp in enumerate(self.get_connected_components()):
            for nid in comp:
                comp_of[nid] = ci

        def tangent(nid):
            nb = self.adjacency.get(nid, [])
            if not nb:
                return None
            p = np.array(node_by_id[nid].position, float)
            q = np.array(node_by_id[nb[0]].position, float)
            v = p - q
            nrm = np.linalg.norm(v)
            return v / nrm if nrm > 1e-6 else None

        pos = {nid: np.array(node_by_id[nid].position, float) for nid in endpoints}
        tan = {nid: tangent(nid) for nid in endpoints}

        # candidate endpoint pairs across different components
        cands = []
        for i in range(len(endpoints)):
            a = endpoints[i]
            ta = tan[a]
            if ta is None:
                continue
            for j in range(i + 1, len(endpoints)):
                b = endpoints[j]
                if comp_of.get(a) == comp_of.get(b):
                    continue
                tb = tan[b]
                if tb is None:
                    continue
                gap = pos[b] - pos[a]
                d = np.linalg.norm(gap)
                if d < 1e-6 or d > max_gap:
                    continue
                g = gap / d
                # gap must extend a outward and arrive at b head-on
                align_a = float(np.dot(ta, g))
                align_b = float(np.dot(tb, -g))
                if align_a < min_align or align_b < min_align:
                    continue
                score = (align_a + align_b) - d / max_gap  # closer + straighter
                cands.append((score, d, a, b, align_a, align_b))

        if not cands:
            return self
        cands.sort(reverse=True)

        g2 = _copy.deepcopy(self)
        adj = g2.adjacency
        deg2 = {n.id: len(adj.get(n.id, [])) for n in g2.nodes}

        # track merged component reps via union of component ids
        comp_rep = {ci: ci for ci in set(comp_of.values())}

        def croot(c):
            while comp_rep[c] != c:
                comp_rep[c] = comp_rep[comp_rep[c]]
                c = comp_rep[c]
            return c

        eid = (max((e.id for e in g2.edges), default=-1)) + 1
        added = 0
        nbyid = {n.id: n for n in g2.nodes}
        for score, d, a, b, aa, ab in cands:
            if max_added is not None and added >= max_added:
                break
            ca, cb = croot(comp_of[a]), croot(comp_of[b])
            if ca == cb:
                continue
            if deg2[a] != 1 or deg2[b] != 1:
                continue
            na, nb = nbyid[a], nbyid[b]
            e = GraphEdge(id=eid, source=a, target=b, length=float(d),
                          pixel_count=max(1, int(d)),
                          mean_width=(na.width + nb.width) / 2,
                          mean_intensity=(na.intensity + nb.intensity) / 2,
                          curvature=0.0)
            g2.edges.append(e)
            adj.setdefault(a, []).append(b)
            adj.setdefault(b, []).append(a)
            deg2[a] += 1
            deg2[b] += 1
            comp_rep[ca] = cb
            eid += 1
            added += 1

        for n in g2.nodes:
            n.degree = len(adj.get(n.id, []))
        g2.adjacency = dict(adj)
        return g2

    def add_leaf_node(self, position, width, intensity, orientation,
                      node_type, nearest_node_id, edge_length):
        """
        Append a leaf node (e.g. optimizer-added point) connected to nearest_node_id.
        Returns the new node's id.
        """
        new_id = len(self.nodes)
        node = GraphNode(
            id=new_id, position=position,
            width=width, intensity=intensity,
            orientation=orientation, node_type=node_type,
            degree=1
        )
        self.nodes.append(node)

        edge = GraphEdge(
            id=len(self.edges), source=new_id, target=nearest_node_id,
            length=max(edge_length, 1.0),
            pixel_count=max(1, int(edge_length)),
            mean_width=(width + self.nodes[nearest_node_id].width) / 2,
            mean_intensity=(intensity + self.nodes[nearest_node_id].intensity) / 2,
            curvature=0.0
        )
        self.edges.append(edge)

        if new_id not in self.adjacency:
            self.adjacency[new_id] = []
        self.adjacency[new_id].append(nearest_node_id)
        if nearest_node_id not in self.adjacency:
            self.adjacency[nearest_node_id] = []
        self.adjacency[nearest_node_id].append(new_id)

        # Refresh degrees for both endpoints
        node.degree = len(self.adjacency[new_id])
        self.nodes[nearest_node_id].degree = len(self.adjacency[nearest_node_id])

        return new_id

    def summary(self) -> dict:
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


# ============================================================
#  PHASE 1: Trace skeleton branches between critical points
# ============================================================

def _get_8_neighbors(y, x, h, w):
    """Return all valid 8-connected neighbor coordinates."""
    neighbors = []
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                neighbors.append((ny, nx))
    return neighbors


def trace_skeleton_branches(skeleton, ep_mask, jn_mask):
    """
    Trace all branches of the skeleton between critical points.
    
    A branch is a path along the skeleton from one critical point
    (endpoint or junction) to another critical point, or from a
    critical point to a dead end.
    
    Returns:
        branches: list of dicts with keys:
            'start': (y, x) of start critical pixel
            'end': (y, x) of end critical pixel (or dead end)
            'path': list of (y, x) tuples for all pixels in this branch
    """
    h, w = skeleton.shape
    
    # Build critical pixel set
    critical_pixels = set()
    for y, x in np.argwhere(ep_mask):
        critical_pixels.add((int(y), int(x)))
    for y, x in np.argwhere(jn_mask):
        critical_pixels.add((int(y), int(x)))
    
    # Track which non-critical skeleton pixels have been used
    visited = set()
    branches = []
    
    # For each critical pixel, walk in each connected direction
    for cp in critical_pixels:
        cy, cx = cp
        if skeleton[cy, cx] == 0:
            continue
            
        # Find skeleton neighbors of this critical pixel
        neighbors = _get_8_neighbors(cy, cx, h, w)
        skel_neighbors = [(ny, nx) for ny, nx in neighbors
                          if skeleton[ny, nx] > 0]
        
        for start_ny, start_nx in skel_neighbors:
            start = (start_ny, start_nx)
            
            # If this neighbor is another critical pixel, that's a 
            # direct edge (branch of length 1)
            if start in critical_pixels:
                # Only record once (from lower to higher to avoid duplicates)
                if cp < start:
                    branches.append({
                        'start': cp,
                        'end': start,
                        'path': [cp, start]
                    })
                continue
            
            # If this non-critical pixel was already visited, skip
            if start in visited:
                continue
            
            # Walk along the skeleton from this neighbor
            path = [cp, start]
            visited.add(start)
            current = start
            prev = cp
            
            while True:
                # Get skeleton neighbors of current pixel
                nbrs = _get_8_neighbors(current[0], current[1], h, w)
                
                # Filter to skeleton pixels that aren't the previous pixel
                next_candidates = []
                for ny, nx in nbrs:
                    if (ny, nx) == prev:
                        continue
                    if skeleton[ny, nx] == 0:
                        continue
                    next_candidates.append((ny, nx))
                
                if len(next_candidates) == 0:
                    # Dead end — branch terminates here
                    branches.append({
                        'start': cp,
                        'end': current,
                        'path': path
                    })
                    break
                
                # Check if any candidate is a critical pixel
                critical_next = [c for c in next_candidates if c in critical_pixels]
                if critical_next:
                    # Reached another critical pixel — branch complete
                    dest = critical_next[0]
                    path.append(dest)
                    branches.append({
                        'start': cp,
                        'end': dest,
                        'path': path
                    })
                    break
                
                # Filter out already-visited pixels
                unvisited = [c for c in next_candidates if c not in visited]
                
                if len(unvisited) == 0:
                    # All neighbors visited — terminate
                    branches.append({
                        'start': cp,
                        'end': current,
                        'path': path
                    })
                    break
                elif len(unvisited) == 1:
                    # Simple continuation
                    nxt = unvisited[0]
                    visited.add(nxt)
                    path.append(nxt)
                    prev = current
                    current = nxt
                else:
                    # Multiple unvisited neighbors — undetected junction
                    # Pick the one most aligned with current direction
                    dy0 = current[0] - prev[0]
                    dx0 = current[1] - prev[1]
                    best = None
                    best_dot = -999
                    for ny, nx in unvisited:
                        dy1 = ny - current[0]
                        dx1 = nx - current[1]
                        dot = dy0 * dy1 + dx0 * dx1
                        if dot > best_dot:
                            best_dot = dot
                            best = (ny, nx)
                    visited.add(best)
                    path.append(best)
                    prev = current
                    current = best
    
    # Also handle isolated skeleton loops (no critical pixels)
    # Find any unvisited skeleton pixels
    all_skel = set((int(y), int(x)) for y, x in np.argwhere(skeleton > 0))
    covered = visited | critical_pixels
    uncovered = all_skel - covered
    
    while uncovered:
        start = uncovered.pop()
        path = [start]
        visited.add(start)
        current = start
        prev = None
        
        while True:
            nbrs = _get_8_neighbors(current[0], current[1], h, w)
            next_opts = [(ny, nx) for ny, nx in nbrs
                         if skeleton[ny, nx] > 0 and (ny, nx) != prev
                         and (ny, nx) not in visited]
            if not next_opts:
                break
            nxt = next_opts[0]
            visited.add(nxt)
            uncovered.discard(nxt)
            path.append(nxt)
            prev = current
            current = nxt
        
        if len(path) >= 3:
            # Check if this is a closed loop (last pixel adjacent to first)
            end_nbrs = _get_8_neighbors(current[0], current[1], h, w)
            if start in [(ny, nx) for ny, nx in end_nbrs
                         if skeleton[ny, nx] > 0]:
                path.append(start)  # close the loop
            branches.append({
                'start': path[0],
                'end': path[-1],
                'path': path
            })
        elif len(path) >= 2:
            branches.append({
                'start': path[0],
                'end': path[-1],
                'path': path
            })
    
    # Deduplicate branches (same path traced from both ends)
    seen = set()
    unique_branches = []
    for b in branches:
        # Create a canonical key: sorted tuple of (start, end)
        key = (min(b['start'], b['end']), max(b['start'], b['end']),
               len(b['path']))
        if key not in seen:
            seen.add(key)
            unique_branches.append(b)
    
    # Coverage check
    n_skeleton = np.count_nonzero(skeleton)
    n_covered = len(visited | critical_pixels)
    if n_skeleton > 0:
        coverage = n_covered / n_skeleton
        if coverage < 0.9:
            print(f'  [GRAPH WARNING] Skeleton coverage: {coverage:.1%} '
                  f'({n_covered}/{n_skeleton} pixels)')
    
    return unique_branches


# ============================================================
#  PHASE 2 & 3: Assign nodes to branches, build edges
# ============================================================

def _compute_edge_attributes(path_pixels, dist_transform, original_img):
    """Compute length, mean_width, mean_intensity, curvature for a path."""
    path_arr = np.array(path_pixels)
    
    if len(path_arr) < 2:
        return 0.0, 0.0, 0.0, 0.0
    
    # Length
    diffs = np.diff(path_arr, axis=0)
    seg_lengths = np.sqrt(np.sum(diffs ** 2, axis=1))
    length = float(np.sum(seg_lengths))
    
    # Mean width
    widths = [dist_transform[y, x] for y, x in path_pixels]
    mean_width = float(np.mean(widths))
    
    # Mean intensity
    img_f = original_img.astype(float) / 255.0
    intens = [img_f[y, x] for y, x in path_pixels]
    mean_intensity = float(np.mean(intens))
    
    # Curvature
    if len(path_arr) > 2:
        angles = np.arctan2(diffs[:, 0].astype(float), diffs[:, 1].astype(float))
        angle_diffs = np.abs(np.diff(angles))
        angle_diffs = np.minimum(angle_diffs, 2 * np.pi - angle_diffs)
        total_curvature = float(np.sum(angle_diffs))
        curvature = total_curvature / max(length, 1e-6)
    else:
        curvature = 0.0
    
    return length, mean_width, mean_intensity, curvature


def build_nanograph(points, intensities, widths, orientations, types,
                    skeleton, dist_transform, original_img, shape,
                    image_type='sparse', cfg=None):
    """
    Build a formal Nanograph using skeleton-walk approach.
    
    Phase 1: Trace skeleton branches between critical points
    Phase 2: Assign every node to a branch by pixel proximity
    Phase 3: Build edges between consecutive nodes along each branch
    """
    from .config import DEFAULT_CONFIG, NanographConfig, GraphConfig
    from .skeleton import skeletonize_and_classify

    gc = cfg if isinstance(cfg, GraphConfig) else (
         cfg.graph if isinstance(cfg, NanographConfig) else DEFAULT_CONFIG.graph)

    if len(points) == 0:
        return Nanograph(nodes=[], edges=[], adjacency={},
                         shape=shape, image_type=image_type)

    # --- Create nodes ---
    nodes = []
    for i, (pt, w, inten, ori, t) in enumerate(
            zip(points, widths, intensities, orientations, types)):
        nodes.append(GraphNode(
            id=i, position=(int(pt[0]), int(pt[1])),
            width=float(w), intensity=float(inten),
            orientation=float(ori), node_type=str(t)
        ))

    # --- Phase 1: Trace skeleton branches ---
    # Recompute ep_mask and jn_mask from skeleton to ensure consistency
    conn_kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    deg = cv2.filter2D(skeleton.astype(np.uint8), -1, conn_kernel) * skeleton
    ep_mask = (deg == 1) & (skeleton > 0)
    jn_mask = (deg >= 3) & (skeleton > 0)
    
    branches = trace_skeleton_branches(skeleton, ep_mask, jn_mask)
    
    # --- Phase 2: Assign nodes to branches ---
    # Build pixel-to-branch lookup: (y, x) -> (branch_idx, position_in_path)
    pixel_to_branch = {}
    for bi, branch in enumerate(branches):
        for pi, (y, x) in enumerate(branch['path']):
            key = (y, x)
            # If pixel is in multiple branches (junction), keep the first
            if key not in pixel_to_branch:
                pixel_to_branch[key] = (bi, pi)
    
    # Assign each node to a branch
    node_branch = {}  # node_id -> (branch_idx, position_in_path)
    orphan_nodes = []
    
    for node in nodes:
        pos = node.position
        
        # Direct match
        if pos in pixel_to_branch:
            node_branch[node.id] = pixel_to_branch[pos]
            continue
        
        # Search nearby pixels (within tolerance)
        tolerance = max(int(node.width), 5)
        found = False
        for radius in range(1, tolerance + 1):
            if found:
                break
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if abs(dy) != radius and abs(dx) != radius:
                        continue  # only check the perimeter of expanding square
                    check = (pos[0] + dy, pos[1] + dx)
                    if check in pixel_to_branch:
                        node_branch[node.id] = pixel_to_branch[check]
                        found = True
                        break
        
        if not found:
            orphan_nodes.append(node.id)
    
    # --- Phase 3: Build edges between consecutive nodes on each branch ---
    edges = []
    adjacency = defaultdict(list)
    edge_id = 0
    
    # Build KDTree of all node positions for cycle-closing lookups
    node_pos_arr = np.array([n.position for n in nodes])
    node_kd = KDTree(node_pos_arr)
    
    for bi, branch in enumerate(branches):
        # Find all nodes assigned to this branch
        branch_nodes = [(nid, pos_in_path)
                        for nid, (b_idx, pos_in_path) in node_branch.items()
                        if b_idx == bi]
        
        # --- Cycle-closing: include start/end critical nodes even if
        #     assigned to a different branch ---
        existing_ids = {nid for nid, _ in branch_nodes}
        
        d_s, i_s = node_kd.query(branch['start'])
        if i_s not in existing_ids and d_s <= 5:
            branch_nodes.append((i_s, 0))
            existing_ids.add(i_s)
        
        d_e, i_e = node_kd.query(branch['end'])
        if i_e not in existing_ids and d_e <= 5:
            branch_nodes.append((i_e, len(branch['path']) - 1))
            existing_ids.add(i_e)
        
        # For closed loops (start == end pixel), add closing edge after
        is_closed_loop = (branch['start'] == branch['end'] and
                          len(branch['path']) >= 4)
        
        if len(branch_nodes) < 2:
            continue
        
        # Sort by position along the branch path
        branch_nodes.sort(key=lambda x: x[1])
        
        # Create edges between consecutive nodes
        for k in range(len(branch_nodes) - 1):
            nid_a, pos_a = branch_nodes[k]
            nid_b, pos_b = branch_nodes[k + 1]
            
            # Get the path segment between these two nodes
            path_segment = branch['path'][pos_a:pos_b + 1]
            
            if len(path_segment) < 2:
                # Adjacent pixels — create minimal edge
                na = nodes[nid_a]
                nb = nodes[nid_b]
                dy = nb.position[0] - na.position[0]
                dx = nb.position[1] - na.position[1]
                length = float(np.sqrt(dy ** 2 + dx ** 2))
                edge = GraphEdge(
                    id=edge_id, source=nid_a, target=nid_b,
                    length=max(length, 1.0), pixel_count=1,
                    mean_width=(na.width + nb.width) / 2,
                    mean_intensity=(na.intensity + nb.intensity) / 2,
                    curvature=0.0
                )
            else:
                length, mean_w, mean_i, curvature = _compute_edge_attributes(
                    path_segment, dist_transform, original_img)
                edge = GraphEdge(
                    id=edge_id, source=nid_a, target=nid_b,
                    length=length, pixel_count=len(path_segment),
                    mean_width=mean_w, mean_intensity=mean_i,
                    curvature=curvature,
                    path_pixels=np.array(path_segment) if gc.compute_edge_features else None
                )
            
            edges.append(edge)
            adjacency[nid_a].append(nid_b)
            adjacency[nid_b].append(nid_a)
            edge_id += 1
        
        # Close loop: connect last node back to first node
        if is_closed_loop and len(branch_nodes) >= 2:
            first_nid = branch_nodes[0][0]
            last_nid = branch_nodes[-1][0]
            if first_nid != last_nid:
                na = nodes[first_nid]
                nb = nodes[last_nid]
                dy = nb.position[0] - na.position[0]
                dx = nb.position[1] - na.position[1]
                length = float(np.sqrt(dy ** 2 + dx ** 2))
                edge = GraphEdge(
                    id=edge_id, source=last_nid, target=first_nid,
                    length=max(length, 1.0), pixel_count=max(1, int(length)),
                    mean_width=(na.width + nb.width) / 2,
                    mean_intensity=(na.intensity + nb.intensity) / 2,
                    curvature=0.0
                )
                edges.append(edge)
                adjacency[last_nid].append(first_nid)
                adjacency[first_nid].append(last_nid)
                edge_id += 1
    
    # --- Connect orphan nodes to nearest non-orphan node ---
    if orphan_nodes and (len(nodes) - len(orphan_nodes)) > 0:
        connected_positions = []
        connected_ids = []
        for node in nodes:
            if node.id not in orphan_nodes:
                connected_positions.append(node.position)
                connected_ids.append(node.id)
        
        if connected_positions:
            tree = KDTree(connected_positions)
            for oid in orphan_nodes:
                onode = nodes[oid]
                dist, idx = tree.query(onode.position)
                target_id = connected_ids[idx]
                
                edge = GraphEdge(
                    id=edge_id, source=oid, target=target_id,
                    length=float(dist), pixel_count=max(1, int(dist)),
                    mean_width=(onode.width + nodes[target_id].width) / 2,
                    mean_intensity=(onode.intensity + nodes[target_id].intensity) / 2,
                    curvature=0.0
                )
                edges.append(edge)
                adjacency[oid].append(target_id)
                adjacency[target_id].append(oid)
                edge_id += 1
    
    # Update node degrees
    for node in nodes:
        node.degree = len(adjacency.get(node.id, []))
    
    # Build result
    graph = Nanograph(
        nodes=nodes, edges=edges, adjacency=dict(adjacency),
        shape=shape, image_type=image_type
    )
    
    # Connectivity check
    n_components = len(graph.get_connected_components())
    n_branches = len(branches)
    n_assigned = len(node_branch)
    n_orphans = len(orphan_nodes)
    
    print(f'  [GRAPH] {n_branches} branches traced, '
          f'{n_assigned}/{len(nodes)} nodes assigned, '
          f'{n_orphans} orphans, '
          f'{len(edges)} edges, {n_components} components')
    
    return graph


def nanograph_morphometry(graph: Nanograph) -> dict:
    """Extract biological descriptors from the formal graph."""
    if graph.n_nodes == 0:
        return {}

    summary = graph.summary()
    
    widths = graph.node_widths
    summary['mean_width_px'] = float(np.mean(widths))
    summary['std_width_px'] = float(np.std(widths))
    summary['max_width_px'] = float(np.max(widths))
    summary['min_width_px'] = float(np.min(widths))
    summary['mean_intensity'] = float(np.mean(graph.node_intensities))

    if graph.edges:
        edge_lengths = [e.length for e in graph.edges]
        edge_curvatures = [e.curvature for e in graph.edges]
        summary['mean_edge_length'] = float(np.mean(edge_lengths))
        summary['std_edge_length'] = float(np.std(edge_lengths))
        summary['max_edge_length'] = float(np.max(edge_lengths))
        summary['mean_edge_curvature'] = float(np.mean(edge_curvatures))
        summary['max_edge_curvature'] = float(np.max(edge_curvatures))
        summary['network_length'] = float(np.sum(edge_lengths))

    junctions = graph.get_junctions()
    if junctions:
        degrees = [n.degree for n in junctions]
        summary['mean_junction_degree'] = float(np.mean(degrees))
        summary['max_junction_degree'] = int(np.max(degrees))

    pts = graph.node_positions
    summary['spatial_extent_rows'] = float(np.ptp(pts[:, 0]))
    summary['spatial_extent_cols'] = float(np.ptp(pts[:, 1]))
    summary['bounding_box_area'] = (
        float(np.ptp(pts[:, 0])) * float(np.ptp(pts[:, 1]))
    )

    from collections import Counter
    summary['type_distribution'] = dict(Counter(graph.node_types))
    
    return summary