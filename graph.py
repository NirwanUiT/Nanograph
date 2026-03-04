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


def _trace_edge_between_nodes(skeleton, start_yx, end_yx, node_positions_set,
                               max_trace_len=5000):
    """
    Trace skeleton pixels between two adjacent nodes using BFS.
    Returns the pixel path (list of (y, x) tuples) or None if no path found.
    """
    h, w = skeleton.shape
    visited = set()
    visited.add(start_yx)
    queue = [(start_yx, [start_yx])]

    while queue:
        (cy, cx), path = queue.pop(0)
        if len(path) > max_trace_len:
            return None

        # Check 8-connected neighbors
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and (ny, nx) not in visited:
                    if skeleton[ny, nx] > 0:
                        new_path = path + [(ny, nx)]
                        if (ny, nx) == end_yx:
                            return new_path
                        # Don't cross through other nodes
                        if (ny, nx) not in node_positions_set:
                            visited.add((ny, nx))
                            queue.append(((ny, nx), new_path))
    return None


def build_nanograph(points, intensities, widths, orientations, types,
                    skeleton, dist_transform, original_img, shape,
                    image_type='sparse', cfg=None):
    """
    Build a formal Nanograph from extracted points and skeleton.
    
    This is the core v4 function that transforms a point cloud into
    a proper graph with edges traced along the skeleton.
    
    Strategy:
    1. Place all extracted points as nodes
    2. For critical points (endpoints, junctions), trace skeleton paths
       between nearby critical points to form edges
    3. For sampled points on edges, assign them to the nearest edge
    4. Compute edge attributes (length, curvature, mean width)
    """
    from .config import DEFAULT_CONFIG, NanographConfig, GraphConfig

    gc = cfg if isinstance(cfg, GraphConfig) else (
         cfg.graph if isinstance(cfg, NanographConfig) else DEFAULT_CONFIG.graph)

    if len(points) == 0:
        return Nanograph(nodes=[], edges=[], adjacency={},
                         shape=shape, image_type=image_type)

    # --- Step 1: Create nodes ---
    nodes = []
    for i, (pt, w, inten, ori, t) in enumerate(
            zip(points, widths, intensities, orientations, types)):
        nodes.append(GraphNode(
            id=i, position=(int(pt[0]), int(pt[1])),
            width=float(w), intensity=float(inten),
            orientation=float(ori), node_type=str(t)
        ))

    # --- Step 2: Build edges using skeleton connectivity ---
    # Find which nodes are on the skeleton and close to each other
    node_positions = np.array([n.position for n in nodes])
    node_pos_set = set(tuple(p) for p in node_positions)

    # Build a KDTree of node positions for fast neighbor lookup
    tree = KDTree(node_positions)

    # For each pair of nearby nodes, check if they're connected via skeleton
    edges = []
    adjacency = defaultdict(list)
    edge_id = 0

    # Find candidate pairs: nodes within reasonable skeleton distance
    max_edge_dist = 50  # max Euclidean distance to consider
    pairs_checked = set()

    for i, node in enumerate(nodes):
        # Find nearby nodes
        nearby_idx = tree.query_ball_point(node.position, r=max_edge_dist)

        for j in nearby_idx:
            if i >= j:
                continue
            pair_key = (min(i, j), max(i, j))
            if pair_key in pairs_checked:
                continue
            pairs_checked.add(pair_key)

            ni, nj = nodes[i], nodes[j]
            pi = ni.position
            pj = nj.position

            # Quick check: both must be on or very near the skeleton
            if skeleton[pi[0], pi[1]] == 0 and skeleton[pj[0], pj[1]] == 0:
                continue

            # Trace path along skeleton between these two nodes
            # Only trace between critical points (endpoints/junctions)
            # or between a critical point and a nearby sampled point
            path = _trace_edge_between_nodes(
                skeleton, pi, pj, 
                node_pos_set - {pi, pj},
                max_trace_len=max_edge_dist * 3
            )

            if path is not None and len(path) >= gc.min_edge_length:
                path_arr = np.array(path)

                # Edge length (sum of pixel-to-pixel distances)
                diffs = np.diff(path_arr, axis=0)
                seg_lengths = np.sqrt(np.sum(diffs**2, axis=1))
                edge_length = float(np.sum(seg_lengths))

                # Mean width along edge
                edge_widths = [dist_transform[y, x] for y, x in path]
                mean_w = float(np.mean(edge_widths))

                # Mean intensity along edge
                img_f = original_img.astype(float) / 255.0
                edge_intens = [img_f[y, x] for y, x in path]
                mean_i = float(np.mean(edge_intens))

                # Curvature: total angular change / length
                if len(path) > 2:
                    angles = np.arctan2(diffs[:, 0], diffs[:, 1])
                    angle_diffs = np.abs(np.diff(angles))
                    # Wrap around
                    angle_diffs = np.minimum(angle_diffs, 2 * np.pi - angle_diffs)
                    total_curvature = float(np.sum(angle_diffs))
                    curvature = total_curvature / max(edge_length, 1e-6)
                else:
                    curvature = 0.0

                edge = GraphEdge(
                    id=edge_id, source=i, target=j,
                    length=edge_length, pixel_count=len(path),
                    mean_width=mean_w, mean_intensity=mean_i,
                    curvature=curvature,
                    path_pixels=path_arr if gc.compute_edge_features else None
                )
                edges.append(edge)
                adjacency[i].append(j)
                adjacency[j].append(i)
                edge_id += 1

    # Update node degrees
    for node in nodes:
        node.degree = len(adjacency.get(node.id, []))

    # --- Step 3: Connect isolated sampled points to nearest connected node ---
    connected_nodes = set()
    for nid in adjacency:
        connected_nodes.add(nid)
    
    for node in nodes:
        if node.id not in connected_nodes and node.node_type == 'sampled':
            # Find nearest connected node
            dists, idxs = tree.query(node.position, k=min(10, len(nodes)))
            if not isinstance(idxs, np.ndarray):
                idxs = [idxs]
                dists = [dists]
            for d, idx in zip(dists, idxs):
                if idx != node.id and idx in connected_nodes:
                    # Create a simple edge
                    ni = node
                    nj = nodes[idx]
                    edge_length = float(d)
                    edge = GraphEdge(
                        id=edge_id, source=ni.id, target=nj.id,
                        length=edge_length, pixel_count=max(1, int(d)),
                        mean_width=(ni.width + nj.width) / 2,
                        mean_intensity=(ni.intensity + nj.intensity) / 2,
                        curvature=0.0
                    )
                    edges.append(edge)
                    adjacency[ni.id].append(nj.id)
                    adjacency[nj.id].append(ni.id)
                    ni.degree = len(adjacency[ni.id])
                    nj.degree = len(adjacency[nj.id])
                    connected_nodes.add(ni.id)
                    edge_id += 1
                    break

    return Nanograph(
        nodes=nodes, edges=edges, adjacency=dict(adjacency),
        shape=shape, image_type=image_type
    )


def nanograph_morphometry(graph: Nanograph) -> dict:
    """
    v4: Extract biological descriptors from the formal graph.
    Much richer than v3 since we have edge information.
    """
    if graph.n_nodes == 0:
        return {}

    summary = graph.summary()
    
    # Add distribution stats
    widths = graph.node_widths
    summary['mean_width_px'] = float(np.mean(widths))
    summary['std_width_px'] = float(np.std(widths))
    summary['max_width_px'] = float(np.max(widths))
    summary['min_width_px'] = float(np.min(widths))
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
