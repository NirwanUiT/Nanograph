# Nanograph v4 — Changelog

## Architecture: Modular Package

The monolithic notebook has been refactored into **12 modules** for VS Code development:

```
nanograph_v4/
├── __init__.py        # Public API
├── config.py          # 92 configurable parameters (was 78)
├── detect.py          # Image type auto-detection
├── preprocess.py      # CLAHE, BG subtraction, reflectivity + bg_grid
├── segment.py         # Multi-segmenter with cascading early-exit
├── skeleton.py        # Skeletonization, point extraction, orientations
├── graph.py           # NEW: Formal graph with nodes, edges, adjacency
├── reconstruct.py     # Oriented PSF reconstruction + topology optimizer
├── compress.py        # Quantise + delta + zlib with orientations + bg grid
├── classify.py        # Structure shape classification
├── api.py             # nanograph_encode() / nanograph_decode()
└── evaluate.py        # Batch benchmarking + JPEG comparison + CSV export
```

## Key Improvements

### 1. Formal Graph Structure (`graph.py`) — NEW
**What:** Nanograph is now an explicit graph, not just a point cloud.
- **Nodes** have: position, width, intensity, orientation, type, degree
- **Edges** have: source, target, length, pixel_count, mean_width, mean_intensity, curvature
- **Adjacency** stored explicitly for graph traversal
- Built-in: connected components, subgraph extraction, morphometry

**Why:** This is the intellectual core of the thesis pivot. It enables:
- Structural queries ("find all branched networks with width > 4px")
- Temporal tracking (compare graph topology across timepoints)
- Graph compression (delta-encode along edges)
- GNN-based analysis
- Database-searchable microscopy

### 2. Oriented Elliptical PSF (`reconstruct.py`)
**What:** Reconstruction uses anisotropic kernels aligned to skeleton tangent.
- Computes local orientation at each skeleton pixel via structure tensor
- Groups points by (sigma_bin, orientation_bin) for efficient FFT convolution
- Configurable aspect ratio (default 2.5x elongation along filament)

**Why:** Filaments are not circular. Isotropic Gaussians waste energy perpendicular to the structure. Oriented PSFs should significantly improve FG-SSIM on dense networks.

### 3. Low-Rank Background Grid (`preprocess.py`, `compress.py`)
**What:** Instead of storing 1 byte (mean background), stores a 16×16 grid (256 bytes).
- Captured via cv2.resize(background_model, (16, 16))
- Upsampled with bicubic interpolation + Gaussian smoothing
- Stored in compressed payload alongside point data

**Why:** Full-image PSNR was terrible (13-18 dB) because a single mean value can't represent spatial background variation. A 256-byte grid captures most of the background structure at negligible cost, potentially adding 5-10 dB to full-image PSNR.

### 4. Cascading Segmentation (`segment.py`)
**What:** Run cheap methods (Otsu, Frangi, Meijering) first. Only invoke SAM if best cheap score < 0.55.

**Impact on runtime:**
- Sparse image: was ~6.4s (99% in SAM), now ~170ms (SAM skipped)
- Dense image: was ~11.8s, now ~1.5s or less (SAM skipped when Otsu scores well)

### 5. Per-Point Orientation Storage (`compress.py`)
**What:** Each point now stores a 1-byte quantized orientation angle (256 levels over [0, π)).
- Adds 1 byte/point to raw encoding (7 bytes vs 6)
- Enables oriented reconstruction from compressed data (round-trip)
- Quantization error: ~0.006 radians (< 0.4°)

### 6. Batch Evaluation Framework (`evaluate.py`)
**What:** `batch_evaluate(paths, ...)` runs the full pipeline on N images and exports:
- Per-image metrics CSV with all PSNR/SSIM/IoU/topology/graph stats
- Byte-matched JPEG comparison for every image
- Aggregate statistics (mean, std, min, max)

### 7. Enhanced Morphometry (`graph.py`)
**What:** `nanograph_morphometry(graph)` now reports edge-based metrics:
- Mean/std/max edge length
- Mean/max curvature
- Network total length
- Junction degree statistics
- Branching analysis

## Header Format Change (v3 → v4)

| Field | v3 | v4 |
|-------|----|----|
| H, W | uint16 × 2 | uint16 × 2 |
| n_points | uint32 | uint32 |
| flags | uint8 (1 bit: has_types) | uint8 (3 bits: types, orientation, bg_grid) |
| mean_bg | uint8 | uint8 |
| bg_grid_size | — | uint8 |
| reserved | — | uint8 |
| **Total** | **10 bytes** | **12 bytes** |

## Migration Guide

```python
# v3 (notebook)
result = nanograph_encode('image.png', sam_model=sam)

# v4 (package) — same API, new features
from nanograph_v4 import nanograph_encode, NanographConfig
result = nanograph_encode('image.png', sam_model=sam)

# New: access the graph
graph = result.graph
print(graph.summary())
components = graph.get_connected_components()

# New: batch evaluation
from nanograph_v4 import batch_evaluate
results = batch_evaluate(image_paths, output_csv='results.csv')
```
