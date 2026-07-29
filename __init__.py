"""
Nanograph v5 — Adaptive structural encoding for microscopy.

v5 innovations:
  - Spatial background grid (16×16 bilinearly interpolated)
  - Persistent homology preservation metrics (β₀, β₁, Wasserstein distance)
  - Graph-predictive compression (neighbor-predicted residuals)
  - Foreground residual coding (DCT-based detail recovery)
  - Multi-codec rate-distortion evaluation framework

Usage:
    from nanograph_v4 import nanograph_encode, nanograph_decode, NanographConfig

    result = nanograph_encode('image.png')
    recon, pts, ints, widths, types, oris, shape = nanograph_decode(result.compressed)
"""

from .config import NanographConfig, DEFAULT_CONFIG
from .api import nanograph_encode, nanograph_decode, NanographResult
from .graph import Nanograph, nanograph_morphometry, build_nanograph
from .track import (track_mitochondria, MitoTrack, CurveTrack,
                    propagate_curve, seed_curve_from_graph)
from .evaluate import (batch_evaluate, evaluate_result, print_comparison,
                       rate_distortion_curve)
from .utils import (compute_betti_numbers, betti_error,
                    topology_preservation_score, graph_topology_score)

__version__ = '5.0.0'

__all__ = [
    'nanograph_encode', 'nanograph_decode', 'NanographResult',
    'NanographConfig', 'DEFAULT_CONFIG',
    'Nanograph', 'nanograph_morphometry', 'build_nanograph',
    'track_mitochondria', 'MitoTrack', 'CurveTrack',
    'propagate_curve', 'seed_curve_from_graph',
    'batch_evaluate', 'evaluate_result', 'print_comparison',
    'rate_distortion_curve',
    'compute_betti_numbers', 'betti_error', 'topology_preservation_score',
]
