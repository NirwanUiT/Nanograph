"""
Nanograph v4 — Adaptive structural encoding for microscopy.

Usage:
    from nanograph_v4 import nanograph_encode, nanograph_decode, NanographConfig

    result = nanograph_encode('image.png')
    recon, pts, ints, widths, types, oris, shape = nanograph_decode(result.compressed)
"""

from .config import NanographConfig, DEFAULT_CONFIG
from .api import nanograph_encode, nanograph_decode, NanographResult
from .graph import Nanograph, nanograph_morphometry, build_nanograph
from .evaluate import batch_evaluate, evaluate_result, print_comparison

__version__ = '4.0.0'

__all__ = [
    'nanograph_encode', 'nanograph_decode', 'NanographResult',
    'NanographConfig', 'DEFAULT_CONFIG',
    'Nanograph', 'nanograph_morphometry', 'build_nanograph',
    'batch_evaluate', 'evaluate_result', 'print_comparison',
]
