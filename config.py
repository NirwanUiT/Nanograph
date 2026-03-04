"""
Nanograph v4 Configuration
All tunable parameters in one place. Override any subset:
    cfg = NanographConfig()
    cfg.segment.w_contrast = 0.35
    result = nanograph_encode(img, config=cfg)
"""

from copy import deepcopy
from dataclasses import dataclass, field, asdict


@dataclass
class PreprocessConfig:
    """Preprocessing parameters."""
    clahe_clip_limit: float = 2.0
    clahe_tile_size: int = 8
    bg_kernel_size: int = 50
    density_sigma: float = 20.0
    mean_sigma: float = 10.0
    edge_boost: float = 0.3
    bilateral_d: int = 9
    bilateral_sigma_color: float = 75.0
    bilateral_sigma_space: float = 75.0


@dataclass
class SegmentConfig:
    """Segmentation selection parameters."""
    sam_points_per_side: int = 32
    sam_pred_iou_thresh: float = 0.60
    sam_stability_thresh: float = 0.65
    sam_min_area_frac: float = 0.0003
    sam_max_area_frac: float = 0.6
    sam_crop_n_layers: int = 1
    sam_crop_downscale: int = 2
    sam_min_region_area: int = 30
    sam_bridge_kernel: int = 5
    frangi_sigmas: tuple = (1, 2, 3, 5)
    frangi_beta: float = 0.5
    frangi_gamma: float = 15.0
    meijering_sigmas: tuple = (1, 2, 3, 5)
    morph_kernel_size: int = 3
    morph_min_area_frac: float = 0.001
    # Scoring weights (sum to 1.0)
    w_contrast: float = 0.25
    w_coverage: float = 0.20
    w_coherence: float = 0.15
    w_skeleton: float = 0.15
    w_sharpness: float = 0.10
    w_shape_match: float = 0.15
    coherence_floor: float = 0.3
    boundary_sobel_ksize: int = 3
    expected_fg_lo: float = 0.5
    expected_fg_hi: float = 35.0
    # v4: Cascading early-exit
    cascade_confidence_thresh: float = 0.55  # skip SAM if best cheap method exceeds this
    cascade_enable: bool = True


@dataclass
class ReconConfig:
    """Reconstruction parameters."""
    sigma_scale: float = 1.0
    min_sigma: float = 1.0
    max_sigma: float = 30.0
    sigma_quantize: float = 0.5
    blend_sigma: float = 5.0
    fg_presence_thresh: float = 0.01
    psf_max_radius: int = 30
    # v4: Oriented elliptical PSF
    use_oriented_psf: bool = True
    orientation_aspect_ratio: float = 2.5   # major/minor axis ratio for filament PSF
    orientation_smooth_sigma: float = 3.0   # smoothing for tangent estimation
    # v4: Low-rank background grid
    bg_grid_size: int = 16   # NxN downsampled background grid (0 = use mean only)
    bg_grid_sigma: float = 2.0  # smoothing when upsampling the grid


@dataclass
class OptimizerConfig:
    """Topology optimiser parameters."""
    max_iter: int = 20
    pts_per_iter: int = 30
    min_error: float = 0.02
    plateau_thresh: float = 0.001
    plateau_min_iter: int = 3
    error_blur_ksize: int = 7
    min_point_spacing: float = 2.0
    dense_pts_per_iter: int = 50
    dense_min_error: float = 0.015
    dense_max_iter: int = 25


@dataclass
class CompressConfig:
    """Compression parameters."""
    header_size: int = 10  # base header (shape + n_points + flags + mean_bg)
    width_quant_scale: float = 4.0
    zlib_level: int = 9
    raw_bytes_per_point: int = 16
    # v4: Graph-aware compression
    use_edge_delta: bool = True   # delta-encode along edges instead of row-sort
    store_bg_grid: bool = True    # embed the low-rank background grid
    store_orientation: bool = True  # store per-point orientation angle (1 byte)


@dataclass
class DetectConfig:
    """Image type auto-detection thresholds."""
    adapt_block_size: int = 51
    adapt_C: int = -5
    min_cc_area_abs: int = 20
    min_cc_area_frac: float = 0.0005
    dense_fg_pct_thresh: float = 10.0
    dense_mean_fg_thresh: float = 0.3
    dense_largest_cc_ratio: float = 0.3
    dense_max_n_components: int = 20
    sparse_bg_kernel: int = 50
    sparse_spacing: int = 3
    sparse_width_cap: float = 30.0
    dense_bg_kernel: int = 25
    dense_spacing: int = 2
    dense_width_cap: float = 15.0


@dataclass
class ClassifyConfig:
    """Structure classification thresholds."""
    min_area: int = 5
    punctate_max_area: int = 50
    punctate_min_compactness: float = 0.6
    curved_min_aspect: float = 2.5
    curved_min_skel_len: int = 15
    linear_min_aspect: float = 1.5


@dataclass
class EvalConfig:
    """Evaluation / comparison thresholds."""
    recon_mask_thresh: float = 0.05
    roundtrip_pass_psnr: float = 15.0
    png_compression: int = 9


@dataclass
class GraphConfig:
    """v4: Explicit graph structure parameters."""
    min_edge_length: int = 3          # minimum skeleton pixels between nodes to form edge
    edge_sample_spacing: int = 5      # sample edge attributes every N pixels
    store_adjacency: bool = True      # store edge list in compressed output
    compute_edge_features: bool = True  # compute curvature, mean width, etc. per edge


@dataclass
class NanographConfig:
    """Master configuration — every tunable parameter in one place."""
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    segment: SegmentConfig = field(default_factory=SegmentConfig)
    recon: ReconConfig = field(default_factory=ReconConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    compress: CompressConfig = field(default_factory=CompressConfig)
    detect: DetectConfig = field(default_factory=DetectConfig)
    classify: ClassifyConfig = field(default_factory=ClassifyConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)

    def for_image_type(self, image_type: str) -> 'NanographConfig':
        """Return a copy with type-specific overrides applied."""
        cfg = deepcopy(self)
        d = cfg.detect
        if image_type == 'dense':
            cfg.preprocess.bg_kernel_size = d.dense_bg_kernel
            cfg.segment.expected_fg_lo = 3.0
            cfg.segment.expected_fg_hi = 55.0
        else:
            cfg.preprocess.bg_kernel_size = d.sparse_bg_kernel
            cfg.segment.expected_fg_lo = 0.5
            cfg.segment.expected_fg_hi = 35.0
        return cfg

    def param_count(self) -> int:
        return sum(len(asdict(getattr(self, f)))
                   for f in ['preprocess', 'segment', 'recon', 'optimizer',
                             'compress', 'detect', 'classify', 'eval', 'graph'])


DEFAULT_CONFIG = NanographConfig()
