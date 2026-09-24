"""
Nanograph v5 Configuration
All tunable parameters in one place. Override any subset:
    cfg = NanographConfig()
    cfg.segment.w_contrast = 0.35
    result = nanograph_encode(img, config=cfg)

v5 additions: bg_grid_size, residual encoding, graph-predictive compression.
"""

from copy import deepcopy
from dataclasses import dataclass, field, asdict
import os

# Bundled learned-segmenter weight (clDice U-Net trained on independent NMI
# organelle data). Resolved at import; empty string if the weight is absent so
# the pipeline gracefully falls back to the classical cascade.
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_LEARNED_CKPT = os.path.join(_PKG_DIR, 'weights', 'unet_organelle_cldice.pt')
if not os.path.isfile(_DEFAULT_LEARNED_CKPT):
    _DEFAULT_LEARNED_CKPT = ''

# Curvilinear multi-domain weight (clDice U-Net trained on retinal vessels
# STARE/DRIVE, EPFL EM mitochondria and IRM microtubules, with polarity-agnostic
# augmentation). Used by NanographConfig.for_curvilinear() for dark-on-bright
# vessel/EM/filament imagery the organelle net transfers poorly to.
_CURVILINEAR_LEARNED_CKPT = os.path.join(_PKG_DIR, 'weights', 'unet_curvilinear_cldice.pt')
if not os.path.isfile(_CURVILINEAR_LEARNED_CKPT):
    _CURVILINEAR_LEARNED_CKPT = ''


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
    # Polarity: structures of interest may be bright-on-dark (fluorescence) or
    # dark-on-light (brightfield/absorption, e.g. retinal vasculature, EM).
    # 'auto' picks via a morphological top-hat ratio; 'bright' = no inversion;
    # 'dark' = invert the image so dark structures become the bright foreground.
    polarity: str = 'auto'            # 'auto' | 'bright' | 'dark'
    polarity_tophat_size: int = 15    # struct-element size for top-hat polarity test
    polarity_tophat_thresh: float = 1.3  # black/white top-hat ratio above which = dark


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
    morph_kernel_size: int = 5
    morph_iterations: int = 2
    morph_min_area_frac: float = 0.0005
    # Scoring weights (sum to 1.0)
    w_contrast: float = 0.30
    w_coverage: float = 0.10
    w_coherence: float = 0.15
    w_skeleton: float = 0.15
    w_sharpness: float = 0.15
    w_shape_match: float = 0.15
    coherence_floor: float = 0.3
    boundary_sobel_ksize: int = 3
    expected_fg_lo: float = 2.0
    expected_fg_hi: float = 20.0
    # v4: Cascading early-exit
    cascade_confidence_thresh: float = 0.65  # skip SAM if best cheap method exceeds this
    cascade_enable: bool = True
    # Learned U-Net foreground segmenter (trained on independent data, applied
    # zero-shot). Recovers dim/thin structure the classical cascade under-traces.
    # Enabled by default in 'candidate' mode: it competes with the classical
    # cascade by quality score, so it wins on organelle-like structure it was
    # trained for and falls back to classical on very different modalities.
    use_learned: bool = bool(_DEFAULT_LEARNED_CKPT)  # add U-Net mask as a candidate
    learned_ckpt: str = _DEFAULT_LEARNED_CKPT        # path to trained checkpoint (.pt)
    learned_thresh: float = 0.5      # sigmoid threshold for U-Net mask
    learned_mode: str = 'candidate'  # 'candidate' (compete by score) |
                                     # 'union' (learned | best-cheap) |
                                     # 'replace' (use learned directly)
    learned_union_with: str = 'best'  # 'best' cheap method or 'otsu'
    # A learned mask is already clean; the erosive OPEN in morphological_clean
    # destroys thin structure (e.g. NMI organelles ~4px wide -> recall collapse).
    # When True, learned masks get small-component removal ONLY (no open/close).
    learned_light_clean: bool = True
    # Plausibility gate (candidate mode only): the learned candidate competes
    # only when its foreground fraction lies in
    # [expected_fg_lo*lo_mult, expected_fg_hi*hi_mult] %, blocking empty/flood
    # masks a single-domain net can emit out-of-domain.
    learned_gate_lo_mult: float = 0.25
    learned_gate_hi_mult: float = 1.75
    force_segmenter: str = ''  # T4: restrict cascade to one candidate ('otsu', 'frangi', 'meijering', 'learned')


@dataclass
class ReconConfig:
    """Reconstruction parameters."""
    sigma_scale: float = 1.0
    min_sigma: float = 1.0
    max_sigma: float = 30.0
    sigma_quantize: float = 0.5
    blend_sigma: float = 5.0
    fg_presence_thresh: float = 0.05
    psf_max_radius: int = 30
    # v4: Oriented elliptical PSF
    use_oriented_psf: bool = True
    orientation_aspect_ratio: float = 2.5   # major/minor axis ratio for filament PSF
    orientation_smooth_sigma: float = 3.0   # smoothing for tangent estimation
    # v5: Spatial background grid
    bg_grid_size: int = 16                # background grid resolution
    # v5: Foreground residual encoding
    use_fg_residual: bool = True          # encode DCT residual in foreground
    residual_quality: int = 30            # DCT quality (1-100, lower=more compression)
    residual_block_size: int = 8          # DCT block size
    residual_max_fraction: float = 0.5    # max residual size as fraction of point data


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
    header_size: int = 13  # v5 header size
    width_quant_scale: float = 4.0
    zlib_level: int = 9
    raw_bytes_per_point: int = 16
    # v4: Graph-aware compression
    use_edge_delta: bool = True   # delta-encode along edges instead of row-sort
    store_orientation: bool = True  # store per-point orientation angle (1 byte)
    # v5: Enhanced compression
    format_version: int = 5       # bitstream format version
    store_bg_grid: bool = True    # store spatial background grid
    store_fg_residual: bool = True  # store DCT foreground residual
    store_edges: bool = True      # v6: store graph edge connectivity
    graph_predictive: bool = True   # predict from graph neighbors, encode residuals


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
    # v5: Enhanced evaluation
    compute_betti: bool = True          # compute Betti number metrics
    compute_persistence: bool = True    # compute persistence diagram distance
    persistence_thresholds: int = 30    # sublevel filtration resolution
    multi_codec_compare: bool = True    # compare against JPEG2000, WebP
    rate_distortion_points: int = 8     # number of points on R-D curve


@dataclass
class GraphConfig:
    """v4: Explicit graph structure parameters."""
    min_edge_length: int = 3          # minimum skeleton pixels between nodes to form edge
    edge_sample_spacing: int = 5      # sample edge attributes every N pixels
    store_adjacency: bool = True      # store edge list in compressed output
    compute_edge_features: bool = True  # compute curvature, mean width, etc. per edge
    spur_min_length: int = 5          # prune skeleton spur branches shorter than this (0=off)
    attach_optimizer_points: bool = False  # add optimizer reconstruction points to the
                                           # structural graph. Off => graph stays a clean
                                           # skeleton network; optimizer points still used
                                           # for PSF reconstruction (kept in points array).
    min_component_nodes: int = 3      # drop graph components with fewer nodes than this
                                      # (removes tiny vesicle/noise blobs). 0 => keep all.
    bridge_gaps: bool = True          # reconnect filament fragments split by skeleton breaks
    bridge_max_gap: float = 12.0      # max endpoint distance (px) to bridge
    bridge_min_align: float = 0.6     # min collinearity cosine at both endpoints to bridge
    # v7: graph builder. 'branch' builds the stored graph from the branch
    # decomposition of the mask's full skeleton (one node per junction,
    # Douglas-Peucker branch polylines) and writes a layered payload
    # (structure layer + graph-free appearance layer); spur_min_length,
    # bridge_* and min_component_nodes then affect only the render points.
    # 'pixel' is the v6 builder (greedy skeleton-pixel nodes, v6 payload).
    builder: str = 'branch'           # 'branch' (v7) | 'pixel' (v6)
    simplify_eps: float = 0.75        # Douglas-Peucker tolerance for branch polylines (px)
    max_segment: float = 8.0          # max chord between stored branch points (px)
    width_mode: str = 'dt'            # 'dt' (mask distance transform) | 'profile' (image fit)


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
            cfg.segment.expected_fg_lo = 5.0
            cfg.segment.expected_fg_hi = 45.0
            cfg.recon.use_oriented_psf = True   # filaments benefit from oriented PSF
        else:
            cfg.preprocess.bg_kernel_size = d.sparse_bg_kernel
            cfg.segment.expected_fg_lo = 2.0
            cfg.segment.expected_fg_hi = 20.0
            cfg.recon.use_oriented_psf = True   # oriented PSF helps all structures
        return cfg

    def for_curvilinear(self) -> 'NanographConfig':
        """Return a copy configured for dark-on-bright curvilinear imagery
        (retinal vessels, EM mitochondria, IRM microtubules).

        Uses the multi-domain clDice U-Net (trained with polarity-agnostic
        augmentation, so no polarity detection is needed) in 'replace' mode:
        the organelle-calibrated quality score cannot reliably rank masks on
        these modalities, so the learned mask is used directly rather than
        competing in the cascade. Held-out val IoU: STARE 0.64, DRIVE 0.56,
        microtubules 0.60, EPFL EM 0.83 (vs classical 0.06/0.10/0.02/0.001).
        """
        cfg = deepcopy(self)
        if not _CURVILINEAR_LEARNED_CKPT:
            raise FileNotFoundError(
                'curvilinear weight not found: '
                "Nanograph/weights/unet_curvilinear_cldice.pt")
        cfg.segment.use_learned = True
        cfg.segment.learned_ckpt = _CURVILINEAR_LEARNED_CKPT
        cfg.segment.learned_mode = 'replace'
        cfg.segment.learned_light_clean = True
        cfg.recon.use_oriented_psf = True
        return cfg

    def param_count(self) -> int:
        return sum(len(asdict(getattr(self, f)))
                   for f in ['preprocess', 'segment', 'recon', 'optimizer',
                             'compress', 'detect', 'classify', 'eval', 'graph'])


DEFAULT_CONFIG = NanographConfig()
