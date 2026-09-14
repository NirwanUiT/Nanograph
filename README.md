# Nanograph

A structural graph representation for curvilinear microscopy images
(mitochondrial networks, vessels, microtubules, neurites). Nanograph encodes an
image once into a compact **attributed graph** — nodes carry position, width,
intensity and orientation; edges carry connectivity, length, curvature and mean
width — from which morphometry and graph-exact topology (β₀, β₁) are read
directly, and an approximate image can be reconstructed via oriented
point-spread functions.

Companion code for the paper *"Nanograph: A Shape-Preserving Graph
Representation of Curvilinear Structures in High-Content Microscopy"*
(under review).

## Setup

The code is a plain Python package imported as `nanograph_v4`, so clone into a
directory of that name (or symlink):

```bash
git clone https://github.com/NirwanUiT/Nanograph.git nanograph_v4
pip install numpy opencv-python scikit-image scipy matplotlib
pip install torch          # optional: learned U-Net segmenter (enabled by default)
```

Python ≥ 3.9. No build step; run scripts from the directory *containing*
`nanograph_v4/` (the package inserts its parent on `sys.path`). The two trained
segmenter weights ship in `weights/`:

| Checkpoint | Trained on | Used for |
|---|---|---|
| `unet_organelle_cldice.pt` | fluorescence organelles | default learned candidate |
| `unet_curvilinear_cldice.pt` | STARE/DRIVE/EM/microtubules, polarity-agnostic | `for_curvilinear()` preset |

Without torch the learned segmenter is skipped and the classical
Otsu/Frangi/Meijering cascade is used alone.

## Quick start

```python
from nanograph_v4 import nanograph_encode, nanograph_decode, nanograph_morphometry

result = nanograph_encode('image.png')          # path or uint8 grayscale array

result.compressed_bytes    # payload size
result.psnr_fg             # foreground reconstruction fidelity
result.mask                # selected foreground mask
result.graph               # attributed Nanograph object

morpho = nanograph_morphometry(result.graph)    # widths, lengths, junctions, ...
recon, *rest = nanograph_decode(result.compressed)
```

Configuration (121 explicit parameters in 9 dataclasses):

```python
from nanograph_v4 import NanographConfig

cfg = NanographConfig()
cfg.segment.learned_mode = 'replace'    # trust the learned mask outright
result = nanograph_encode('image.png', config=cfg)

cfg = NanographConfig().for_curvilinear()   # dark-on-bright vessels / EM / filaments
```

## Reproducing the paper's evaluations

Dataset evaluation (per-image metrics CSV, byte-matched JPEG/WebP/JPEG 2000
comparison, GT-referenced IoU/PSNR, plots):

```bash
python run_dataset.py --images <dir/*.png> --masks <gt_dir> --outdir dataset_results
python run_dataset.py ... --preset learned-replace   # learned mask forced (Sec. on the
                                                     # segmentation bottleneck)
python run_dataset.py ... --preset curvilinear       # multi-domain segmenter, for the
                                                     # out-of-domain datasets
```

Ablations reported in the paper:

```bash
# background-grid + DCT-residual contribution, paired over the full set
python experiments/ablation_bg_residual.py --images <dir> --out ablation_bg_residual.csv

# sensitivity of stored invariants to segmentation error (dilate/erode/drop/noise)
python experiments/seg_perturbation_ablation.py --images <dir> --masks <gt_dir> \
    --out seg_perturbation_ablation.csv
```

Committed result CSVs backing the paper's tables:
`dataset_results/` (726-image organelle run), `dataset_results_learned_replace/`,
`cross_dataset_results/` (four sample-data modalities, no GT),
`cross_gt_results/` (four annotated out-of-domain datasets),
`ablation_bg_residual.csv`, `seg_perturbation_ablation.csv`.

## Package layout

| File | Purpose |
|---|---|
| `api.py` | `nanograph_encode` / `nanograph_decode` entry points |
| `config.py` | all parameters, presets (`for_image_type`, `for_curvilinear`) |
| `preprocess.py` / `segment.py` | background subtraction; cascading segmentation + learned U-Net |
| `skeleton.py` / `graph.py` | skeletonisation; attributed-graph construction and queries |
| `reconstruct.py` | oriented-PSF rendering, background grid, DCT residual |
| `compress.py` | graph-predictive serialisation + zlib |
| `evaluate.py` | metrics incl. byte-matched codec comparison and GT-referenced IoU |
| `unet_seg.py` | compact clDice U-Net used by the learned segmenter |
| `track.py` | temporal centreline tracking (implemented, **unvalidated**) |
| `run_dataset.py` | dataset evaluation CLI |
| `experiments/` | training, ablation and diagnostic scripts |

## Notes

- The evaluation datasets themselves (726-image organelle set from live-cell
  mitochondrial imaging, STARE, DRIVE, EPFL/Lucchi EM, IRM microtubules) are
  not redistributed here; see the paper's data availability statement for
  sources.
- `track.py` is engineering without quantitative validation — treat its design
  claims as hypotheses (see module docstring).
- No automated test suite accompanies the codebase; `test_v4.py` is a
  demo/benchmark CLI entry point.
