# T12 — method improvements (experimental branch `method-v7`)

Everything here lives on branch `method-v7` (worktree), built on `cf3a10c`. The `paper-v2` code paths and the shipped defaults are unchanged; v7 is a separate module (`graph_branch.py`) evaluated next to them. Organelle set, 715 images where a byte-matched JPEG fits; "held-out" = the 105 of the 108 segmenter-held-out images with a fitting JPEG. Descriptors use the T11.2 definitions (degree-based junctions, shared terminal-branch pruning at L; L = 5 unless stated).

## #1 Branch-level graph construction

`graph_branch.extract_branches`:
- Takes the skan branch decomposition of the skeleton.
- Each connected cluster of degree ≥ 3 pixels becomes **one** junction node.
- Each branch is a Douglas–Peucker polyline (tolerance ε = 0.75 px, chords ≤ 8 px).
- A junction-to-junction branch always keeps an interior node, so two junctions never read as one.
- The graph is built from the mask's **unpruned** skeleton: spur pruning is an analysis-time choice (L), not baked into the stored object.

**Validation on the annotation itself** (all 726 masks, after encode → decode): component, branch, junction and cycle counts are **identical to skan on 726/726**. Length ratio 1.002 (min 0.9985); width ratio 0.996.

**Result.** Graph construction and storage now cost nothing. On the encoder's own mask, the decoded v7 graph equals the lossless mask (SEG) on every count at every L. It also equals the encoder-side v7 graph (PRE7): length within 0.1 %, width within 0.1 pp. The Douglas–Peucker tolerance does not change any count between 0.5 and 3 px; it only trades bytes (195 → 155 B).

## #2 Structure / appearance split

`graph_branch.encode_structure` stores vertices, branch end-vertex pairs, interior points as coordinate deltas, a width per point and each branch's pixel-path length (all varints, zlib).

| layer | mean bytes |
|---|---|
| structure (v7, ε 0.75) | **171** |
| appearance: v6 render points + background + residual, graph-free | 2018 |
| structure + appearance | 2189 |
| paper-v2 payload | 2198 |
| lossless mask (bit-packed + zlib) | 366 |

The split costs nothing overall (2189 ≤ 2198 B). The analysis-facing layer is 8 % of the payload and **half the size of a lossless mask** (171 vs 366 B), while carrying the same descriptors.

**all 715 images, L = 5**

| arm | bytes | branches bias % (CCC) | junctions bias % (CCC) | cycle rank % | length % | width % | junction F1 | endpoint F1 | branches: closer than byte-matched JPEG / JPEG closer |
|---|---|---|---|---|---|---|---|---|---|
| current graph (paper-v2 payload) | 2216 | -17.5 (0.66) | -38.4 (0.60) | -44.8 | -8.4 | +2.6 | 0.657 | 0.800 | 329/197 (p 5e-16) |
| lossless mask (skan on the encoder mask) | 366 | -10.1 (0.76) | -19.9 (0.74) | -36.6 | -3.9 | +4.3 | 0.692 | 0.803 | 336/130 (p 3e-29) |
| v7 structure layer, eps 0.75 | 172 | -10.1 (0.76) | -19.9 (0.74) | -36.6 | -3.8 | +4.2 | 0.693 | 0.803 | 336/130 (p 3e-29) |
| v7, eps 3 | 155 | -10.1 (0.76) | -19.9 (0.74) | -36.6 | -3.8 | +4.2 | 0.693 | 0.803 | 336/130 (p 3e-29) |
| v7, profile widths | 180 | -10.1 (0.76) | -19.9 (0.74) | -36.6 | -3.8 | +18.8 | 0.693 | 0.803 | 336/130 (p 3e-29) |
| byte-matched JPEG, same segmenter | 2148 | +58.9 (-0.01) | +45.5 (0.06) | +20.7 | +24.2 | +4.3 | 0.581 | 0.628 | — |
| JPEG q20 | 1608 | +423.7 (-0.00) | +401.1 (0.00) | +301.0 | +191.1 | +5.4 | 0.384 | 0.358 | — |
| JPEG q30 | 2579 | +3.0 (0.51) | -2.4 (0.65) | -19.4 | +0.8 | +4.1 | 0.630 | 0.713 | — |
| JPEG q50 | 4784 | -8.4 (0.78) | -16.9 (0.76) | -31.5 | -3.5 | +3.6 | 0.680 | 0.780 | — |
| JPEG q75 | 9206 | -9.2 (0.78) | -19.7 (0.75) | -35.0 | -3.5 | +4.3 | 0.694 | 0.798 | — |

**105 held-out images, L = 5**

| arm | bytes | branches bias % (CCC) | junctions bias % (CCC) | cycle rank % | length % | width % | junction F1 | endpoint F1 | branches: closer than byte-matched JPEG / JPEG closer |
|---|---|---|---|---|---|---|---|---|---|
| current graph (paper-v2 payload) | 2245 | -19.5 (0.67) | -39.9 (0.64) | -43.5 | -8.4 | +2.2 | 0.654 | 0.778 | 54/31 (p 1e-03) |
| lossless mask (skan on the encoder mask) | 370 | -10.3 (0.75) | -18.3 (0.75) | -29.0 | -3.6 | +3.9 | 0.693 | 0.787 | 55/16 (p 9e-07) |
| v7 structure layer, eps 0.75 | 177 | -10.3 (0.75) | -18.3 (0.75) | -29.0 | -3.5 | +3.9 | 0.693 | 0.787 | 55/16 (p 9e-07) |
| v7, eps 3 | 159 | -10.3 (0.75) | -18.3 (0.75) | -29.0 | -3.5 | +3.8 | 0.693 | 0.787 | 55/16 (p 9e-07) |
| v7, profile widths | 185 | -10.3 (0.75) | -18.3 (0.75) | -29.0 | -3.5 | +21.7 | 0.693 | 0.787 | 55/16 (p 9e-07) |
| byte-matched JPEG, same segmenter | 2169 | +28.7 (0.19) | +14.3 (0.61) | +6.5 | +9.4 | +4.6 | 0.584 | 0.623 | — |
| JPEG q20 | 1619 | +432.5 (-0.01) | +396.0 (0.01) | +258.1 | +202.8 | +6.0 | 0.383 | 0.353 | — |
| JPEG q30 | 2603 | +2.3 (0.60) | -0.7 (0.67) | -6.5 | +0.7 | +4.6 | 0.605 | 0.691 | — |
| JPEG q50 | 4817 | -7.3 (0.81) | -13.6 (0.79) | -16.1 | -3.0 | +3.1 | 0.632 | 0.757 | — |
| JPEG q75 | 9273 | -9.5 (0.78) | -18.7 (0.75) | -29.0 | -3.1 | +4.0 | 0.672 | 0.789 | — |

- **Against the current graph (GRAPH6), at L = 5:**

| bias vs REF | GRAPH6 | GRAPH7 |
|---|---|---|
| branches | −17.5 % | −10.1 % |
| junctions | −38.4 % | −19.9 % |
| cycle rank | −44.8 % | −36.6 % |
| length | −8.4 % | −3.8 % |

  Junction F1 rises from 0.657 to 0.693 (at 3 px). Bytes fall from 2216 to 172.
- **Against byte-matched JPEG (same segmenter):** GRAPH7 is closer on branches on 336 vs 130 images (p = 3e-29); held-out 55 vs 16 (p = 9e-7).
- **What JPEG needs to match:** about quality 50 (4.8 kB, 28× the structure layer). At ≤ 1.6 kB (q ≤ 20) JPEG counts are off by several hundred per cent.
- **The whole remaining deficit is segmentation:** SEG = GRAPH7.

## #3 Width from the image profile

`graph_branch.profile_width` fits a Gaussian plus offset to the intensity profile across the centreline and reports FWHM/2 as the radius. Robustness test: 150 images; the encoder mask dilated by 1–2 px or eroded by 1 px; width recomputed.

| width estimator | mask perturbation | width change vs unperturbed (median) | CCC vs REF | r vs REF | bias vs REF |
|---|---|---|---|---|---|
| dt | none | +0.0 % | 0.751 | 0.811 | +4.0 % |
| dt | dilate1 | +31.0 % | 0.169 | 0.759 | +35.5 % |
| dt | dilate2 | +63.2 % | 0.057 | 0.731 | +68.2 % |
| dt | erode1 | -27.9 % | 0.289 | 0.764 | -24.2 % |
| profile | none | +0.0 % | 0.304 | 0.565 | +17.8 % |
| profile | dilate1 | +1.8 % | 0.225 | 0.467 | +20.8 % |
| profile | dilate2 | +3.9 % | 0.186 | 0.450 | +25.2 % |

**Mixed result.** The profile width is almost immune to mask error: +2 % / +4 % under 1 / 2 px dilation, −1 % under erosion. The mask-distance-transform width moves by +31 % / +63 % / −28 %. But the profile width agrees **less** with the annotation (r 0.57 vs 0.81; bias +18 %), because it measures the optical (blurred) width, not the drawn mask width.

**Recommendation:** store both. That costs about 8 B per image (180 vs 172 B structure layer). Use the profile width when the mask is uncertain, or when comparing across segmenters. Calibrating the profile width against the PSF (deconvolving the FWHM) is the obvious next step; it is not done here.

## #4 Topology-aware segmentation (clean split)

`experiments/train_unet_topo.py`:
- The 108 held-out images are **never seen**. Checkpoint selection uses 10 % of the remaining 618; the shipped model selected its checkpoint on the held-out images.
- Baseline loss: Dice + BCE + clDice. Treatment: baseline + skeleton-recall loss (Kirchhoff et al. 2024).
- 60 epochs each, 3 seeds each.
- Evaluation (`experiments/eval_unet_topo.py`): the v7 graph from each model's mask, on the held-out images.
| profile | erode1 | -1.1 % | 0.341 | 0.605 | +16.2 % |

| segmenter (held-out 108, v7 graph, L = 5) | seeds | IoU | branches % | junctions % | cycle rank % | length % | junction F1 | endpoint F1 |
|---|---|---|---|---|---|---|---|---|
| shipped U-Net (checkpoint chosen on these images) | 1 | 0.886 | -10.2 | -18.6 | -29.7 | -2.8 | 0.700 | 0.794 |
| clean split, Dice+BCE+clDice | 3 | 0.882 ± 0.005 | -11.1 ± 1.5 | -20.4 ± 2.7 | -28.6 ± 2.4 | -3.2 ± 0.5 | 0.662 ± 0.018 | 0.781 ± 0.013 |
| clean split, + skeleton recall | 3 | 0.877 ± 0.001 | -9.9 ± 1.3 | -17.0 ± 2.5 | -24.5 ± 5.0 | -2.0 ± 0.4 | 0.669 ± 0.015 | 0.798 ± 0.011 |

- **Skeleton recall helps a little, consistently, on every structural measure:**

| vs baseline | change |
|---|---|
| branches | +1.2 pp |
| junctions | +3.4 pp |
| cycle rank | +4.2 pp |
| length | +1.2 pp |
| endpoint F1 | +0.017 |

  Each is about one seed-SD, so suggestive, not conclusive, with 3 seeds. IoU falls by 0.005.
- **It does not close the gap:** with it, the v7 graph still under-counts branches by ~10 % and junctions by ~17 %.
- **Selection optimism is visible:** the shipped model's IoU (0.886) is above the clean-split baseline mean (0.882 ± 0.005), consistent with a small optimism from selecting on the held-out images.

## #5 Evaluation the field uses

- **Junction and endpoint F1** (one-to-one Hungarian matching within 3 and 5 px, after pruning at L) are in every v7 table (`experiments/downstream_morphometry.py::vertex_positions`, `point_f1`).
- **Rate–distortion:** JPEG at q = 5–75 plus the byte-matched JPEG, lossless mask, GRAPH6 and GRAPH7 at four tolerances, all in `results/v7/summary.csv` (bytes vs descriptor bias, CCC, F1).
- **Inter-annotator ceiling: NOT DONE — needs a person.**
  - `experiments/annotator_agreement.py` implements the protocol and the comparison, with the same statistics as every pipeline arm.
  - `results/annotator/subset.txt` holds the 50-image subset (seed 0) for a second, blind annotation.
  - `--selftest` only checks that the code runs; its output is not a result.

## #6 Pipeline simplification

**Not done:** it changes shipped behaviour and defaults, which T11.3 said to leave alone. The v7 results bear on it: the knobs that matter for structure are the segmenter and the simplification tolerance; the rest (bridging, minimum component size, encoder spur pruning) are no-ops for the stored structure.

## What this means for the method
1. **Adopt the v7 structure layer.** It removes graph construction and storage as error sources, at 8 % of the payload and half a lossless mask. The downstream claim can then be "the stored graph is as accurate as analysing the mask itself, and more accurate than a byte-matched JPEG".
2. **All remaining error is segmentation:** held-out, −10 % branches, −18 % junctions, −29 % cycle rank. Skeleton-recall training moves this by a few points. The annotator ceiling is needed to know how much of it is real.
3. **Caveats:**
   - One dataset (organelles).
   - Appearance layer and rendering unchanged.
   - v7 not yet wired into `nanograph_encode` (it runs beside it).
   - U-Net checkpoints kept locally (31 MB each, not in git).
