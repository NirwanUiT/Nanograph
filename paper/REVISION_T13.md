# Revision plan (T13) — story, evidence, required changes

Branch `method-v7`. Every number below comes from a committed CSV; paths are given in the evidence map.

## 1. The story

**Nanograph is a structure-first storage format, not an image codec.** A pixel codec (JPEG, WebP, JPEG 2000) is built to reproduce what an image *looks like*. Nanograph is built to keep what an image *contains* — the curvilinear network, as a graph that answers morphometric and topological queries directly — and to make that affordable at archive scale. The image is an optional second layer.

The two are built for different jobs, so the paper should not be framed as "Nanograph beats JPEG". It should make three claims, each backed below:

1. **What is stored is exactly what analysis needs.** The v7 structure layer reproduces the descriptors of the mask it encodes (components, branches, junctions, cycle rank, length, width) on 726/726 images, at 170–560 B. Storage adds no error to the analysis.
2. **For analysis-only storage it is several times cheaper than any pixel route.** A pixel codec must be decoded and re-segmented before any analysis. Two comparisons show this:
   - On real annotated mitochondria, JPEG needs roughly 4–5× the bytes of the structure layer to support the same analysis quality.
   - Even the smallest possible JPEG file is at least twice the structure layer's size and is markedly worse.
3. **It is queryable.** Descriptors are read in about 1 ms from the payload:
   - 4.3× faster than re-analysing a stored mask;
   - about 240× faster than the JPEG route (decode + segment + skeletonise).

**Where JPEG wins, and why that is a different purpose.** State this plainly in the paper:

- **Photometric fidelity.** Full-image PSNR/SSIM and GT-FG-PSNR favour the codecs. Nanograph summarises the background by design (256-byte field) and renders structure from nodes. It is not trying to reproduce pixels.
- **Analysis when both are given kilobytes.** On real data at the *full* payload (4–6.5 kB, structure + appearance layers), a byte-matched JPEG is nearly lossless. Re-segmenting it therefore gives the same analysis as the original:
  - ties on UiT-Rat and EP-UiT-Human;
  - JPEG slightly closer on CBMI branches/junctions and MITO length/width.

  At that budget both routes reach "analysis of the original". Nanograph's structure layer gets there with ~0.5 kB and no re-segmentation.
- **Raw length on unbranched simulated tubes.** JPEG's mean length bias vs the true geometry is near zero. This comes from two errors cancelling (fragmentation adds length, merging removes it): JPEG's per-image concordance with truth is ≈ 0 (CCC 0.006). Nanograph's error is a consistent, calibratable bias. After calibration it is closer to the truth than JPEG (MdAPE 10.8 vs 14.6 %, 255 vs 142 images).

**What to compare against instead of JPEG (as the headline).** The natural alternatives for "store a curvilinear image so it can be analysed later" are:

- **(a) storing the segmentation mask losslessly** (366–660 B on organelles): the v7 structure layer is about half that size, carries the same descriptors, adds per-node width and intensity, is directly queryable, and — with its appearance layer — regenerates the image;
- **(b) storing the image and re-running the analysis pipeline:** 20–30× the bytes and ~250 ms per image;
- **(c) graph-extraction tools that discard the pixels** (Vaa3D/image-to-graph predictors): no image regeneration, no storage format.

Keep JPEG / WebP / JPEG 2000 as reference points, not as the opponent.

## 2. Evidence map

| claim | result | file |
|---|---|---|
| structure layer = mask analysis | GRAPH7 = SEG on every descriptor, 726/726 organelle images, all L; built from annotation skeletons, counts identical to skan on 726/726 after encode/decode, length within ±0.45 % | `results/v7i/downstream/summary.csv`, `test_v4.py::test_v7_structure_matches_skan_on_annotations` |
| topology of the stored graph = topology of the mask | components match 99.9 % (v6: 95.6 %); cycle rank = mask holes 95.2 % (v6: 16 %) | `results/v7i/org_default/metrics.csv` |
| structure layer size | organelles 171 B (mask: 366 B bit-packed, 659 B PNG); real sets 367–564 B | `results/v7/summary.csv`, `results/real/real_downstream/summary.csv` |
| layered payload costs nothing | structure 171 + appearance 2018 = 2189 B vs v6 payload 2198 B; rendering identical | `results/v7/T12_REPORT.md` |
| analysis-only storage vs JPEG (real data) | branches CCC vs expert masks: structure layer 0.94 / 0.81 (UiT / Human) vs smallest JPEG (q1, 1.2 kB) 0.73 / 0.35; JPEG needs q≈20 (2.2–2.9 kB) to match | `results/real/real_downstream/jpeg_low_quality.csv.gz`, `per_tile.csv.gz` |
| query cost | 1.05 ms (structure layer) vs 4.5 ms (mask re-analysis) vs 252 ms (JPEG decode + U-Net + skeleton), single thread | `results/v7i/downstream/cost_summary.csv` |
| accuracy against the true geometry (simulation) | after 2-fold calibration: GRAPH7 ≈ the simulator's own mask; better than JPEG on components, length, branches and junctions (all p ≤ 6e-6) | `results/real/sim_truth/calibrated.csv` |
| organelle downstream vs reference mask | GRAPH7 closer than byte-matched JPEG on all six descriptors (e.g. length 447/269, branches 341/141 images) | `results/v7i/downstream/summary.csv` |
| real-data segmentation | `real-mito` preset: Seg-IoU UiT 0.55→0.78, CBMI 0.49→0.58, MITO 0.09→0.32, Human (untouched) 0.37→0.41 | `results/real/pipeline/*/*/metrics.csv` |
| downstream on an untouched real dataset (Human) | real-trained vs simulation-trained: branch bias +8 % vs +91 %, junction bias +5 % vs +116 %, branch CCC 0.88 vs 0.49 (3 seeds) | `results/real/eval_summary.csv` |

## 3. Factual corrections (already applied in `nanograph_main.tex` on this branch)

- **Organelle set.** Previously "in-house live-cell acquisition with independent annotations". Now: **simulated** with the Sekh et al. (2021) simulator, Airyscan variant, as 2×2 mosaics of 128×128 simulations of 1–2 unbranched tubes. The masks are simulator ground truth.
- **Temporal clip.** Previously "from the same acquisition". Now: **simulated** with a separate dynamic simulator (CODS; 4 unbranched mitochondria, elastic motion), with per-frame ground-truth skeletons.
- **Metrics, captions, data availability:** "annotation" is now "reference mask (simulator ground truth / expert annotation)".

## 4. Claims to change or drop (manuscript still says them)

| # | current claim | problem | replace with |
|---|---|---|---|
| 1 | Byte-matched JPEG Betti agreement 0.105, inflated β0/β1 ("decisive difference") | Otsu is compared against a U-Net mask: Otsu on the *uncompressed* image scores 0.000 | same-segmenter comparison (organelles: JPEG components +68 %, CCC ≈ 0; GRAPH7 −1.4 %, CCC 0.81) |
| 2 | Table 2/5 "graph" β0/β1 | they are the mask's Betti numbers | with v7 the stored graph matches them (§2); report the stored-graph values |
| 3 | skeleton-graph cycle rank 3.66 ("path redundancy of the network") | v6 junction-pixel artefacts; true holes 0.32; structures are unbranched | v7 cycle rank 0.34 ≈ holes 0.32; drop "network redundancy" wording for the simulated set |
| 4 | Junction-degree statistics, branching network (organelles) | simulated tubes are unbranched: junctions are projection overlaps | show network statistics on real data (UiT, Human) instead |
| 5 | "localises the foreground more accurately than JPEG" (GT-IoU) | mean paired difference −0.0007 (p = 0.04 sign test) | parity; not a claim the paper needs (§1) |
| 6 | classical cascade "at parity" with JPEG | it loses on 650/692 images | state that GT-IoU follows segmentation quality |
| 7 | nodes spaced proportional to width | fixed spacing (v6); v7 uses Douglas–Peucker branch polylines | describe the v7 builder |
| 8 | error-guided refinement never triggers | it runs on every image (+120 render points) | describe it as part of the appearance layer |
| 9 | edges store length and curvature; "all descriptors read from the payload" | v6 payload stored neither | v7 stores branch path length; curvature derivable from the polylines |
| 10 | temporal clip transfers "across time without adjustment" as real-data evidence | clip is simulated | keep as simulated evidence with exact truth; real-data evidence = §2 real rows |
| 11 | STARE/DRIVE/EM/microtubule, MITO results | unchanged by T13, but the v7 builder changes their graph statistics | re-run tables with v7 (bytes within ±3 %, rendering identical; stored components = mask on 100 %) |

## 5. New Results subsections to add

1. **A layered format: structure first.** v7 builder and codec; the structure layer equals mask analysis; its size vs a lossless mask; the layered payload costs nothing.
2. **Analysis from the stored object.** Downstream descriptors on organelles (vs reference mask) and on real annotated data (vs expert masks):
   - the analysis-only byte comparison with JPEG at q1–q20;
   - query cost;
   - spur pruning and the one-diameter cleanup rule (declared before the truth evaluation), with the L sweep in the supplement.
3. **Against the true geometry.** Simulation lets the paper measure what no annotation can:
   - even the simulator's own mask misses 27 % of length and overstates width by 53 % (unbranched 20–300 nm tubes at 42 nm pixels, overlapping in projection);
   - after calibration the stored graph carries as much information about the true geometry as that mask, and more than JPEG.

   The temporal clip shows the fragmentation failure mode and how the analysis-time rule removes it (components +58 % → +10 %).
4. **Real mitochondria.** Four annotated real datasets, frame/cell-level splits, 36 trained models:
   - in-domain gains: UiT IoU 0.68 → 0.78, MITO 0.17 → 0.30;
   - on the untouched Human set, real training fixes the topology counts (branch bias +91 % → +8 %);
   - cross-dataset transfer between microscopes is mixed (CBMI is better with the simulation model). Report it honestly as the limit.

## 6. Limitations to state

- Width is overstated by every 2-D arm, including the simulator's own mask. Near the optical limit, 2-D width is a resolution-limited quantity.
- The branch/junction under-count relative to reference masks comes from segmentation, not storage.
- Real-data test sets are small (33–133 tiles; tiles within a frame are correlated).
- The regenerated simulation matches the organelle set closely, but not exactly (U-Net IoU 0.85 vs 0.88).
- Timings on the organelle runs were re-measured sequentially; the other datasets are being re-timed.

## 7. Decisions for the authors

1. Adopt the v7 structure layer as the paper's method (recommended), with v6 described only as the ablation.
2. Adopt the one-diameter analysis rule as the default cleanup (it was declared before the truth evaluation), and report L = 0 and L = 5 in the supplement.
3. Ship `weights/unet_mito_real.pt` as a `real-mito` preset and move the real-data section into the main text.
4. Confirm the simulator provenance wording and cite the CODS simulator used for the temporal clip.
5. Optional: DRIVE/STARE second-observer labels (public) for an inter-annotator ceiling on vessels. For mitochondria, the simulation truth replaces it.
