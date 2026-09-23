# Nanograph paper re-run — results report

**Commit at time of runs:** `39a8c5da94136f01d19d2216c97e98a0819ccfbf` (tag **`paper-v1`**, the T4 commit).
Every `results/paper/**/commit.txt` records this hash. No SAM was used in any run (`--sam` never passed).

Ground rules honoured: no prose edited in the manuscript; metric definitions unchanged; contradictions are **reported here, not silently fixed**.

---

## T1 — edge connectivity in the payload
Edges are now stored (payload flag bit 5) and `decode_graph` rebuilds the `Nanograph` from decoded nodes **and** edges. Acceptance over the organelle set:
- edge section ≈ **317.6 ± 73.4** raw bytes → **111.7 ± 34.7** bytes after zlib.
- 5/5 `test_v4.py` round-trip tests pass (re-run this session, 136.9 s).

## T2 — invertible edge predictive coding
Shared-predictor, unclipped zig-zag/varint residuals. Acceptance: **0/726** decode mismatches (lossless round-trip of the edge stream).

## T3 — all fidelity metrics on the decoded payload render
`nanograph_encode` now recomputes PSNR/SSIM/IoU on the **decoded** reconstruction; the pre-compression values are retained only as `pre_*` debug fields. Hard-coded `ng_iou=1.0` replaced with Otsu **self-IoU**; `ng_recon_iou` kept as an alias (evaluate.py:288, CSV col 35).

**Reported finding (T3):** `jpeg_fg_psnr` (evaluate.py:253) is measured on the **Nanograph predicted mask** (`result.mask>0`), whereas `jpeg_gt_fg_psnr` (evaluate.py:555) is measured on the **GT foreground** (`gt_fg`). They use different masks, so the two JPEG FG-PSNR columns are not directly comparable — only `gt_fg_psnr` vs `jpeg_gt_fg_psnr` is an apples-to-apples pair.

## T5 — all evaluation runs (15 outputs)
`org_{default,classical,replace}`, `ablation_bg_residual.csv`, `seg_perturbation_ablation.csv` (13 794 rows), `cross/{cells3d_membrane,cells3d_nuclei,retina,cell}`, `cross_gt/{stare,drive,epfl_mito,microtubules}`, `mito/{temporal_clip,temporal_clip_replace,sted,mito_mip}`, `heldout_organelle.txt` (108/726 stems). All completed; each dir has `commit.txt`.

## T6 — polarity (retrained multi-domain U-Net)
`results/paper/polarity.csv` (as-is vs GT-oracle best-of-2):

| dataset | as-is | oracle | \|Δ\| |
|---|---|---|---|
| stare | 0.643 | 0.643 | 0.000 |
| drive | 0.564 | 0.564 | 0.000 |
| microtubules | 0.599 | 0.607 | 0.008 |
| epfl | 0.822 | 0.831 | 0.009 |

Network is polarity-agnostic (no inference-time polarity detector needed).

## T7 — MITO (Zenodo 7724799 MIP tiles) failure diagnostics
`results/paper/mito_diag/summary.csv` (228 tiles; forced segmenter):

| segmenter | n | Seg-IoU | precision | recall | P@dil20 |
|---|---|---|---|---|---|
| otsu | 228 | 0.221 | 0.282 | 0.655 | 0.380 |
| frangi | 132 | 0.006 | 0.182 | 0.007 | 0.236 |
| meijering | 228 | 0.162 | 0.251 | 0.394 | 0.357 |
| learned | 228 | 0.171 | 0.206 | 0.639 | 0.356 |

**Conclusion:** on the sparse-foreground MIP tiles every segmenter collapses (best is Otsu at 0.22 IoU; even within a 20-px dilation of GT, precision ≤ 0.38). Frangi degenerates to an **empty mask on 96/228 tiles** (only 132 non-empty, IoU 0.006). This is the mitochondria-MIP failure mode; 10 TP/FP/FN overlays written.

## T8 — figures (no PDF build, per request)
Regenerated from the runs: `fig_codec_rd`, `fig_seg_bottleneck`, `fig_topology`, `cross_dataset_comparison`, `fig_modality_scaling`, `fig_prior_positioning`, `fig_seg_perturb`.
- `pipeline_overview.png` re-rendered on organelle **7378** (stage 7 from the decoded payload) + JSON sidecar (`image 7378, raw 65536 B, payload 2077 B, 144 nodes, 141 edges, 4 components, 264 points, psnr 30.93, fg_psnr 28.76, seg_iou 0.902`); `fig_pipeline.png` recomposed.
- New `experiments/render_paper_panels.py` regenerates `fig_qualitative_organelle.png`, `fig_ood_failure.png` (OOD EM slice036, Seg-IoU 0.002 — genuine failure), `fig_mito_generalisation.png`.
- **MITO row fix:** replaced the near-empty tile (6 nodes, 251 B) with the **median tile by annotated foreground fraction, `M2_019_y256x256` (fg = 0.073)** → 20 nodes / 18 edges / 956 B, PSNR 29.7, FG-PSNR 28.9.

## T9 — numbers
`paper/make_numbers.py --runs results/paper --out paper` wrote **433 macros** + `tables/{tab_cross_body,tab_topoall_body,tab_mitogen_body}.tex`. All required CSV columns present (0 missing). vs the shipped placeholder `numbers.tex`, **~338 of 433 macros changed** (the shipped file was a placeholder). The 15 remaining `\tbd` values are all the intentionally-skipped bootstrap-CI macros (`...WinsCI`, `boot=False`), not missing data.

---

## Claims to verify (11 HOLD, 3 flagged)

| # | verdict | evidence |
|---|---|---|
| 1 | **HOLDS** | Seg-IoU 0.870 (D) > 0.366 (C); payload 2200 B (D) < 2871 B (C) |
| 2 | **HOLDS** | cycles 3.66 < 7.65; mean width 2.98 < 5.55 (default cleaner) |
| 3 | **FLAG** | held-out win-rate 51.4 % > 50 % ✓, **but** default GT-IoU-vs-JPEG p = **0.04**, not < 0.01 (385/715 wins). Significance claim does **not** hold at p<0.01. |
| 4 | **FLAG** | classical GT-IoU vs JPEG p = **6.1×10⁻¹⁴⁰** (43/692 wins) — highly significant, classical **loses** badly; contradicts "not significant, p>0.05". |
| 5 | **HOLDS** | GT-FG-PSNR 27.30 (Nanograph) < 27.90 (JPEG): JPEG favoured under default |
| 6 | **HOLDS** | JPEG Betti 0.105 ≤ 0.2; JPEG β₀ 9.29 > 4.41, β₁ 3.97 > 0.32 |
| 7 | **HOLDS** | every dataset's JPEG Betti ≤ 0.2 (DRIVE 0.177 max) except Temporal clip 0.758 |
| 8 | **HOLDS** | BRR 0.0336 within ±25 % of 0.0365 (band [0.0274, 0.0456]) |
| 9 | **HOLDS** | org_replace held-out Seg-IoU 0.886, \|0.886−0.875\| = 0.011 ≤ 0.02 |
| 10 | **HOLDS** | polarity \|as_is−oracle\| ≤ 0.009 on all four (0.000/0.000/0.008/0.009) |
| 11 | **HOLDS** | dilate 1 px: β₀ changed on 8.3 % of images, width +31.2 %; boundary noise 1 px: β₀ 78.2 %, median\|Δβ₁\| 281; baseline Seg-IoU median 0.890 |
| 12 | **HOLDS** | microtubules 30/66 & EPFL 2/10 below JPEG byte floor; STARE JPEG GT-FG-PSNR 42.9 > Nanograph 28.7 |
| 13 | **FLAG** | temporal-clip learned-replace Seg-IoU 0.696 **<** default 0.750; contradicts "≥ default" |
| 14 | **HOLDS** | bg grid + residual improve FG-PSNR on **726/726** images |

---

## Addendum

**A1 — `delta_gt_iou` is correct.** The apparent whole-column mismatch was mixing subsets: `gt_iou` is over all 726, but `jpeg_gt_iou` exists only where JPEG met the byte budget. The per-image `delta_gt_iou` (written only when JPEG fits) is right.
- org_default: n=726, paired=715, JPEG no-fit=11, mean(gt−jpeg)=−0.0007, wins 385 / losses 330, sign-test p=0.04.
- org_classical: paired=692, no-fit=34, mean=−0.0736, wins 43 / losses 649, p≈6×10⁻¹⁴⁰.

**A2 — decode cost is negligible; no decode bug.** Over all 726 images (pre-compression render minus decoded payload render):
- GT-IoU: mean +0.0009, max 0.022
- FG-PSNR: mean +0.002 dB, max 0.029 dB
- full PSNR: mean +0.003 dB, max 0.008 dB
- SSIM: mean +0.00002
The decode path uses the decoded bg grid + residual (verified). The historic "432/726 GT-IoU wins, p=3.4×10⁻⁷" was the **learned-replace** arm, not default; default is a near-tie (now 385/715, p=0.04). So there is no regression from moving metrics onto the decoded payload.

**A4 — extra reporting (default vs classical):**
- precision / recall: 0.932 / 0.928 (default) vs 0.437 / 0.849 (classical)
- cycle rank: 3.66 vs 7.65 · mean width: 2.98 vs 5.55
- BRR: 0.0336 vs 0.0438 · JPEG byte-floor no-fit: 11/726 vs 34/726

---

## Failed / skipped
- The initial automated T6/T7 watcher run crashed with `ModuleNotFoundError: nanograph_v4` — `experiments/{eval_retrained,mito_diagnostics,render_pipeline_diagram}.py` used two `dirname` levels but the `nanograph_v4` symlink lives three levels up. Fixed (added the third level; T6 `CKPT` now also finds the checkpoint in the sibling `experiments/`) and re-run successfully. No data runs were lost.
- PDF build (`latexmk`) intentionally skipped per request — numbers and figures only.

---

## T10 — downstream morphometry (graph vs pixels vs byte-matched JPEG)

Script `experiments/downstream_morphometry.py` (stages encode → measure → stats); outputs `results/paper/downstream/` (`per_image.csv`, `branch_lengths/`, `summary.csv`, `branch_distances.csv`, `timing.csv`, `cost_summary.csv`, `commit.txt`); figure `paper/figures/fig_downstream.png`; 100 new `\Ds*` macros in `paper/numbers.tex` (0 existing macros changed).
Code at `4aed371` (codec byte-identical to `paper-v1`: re-encoded payload bytes = `org_default` `ng_bytes` on 726/726, selected segmenter 726/726). Stats on the 715 images where byte-matched JPEG fits (11 do not).

### Codec bug found (affects the GRAPH arm)
- `Nanograph.remove_small_components()` → `subgraph()` re-indexes node ids; `compress_nanograph` serialises `graph.edges` assuming node id = index into `points`. After any component is dropped, stored edges connect the wrong points.
- Affected: **15/726 images (2.1 %)**. Topology counts survive (isomorphic relabelling); geometry does not (decoded total length median **28.9×** the encoder graph's).
- T1 round-trip test cannot catch it: it maps encoder edges through the same wrong index.
- With edges remapped to point indices (same `compress_nanograph`, correct ids): decoded graph == encoder graph on **726/726**; payload mean 2199.9 → **2198.0 B** (−91 B mean on the 15).
- **GRAPH arm below uses the remapped payload.** As-tagged payload (`GRAPH_TAGGED`): total length CCC **0.006**, bias **+34.2 %**; all other descriptors identical to GRAPH within ±0.01 CCC.
- Codec not changed in this task; fix + re-tag + T5 re-run pending decision.

### Definitions (fixed before any run, one function for every arm)
- Pixel arms (REF, JPEG, SEG): `skimage.morphology.skeletonize` → `skan.Skeleton` pixel graph. Graph arms (GRAPH, PRE): Nanograph nodes/edges.
- Branch decomposition validated: with junction contraction off, identical to `skan.summarize` (count, every `branch-distance`, every `branch-type`) on **726/726** annotation masks.
- **Width, definitional mismatch fixed:** the stored width is the distance-transform **radius** (`dist_transform[y,x]`), not a diameter. All arms use diameter = 2 × cv2 L2 distance transform; GRAPH/PRE = 2 × stored value. Length-weighted over edges.
- Junction = connected cluster of degree ≥ 3 nodes (contracted to one vertex, internal links dropped) in every arm; shifts REF by 0.35 branches/image.
- Branch types 0–3 all counted (type 3 = loop on one vertex or vertex-free cycle; 3 type-3 branches in 104 REF masks sampled).
- Degree-0 nodes ignored (GRAPH: optimizer PSF points carry no edges).
- Edge length = Euclidean between node positions (pixel graph: = skan branch-distance; GRAPH: chord between stored nodes; PRE: encoder's traced path length).
- JPEG: `evaluate.jpeg_for_budget(img, payload_bytes)` → decode → the image's recorded segmenter via the encoder's own path (polarity → type → [preprocess] → segmenter → its clean-up).
- SEG (auxiliary): same segmenter on the original pixels. Decomposition: SEG−REF = segmentation; PRE−SEG = graph construction (spur prune, bridging, min-component, sampling); GRAPH−PRE = storage.

### Summary (vs REF, n = 715; width JPEG n = 709: 6 empty JPEG masks)

| descriptor | arm | n | CCC | r | bias (% REF mean) | 95% LoA (% REF mean) | MdAPE % | GRAPH closer / JPEG closer / tie | Wilcoxon p |
|---|---|---|---|---|---|---|---|---|---|
| n_components | GRAPH | 715 | 0.766 | 0.776 | -0.10 (-2.2) | -0.94 to +0.74 (-21 to +16) | 0.0 | 251 / 23 / 441 | 7.3e-41 |
| n_components | JPEG | 715 | -0.009 | -0.090 | +3.11 (+69.0) | -19.52 to +25.74 (-433 to +571) | 0.0 |  |  |
| total_length_px | GRAPH | 715 | 0.724 | 0.856 | -30.13 (-11.6) | -83.06 to +22.79 (-32 to +9) | 10.0 | 216 / 499 / 0 | 5.0e-09 |
| total_length_px | JPEG | 715 | -0.028 | -0.087 | +59.56 (+22.9) | -548.24 to +667.36 (-211 to +257) | 5.0 |  |  |
| mean_width_px | GRAPH | 715 | 0.868 | 0.905 | +0.18 (+3.2) | -0.47 to +0.84 (-8 to +15) | 2.9 | 517 / 192 / 0 | 8.2e-43 |
| mean_width_px | JPEG | 709 | 0.792 | 0.865 | +0.28 (+5.0) | -0.39 to +0.95 (-7 to +17) | 5.1 |  |  |
| n_branches | GRAPH | 715 | 0.292 | 0.694 | -5.46 (-41.0) | -13.05 to +2.14 (-98 to +16) | 40.0 | 191 / 434 / 90 | 4.3e-06 |
| n_branches | JPEG | 715 | -0.017 | -0.051 | +4.96 (+37.3) | -52.88 to +62.79 (-398 to +472) | 27.3 |  |  |
| n_junctions | GRAPH | 715 | 0.354 | 0.695 | -2.44 (-54.3) | -6.39 to +1.50 (-142 to +33) | 50.0 | 177 / 354 / 184 | 5.4e-05 |
| n_junctions | JPEG | 715 | 0.021 | 0.042 | +1.01 (+22.5) | -18.86 to +20.88 (-420 to +465) | 40.0 |  |  |
| cycle_rank | GRAPH | 715 | 0.507 | 0.578 | -0.24 (-44.8) | -1.54 to +1.05 (-282 to +192) | 50.0 | 141 / 46 / 528 | 2.1e-12 |
| cycle_rank | JPEG | 715 | 0.183 | 0.210 | +0.11 (+20.7) | -2.67 to +2.90 (-489 to +530) | 50.0 |  |  |

Branch-length distributions (per image vs REF, mean over images): 1-Wasserstein GRAPH **10.79** px vs JPEG **7.43** px (GRAPH closer on 192, JPEG on 517, p = 1.5e-43); KS GRAPH **0.416** vs JPEG **0.333** (154 / 488, p = 7.9e-37). PRE 11.15 / 0.422; SEG 8.14 / 0.340.

JPEG failure tail: 78/715 images with > 2× REF component count; 6 empty masks. Mean |error| total length GRAPH 30.4 vs JPEG 76.7 px; n_branches 5.5 vs 9.5.

### Where GRAPH's error comes from (bias %, CCC)

| descriptor | SEG vs REF (segmentation) | PRE vs SEG (graph construction) | GRAPH vs PRE (storage) |
|---|---|---|---|
| n_components | -1.4, 0.811 | -0.8, 0.956 | +0.00, 1.000 |
| total_length_px | -6.0, 0.841 | -4.0, 0.965 | -1.97, 0.995 |
| mean_width_px | +5.0, 0.829 | -1.2, 0.989 | -0.53, 0.999 |
| n_branches | -27.6, 0.472 | -18.5, 0.711 | +0.00, 1.000 |
| n_junctions | -39.5, 0.513 | -24.5, 0.801 | +0.00, 1.000 |
| cycle_rank | -36.6, 0.514 | -12.9, 0.901 | +0.00, 1.000 |

Storage is exact for topology; costs −2.0 % length (chord vs traced path) and −0.5 % width (¼-px quantisation).

### Cost (median over 726; single-threaded, CPU only; Intel Xeon Gold 6342 @ 2.80 GHz; Python 3.11.3)

| | bytes (mean) | time to descriptors (median) |
|---|---|---|
| Nanograph payload / GRAPH | 2198 | **4.9 ms** (decode_graph 4.1 + descriptors 0.8) |
| byte-matched JPEG / JPEG | 2148 | **274 ms** (decode 0.2 + segment 268 + skeleton/skan 3.7 + descriptors 1.1) |
| raw array | 65 536 | — |
| lossless PNG | 45 372 | — |
| REF (annotation mask) | — | **4.4 ms** (skeleton/skan 3.4 + descriptors 1.0) |

Ratios: JPEG / GRAPH **56×**; REF / GRAPH **0.91×**; JPEG without segmentation / GRAPH 1.03×. Speed-up is the segmenter, not the representation. "Three to four orders of magnitude" **does not hold** (1.7 orders vs JPEG; none vs REF).

### Outcome: **3** — GRAPH worse than JPEG on some descriptors
- GRAPH better than JPEG: **n_components** (251 vs 23, p = 7e-41), **mean_width** (517 vs 192, p = 8e-43; MdAPE 2.9 vs 5.1 %), **cycle_rank** (141 vs 46, p = 2e-12). GRAPH has higher CCC than JPEG on all six descriptors.
- GRAPH worse than JPEG (per-image paired error): **total_length** (MdAPE 10.0 vs 5.0 %; 216 vs 499, p = 5e-9), **n_branches** (40.0 vs 27.3 %; 191 vs 434, p = 4e-6), **n_junctions** (50 vs 40 %; 177 vs 354, p = 5e-5), **branch-length distribution** (W1 10.8 vs 7.4 px).
- GRAPH is not ≈ REF on topology counts: systematic under-count (branches −41 %, junctions −54 %, cycle rank −45 %). The cause is segmentation plus graph construction; storage adds nothing to it (table above). JPEG's median error is smaller but it has an unbounded failure tail (CCC ≈ 0 on every count).
- Optional temporal-clip extension: **not run** (main result did not land cleanly).
