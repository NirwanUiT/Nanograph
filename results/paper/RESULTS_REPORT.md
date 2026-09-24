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

---

## T11.1 — node-index bug fixed, re-tagged `paper-v2`, everything re-run

**Fix (commit `35b90f6`, tag `paper-v2`).**
- `compress_nanograph` maps each graph edge to point indices by node coordinate at serialisation time. Every call site is covered.
- `nanograph_encode` asserts that every stored edge index is `< len(points)`. It also asserts that the decoded edge set equals the encoder graph's edge set, compared by node coordinate (index-identical, not just isomorphic).
- `test_v4.py` round trip now derives expected edges from `graph.nodes[i].position` matched to stored points by coordinate. New `test_edge_roundtrip_small_components` covers 6 images that trigger `remove_small_components()` re-indexing, and asserts the re-indexing actually happens on them.
- The new test **fails on the old codec** (verified) and passes on the fix. Suite: 6/6 pass (143 s).
- Fixed payloads are byte-identical to the T10 remapped payloads (726/726).
- Same latent assumption, not fixed: `api.py` `attach_optimizer_points` path indexes `graph.nodes[i]` by point index. It is off by default and never used in any run.

**Re-run.** All T5 outputs (org ×3, ablation, perturbation, cross ×4, cross_gt ×4, mito ×4, held-out list) via `experiments/run_paper_t11.sh`: the same commands as T5, in 5 concurrent lanes (2:31 h instead of 5:22 h). Every `commit.txt` = `35b90f6`. Then `make_numbers.py`, `make_figures.py`, `render_pipeline_diagram.py` (7378: unchanged, 2077 B), `make_fig_pipeline.py`.
- **Timing columns were measured under concurrent load.** 14 timing macros moved for that reason alone, e.g. `\DTimem` 1154 → 1248 ms. Re-time sequentially if the paper quotes run times.

**Images changed beyond rounding** (any numeric column, |Δ| > 1e-6 relative, timing excluded; `experiments/diff_runs.py`):

| run | images | changed | top changed columns (n images) |
|---|---|---|---|
| cross/cell | 5 | 3 | ng_bytes (3), ng_vs_raw (3), ng_vs_png (3), delta_iou (2), delta_fg_ssim (2), jpeg_quality (2) |
| cross/cells3d_membrane | 60 | 46 | ng_bytes (46), ng_vs_png (46), ng_vs_raw (46), delta_fg_ssim (45), delta_iou (45), jpeg_quality (45) |
| cross/cells3d_nuclei | 60 | 41 | ng_bytes (41), ng_vs_raw (41), ng_vs_png (41), delta_iou (37), delta_fg_ssim (37), jpeg_quality (37) |
| cross/retina | 25 | 5 | ng_bytes (5), ng_vs_raw (5), ng_vs_png (5), delta_iou (2), delta_fg_ssim (2), jpeg_quality (2) |
| cross_gt/drive | 20 | 0 | — |
| cross_gt/epfl_mito | 10 | 2 | ng_bytes (2), ng_vs_png (2), ng_vs_raw (2) |
| cross_gt/microtubules | 66 | 0 | — |
| cross_gt/stare | 20 | 0 | — |
| mito/mito_mip | 228 | 154 | ng_bytes (154), ng_vs_raw (154), ng_vs_png (154), delta_gt_fg_psnr (139), delta_fg_ssim (139), delta_iou (139) |
| mito/sted | 345 | 1 | ng_bytes (1), delta_fg_ssim (1), delta_iou (1), ng_vs_png (1), ng_vs_raw (1), jpeg_quality (1) |
| mito/temporal_clip | 73 | 57 | ng_bytes (57), delta_gt_fg_psnr (57), delta_gt_fg_ssim (57), delta_fg_ssim (57), ng_vs_png (57), jpeg_quality (57) |
| mito/temporal_clip_replace | 73 | 68 | ng_bytes (68), delta_gt_iou (68), delta_gt_fg_psnr (68), delta_gt_fg_ssim (68), delta_fg_ssim (68), ng_vs_png (68) |
| org_classical | 726 | 463 | ng_bytes (463), ng_vs_raw (463), ng_vs_png (463), delta_gt_fg_psnr (385), jpeg_quality (385), jpeg_fg_psnr (385) |
| org_default | 726 | 15 | ng_bytes (15), ng_vs_raw (15), ng_vs_png (15), delta_gt_iou (9), delta_gt_fg_psnr (9), delta_fg_ssim (9) |
| org_replace | 726 | 11 | ng_bytes (11), ng_vs_raw (11), ng_vs_png (11), delta_gt_iou (10), delta_gt_fg_psnr (10), delta_fg_ssim (10) |

- **org_default: 15/726 images changed, exactly the 15 bug-affected ones; 711 are bit-identical.**
  - Changed columns: only payload bytes and the byte-matched codec comparisons (9 of the 15 got a different JPEG quality).
  - No segmentation, graph, fidelity or GT column changed. The graph descriptors in `metrics.csv` come from the encoder-side graph, which was always correct.
- **T10 descriptors (GRAPH arm):** identical to the T10 remapped-payload GRAPH on 726/726. Against the as-tagged payload, 15 images change (geometry only).
- **The bug was far more widespread outside `org_default`.** It fires whenever `remove_small_components()` drops a component.
- Mean payload change:

| run | payload change |
|---|---|
| org_classical | −3.45 % |
| Cells3D membrane | −8.85 % |
| Cells3D nuclei | −5.27 % |
| retina | −2.33 % |
| MITO MIP | −8.18 % |
| temporal clip | −5.33 % |
| temporal clip (learned-replace) | −10.92 % |
| org_default / org_replace | −0.09 % |
| STED | −0.03 % |
| STARE / DRIVE / microtubules | 0 % |

Maximum per-image reduction: 14.4 %.
- **perturbation CSV:** bit-identical. **ablation CSV:** only `bytes_with/without` changed (15 images).

**`numbers.tex`:** 138 of 533 existing macros changed (14 timing, `\RunCommit`, the rest bytes / codec comparisons); 77 added (T11.2 `\DsP*`). Largest substantive moves:
- `\CBytesm` 2871 → 2772
- `\CBRR` 0.0438 → 0.0423; `\CBRRvsGU` +20 → +16
- `\CFGPSNRWinsP` 1.00 → 0.10
- `\CSelfIoUWinsJtwoP` 0.27 → **0.01** (classical self-IoU vs JPEG 2000 becomes significant)
- `\ClipLBytesm` 6.2 → 5.5 kB; `\MitoBytesm` 6.7 → 6.1 kB
- Default arm: bytes 2200 → 2198, GT-IoU wins 385 → 386/715 (p still 0.04)

**Claims: no verdict flips** (still 11 HOLD, 3 FAIL). Evaluated by `experiments/check_claims.py`, which reproduces the paper-v1 verdicts exactly on the old results:

| # | verdict | evidence |
|---|---|---|
| 1 | **HOLDS** | Seg-IoU 0.870 (D) vs 0.366 (C); payload 2198 B (D) vs 2772 B (C) |
| 2 | **HOLDS** | cycles 3.66 vs 7.65; width 2.98 vs 5.55 |
| 3 | **FAILS** | GT-IoU vs JPEG wins 386/715, p = 0.04; held-out wins 51.4 % |
| 4 | **FAILS** | classical GT-IoU vs JPEG wins 42/692, p = 4.0\times10^{-141} |
| 5 | **HOLDS** | GT-FG-PSNR 27.30 (NG) vs 27.90 (JPEG) |
| 6 | **HOLDS** | JPEG Betti 0.106; beta0 9.30 vs 4.41; beta1 3.97 vs 0.32 |
| 7 | **HOLDS** | max JPEG Betti excl. clip 0.177 (DRIVE); clip 0.768 |
| 8 | **HOLDS** | BRR 0.0335 (band 0.0274-0.0456); FewerX 7.6, RicherX 6.9 |
| 9 | **HOLDS** | org_replace held-out Seg-IoU 0.886 |
| 10 | **HOLDS** | max |as_is - oracle| 0.009 |
| 11 | **HOLDS** | dilate 1 px: beta0 changed 8.3 %, width +31.2 %; boundary noise 1 px: beta0 changed 78.2 %, median |dbeta1| 281; baseline Seg-IoU median 0.890 |
| 12 | **HOLDS** | no-JPEG microtubules 30/66, EPFL 2/10; STARE JPEG GT-FG-PSNR 42.9 vs 28.7 |
| 13 | **FAILS** | clip Seg-IoU learned-replace 0.696 vs default 0.750 |
| 14 | **HOLDS** | bg grid + residual improve FG-PSNR on 726/726 |

## T11.2 — shared spur pruning (every arm), then T10 re-run at `paper-v2`

Code: `experiments/downstream_morphometry.py` (`descriptors_at`, `SETTINGS`); outputs `results/paper/downstream/{per_image,summary,branch_distances,spur_fractions,timing,cost_summary}.csv` (columns `junction_def`, `L`). n = 715 (JPEG width 709). Cost unchanged: GRAPH 4.9 ms, JPEG 279 ms (57×), REF 4.4 ms.

**Pruning rule (identical in REF, JPEG, SEG, PRE, GRAPH):** one pass over the branch graph removes every terminal branch (free end of degree 1 at one end, a vertex with ≥ 3 branches at the other) shorter than `L` px; if every branch at a vertex would go, its longest is kept (no component disappears); vertices left with two branches are merged; branches, junctions, cycle rank, length and width are recounted. Cycle rank and component count are invariant under this pruning by construction (verified on all 726 REF masks).

**Second definitional mismatch found and fixed (junctions).** The T10 definition counts a junction-pixel cluster as a junction even when only two branches leave it (a pass-through), and splits the branch there. REF has no such clusters (4.53 → 4.53 junctions per image on 242 images checked); GRAPH has many (1.99 → 1.48 junctions, 7.73 → 7.32 branches): in the Nanograph, some junction-pixel nodes end up with two edges. The T10 definition therefore **flattered GRAPH**. From here on a junction is a vertex with ≥ 3 branches, and degree-2 vertices are merged (`junction_def = degree`), at every `L` including 0. The T10 numbers are kept as the first table.

### Terminal branches shorter than L (share of each arm at L = 0; pooled over images)

| L | REF branches | REF length | JPEG branches | SEG branches | PRE branches | GRAPH branches |
|---|---|---|---|---|---|---|
| 2 | 6.4 % | 0.4 % | 2.0 % | 1.4 % | 0.0 % | 0.0 % |
| 5 | 25.8 % | 3.4 % | 14.6 % | 10.8 % | 0.1 % | 0.1 % |
| 10 | 32.9 % | 6.0 % | 26.6 % | 19.3 % | 8.0 % | 8.5 % |

### L = 0, T10 junction definition (= T10 report)

| descriptor | CCC GRAPH / JPEG | bias % GRAPH / JPEG | 95 % LoA % GRAPH | MdAPE % GRAPH / JPEG | GRAPH closer / JPEG closer / tie | Wilcoxon p | better per image |
|---|---|---|---|---|---|---|---|
| n_components | 0.766 / -0.009 | -2.2 / +69.0 | -21 to +16 | 0.0 / 0.0 | 251 / 23 / 441 | 7.3e-41 | GRAPH |
| total_length_px | 0.724 / -0.028 | -11.6 / +22.9 | -32 to +9 | 10.0 / 5.0 | 216 / 499 / 0 | 5.0e-09 | JPEG |
| mean_width_px | 0.868 / 0.792 | +3.2 / +5.0 | -8 to +15 | 2.9 / 5.1 | 517 / 192 / 0 | 8.2e-43 | GRAPH |
| n_branches | 0.292 / -0.017 | -41.0 / +37.3 | -98 to +16 | 40.0 / 27.3 | 191 / 434 / 90 | 4.3e-06 | JPEG |
| n_junctions | 0.354 / 0.021 | -54.3 / +22.5 | -142 to +33 | 50.0 / 40.0 | 177 / 354 / 184 | 5.4e-05 | JPEG |
| cycle_rank | 0.507 / 0.183 | -44.8 / +20.7 | -282 to +192 | 50.0 / 50.0 | 141 / 46 / 528 | 2.1e-12 | GRAPH |
| branch-length distribution | W1 10.79 / 7.43 px; KS 0.416 / 0.333 | | | | 192 / 517 (W1) | 1.5e-43 | JPEG |

Decomposition (bias %, CCC): segmentation = SEG vs REF · graph construction = PRE vs SEG · storage = GRAPH vs PRE

| descriptor | segmentation | graph construction | storage |
|---|---|---|---|
| n_components | -1.4, 0.811 | -0.8, 0.956 | +0.00, 1.000 |
| total_length_px | -6.0, 0.841 | -4.0, 0.965 | -1.97, 0.995 |
| mean_width_px | +5.0, 0.829 | -1.2, 0.989 | -0.53, 0.999 |
| n_branches | -27.6, 0.472 | -18.5, 0.711 | +0.00, 1.000 |
| n_junctions | -39.5, 0.513 | -24.5, 0.801 | +0.00, 1.000 |
| cycle_rank | -36.6, 0.514 | -12.9, 0.901 | +0.00, 1.000 |

### L = 0 (no pruning), degree-based junctions

| descriptor | CCC GRAPH / JPEG | bias % GRAPH / JPEG | 95 % LoA % GRAPH | MdAPE % GRAPH / JPEG | GRAPH closer / JPEG closer / tie | Wilcoxon p | better per image |
|---|---|---|---|---|---|---|---|
| n_components | 0.766 / -0.009 | -2.2 / +69.0 | -21 to +16 | 0.0 / 0.0 | 251 / 23 / 441 | 7.3e-41 | GRAPH |
| total_length_px | 0.724 / -0.028 | -11.6 / +22.9 | -32 to +9 | 10.0 / 5.0 | 216 / 499 / 0 | 5.0e-09 | JPEG |
| mean_width_px | 0.868 / 0.792 | +3.2 / +5.0 | -8 to +15 | 2.9 / 5.1 | 517 / 192 / 0 | 8.2e-43 | GRAPH |
| n_branches | 0.249 / -0.017 | -44.3 / +37.3 | -104 to +15 | 42.9 / 27.3 | 179 / 450 / 86 | 2.9e-09 | JPEG |
| n_junctions | 0.262 / 0.021 | -66.5 / +22.5 | -157 to +24 | 66.7 / 40.0 | 142 / 437 / 136 | 1.5e-16 | JPEG |
| cycle_rank | 0.507 / 0.183 | -44.8 / +20.7 | -282 to +192 | 50.0 / 50.0 | 141 / 46 / 528 | 2.1e-12 | GRAPH |
| branch-length distribution | W1 12.38 / 7.43 px; KS 0.436 / 0.333 | | | | 164 / 545 (W1) | 1.1e-56 | JPEG |

Decomposition (bias %, CCC): segmentation = SEG vs REF · graph construction = PRE vs SEG · storage = GRAPH vs PRE

| descriptor | segmentation | graph construction | storage |
|---|---|---|---|
| n_components | -1.4, 0.811 | -0.8, 0.956 | +0.00, 1.000 |
| total_length_px | -6.0, 0.841 | -4.0, 0.965 | -1.97, 0.995 |
| mean_width_px | +5.0, 0.829 | -1.2, 0.989 | -0.53, 0.999 |
| n_branches | -27.6, 0.472 | -23.1, 0.598 | +0.00, 1.000 |
| n_junctions | -39.5, 0.513 | -44.7, 0.578 | +0.00, 1.000 |
| cycle_rank | -36.6, 0.514 | -12.9, 0.901 | +0.00, 1.000 |

### L = 2

| descriptor | CCC GRAPH / JPEG | bias % GRAPH / JPEG | 95 % LoA % GRAPH | MdAPE % GRAPH / JPEG | GRAPH closer / JPEG closer / tie | Wilcoxon p | better per image |
|---|---|---|---|---|---|---|---|
| n_components | 0.766 / -0.009 | -2.2 / +69.0 | -21 to +16 | 0.0 / 0.0 | 251 / 23 / 441 | 7.3e-41 | GRAPH |
| total_length_px | 0.731 / -0.028 | -11.2 / +23.2 | -32 to +9 | 9.6 / 4.9 | 221 / 494 / 0 | 1.1e-07 | JPEG |
| mean_width_px | 0.870 / 0.795 | +3.1 / +4.8 | -8 to +14 | 2.8 / 4.9 | 519 / 190 / 0 | 1.2e-42 | GRAPH |
| n_branches | 0.326 / -0.018 | -37.3 / +49.1 | -93 to +18 | 33.3 / 25.0 | 219 / 388 / 108 | 2.9e-02 | JPEG |
| n_junctions | 0.333 / 0.022 | -60.4 / +37.0 | -152 to +31 | 60.0 / 40.0 | 174 / 384 / 157 | 1.0e-06 | JPEG |
| cycle_rank | 0.507 / 0.183 | -44.8 / +20.7 | -282 to +192 | 50.0 / 50.0 | 141 / 46 / 528 | 2.1e-12 | GRAPH |
| branch-length distribution | W1 10.37 / 7.44 px; KS 0.388 / 0.327 | | | | 228 / 481 (W1) | 1.0e-26 | JPEG |

Decomposition (bias %, CCC): segmentation = SEG vs REF · graph construction = PRE vs SEG · storage = GRAPH vs PRE

| descriptor | segmentation | graph construction | storage |
|---|---|---|---|
| n_components | -1.4, 0.811 | -0.8, 0.956 | +0.00, 1.000 |
| total_length_px | -5.7, 0.845 | -4.0, 0.966 | -1.97, 0.995 |
| mean_width_px | +4.9, 0.832 | -1.2, 0.990 | -0.53, 0.999 |
| n_branches | -20.5, 0.574 | -21.2, 0.647 | +0.00, 1.000 |
| n_junctions | -31.4, 0.599 | -42.3, 0.619 | +0.00, 1.000 |
| cycle_rank | -36.6, 0.514 | -12.9, 0.901 | +0.00, 1.000 |

### L = 5 (recommended)

| descriptor | CCC GRAPH / JPEG | bias % GRAPH / JPEG | 95 % LoA % GRAPH | MdAPE % GRAPH / JPEG | GRAPH closer / JPEG closer / tie | Wilcoxon p | better per image |
|---|---|---|---|---|---|---|---|
| n_components | 0.766 / -0.009 | -2.2 / +69.0 | -21 to +16 | 0.0 / 0.0 | 251 / 23 / 441 | 7.3e-41 | GRAPH |
| total_length_px | 0.793 / -0.025 | -8.4 / +24.2 | -28 to +11 | 7.0 / 4.8 | 300 / 415 / 0 | 9.5e-01 | — |
| mean_width_px | 0.882 / 0.809 | +2.6 / +4.3 | -9 to +14 | 2.6 / 4.6 | 505 / 204 / 0 | 3.2e-40 | GRAPH |
| n_branches | 0.656 / -0.013 | -17.5 / +58.9 | -63 to +28 | 15.4 / 22.2 | 329 / 197 / 189 | 4.6e-16 | GRAPH |
| n_junctions | 0.602 / 0.061 | -38.4 / +45.5 | -140 to +63 | 42.9 / 33.3 | 241 / 202 / 272 | 2.2e-04 | GRAPH |
| cycle_rank | 0.507 / 0.183 | -44.8 / +20.7 | -282 to +192 | 50.0 / 50.0 | 141 / 46 / 528 | 2.1e-12 | GRAPH |
| branch-length distribution | W1 6.19 / 7.83 px; KS 0.309 / 0.343 | | | | 421 / 288 (W1) | 1.1e-07 | GRAPH |

Decomposition (bias %, CCC): segmentation = SEG vs REF · graph construction = PRE vs SEG · storage = GRAPH vs PRE

| descriptor | segmentation | graph construction | storage |
|---|---|---|---|
| n_components | -1.4, 0.811 | -0.8, 0.956 | +0.00, 1.000 |
| total_length_px | -3.9, 0.868 | -2.8, 0.979 | -1.98, 0.995 |
| mean_width_px | +4.3, 0.851 | -1.1, 0.992 | -0.53, 0.999 |
| n_branches | -10.0, 0.757 | -8.2, 0.899 | -0.11, 0.999 |
| n_junctions | -19.8, 0.739 | -23.0, 0.836 | -0.28, 0.999 |
| cycle_rank | -36.6, 0.514 | -12.9, 0.901 | +0.00, 1.000 |

### L = 10

| descriptor | CCC GRAPH / JPEG | bias % GRAPH / JPEG | 95 % LoA % GRAPH | MdAPE % GRAPH / JPEG | GRAPH closer / JPEG closer / tie | Wilcoxon p | better per image |
|---|---|---|---|---|---|---|---|
| n_components | 0.766 / -0.009 | -2.2 / +69.0 | -21 to +16 | 0.0 / 0.0 | 251 / 23 / 441 | 7.3e-41 | GRAPH |
| total_length_px | 0.799 / -0.024 | -7.9 / +21.2 | -28 to +13 | 6.4 / 4.2 | 292 / 423 / 0 | 1.7e-01 | — |
| mean_width_px | 0.884 / 0.811 | +2.5 / +4.5 | -9 to +14 | 2.7 / 4.7 | 504 / 205 / 0 | 1.7e-40 | GRAPH |
| n_branches | 0.625 / -0.018 | -17.4 / +49.1 | -65 to +30 | 12.5 / 18.2 | 266 / 190 / 259 | 4.6e-11 | GRAPH |
| n_junctions | 0.578 / 0.117 | -42.8 / +20.3 | -167 to +81 | 50.0 / 33.3 | 180 / 185 / 350 | 3.5e-01 | — |
| cycle_rank | 0.507 / 0.183 | -44.8 / +20.7 | -282 to +192 | 50.0 / 50.0 | 141 / 46 / 528 | 2.1e-12 | GRAPH |
| branch-length distribution | W1 6.58 / 8.08 px; KS 0.327 / 0.355 | | | | 395 / 314 (W1) | 5.5e-05 | GRAPH |

Decomposition (bias %, CCC): segmentation = SEG vs REF · graph construction = PRE vs SEG · storage = GRAPH vs PRE

| descriptor | segmentation | graph construction | storage |
|---|---|---|---|
| n_components | -1.4, 0.811 | -0.8, 0.956 | +0.00, 1.000 |
| total_length_px | -3.7, 0.869 | -2.3, 0.981 | -2.09, 0.994 |
| mean_width_px | +4.0, 0.861 | -0.9, 0.992 | -0.54, 0.999 |
| n_branches | -11.0, 0.727 | -6.4, 0.902 | -0.85, 0.987 |
| n_junctions | -25.8, 0.700 | -20.9, 0.853 | -2.50, 0.987 |
| cycle_rank | -36.6, 0.514 | -12.9, 0.901 | +0.00, 1.000 |

### CCC against REF, GRAPH vs JPEG side by side

| descriptor | T10 (L = 0, T10 def) GRAPH / JPEG | L = 5 GRAPH / JPEG |
|---|---|---|
| n_components | 0.766 / -0.009 | 0.766 / -0.009 |
| total_length_px | 0.724 / -0.028 | 0.793 / -0.025 |
| mean_width_px | 0.868 / 0.792 | 0.882 / 0.809 |
| n_branches | 0.292 / -0.017 | 0.656 / -0.013 |
| n_junctions | 0.354 / 0.021 | 0.602 / 0.061 |
| cycle_rank | 0.507 / 0.183 | 0.507 / 0.183 |

GRAPH has the higher CCC on **all six descriptors at every L** (and every junction definition); JPEG CCC is ≤ 0.12 on every count. In T10, JPEG nevertheless had the smaller **median** error on length, branches and junctions: JPEG is right on the typical image and wildly wrong on a minority (78/715 images with > 2× REF components; 6 empty masks), GRAPH is biased but tightly concordant.

### Recommended L: 5 px

Rationale, fixed from REF's own statistics and the encoder's definition, not from GRAPH agreement:
- `L = 5` is the encoder's own `spur_min_length`.
- It is about one mean structure diameter (REF mean width 5.7 px). A terminal branch shorter than one diameter cannot be told apart from boundary roughness.
- In REF it removes 25.8 % of branches but only 3.4 % of skeleton length.
- `L = 2` leaves most roughness spurs (removes 6.4 %). At `L = 10`, pruning starts to cut GRAPH's own branches (8.5 %, spur pruning cannot explain those) and a third of REF's.

**Caveat:** `L = 5` is also the most favourable `L` for GRAPH on paired wins. `L = 10` gives the same qualitative picture: GRAPH better on branches (266 vs 190, p = 5e-11), a tie on junctions (p = 0.35) and total length (p = 0.17).

### Outcome of the fair comparison: **2** — GRAPH still under-counts; the deficit is segmentation plus graph construction, and GRAPH now beats JPEG

At L = 5, against REF:

| descriptor | GRAPH bias (CCC) | vs JPEG per image | segmentation / construction / storage |
|---|---|---|---|
| branches | −17.5 % (0.656) | **better**: 329 vs 197, p = 5e-16 | −10.0 / −8.2 / −0.1 % |
| junctions | −38.4 % (0.602) | **better**: 241 vs 202, p = 2e-4 (JPEG has the lower MdAPE, 33 vs 43 %) | −19.8 / −23.0 / −0.3 % |
| cycle rank | −44.8 % (0.507) | **better**: 141 vs 46, p = 2e-12 | −36.6 / −12.9 / 0 % |
| total length | −8.4 % (0.793) | **no difference**: p = 0.95 (JPEG MdAPE 4.8 vs 7.0 %) | −3.9 / −2.8 / −2.0 % |
| components | −2.2 % (0.766) | **better** | — |
| mean width | +2.6 % (0.882) | **better** | — |
| branch-length distribution | W1 6.19 vs 7.83 px | **better**: 421 vs 288, p = 1e-7 | — |

(Pruning only removes terminal branches, so cycle rank does not depend on L; it is the same as in T10.)

- Versus T10: the branch bias falls from −41 % to −17.5 %, and branches, junctions and branch-length distributions flip from JPEG to GRAPH.
- GRAPH is **not ≈ REF on counts**. Segmentation is the larger share for cycle rank (74 %) and branches (55 %). For junctions, graph construction contributes as much as segmentation.
- The junction-construction loss is not exposed as a parameter (T11.3). Evidence of where it comes from: 26 % of GRAPH's T10 "junctions" are junction-pixel nodes left with only two edges.
- **Paper implication:** the downstream section can claim superiority over byte-matched JPEG on six of seven comparisons and equivalence on length. The branch/junction under-count must be stated as a limitation with the decomposition above.

---

## T11.3 — is the construction deficit tunable?

`experiments/graph_param_sweep.py`: 150 organelle images (random, seed 0; `results/paper/graph_sweep/subset.txt`), default config otherwise, codec at `paper-v2`, U-Net on CPU for every setting. One parameter at a time. Descriptors from the **decoded payload** vs REF, both pruned at L = 5 with degree-based junctions. FG-PSNR on the decoded render. Shipped defaults unchanged.

| setting | payload B (Δ) | FG-PSNR dB | branches bias % | junctions bias % | total length bias % | cycle rank bias % | components bias % | branches CCC |
|---|---|---|---|---|---|---|---|---|
| default | 2203 (+0) | 28.08 | -18.1 | -40.0 | -9.2 | -48.1 | -2.1 | 0.620 |
| spur_min_length=0 | 2216 (+12) | 28.08 | -17.3 | -38.8 | -9.1 | -49.4 | -2.1 | 0.633 |
| spur_min_length=2 | 2212 (+9) | 28.08 | -17.3 | -38.8 | -9.0 | -49.4 | -2.1 | 0.633 |
| spur_min_length=10 | 2186 (-18) | 28.07 | -29.5 | -58.8 | -11.6 | -41.8 | -2.2 | 0.359 |
| bridge=off | 2203 (-0) | 28.08 | -18.0 | -40.0 | -9.2 | -48.1 | -1.9 | 0.620 |
| bridge_max_gap=6.0 | 2203 (-0) | 28.08 | -18.0 | -40.0 | -9.2 | -48.1 | -1.9 | 0.620 |
| bridge_max_gap=20.0 | 2203 (+0) | 28.08 | -18.2 | -40.0 | -9.1 | -48.1 | -2.2 | 0.621 |
| min_component_nodes=0 | 2203 (+0) | 28.08 | -17.9 | -40.0 | -9.2 | -48.1 | -1.6 | 0.634 |
| min_component_nodes=6 | 2203 (-0) | 28.08 | -18.5 | -40.0 | -9.3 | -48.1 | -2.8 | 0.604 |
| min_component_nodes=10 | 2202 (-1) | 28.08 | -19.5 | -40.0 | -9.9 | -48.1 | -4.9 | 0.597 |
| spacing=2 | 2309 (+106) | 28.09 | -17.1 | -36.5 | -8.1 | -48.1 | -1.9 | 0.639 |
| spacing=5 | 2060 (-144) | 28.07 | -19.8 | -43.8 | -10.7 | -49.4 | -3.1 | 0.583 |
| spacing=8 | 1896 (-308) | 28.06 | -22.5 | -49.0 | -13.9 | -57.0 | -5.3 | 0.528 |

Without shared pruning (L = 0, T10 definition) the encoder spur prune is visible: branch bias default -41.9 %, spur_min_length=0 -31.9 %, =2 -34.6 %, =10 -47.7 %.

**Findings.** The under-count is a **defensible design choice, not an untuned default**, but the choice is not what causes it:
- **Encoder spur pruning (5 px):** costs nothing once spurs are matched (−18.1 vs −17.3 % branches with pruning off) and saves 12 B. Its only visible effect is at L = 0, where it removes exactly the roughness spurs REF counts. `spur_min_length = 10` does hurt: −29.5 % branches, −58.8 % junctions.
- **Gap bridging (off / 6 / 12 / 20 px) and minimum component size (0–6 nodes):** within ±0.5 pp on every count, and payload within ±1 B. `min_component_nodes = 10` starts dropping real components (−4.9 %).
- **Node spacing is the only real storage trade-off:**

| spacing | payload | branches | junctions | length |
|---|---|---|---|---|
| 2 | +106 B (+4.8 %) | +1.0 pp | +3.5 pp | +1.1 pp |
| 5 | −144 B (−6.5 %) | −1.7 pp | −3.8 pp | −1.5 pp |
| 8 | −308 B (−14 %) | −4.4 pp | −9.0 pp | −4.7 pp |

(changes relative to the default spacing of 3.)
- **FG-PSNR is flat across every setting** (28.06–28.09 dB).
- **Where the construction loss lives:** none of the four knobs moves the junction deficit by more than 3.5 pp. The construction share (PRE vs SEG: −8 % branches, −23 % junctions at L = 5) sits in `build_nanograph`'s edge building at junction-pixel clusters, which has no parameter. This is inferred from the degree-2 junction evidence in T11.2, not tested directly.

---

## T11.4 — robustness of the T11.2 comparison

Script `experiments/downstream_robustness.py`; outputs `results/paper/downstream/robustness/`.

**Pruning rule** (`pruning_rule.csv`). At L = 5, alternatives to the one-pass guarded rule change every bias by < 0.7 pp and no GRAPH-vs-JPEG verdict:
- **one pass, no guard:** branches −17.5 %, junctions −38.4 %
- **iterative:** branches −16.8 %, junctions −37.6 %; GRAPH closer on branches 334 vs 191, junctions 246 vs 198.
- Same at L = 2 and 10.

**Holm correction** (`holm.csv`, 7 paired tests per (definition, L), 35 in total). At L = 5 every significant result survives correction across all 35: components, width, branches (p_adj 1e-14), junctions (1e-3), cycle rank, branch-length W1 (1e-6). Total length stays n.s. At L = 10, junctions and length are n.s.

**SEG arm CPU vs GPU masks** (`seg_masks_cpu_gpu.csv`, `decomposition_gpu_mask.csv`):
- Mask IoU 0.99996 (min 0.9990); 675/715 masks bit-identical.
- The GPU re-segmentation equals the encoder's own mask on 13/13 images checked (both segmenters).
- The decomposition moves by ≤ 0.06 pp (L = 5 junctions: graph construction −23.03 → −22.97 %). **The graph-construction share is not a device artefact.**

**Held-out 108 images only** (105 with a fitting JPEG; U-Net never trained on them), L = 5:

| descriptor | GRAPH bias % (CCC) | JPEG bias % (CCC) | GRAPH / JPEG closer | p |
|---|---|---|---|---|
| components | −3.8 (0.724) | +41.2 (−0.005) | 36 / 5 | 4e-7 |
| total length | −8.4 (0.825) | +9.4 (0.347) | 45 / 60 | 0.81 |
| mean width | +2.2 (0.927) | +4.6 (0.797) | 78 / 27 | 5e-10 |
| branches | −19.5 (0.665) | +28.7 (0.186) | 54 / 31 | 1e-3 |
| junctions | −39.9 (0.636) | +14.3 (0.614) | 35 / 33 | 0.44 |
| cycle rank | −43.5 (0.431) | +6.5 (0.440) | 18 / 7 | 0.07 |

On held-out images GRAPH still wins on components, width and branches, but junctions and cycle rank are ties. **"Higher CCC on all six" does not hold on held-out images** (cycle rank 0.431 vs 0.440). Seg-IoU itself barely depends on the split (held-out 0.863, training 0.872).

**Checks bearing on manuscript claims** (not T10 outputs; recorded for the revision):
- **Betti agreement vs JPEG is a segmenter mismatch, not compression.** Against the pipeline mask on 197 images:
  - Otsu of the **uncompressed** image: **0.000**
  - Otsu of byte-matched JPEG (the paper's protocol): 0.124
  - the pipeline's own segmenter on the JPEG: 0.777
- **Paper's graph "β1" is the mask's hole count.** `\DBetaZero`/`\DBetaOne` are mask Betti numbers (`ng_seg_beta_*`), not stored-graph quantities; the mask is not in the payload.
- **Paper's cycle rank is dominated by junction-cluster triangles.** `graph_n_cycles` mean 3.66; mask holes 0.32; stored-graph cycle rank with junction clusters contracted 0.30 (annotation 0.55).
- **Refinement runs on every image.** 726/726 images carry optimizer PSF points in the payload (exactly 120 on 711 images, 122–124 on the rest; ~43 % of stored points). Methods says the stage is never triggered.
- **Node spacing is fixed,** 3 px (sparse) / 2 px (dense), not width-proportional as Methods states.
- **Decoded edges keep only the endpoints:** length is the chord between stored nodes (−2 % vs the traced path), and curvature is not stored (decoded as 0).
- **A lossless mask costs 3–6× less than the payload.** The pipeline's mask stored losslessly takes 366 B (bit-packed + zlib) or 659 B (PNG) vs 2211 B for the payload (182 images).
- **GT-IoU vs byte-matched JPEG is a tie in effect size.** Mean paired difference −0.0007 (all) / +0.0002 (held-out).
