# Nanograph — agent task list for the paper re-run

You are working in the `Nanograph` repository (package `nanograph_v4`). The manuscript lives in `paper/` and is **fully data-driven**: every run-dependent number is a macro in `paper/numbers.tex`, and three tables are generated bodies in `paper/tables/`. Both are produced by `paper/make_numbers.py`; data-driven figures come from `paper/make_figures.py`. Your job is to make the code match what the paper describes, produce every result at **one tagged commit**, regenerate numbers/figures, and report back.

## Ground rules

- Do **not** edit prose in `paper/nanograph_main.tex`. If a result contradicts a statement listed under "Claims to verify", report it; do not change the text.
- Do **not** change metric definitions (IoU, PSNR/SSIM masks, Otsu protocol, Betti agreement) except where a task below says so.
- One task = one commit, message prefixed with the task id (`T1: ...`).
- Every evaluation output directory gets a `commit.txt` containing `git rev-parse HEAD`.
- No SAM in any run (do not pass `--sam`).
- Data roots (from existing scripts): organelle images `/mnt/nas1/nba055-2/idea_1/nmi_data/org`, masks `/mnt/nas1/nba055-2/idea_1/nmi_data/seg`. Use the same roots the existing `cross_gt_results/`, `cross_dataset_results/`, `mito_*_results/` and `sted_results/` runs used; if a path is not recorded in the repo, find it in shell history or the scripts and **record it in `experiments/DATA_PATHS.md`**.

## Setup

Copy the delivered paper folder into the repo:
```
paper/
  nanograph_main.tex   references.bib   naturemag.bst
  make_numbers.py      make_figures.py
  figures/             tables/          numbers.tex (placeholder, will be regenerated)
```
Move the old LNCS draft to `paper/legacy/`.

---

## T1 — Store the edges in the payload

**Problem.** `compress.py` (v5) serialises points, width, intensity, orientation, types, background grid and DCT residual. It stores no connectivity, so the decoder cannot rebuild the graph. The paper states the stream contains the edge list and that the decoder rebuilds `G`.

**Do:**
1. Confirm point index `i` in `compress_nanograph(points, ...)` corresponds to `graph.nodes[i]` (the CSVs show `n_points == graph_n_nodes`). If not, build the mapping explicitly.
2. After row-sorting (`order = np.argsort(rows)`), map each graph edge `(u, v)` to sorted positions `(pu, pv)` with `pu < pv`. Sort edges by `pu`, then `pv`.
3. Encode each edge as two zigzag varints: `pu - pu_prev` and `pv - pu`. Append as a new section inside the zlib'd payload. Set flag bit 5 (`has_edges`). Bump the stream version to 6; keep v4/v5 decoding working.
4. `decompress_nanograph` returns the decoded edge list and an adjacency dict in addition to what it returns now.
5. Add `decode_graph(payload) -> Nanograph` in `api.py` that rebuilds the `Nanograph` object from decoded nodes + edges. Edge attributes that are functions of node attributes (length, mean width, mean intensity, curvature) are recomputed from the decoded nodes.

**Acceptance (add to `test_v4.py`, runnable with `pytest -q`):**
- Round trip on 20 organelle images and 5 images from each cross-modality set: decoded edge set == encoder edge set; `beta0`, cycle rank `|E|-|V|+beta0`, total edge length (±1e-6 relative) equal between encoder-side graph and `decode_graph`.
- Print mean±sd byte cost of the edge section on the 726 organelle images (report it).

## T2 — Make predictive coding invertible

**Problem.** `_graph_predictive_encode` predicts width/intensity from the mean of already-encoded graph neighbours; `_graph_predictive_decode` predicts from the previous point in row order. They disagree, so decoded attributes are wrong whenever a node has encoded neighbours. Residuals are also clipped to int8 (lossy). Demonstrated on a 200-node path graph: mean |ΔI| ≈ 0.028 (≈7 grey levels), max 0.09; max |ΔW| 0.25 px.

**Do:**
1. Predict along the **stored** edges (from T1): for node at sorted position `p`, predictor = mean of neighbours with sorted position `< p`; if none, previous node; if `p == 0`, 128. Use the identical function on both sides (decoder has the adjacency from T1).
2. Encode residuals as zigzag varints (or int8 with an escape byte followed by int16) — no clipping.

**Acceptance:**
- Unit test: synthetic path, branching tree and graph with a cycle; decoded quantised width/intensity == encoder quantised values exactly.
- On all 726 organelle images: exact equality of decoded vs encoder-side quantised attributes (log the count of mismatches; must be 0).

## T3 — Compute every metric on the decoded payload

**Problem.** `api.py` computes PSNR/SSIM on the in-memory reconstruction *before* compression. `evaluate.py` hard-codes `ng_iou = 1.0`.

**Do:**
1. In `nanograph_encode`, after compression, render the image from `decode_graph(payload)` (plus decoded background grid and residual) and compute all fidelity metrics on that render. Keep the pre-compression values only as debug columns prefixed `pre_`.
2. In `evaluate.py`, remove the hard-coded `ng_iou = 1.0`. Add column `self_iou` = IoU(Otsu(decoded recon), selected mask). Keep `ng_recon_iou` as an alias with the same value.
3. Verify `jpeg_gt_fg_psnr` is computed inside the GT mask and `jpeg_fg_psnr` inside the pipeline mask (in the existing learned-replace CSV their means coincide at 27.85; confirm this is a coincidence and not the same computation). Report what you find.

**Acceptance:** on 20 images, `ng_fg_psnr` from the decoded render equals the pre-compression value within 0.01 dB after T1+T2 (prediction is lossless, quantisation unchanged). Report the max difference.

## T4 — Presets and held-out list

1. Add `--preset classical` to `run_dataset.py`: `cfg.segment.use_learned = False`.
2. Add `experiments/write_heldout.py` that reproduces the organelle U-Net split from `experiments/train_unet.py` (sorted basenames of `org/*.png`, `np.random.default_rng(0).shuffle`, first `int(0.15*n)` = 108) and writes the stems to `results/paper/heldout_organelle.txt`. Sanity check: under the existing learned-replace CSV, this list gives 62/108 GT-IoU wins vs JPEG.
3. Add `--force-segmenter {otsu,frangi,meijering,learned}` to `run_dataset.py` (sets the cascade to that single candidate) — needed for T7.

## T5 — The paper run (single tagged commit)

After T1–T4 are merged and tests pass: `git tag paper-v1`. Then produce, **all at that commit**, into `results/paper/`:

| Output dir / file | Command (sketch) |
|---|---|
| `org_default/` | `run_dataset.py --images $ORG --masks $SEG --outdir results/paper/org_default` |
| `org_classical/` | same + `--preset classical` |
| `org_replace/` | same + `--preset learned-replace` |
| `ablation_bg_residual.csv` | `experiments/ablation_bg_residual.py --images $ORG --out results/paper/ablation_bg_residual.csv` |
| `seg_perturbation_ablation.csv` | `experiments/seg_perturbation_ablation.py --images $ORG --masks $SEG --out results/paper/seg_perturbation_ablation.csv` |
| `cross/{cells3d_membrane,cells3d_nuclei,retina,cell}/` | `run_dataset.py` on each (default config, same inputs as `cross_dataset_results/`) |
| `cross_gt/{stare,drive,epfl_mito,microtubules}/` | `run_dataset.py --preset curvilinear` with masks (same inputs as `cross_gt_results/`) |
| `mito/temporal_clip/` | default config (same inputs as `mito_aaron_results/`) |
| `mito/temporal_clip_replace/` | `--preset learned-replace` |
| `mito/sted/` | `--preset learned-replace` (same inputs as `sted_results/`) |
| `mito/mito_mip/` | default config (same inputs as `mito_mip_results/`) |
| `heldout_organelle.txt` | `experiments/write_heldout.py` |
| `commit.txt` | `git rev-parse HEAD` |

Every `run_dataset.py` output must contain `metrics.csv` with at least the columns currently written plus `self_iou`. Byte-matched JPEG, WebP and JPEG 2000 comparisons must be on for the organelle runs; JPEG on for all others.

## T6 — Polarity verification

Run `experiments/eval_retrained.py` for `stare`, `drive`, `microtubules`, `epfl` (as-is vs best-of-2 oracle) and write `results/paper/polarity.csv` with columns `dataset,as_is,oracle` (dataset names exactly as listed).

## T7 — MITO dataset diagnostics (report only, no paper numbers)

1. Commit the tile and max-projection preparation scripts under `experiments/prep_mito_zenodo.py` (whatever produced the inputs of `mito_zenodo_results/`, `mito_zenodo_tiles_results/`, `mito_mip_results/`), including how per-slice masks were turned into MIP masks.
2. Run `run_dataset.py --force-segmenter X` for X in otsu, frangi, meijering, learned on the 228 MIP tiles → `results/paper/mito_diag/X/`. Report mean Seg-IoU, precision, recall per segmenter.
3. Save 10 overlays (input, prediction, annotation; TP/FP/FN coloured) to `results/paper/mito_diag/overlays/`.
4. Compute precision restricted to a 20-px dilation of the annotated foreground ("precision near annotated structure") per segmenter. If precision rises sharply when restricted, the low global precision is mostly unannotated structure — say so in the report with the numbers.

## T8 — Figures

1. `python paper/make_figures.py --runs results/paper --out paper/figures` (regenerates `fig_codec_rd`, `fig_seg_bottleneck`, `fig_topology`, `cross_dataset_comparison`, `fig_modality_scaling`, `fig_prior_positioning`, `fig_seg_perturb`).
2. Regenerate `paper/figures/pipeline_overview.png` (the eight-stage montage, Methods figure) with `experiments/render_pipeline_diagram.py` on organelle image `7378`, default config, rendering stage 7 from the **decoded** payload; panel titles must show the new payload bytes and ratios. Also write a JSON next to it with `image, raw_bytes, payload, nodes, edges, components, points, psnr, fg_psnr, seg_iou` for that image.
3. Rebuild Figure 1: `python paper/make_fig_pipeline.py --stages paper/figures/pipeline_overview.png --meta paper/figures/pipeline_overview.json --out paper/figures/fig_pipeline.png`. It crops the montage panels and re-lays them out; if the montage layout changes, fix the crop in `crop_panels` rather than hand-editing the figure.
4. `fig_qualitative_organelle.png`, `fig_ood_failure.png`, `fig_mito_generalisation.png`: no generating script is in the repo. Write `experiments/render_paper_panels.py` that reproduces them (same layout as the current files) from the new code, and regenerate. **The MITO row of `fig_mito_generalisation.png` currently shows an almost empty tile (6 nodes, 251 B)** — pick a tile whose annotation contains a reasonable amount of structure (e.g. the median tile by annotated foreground fraction) and say which tile you used.

## T9 — Numbers and build

```
python paper/make_numbers.py --runs results/paper --out paper
cd paper && latexmk -pdf nanograph_main.tex
grep -c '\[TBD\]' <(pdftotext nanograph_main.pdf -)   # must be 1 (the draft-page mention) or 0 with \finaltrue
```

## Claims to verify (report each as HOLDS / FAILS with the numbers)

The prose asserts these; they depend on T5 outputs. Macro names in brackets.

1. Default Seg-IoU > classical Seg-IoU, and default payload < classical payload [`\DSegIoUm`, `\CSegIoUm`, `\DBytesm`, `\CBytesm`].
2. Default graphs are cleaner than classical: lower cycle rank and lower mean width [`\DCyclesm` < `\CCyclesm`, `\DWidthm` < `\CWidthm`].
3. Default GT-IoU beats byte-matched JPEG with p < 0.01 [`\DGTIoUWinsP`], and wins > 50% on the held-out 108 [`\DHeldGTIoUWinsPct`].
4. Classical GT-IoU vs JPEG is not significant (p > 0.05) [`\CGTIoUWinsP`].
5. GT-FG-PSNR favours JPEG under the default [`\DGTFGPSNRm` < `\DJpegGTFGPSNRm`] (the text says the codecs lead photometrically).
6. JPEG Betti agreement on organelles ≤ 0.2 [`\DJpegBettim`], JPEG β0 and β1 both exceed the graph's.
7. JPEG Betti agreement ≤ 0.2 on every dataset in `tab_topoall_body.tex` except the temporal clip.
8. Storage budget: BRR within ±25% of 0.0365 after adding the edge section [`\DBRR`, `\DBRRvsGU`]; report `\DFewerX`, `\DRicherX`.
9. `org_replace` held-out Seg-IoU within 0.02 of 0.875 [`\LHeldSegIoUm`].
10. Polarity: |as_is − oracle| ≤ 0.02 on all four datasets.
11. Perturbation CSV reproduces the current per-arm medians (dilation 1 px: β0 changed on ~8% of images, width +31%; boundary noise 1 px: β0 changed on ~78%, median |Δβ1| ~281). Default-config baseline Seg-IoU ≈ 0.870.
12. Cross-GT: on microtubules and EM some images are below JPEG's byte floor [`\CgMtNoJpeg`, `\CgEpflNoJpeg` non-zero]; STARE JPEG GT-FG-PSNR > Nanograph's.
13. Temporal clip, learned-replace: Seg-IoU ≥ default Seg-IoU on the clip [`\ClipLSegIoUm` vs `\ClipSegIoUm`].
14. Ablation: background grid + residual improve FG-PSNR on all 726 images [`\AblImproved`].

## Report back

Write `results/paper/RESULTS_REPORT.md` containing:
- commit hash and tag;
- T1 edge-section byte cost; T2/T3 round-trip test outputs;
- the full `paper/numbers.tex` diff against the placeholder;
- the 14 claims with HOLDS/FAILS and numbers;
- T3 finding on `jpeg_gt_fg_psnr`;
- T7 MITO diagnostics table and conclusion;
- any run that failed or was skipped, with the error.

Then commit `paper/` (tex, numbers.tex, tables, figures, scripts) and `results/paper/` summaries (CSVs are fine to commit; skip images other than overlays).
