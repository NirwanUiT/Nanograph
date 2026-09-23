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
