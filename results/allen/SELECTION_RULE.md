# Default segmenter: selection rule (fixed 2026-09-25, before the bake-off results)

## Candidates
- `real_mito`: shipped real-mito clDice U-Net (trained on UiT-Rat, CBMI and MITO training splits plus simulations)
- `allen_ft`: that U-Net fine-tuned on Allen TOMM20 tiles (Allen's segmentation as labels)
- `nnunet`: nnU-Net v2 2-D trained on the same Allen tiles and split
- `nellie`: Nellie (Lefebvre et al., Nat Methods 2025), zero-shot
- `microsam_zs` / `microsam_ft`: micro-SAM, zero-shot and fine-tuned on the Allen tiles (when a GPU is free)

## Benchmarks (each produces one ranking)
1. UiT-Rat test set: expert masks
2. CBMI test set: expert masks
3. MITO test set: expert masks
4. EP-UiT-Human test set: expert masks (never used for training or model selection)
5. Allen held-out plates (300 cells): Allen's segmentation. This is a model output, so it counts as one vote, not ground truth.
6. Blinded pairwise preference on 60 random Allen held-out cells. Masks are shown unlabelled, in random left/right order. Score = win rate (ties split).

## Metric
- Ranking metric: mean per-image clDice against the benchmark's reference; win rate for benchmark 6.
- If two candidates are within 0.01 clDice: rank by junction F1 at 3 px on the structure layers.
- Reported alongside, not used for ranking:
  - Dice;
  - endpoint F1;
  - descriptor CCC / bias from each candidate's structure layer (one-diameter rule);
  - seconds per image.

## Decision
- The default is the candidate with the best mean rank over the six benchmarks.
- Ties are broken by speed (seconds per image, single core).
- A candidate that fails on a benchmark (no output, or crashes on more than 5 % of images) takes the last rank there.

## Known biases, recorded before the results
- `real_mito` was trained on the training splits of benchmarks 1–3: it has an in-domain advantage there. Benchmark 4 is out of its training data.
- `allen_ft` and `nnunet` learn Allen's segmentation style: they are favoured on benchmark 5.
- Nellie was inspected visually on 6 Allen cells before this rule was fixed (it looked better than both `real_mito` and Allen's segmentation). Those 6 cells are excluded from benchmark 6.
- Nellie's filter scales use physical pixel size. The real datasets' pixel sizes are set from their sources where known. Otherwise they are set from the median mitochondrial width in the training masks, fixed before scoring.

## Addendum (2026-09-25, before any variant below was scored)
Every model is entered untrained or zero-shot, and trained or fine-tuned. All variants are candidates under the same rule.

| model | untrained / zero-shot | trained on real expert data | trained on Allen |
|---|---|---|---|
| clDice U-Net | `unet_sim`: shipped simulation-only weights | `real_mito` | `allen_ft` |
| nnU-Net v2 2-D | n/a: no pretrained nnU-Net exists; it always trains from scratch | `nnunet_real`: UiT-Rat + CBMI + MITO training splits, validated on their val splits | `nnunet` |
| micro-SAM | `microsam_zs`: vit_b_lm generalist | `microsam_real` | `microsam_ft` |
| Nellie | `nellie`: defaults | `nellie_tuned_real`: one setting chosen on the pooled UiT-Rat + CBMI + MITO training splits | `nellie_tuned_allen`: one setting chosen on Allen training cells |

- Nellie is not a learned model: its "trained" variant is a setting tuned on training data only.
  - Grid: `otsu_thresh_intensity` in {False, True} × `min_radius_um` in {0.15, 0.25, 0.35}.
  - 60 seeded training images per source; the setting with the best mean clDice is chosen.
  - As for the trained models, one setting per training pool is applied everywhere.
  - Pixel size is the per-dataset calibration and is not tuned.
  - (Corrected before any tuned variant was scored: an earlier draft said "per dataset".)
- Benchmark results on a variant's own training domain are marked in-domain in every table.

## Addendum: benchmark 6 design (2026-09-25, before benchmarks 1–5 were complete)
- 60 Allen held-out cells: seed 0, excluding the 6 showcase cells.
- Per cell: the slab image plus 4 unlabelled masks in random order. The masks are Allen's segmentation and the top 3 candidates by mean rank on benchmarks 1–5.
- The rater picks the best mask (and may mark "no clear winner").
- Score = fraction of cells on which the candidate was picked (no-clear-winner cells are split equally).
- Candidates not shown rank below the shown ones, in their benchmark 1–5 order.
- Two raters where possible; their picks are pooled.

## Reference-free reconstruction check (reported, not a ranking vote)
- Suggested by the author: the best mask should give the best Nanograph reconstruction.
- Tested on the real sets, where expert masks exist (`recon_bench.py`); kept as a reported check.
- Pilot, 2026-09-25 (20 tiles per dataset, 5–6 candidates):
  - Fidelity tracked expert clDice on UiT-Rat and CBMI.
  - It did not on MITO or EP-UiT-Human (Spearman −0.05 to −0.68).
  - The expert mask was never the best-reconstructing mask.
  - So reconstruction fidelity is not a valid proxy for mask correctness here.

## Addendum: identical candidates (2026-09-25, before final results)
- A tuned variant whose chosen setting equals the untuned default is the same segmenter: it is merged into the default, not ranked separately.
- Duplicates would otherwise take two ranks and push every other candidate down.
- This happened for `nellie_tuned_real`: tuning chose Nellie's default.
