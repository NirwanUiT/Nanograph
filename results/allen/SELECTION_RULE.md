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
