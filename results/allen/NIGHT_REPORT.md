# Overnight report: 2026-09-25, 05:00–07:00

## Summary
- **The default segmenter is not decided yet.** The GPU-trained candidates are waiting for GPU 1, and the preference test needs you.
- **Provisional leader: Nellie, zero-shot** (mean rank 2.00 over 5 of the 6 benchmarks).
- Our real-trained U-Net and Allen-tuned Nellie are next, tied at 2.60.
- Nellie is also the fastest candidate: 0.08 s per tile on one core.
- Details: `results/allen/selection.md` (regenerated automatically when the remaining results land).

## Benchmarks so far (mean clDice; * = in-domain)

| candidate | UiT rat | CBMI | MITO | UiT human | Allen held-out |
|---|---|---|---|---|---|
| Nellie, zero-shot | 0.867 | **0.873** | 0.294 | 0.670 | 0.875 |
| Nellie, Allen-tuned | 0.865 | 0.871 | 0.281 | **0.684** | **0.876*** |
| our U-Net, real-trained (`real_mito`) | **0.922*** | 0.819* | **0.507*** | 0.679 | 0.752 |
| our U-Net, sim-only | 0.867 | 0.863 | 0.311 | 0.601 | 0.853 |
| micro-SAM, zero-shot | 0.639 | 0.836 | 0.060 | 0.456 | 0.774 |

- Nellie tuned on the real training data chose Nellie's default setting, so it is the same segmenter and is merged into it (see the rule addendum).
- Still to come:
  - our U-Net trained on Allen;
  - nnU-Net trained on Allen and on real data (needs GPU 1 after the paper run's GPU phase);
  - micro-SAM fine-tuned on Allen and on real data (needs a GPU with 20 GB free);
  - the blinded preference test (needs you).

## What the numbers say
- **Nellie is the best all-rounder with no training.** It is first or second on CBMI and Allen, and close to the best on UiT human.
- **Nellie is weak on MITO** (0.29 vs 0.51 for `real_mito`).
  - MITO is confocal, and its pixel size is not recorded. The width-based calibration may misfit it; the Allen-tuned setting did not help.
- **`real_mito` wins its own training datasets** (UiT rat, MITO) but is the worst non-SAM model on Allen (0.752).
  - It misses dense regions (see figure), so real-data training cost it generality.
- **Our simulation-only U-Net generalises better than the real-trained one** on CBMI and Allen, and is the best of all on dense Allen cells.
- **Zero-shot micro-SAM is last:** its masks are fat and merge neighbouring mitochondria. It is also 200× slower on CPU (16 s per image).
- **Junction and branch agreement is low for every method on UiT human and MITO** (junction F1 ≤ 0.24). Topology on those sets stays hard, whichever segmenter is used.

## Your reconstruction idea: tested, and it does not hold
- The best mask should give the best reconstruction, so I ran the encoder with each candidate's mask, and the expert mask, on 20 tiles per real dataset. Two variants: with and without the residual correction.
- **The expert's own mask never gave the best reconstruction:** MITO SSIM 0.748 for the expert mask vs 0.865 for the best candidate.
- Across candidates, fidelity agrees with expert clDice on UiT rat and CBMI, but not on MITO or UiT human (Spearman −0.05 to −0.68).
- Why: masks that cover more dim signal reproduce more pixels, whether or not those pixels are mitochondria.
- Consequence: it is recorded as a reported check, not a ranking vote.
- Script: `experiments/archive_demo/recon_bench.py`. The full run is after the GPU candidates.

## Figures
- `archive_demo/allen/showcase/fig_compare_v2.png`: 6 cells, one per cell-cycle stage.
  - Columns: image, Allen, and each candidate's mask against Allen, with its stored structure-layer centreline.

## Needs you
1. **Benchmark 6 (blinded preference):** 60 panels, each with 4 unlabelled masks (Allen + the top 3).
   - Built as soon as the top 3 are final: `python experiments/archive_demo/preference_pack.py --top a,b,c`.
   - I can put it on a click-through web page. Two raters are best.
2. **Docker** (optional): `docker rmi nirwan1998/privategptpp:latest` frees space. I was not permitted to run it, and your containers are untouched.

## Found and logged
- **Encoder bug:** `nanograph_encode` crashes on an empty mask (empty payload, then an `IndexError` on decode).
  - Logged in `results/allen/KNOWN_ISSUES.md`. Fix after the paper run; the package is frozen while it runs.
- **Rule corrections, each made before the affected results existed:**
  - one tuned-Nellie setting per training pool;
  - identical candidates merged;
  - benchmark 6 design;
  - mean (not median) clDice on Allen.

## Still running (tmux session `nanograph`)
- **Paper-v3:** STED (the previous run took 2.4 h, so it ends around 06:30), then the downstream analysis and phase 3.
  - Then `finish_v3.sh` (numbers, figures, supplement, direction checks), T14 lossless and T15-A.
- **`bakeoff` → `bench-queue`:** our U-Net on Allen → nnU-Net (Allen) → nnU-Net (real) → all evaluations → selection, on GPU 1 after the paper run's GPU phase.
- **`microsam-queue`:** the micro-SAM fine-tunes, when a GPU has 20 GB free.
