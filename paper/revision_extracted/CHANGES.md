# Nanograph revision — what changed, and what is left

Applied to `main.tex` as supplied; output is `main_revised.tex` (73,797 -> 79,256 characters). Brace balance, `table`/`tabular`/`figure` environment balance and `\ref`->`\label` resolution were all verified unchanged from the original. LaTeX was not compiled — no TeX distribution in this environment — so please build once before reading closely.

Every number introduced below was recomputed from `dataset_results/metrics.csv` (n = 726), the same file behind Tables 1–2 of the draft. The full recomputation is in `nanograph_recomputed_stats.csv`.

## The one-sentence summary of the problem

The draft's headline claim — beating byte-matched JPEG on 77% of images — is measured by Recon-IoU, which compares the reconstruction against the mask *the pipeline itself selected*. It is a self-consistency measure, not a fidelity measure, and §metrics already says so in as many words. Against independent ground truth the two codecs are indistinguishable (GT-IoU 0.6543 vs 0.6537; 354/726 wins; sign test p = 0.53; paired-difference 95% CI [−0.00044, +0.00175]). The candour existed in the metrics section and the tables; it had not propagated up into the abstract, highlights, contributions, or section headlines. That propagation is what these edits do.

The reframing is not a retreat. Parity is now a *provable* claim backed by a sign test and a bootstrap interval, and it is paired with a structural result that is large rather than marginal: Nanograph's Betti numbers are exact by construction, while byte-matched JPEG attains a mean Betti agreement of 0.150 ± 0.275 and inflates β₁ roughly eightfold (4.12 vs 0.53). Equal fidelity, exact topology, fewer bytes, plus morphometry no codec provides — that is a stronger and more defensible thesis than a 0.9-percentage-point IoU margin.

## Corrections to my own earlier notes

Two items from the change list I gave you were wrong or unnecessary. Recording them so you do not act on them:

1. **Table 1 does *not* have two identically labelled rows.** I reported that it did. It was an artifact of my search pattern — `Recon-IoU wins` contains the substring `IoU wins`. The table correctly labels both rows and already reports Recon-IoU 0.433, GT-IoU 0.654 and Seg-IoU 0.489 in its body. It is one of the most honest parts of the paper. I have only bolded the GT-IoU row for emphasis and extended the caption.

2. **§prior-quality needs no fix (my item 21).** I had flagged it for not normalising the reconstruction-quality comparison against prior work. It already declines the comparison outright, explains the bias in *both* directions, notes that the 256×256 fields are sparse with foreground at a few percent of frame, and names the experiment that would settle it. I made no change. Item 20 (the Topology-Q disclaimer) is likewise already adequate.

## Tier 1 — claims that were wrong as written (8 edits)

**Abstract.** Replaced the three win-rate figures with the GT-IoU parity result (with test statistic and CI) and the Betti-number contrast. This is the single most important edit: as written, the abstract's central empirical claim does not survive the repository's own results file.

**Highlights bullet (commented block).** Rewrote to parity + topology. Left commented, as in the original.

**Contribution (2).** Moved the load-bearing claim from reconstruction IoU onto ground-truth parity and exact topology.

**§repr headline.** Now names Recon-IoU as a self-consistency comparison, gives its actual margin (+0.0089, CI [+0.0079, +0.0099] — significant but small), then the GT-IoU parity result. Deleted the causal clause “because it spends its bits on structure rather than background texture”: the data do not support attributing the win to bit allocation when the ground-truth version of the same comparison is a tie.

**Extended codec comparison.** Corrected the FG-PSNR direction — JPEG wins foreground PSNR at matched bytes (28.84 vs 28.38 dB, 693/726 images), as does WebP, and the ordering holds against ground truth (28.10 vs 27.74 dB). **Deleted the JPEG 2000 “understates our advantage” inference**, which was the draft's most exposed sentence: at its 48%-larger budget JPEG 2000 wins foreground PSNR outright (30.68 dB), so an unmatched budget cannot be read as favouring Nanograph. The column is now explicitly disclaimed in both directions.

**§repr summary.** “structural win-rate” -> “self-consistency win-rate”, plus the GT-IoU and foreground-PSNR outcomes.

**§cross.** “confirming that the graph representation is not specific to organelles” -> “consistent with”, and added the two disclosures that matter: GT-IoU was computed for *none* of the four cross-modality datasets, and the 100% figure is over five images.

**§related / Compression.** “wins on foreground/structural fidelity at matched bytes” -> exact structural preservation plus foreground *parity*, and acknowledged the foreground-PSNR loss alongside the full-image one.

## Tier 2 — new table, from numbers already in the results file

**Added Table `tab:topo`** (immediately after `tab:codec`): β₀, β₁ and Betti agreement for Nanograph vs byte-matched JPEG. These columns were already sitting in `metrics.csv` and were never reported. This is the table that carries the reframed thesis, and it trades a 0.9-point margin for an eightfold one. The caption states that WebP and JPEG 2000 Betti numbers were not computed.

## Tier 3 — presentation (8 edits)

**`tab:nmi` caption.** Names both win-rates, marks GT-IoU as the load-bearing fidelity comparison, carries the test and CI.

**`tab:nmi` body.** Bolded the GT-IoU wins row.

**`tab:codec` caption.** Flags the IoU column as self-referenced, states plainly that FG-PSNR favours every codec at these budgets, and gives the Betti figures.

**`tab:cross` caption.** Per-dataset n added (726 / 60 / 60 / 25 / 5), GT-IoU absence stated, n = 5 column footnoted as not interpretable as a rate.

**`tab:cross` body.** IoU row relabelled `Recon-IoU wins/JPEG*` with the self-reference footnote.

**§metrics.** Removed the three-way comparison of GT-IoU against both Recon-IoU and Seg-IoU. Recon-IoU references a different mask, so it is not commensurable with the other two; the GT-IoU vs Seg-IoU comparison is valid and is retained. Good catch territory for a referee — the surrounding paragraph is otherwise the most careful in the paper.

**`fig:codec` caption (c).** Describes the added GT-IoU bar. **The figure itself is not regenerated** — see below.

**bytes/node.** Noted that ~7 bytes/node (pre-entropy-coding) and the 7.3 bytes/primitive of `tab:prior` (post-zlib BRR/APRR) are different quantities that agree by coincidence.

**Availability statement.** “adding one is recommended before wider release” -> “we record its absence here for completeness.” A paper should not issue recommendations to its own authors.

## What I could not do here, and why

**The package index was unreachable for the entire session** (502 at the proxy on every attempt, across conda and pip). No numpy, no matplotlib, no OpenCV. Everything above was computed with the Python standard library — the sign test is an exact two-sided binomial, the intervals are paired bootstraps at 10,000 resamples, seed 20260907.

Two consequences:

1. **`fig:codec` panel (c) is described but not regenerated.** Run `regen_fig_codec_panel_c.py` from the repo root when packages are back; it reads `metrics.csv` directly and needs no other input. Until then that caption promises a bar the figure does not show — the one place where the revision is internally inconsistent, so do this before circulating.

2. **GT-IoU for the cross-modality datasets is still missing.** Not a package problem: the ground-truth masks are not in the committed snapshot. They are reachable on your machine — `experiments/eval_ext_datasets.py` reads them from `{P}/{ds}/labels/{id}.png` and its docstring confirms all four datasets carry them. `patch_gt_iou_cross_modality.py` has the two columns to add, defined to mirror the organelle set exactly so the results drop straight into Table 1's frame of reference. **This is the highest-value remaining experiment**, because §cross currently rests entirely on the self-referenced metric, and on the one dataset where the ground-truth version was computed it came back a tie.

## Still open from the change list (unchanged by this pass)

- **Re-run the 726-image evaluation with `learned_mode='replace'`.** The learned centreline-aware segmenter is selected for zero of 726 organelle images in the default cascade — every headline number comes from the classical Otsu/Meijering path. The paper reports the learned segmenter as a contribution while no reported result uses it.

- **The error-guided refinement stage never fires** on any of 816 images. It occupies a method subsection and configuration parameters. Either find the reduced-node-budget regime where it activates, or cut it.

- Replace the `EDITME` placeholder and the private-test-suite reference before submission.

## Files

- `main_revised.tex` — the revision; drop-in replacement for main.tex
- `nanograph_recomputed_stats.csv` — all seven metric comparisons: means, win rates, exact sign-test p, mean paired difference, bootstrap 95% CI, verdict
- `regen_fig_codec_panel_c.py` — regenerates fig:codec panel (c) with the GT-IoU bar (needs numpy + matplotlib)
- `patch_gt_iou_cross_modality.py` — the GT-IoU columns for the four cross-modality datasets (needs the external masks)
- `CHANGES.md` — this file