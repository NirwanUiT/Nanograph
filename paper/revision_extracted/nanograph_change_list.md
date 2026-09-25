# Nanograph — complete change list before submission

Locations are byte offsets into `main.tex` as supplied, with section labels. Recomputed figures come from `dataset_results/metrics.csv` (n=726) and `cross_dataset_results/*/metrics.csv`.

---

## Tier 1 — claims that are wrong as written (must change)

The single issue behind items 1-9: **Recon-IoU is measured against the pipeline's own selected mask**, so it is a self-consistency metric, not a fidelity metric. It is the only comparison Nanograph wins, by a mean margin of +0.0089 IoU. Every ground-truth-referenced metric ties or favours the codecs, at 2.6 % fewer bytes.

| metric | NG | JPEG | NG win rate |
|---|---|---|---|
| Recon-IoU (vs NG's own mask) | 0.4331 | 0.4243 | 557/726 (77 %) |
| GT-IoU (vs ground truth) | 0.6543 | 0.6537 | 354/726 (**49 %**) |
| FG-PSNR | 28.377 | 28.836 | 33/726 (**5 %**) |
| GT-FG-PSNR | 27.735 | 28.104 | 90/726 (**12 %**) |
| GT-FG-SSIM | 0.9991 | 0.9992 | 50/726 (7 %) |
| Full PSNR / SSIM | 30.74 / 0.564 | 31.24 / 0.620 | 0/726 (0 %) |

**1. Abstract (@5367).** "recovers the foreground more faithfully than byte-matched JPEG on 77 % of images, WebP on 88 % and JPEG 2000 on 82 %" — the word *faithfully* attached to a self-referenced metric is the core problem. Replace with the parity-plus-structure claim (Tier 2, item 11).

**2. Highlights bullet (@2577).** "...4 dB FG-PSNR, beats byte-matched JPEG on IoU 77 % of the time." Both halves are unsafe: JPEG scores 28.84 dB FG-PSNR against your 28.38, and the IoU is the self-referenced one. Rewrite entirely.

**3. Contribution (2) (@11939).** "beats byte-matched JPEG, WebP and JPEG 2000 on reconstruction IoU for 77-88 % of images" — technically true and technically labelled ("reconstruction IoU"), but a contribution bullet is read as a fidelity claim. Move the contribution onto topology.

**4. Sec.~\ref{sec:repr} headline (@37322).** "reconstructs the biological foreground more faithfully than JPEG on 77 % of images by Recon-IoU ... because it spends its bits on structure rather than background texture." The causal clause is a good explanation of a result you do not have. Disclose GT-IoU 49 % in the same paragraph.

**5. Extended comparison (@38319).** Same reframing as item 1.

**6. The JPEG 2000 sentence (@38527)** is the most exposed sentence in the paper: JPEG 2000 used 3661 bytes, 48 % *more* than your budget, "and still lost on IoU 82 % of the time, which if anything understates \method's advantage at truly matched bytes." At those bytes JPEG 2000 reaches **FG-PSNR 30.68 vs your 28.38** — it wins fidelity decisively. The comparison is not byte-matched, so it cannot be used in either direction. Delete the "understates" inference and report JPEG 2000 in a clearly-marked unmatched-budget row.

**7. Sec.~\ref{sec:cross} (@49658)** and **8. the `IoU wins/JPEG` row of the cross-modality table (@51811)** — "82-100 % of images across datasets, confirming that the graph representation is not specific to organelles." **`gt_iou` is not computed in any of the four cross-modality result files.** The entire generality claim rests on the self-referenced metric with no ground-truth counterpart anywhere. Either run item 10, or relabel the row `Self-IoU wins/JPEG` and drop "confirming" to "consistent with".

**9. Sec.~\ref{sec:related}, Compression subsection.** The positioning sentence — that Nanograph wins on foreground and structural fidelity at matched bytes — is contradicted by your own Table~\ref{tab:codec}: FG-PSNR loses to **all three** codecs (28.38 vs 28.84 / 28.52 / 30.68). Restate as parity on foreground fidelity, advantage on structure.

---

## Tier 2 — experiments and additions, cheapest first

**10. Compute GT-IoU for the four cross-modality datasets.** Zero new training; the ground-truth masks exist and `evaluate.py` already has the code path that populates `gt_iou` for the organelle set. This is the cheapest change on the list and it decides whether item 7 is a relabel or a real result. Do it first.

**11. Add topology preservation to Table~\ref{tab:codec}. This is the fix that makes the paper stronger, and the numbers are already in the CSV.**

| | $\beta_0$ | $\beta_1$ | Betti preservation |
|---|---|---|---|
| Nanograph | 5.91 $\pm$ 3.11 | 0.53 $\pm$ 0.80 | exact by construction |
| byte-matched JPEG | 9.54 $\pm$ 3.93 | 4.12 $\pm$ 3.69 | **0.150 $\pm$ 0.275** |

JPEG inflates $\beta_0$ by 61 % and $\beta_1$ by roughly 8x — spurious components and spurious loops — and preserves 15 % of the topology. You currently report your own Betti numbers in Sec.~\ref{sec:repr} but never the codecs'. Proposed replacement claim for the abstract and contribution (2):

> At matched bytes Nanograph reaches foreground-fidelity parity with JPEG, WebP and JPEG 2000 (GT-IoU 0.654 vs 0.654; GT-FG-SSIM 0.999 vs 0.999) while preserving topology exactly, where the same codecs retain 15 % of it and expose no morphometry at all.

This concedes a tie the data already forces and trades a 0.009-IoU margin for a ~6x one, on the axis the paper's thesis is actually about.

**12. Re-run the 726-image evaluation with `learned_mode='replace'` — the highest-value experiment available.** `Counter(segmenter)` over the results is `{Meijering: 567, Otsu: 159}`: the clDice U-Net is selected for **zero of 726 images**, so every headline number in the paper comes from the classical cascade at Seg-IoU 0.489 / precision 0.493. Meanwhile Sec.~\ref{sec:seg-standalone} reports it lifting held-out IoU 0.381 -> 0.875. Since Sec.~\ref{sec:seg} argues segmentation *is* the bottleneck, running the pipeline with the component that fixes it should move GT-IoU, GT-FG-PSNR and the win rates together — and it converts that subsection from a limitation into the paper's main result.

**13. Run the background-grid/residual ablation over the full 726 images** instead of the single image pair. No training, and Sec.~\ref{sec:ablation} currently carries a table off n=1.

**14. Decide the fate of error-guided topology refinement (Sec.~\ref{sec:optimizer}).** It fires on zero of 816 images while occupying a full method subsection and 10 of the 121 config parameters. Either run the reduced-node-budget sweep that makes it act, or cut it to a paragraph in Future work. As written it invites the question of what else in the config is inert.

**15. Put uncertainty on the win rates.** They are bare fractions of 726 paired comparisons; a paired bootstrap CI or a sign test costs nothing and pre-empts the obvious question about a +0.0089 margin. The 100 % win rate on the `cell` dataset is **n=5** and should be dropped or pooled rather than reported as a rate.

---

## Tier 3 — presentation and structure

**16. Rename Recon-IoU** to Self-IoU or Mask-Consistency IoU throughout, and restructure Sec.~\ref{sec:metrics} so ground-truth-referenced metrics are defined first and marked primary, with the self-referenced ones grouped explicitly as self-consistency checks. This one rename removes most of the Tier 1 exposure at the source.

**17. Fig. `fig_codec_rd`, panel (c).** It plots the win rate on Recon-IoU (green) beside full-image SSIM (grey) as "the honest counterpoint" — but full-image SSIM is the metric you have already conceded and explained away, so the panel pairs your best number with a harmless one and omits the decisive middle case. Add a GT-IoU bar at 49 %.

**18. Fig. `fig_seg_bottleneck`, panel (c).** The caption warns that the green line "is a standalone evaluation, not the cascade's selected mask." A caption doing damage control means the panel is juxtaposing two things it shouldn't. Split it, or plot the cascade's actual selected-mask distribution once item 12 exists.

**19. Cross-modality table.** The caption promises mean$\pm$std "unless noted" but several rows carry bare values. Fix or amend the caption.

**20. Consider dropping Topology-Q from the manuscript entirely.** You already concede it is not a topological invariant and keep it because the pipeline emits it. A metric that needs a disclaimer every time it appears costs more than the column is worth — leave it in the CSV.

**21. Sec.~\ref{sec:prior-quality}.** APRR and BRR are resolution-normalised, but they are not protocol-independent: your 256$\times$256 crops and the prior systems' 1024$\times$1024 frames differ in foreground fraction and structure density, which is exactly what APRR is a ratio of. Say so, rather than resting on resolution normalisation alone.

**22. Bytes-per-primitive inconsistency.** Sec.~Compression states "$\approx$7 bytes/node before entropy coding" while the prior-positioning table's 7.3 is BRR/APRR, i.e. *after* zlib. Reconcile the two or label each.

**23. Reorder the contributions.** Your two strongest items — the out-of-domain failure analysis (a well-formed graph of the wrong structure, which does not fail loudly) and the BRR = APRR $\times$ bytes-per-primitive factorisation — are currently behind the codec win rate. Lead with them.

---

## Tier 4 — submission mechanics

**24.** Resolve both `EDITME` placeholders: author order/inclusion carried over from the prior papers, and the repository URL.

**25. Make the repository public.** The availability statement currently reads "it is currently a private GitHub repository." For a paper asking readers to accept a representation in place of pixels, a private repo is a reviewer objection on its own.

**26. Pick a venue and commit to it.** The preamble comments specify Nature-style typography, there is an Elsevier highlights block, and the text refers to MedIA. The content is a methods paper for a journal like MedIA; drop the other two scaffolds.

**27. The test-suite sentence** — "adding one is recommended before wider release" — is a recommendation to yourself sitting in an availability statement. Either add the tests or state the position neutrally.

**28. `track.py`.** The manuscript discloses that this module's internal comments cite validation results from scripts absent from the snapshot. Clean the comments and describe the module as unvalidated forward-looking engineering; as written you are advertising that the codebase contains unverifiable claims, which invites doubt about the parts that are fine.

---

## Suggested order of work

1. Item 10 (GT-IoU cross-modality) and item 11 (topology table) — cheap, and together they determine the wording of every Tier 1 fix.
2. Item 12 (`replace` mode) — the one experiment that could improve the headline numbers rather than only qualify them.
3. Items 1-9 and 16-17 — rewrite the claims once the numbers above are settled.
4. Items 13-15, 18-23.
5. Tier 4 last.

Nothing here questions the method. The representation argument, the factorisation, and the out-of-domain analysis all hold. The problem is confined to which number the paper leads with.
