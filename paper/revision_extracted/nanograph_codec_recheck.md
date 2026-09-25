# Nanograph: the byte-matched codec comparison, recomputed

Recomputed from `dataset_results/metrics.csv` (n = 726, the same file behind Tables 1-2 of the draft). Win rate = fraction of images where Nanograph scores higher than byte-matched JPEG.

| metric | Nanograph | JPEG | NG win rate | mean margin |
|---|---|---|---|---|
| Recon-IoU *(vs NG's own mask)* | 0.4331 | 0.4243 | **557/726 (77%)** | +0.0089 |
| GT-IoU *(vs ground truth)* | 0.6543 | 0.6537 | 354/726 (49%) | +0.0007 |
| FG-PSNR *(vs NG's own mask)* | 28.377 | 28.836 | 33/726 (5%) | -0.459 dB |
| GT-FG-PSNR *(vs ground truth)* | 27.735 | 28.104 | 90/726 (12%) | -0.368 dB |
| GT-FG-SSIM *(vs ground truth)* | 0.9991 | 0.9992 | 50/726 (7%) | -0.0001 |
| Full PSNR | 30.744 | 31.245 | 0/726 (0%) | -0.501 dB |
| Full SSIM | 0.5639 | 0.6196 | 0/726 (0%) | -0.0557 |

Bytes: Nanograph 2469, JPEG 2404 — JPEG achieves the above using **2.6 % fewer bytes**.

## What this means

The abstract and contribution (2) claim Nanograph "recovers the biological foreground more faithfully than byte-matched JPEG on 77 % of images." That is true of exactly one metric, Recon-IoU, which is defined in Sec. 4.2 as IoU between an Otsu threshold of the reconstruction and **the pipeline's own selected segmentation mask**. It is the only comparison in the table where the target is chosen by the method under test, and the margin is +0.0089 IoU.

On every ground-truth-referenced fidelity metric, byte-matched JPEG is equal or better:

- **GT-IoU is a dead tie** — 0.6543 vs 0.6537, a 49 % win rate. This number is already printed in Table 1 (`354/726 (49%)`) and is never mentioned in the text.
- **GT-FG-PSNR favours JPEG on 88 % of images** (-0.37 dB mean).
- **FG-PSNR favours JPEG on 95 % of images** (-0.46 dB mean). Table 2 prints both values (28.38 vs 28.84) without noting the direction, while the abstract quotes 28.4 dB in a sentence about foreground faithfulness.

A referee who opens `metrics.csv` reaches the summary above in about ten minutes. The claim as written will not survive it.

## The stronger claim, fully supported by the same file

`jpeg_betti_preservation` averages **0.1499**. Byte-matched JPEG destroys roughly 85 % of the topology. Nanograph's topology is exact by construction, because it stores the graph.

So the defensible framing is *parity plus structure*, not victory:

> At matched bytes, Nanograph reaches fidelity parity with general-purpose codecs on the biological foreground (GT-IoU 0.654 vs 0.654; GT-FG-SSIM 0.999 vs 0.999) while additionally exposing morphometry and exact topology at no extra cost. The same codecs preserve 15 % of the topology and expose none of the morphometry.

This is a better paper-level argument than the win-rate framing. It concedes a fidelity tie the data already forces, and it moves the contribution onto the axis where the margin is not 0.009 but roughly 6x — which is also the axis the paper's own thesis is about. Recommend rewriting the abstract, contribution (2), Sec. 5 and Table 2's caption around it, and reporting GT-referenced metrics as primary with the self-referenced ones clearly labelled as self-consistency checks.

## Two related items

1. **The learned segmenter is never selected.** Sec. 8.1 reports the clDice U-Net lifting held-out IoU 0.381 -> 0.875, then states it wins the cascade competition for zero of 726 images. Every headline number in the paper therefore comes from the classical cascade at Seg-IoU 0.489, precision 0.493. Re-running the 726-image evaluation in `learned_mode='replace'` is the highest- value experiment available: it would likely move GT-IoU, GT-FG-PSNR and the win rates together, and it converts Sec. 8.1 from a limitation into a result. This should happen before submission.
2. **The refinement stage never fires** on any of 816 images (Sec. 3.5). The paper is admirably explicit about it, but it currently occupies a method subsection and 10 of the 121 config parameters while contributing nothing to any reported number. Either cut it to a paragraph in Future work, or run the reduced-node-budget sweep that makes it act.
