# Direction checks

numbers: `paper/numbers.tex` (commit f623678)

| check | status | claim | manuscript lines |
|---|---|---|---|
| `layer-size` | **PASS** | structure layer smaller than the lossless mask (abstract, layers section) | 89, 215 |
| `layer-topology` | **PASS** | stored components match the mask on >= 99 % and cycles on >= 90 % of images | 227, 229, 1174 |
| `ds-every` | **ERROR (KeyError)** | stored graph closer to the reference than JPEG on every descriptor (tab:downstream) | 453, 454, 455, 456, 457, 458 |
| `ds-components` | **PASS** | JPEG route fragments components more than the stored graph | 425, 426, 428, 453 |
| `ds-wasserstein` | **PASS** | branch-length distribution closer to the reference than JPEG | 430 |
| `cost` | **PASS** | query cost: structure layer < mask re-analysis < JPEG route | 90, 431, 432 |
| `gtiou-parity` | **PASS** | GT-IoU at parity with JPEG (95 % CI of the mean difference contains 0) | 959 |
| `gtiou-classical` | **PASS** | with the classical cascade JPEG is ahead on GT-IoU | - |
| `truth-jpeg-length` | **PASS** | JPEG length bias smaller than the graph's, but its concordance lower | 492, 493, 496 |
| `truth-width` | **PASS** | every route overstates width, the simulator mask included | 491, 492 |
| `truth-calibrated-ref` | **FAIL** | after calibration the graph is as close to truth as the simulator mask (within 2 points) | 500, 501, 502 |
| `truth-calibrated-jpeg` | **PASS** | after calibration the graph is closer to truth than JPEG on components, length, branches, junctions | 500, 501, 502 |
| `clip-rule` | **PASS** | the one-diameter rule reduces the clip's component overcount | 510, 511 |
| `real-seg` | **PASS** | real training raises in-pipeline Seg-IoU on all four datasets | 542, 543, 548, 549 |
| `real-human` | **PASS** | on EP-UiT-Human real training shrinks the branch bias and raises its CCC | 552, 553 |
| `real-q1` | **PASS** | the smallest JPEG is markedly worse than the structure layer (UiT, Human) | 566, 567 |
| `real-q20` | **PASS** | JPEG q20 reaches the structure layer on CBMI and MITO but falls short on UiT and Human | 566, 567, 571 |
| `lossless-smallest` | **FAIL** | T14: the structure layer is smaller than the best lossless alternative (per dataset) | - |

## Values

- `layer-size`: StructVsMaskX=2.1
- `layer-topology`: GraphCompEqPct=99.9, GraphCycEqPct=95.2
- `ds-every`: DsPWinsNComponents=239/716, DsPWinsTotalLength=450/716, DsPWinsMeanWidth=455/708, DsPWinsNBranches=327/716, DsPWinsNJunctions=232/716, DsPWinsCycleRank=142/716, DsPWinsPNComponents=1.4\times10^{-39}, DsPWinsPTotalLength=8.7\times10^{-25}, DsPWinsPMeanWidth=1.8\times10^{-17}, DsPWinsPNBranches=6.6\times10^{-23}, DsPWinsPNJunctions=1.8\times10^{-10}, DsPWinsPCycleRank=3.6\times10^{-10}
- `ds-components`: DsPJpegNComponentsBiasPct=+48.7, DsPGraphNComponentsBiasPct=-2.0, DsPJpegNComponentsCCC=-0.009, DsPGraphNComponentsCCC=0.749
- `ds-wasserstein`: DsPWassGraph=5.64, DsPWassJpeg=7.36
- `cost`: DsTimeGraph=1.3, DsTimeRef=4.9, DsTimeJpeg=266
- `gtiou-parity`: DGTIoUWinsCI=[-0.0050,+0.0023]
- `gtiou-classical`: CGTIoUWinsPct=6.0
- `truth-jpeg-length`: TruthJpegLenBias=-2, TruthGraphLenBias=-30, TruthJpegLenCCC=0.01, TruthGraphLenCCC=0.32
- `truth-width`: TruthRefWidthBias=+53, TruthGraphWidthBias=+67, TruthJpegWidthBias=+65
- `truth-calibrated-ref`: CalGraphComp=12.0, CalGraphLen=11.1, CalGraphWidth=13.2, CalGraphBr=27.5, CalGraphJunc=41.4, CalGraphCyc=36.4, CalRefComp=12.8, CalRefLen=10.6, CalRefWidth=10.7, CalRefBr=28.0, CalRefJunc=39.1, CalRefCyc=42.1
- `truth-calibrated-jpeg`: CalGraphComp=12.0, CalGraphLen=11.1, CalGraphBr=27.5, CalGraphJunc=41.4, CalJpegComp=12.7, CalJpegLen=14.6, CalJpegBr=31.3, CalJpegJunc=45.8
- `clip-rule`: ClipTruthCompZero=+58, ClipTruthCompAuto=+10
- `real-seg`: RealSegIoUSimUit=0.55, RealSegIoUSimCbmi=0.49, RealSegIoUSimMito=0.09, RealSegIoUSimHuman=0.37, RealSegIoURealUit=0.78, RealSegIoURealCbmi=0.58, RealSegIoURealMito=0.31, RealSegIoURealHuman=0.41
- `real-human`: RealHumanBrBiasSim=+91, RealHumanBrBiasReal=+8, RealHumanBrCCCSim=0.49, RealHumanBrCCCReal=0.88
- `real-q1`: RealJpegQoneBrCCCUit=0.73, RealBrCCCUit=0.94, RealJpegQoneBrCCCHuman=0.35, RealBrCCCHuman=0.81
- `real-q20`: RealJpegQtwentyBrCCCUit=0.92, RealJpegQtwentyBrCCCCbmi=0.32, RealJpegQtwentyBrCCCMito=0.38, RealJpegQtwentyBrCCCHuman=0.78, RealBrCCCUit=0.94, RealBrCCCCbmi=0.27, RealBrCCCMito=0.37, RealBrCCCHuman=0.81
- `lossless-smallest`: LLRatioOrganelle=0.72, LLRatioUit=1.11, LLRatioCbmi=1.19, LLRatioMito=1.24, LLRatioHuman=1.07
