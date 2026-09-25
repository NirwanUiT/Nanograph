# Default segmenter: selection (SELECTION_RULE.md applied mechanically)

**Status: PROVISIONAL**

- nellie_tuned_real: tuning kept the default setting, so it is the same segmenter as nellie (merged)
- benchmark 6 (blinded preference): not run yet
- candidates not evaluated yet: allen_ft, nnunet_real, nnunet, microsam_real, microsam_ft

## clDice per benchmark (* = in-domain for that candidate)

| method | CBMI | HUMAN | MITO | UIT | ALLEN | mean rank | s/tile |
|---|---|---|---|---|---|---|---|
| nellie | 0.873 (#1) | 0.670 (#3) | 0.294 (#3) | 0.867 (#2) | 0.875 (#1) | 2.00 | 0.08 |
| nellie_tuned_allen | 0.871 (#2) | 0.684 (#2) | 0.281 (#4) | 0.865 (#3) | 0.876* (#2) | 2.60 | 0.08 |
| real_mito | 0.819* (#5) | 0.679 (#1) | 0.507* (#1) | 0.922* (#1) | 0.752 (#5) | 2.60 | 0.28 |
| unet_sim | 0.863 (#3) | 0.601 (#4) | 0.311 (#2) | 0.867 (#4) | 0.853 (#3) | 3.20 | 0.26 |
| microsam_zs | 0.836 (#4) | 0.456 (#5) | 0.060 (#5) | 0.639 (#5) | 0.774 (#4) | 4.60 | 16.41 |

Order: mean rank, then seconds per tile (SELECTION_RULE.md tie-break).

Leader: **nellie** (mean rank 2.00) - provisional
