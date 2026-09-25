# Default segmenter: selection (SELECTION_RULE.md applied mechanically)

**Status: PROVISIONAL**

- nellie_tuned_real: tuning kept the default setting, so it is the same segmenter as nellie (merged)
- benchmark 6 (blinded preference): not run yet
- candidates not evaluated yet: microsam_real, microsam_ft

## clDice per benchmark (* = in-domain for that candidate)

| method | CBMI | HUMAN | MITO | UIT | ALLEN | mean rank | s/tile |
|---|---|---|---|---|---|---|---|
| nnunet | 0.909 (#1) | 0.760 (#2) | 0.284 (#5) | 0.801 (#6) | 0.910* (#1) | 3.00 | 0.41 |
| nnunet_real | 0.893* (#3) | 0.633 (#6) | 0.371* (#2) | 0.967* (#1) | 0.820 (#6) | 3.60 | 0.42 |
| nellie | 0.873 (#4) | 0.670 (#5) | 0.294 (#4) | 0.867 (#3) | 0.875 (#3) | 3.80 | 0.08 |
| allen_ft | 0.915 (#2) | 0.771 (#1) | 0.271 (#7) | 0.788 (#7) | 0.906* (#2) | 3.80 | 0.26 |
| real_mito | 0.819* (#8) | 0.679 (#3) | 0.507* (#1) | 0.922* (#2) | 0.752 (#8) | 4.40 | 0.28 |
| nellie_tuned_allen | 0.871 (#5) | 0.684 (#4) | 0.281 (#6) | 0.865 (#4) | 0.876* (#4) | 4.60 | 0.08 |
| unet_sim | 0.863 (#6) | 0.601 (#7) | 0.311 (#3) | 0.867 (#5) | 0.853 (#5) | 5.20 | 0.26 |
| microsam_zs | 0.836 (#7) | 0.456 (#8) | 0.060 (#8) | 0.639 (#8) | 0.774 (#7) | 7.60 | 16.41 |

Order: mean rank, then seconds per tile (SELECTION_RULE.md tie-break).

Leader: **nnunet** (mean rank 3.00) - provisional
