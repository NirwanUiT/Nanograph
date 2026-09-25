#!/usr/bin/env bash
# tmux: wait for the Allen training tiles, then fine-tune the real-mito U-Net on them.
set -uo pipefail
cd "$(dirname "$0")/../.."
T=/mnt/nas1/nba055-2/idea_1/archive_demo/allen
until grep -q TRAINSET_END $T/trainset.log 2>/dev/null; do sleep 60; done
# the paper-v3 downstream encode also uses GPU 1: wait until its GPU phase is over
until grep -qE '\] (ablation_bg_residual|ALL PAPER-V3 RUNS DONE)$' results/paper/v3_run.log 2>/dev/null; do sleep 60; done
echo "### [$(date '+%F %T')] training (GPU ${GPU:-1})"
CUDA_VISIBLE_DEVICES=${GPU:-1} REAL_ROOT=$T/trainset python experiments/train_real.py --train-sets ALLEN \
  --strategy finetune --init weights/unet_mito_real.pt --epochs 40 --bs 8 --seed 0 \
  --out results/allen/unet/allen_finetune_s0.pt 2>&1 | tee results/allen/unet/allen_finetune_s0.log
echo "### [$(date '+%F %T')] TRAIN_END"
