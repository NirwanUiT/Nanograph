#!/usr/bin/env bash
# T18 segmenter bake-off (run in tmux). CPU steps start when the Allen tiles
# exist; GPU steps wait for the paper-v3 GPU phase (GPU 1 is the only GPU with
# headroom) and run one at a time.
#   1 export test-cell inputs            (CPU)
#   2 nnU-Net data + plan + preprocess   (CPU, nnunet venv)
#   3 Nellie zero-shot on the test cells (CPU, nellie venv)
#   4 our clDice U-Net fine-tuned on Allen tiles  (GPU)
#   5 nnU-Net 2-D, fold 0 = our split, 250 epochs; predict the test cells (GPU)
#   6 evaluation of every segmenter      (CPU)
set -uo pipefail
cd "$(dirname "$0")/../.."
T=/mnt/nas1/nba055-2/idea_1/archive_demo/allen
NN=/mnt/nas1/nba055-2/idea_1/archive_demo/nnunet
VN=/var/tmp/nba055/venvs/nnunet/bin
VL=/var/tmp/nba055/venvs/nellie/bin
GPU=${GPU:-1}
export TMPDIR=/var/tmp/nba055/tmp nnUNet_raw=$NN/nnUNet_raw nnUNet_preprocessed=$NN/nnUNet_preprocessed \
       nnUNet_results=$NN/nnUNet_results nnUNet_compile=f nnUNet_n_proc_DA=8
say () { echo "### [$(date '+%F %T')] $*"; }
mkdir -p results/allen/unet "$NN"

say "waiting for the Allen training tiles"
until grep -q TRAINSET_END $T/trainset.log 2>/dev/null; do sleep 60; done
say "1 export test cells"
CUDA_VISIBLE_DEVICES= python experiments/archive_demo/allen_export_test.py
say "2 nnU-Net data + plan"
python experiments/archive_demo/allen_nnunet_prep.py --stage data
$VN/nnUNetv2_plan_and_preprocess -d 501 -c 2d -np 8 --verify_dataset_integrity \
  -gpu_memory_target 6 -overwrite_plans_name nnUNetPlans_6G
python experiments/archive_demo/allen_nnunet_prep.py --stage split
say "3 Nellie"
$VL/python experiments/archive_demo/allen_nellie.py

say "waiting for the paper-v3 GPU phase"
until grep -qE '\] (ablation_bg_residual|ALL PAPER-V3 RUNS DONE)$' results/paper/v3_run.log 2>/dev/null; do sleep 60; done
say "4 U-Net fine-tune (GPU $GPU)"
CUDA_VISIBLE_DEVICES=$GPU REAL_ROOT=$T/trainset python experiments/train_real.py --train-sets ALLEN \
  --strategy finetune --init weights/unet_mito_real.pt --epochs 40 --bs 8 --seed 0 \
  --out results/allen/unet/allen_finetune_s0.pt > results/allen/unet/allen_finetune_s0.log 2>&1
say "5 nnU-Net train + predict (GPU $GPU)"
CUDA_VISIBLE_DEVICES=$GPU $VN/nnUNetv2_train 501 2d 0 -tr nnUNetTrainer_250epochs -p nnUNetPlans_6G \
  > $NN/train_fold0.log 2>&1
CUDA_VISIBLE_DEVICES=$GPU $VN/nnUNetv2_predict -i $NN/test_in -o $NN/test_out -d 501 -c 2d -f 0 \
  -tr nnUNetTrainer_250epochs -p nnUNetPlans_6G > $NN/predict.log 2>&1
python experiments/archive_demo/allen_nnunet_prep.py --stage export --pred-in $NN/test_out --pred-out $T/preds/nnunet
say "6 evaluation"
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 python experiments/archive_demo/allen_eval_seg.py \
  --pred nellie=$T/preds/nellie --pred nnunet=$T/preds/nnunet > results/allen/seg_eval.log 2>&1
say "BAKEOFF_DONE"
