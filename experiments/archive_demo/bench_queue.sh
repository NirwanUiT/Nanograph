#!/usr/bin/env bash
# Follow-on to allen_bakeoff.sh (tmux): complete the untrained/trained matrix,
# evaluate every candidate on benchmarks 1-5, apply SELECTION_RULE.md.
set -uo pipefail
cd "$(dirname "$0")/../.."
A=/mnt/nas1/nba055-2/idea_1/archive_demo/allen
R=/mnt/nas1/nba055-2/idea_1/archive_demo/realbench
NN=/mnt/nas1/nba055-2/idea_1/archive_demo/nnunet
VN=/var/tmp/nba055/venvs/nnunet/bin
GPU=${GPU:-1}
export TMPDIR=/var/tmp/nba055/tmp nnUNet_raw=$NN/nnUNet_raw nnUNet_preprocessed=$NN/nnUNet_preprocessed \
       nnUNet_results=$NN/nnUNet_results nnUNet_compile=f nnUNet_n_proc_DA=8
say () { echo "### [$(date '+%F %T')] $*"; }

# nnU-Net prediction with per-image GPU time -> <out_dir>/timing.csv
nnpredict () {  # dataset_id in_dir tmp_out final_out label timing_csv
  mkdir -p "$(dirname "$3")" "$3"
  local t0=$(date +%s.%N)
  CUDA_VISIBLE_DEVICES=$GPU $VN/nnUNetv2_predict -i "$2" -o "$3" -d "$1" -c 2d -f 0 -tr nnUNetTrainer_250epochs \
    -p nnUNetPlans_6G > "$3.log" 2>&1
  local n=$(ls "$2" | wc -l); local t1=$(date +%s.%N)
  python experiments/archive_demo/allen_nnunet_prep.py --stage export --pred-in "$3" --pred-out "$4"
  [ -f "$6" ] || echo "dataset,id,seconds" > "$6"
  python -c "print(','.join(['$5','ALL_GPU',str(($t1-$t0)/max($n,1))]))" >> "$6"
}

say "waiting for the bake-off"
until grep -q BAKEOFF_DONE results/allen/bakeoff.log 2>/dev/null; do sleep 60; done

say "allen_ft on the real tiles (CPU)"
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 python experiments/archive_demo/real_bench.py --stage predict \
  --name allen_ft --ckpt results/allen/unet/allen_finetune_s0.pt

say "nnU-Net (Allen) on the real tiles (GPU $GPU)"
for d in UIT CBMI MITO HUMAN; do
  nnpredict 501 $NN/pred_in/real/$d $NN/out501_real/$d $R/preds/nnunet/$d $d $R/preds/nnunet/timing.csv
done

say "nnU-Net (real) train (GPU $GPU)"
CUDA_VISIBLE_DEVICES=$GPU $VN/nnUNetv2_train 502 2d 0 -tr nnUNetTrainer_250epochs -p nnUNetPlans_6G \
  > $NN/train502_fold0.log 2>&1
say "nnU-Net (real) predict"
for d in UIT CBMI MITO HUMAN; do
  nnpredict 502 $NN/pred_in/real/$d $NN/out502_real/$d $R/preds/nnunet_real/$d $d $R/preds/nnunet_real/timing.csv
done
nnpredict 502 $NN/test_in $NN/out502_allen $A/preds/nnunet_real ALLEN $A/preds/nnunet_real_timing.csv

say "evaluate benchmarks 1-4"
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 python experiments/archive_demo/real_bench.py --stage eval --workers 6
say "evaluate benchmark 5"
PRED=""
for n in nellie nellie_tuned_real nellie_tuned_allen unet_sim nnunet nnunet_real microsam_zs microsam_real microsam_ft; do
  [ -d $A/preds/$n ] && PRED="$PRED --pred $n=$A/preds/$n"
done
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 python experiments/archive_demo/allen_eval_seg.py $PRED --workers 6 \
  > results/allen/seg_eval.log 2>&1
python experiments/archive_demo/select_default.py
say "BENCH_DONE"
