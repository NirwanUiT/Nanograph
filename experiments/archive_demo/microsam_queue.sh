#!/usr/bin/env bash
# tmux: micro-SAM fine-tunes (allen, real) when a GPU has >= NEED MiB free;
# predictions on every benchmark; then re-evaluate and re-select.
set -uo pipefail
cd "$(dirname "$0")/../.."
A=/mnt/nas1/nba055-2/idea_1/archive_demo/allen
R=/mnt/nas1/nba055-2/idea_1/archive_demo/realbench
MS=/mnt/nas1/nba055-2/idea_1/archive_demo/microsam
PY=/home/nba055/micromamba/envs/microsam/bin/python
NEED=${NEED:-20000}
export MICROSAM_CACHEDIR=/mnt/nas1/nba055-2/.microsam_cache TMPDIR=/var/tmp/nba055/tmp
say () { echo "### [$(date '+%F %T')] $*"; }
free_gpu () {   # GPU index with the most free memory if >= NEED, else empty
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits |
    sort -t, -k2 -nr | awk -F', ' -v n=$NEED 'NR==1 && $2>=n {print $1}'
}
for pool in allen real; do
  say "waiting for a GPU with >= $NEED MiB free ($pool)"
  until g=$(free_gpu) && [ -n "$g" ]; do sleep 300; done
  say "training microsam_$pool on GPU $g"
  CUDA_VISIBLE_DEVICES=$g $PY experiments/archive_demo/microsam_train.py --pool $pool > $MS/train_$pool.log 2>&1
  say "exit $?"
done
for spec in "microsam_ft:allen" "microsam_real:real"; do
  n=${spec%%:*}; pool=${spec#*:}; ck=$MS/checkpoints/microsam_$pool/best.pt
  [ -f $ck ] || { say "no checkpoint for $n"; continue; }
  until g=$(NEED=8000 free_gpu) && [ -n "$g" ]; do sleep 120; done
  for d in UIT CBMI MITO HUMAN; do
    CUDA_VISIBLE_DEVICES=$g $PY experiments/archive_demo/microsam_predict.py --img-dir $R/$d/img_raw --out $R/preds/$n/$d \
      --model vit_b --checkpoint $ck --device cuda --workers 1 --timing $R/preds/$n/timing.csv --label $d
  done
  CUDA_VISIBLE_DEVICES=$g $PY experiments/archive_demo/microsam_predict.py --img-dir $A/testset/img_raw --cell-dir $A/testset/cell \
    --out $A/preds/$n --model vit_b --checkpoint $ck --device cuda --workers 1
done
say "waiting for bench_queue to finish its evaluations"
until grep -q BENCH_DONE results/allen/bench_queue.log 2>/dev/null; do sleep 120; done
say "re-evaluate with micro-SAM"
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 python experiments/archive_demo/real_bench.py --stage eval --workers 6
PRED=""
for n in nellie nellie_tuned_real nellie_tuned_allen unet_sim nnunet nnunet_real microsam_zs microsam_real microsam_ft; do
  [ -d $A/preds/$n ] && PRED="$PRED --pred $n=$A/preds/$n"
done
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 python experiments/archive_demo/allen_eval_seg.py $PRED --workers 6 > results/allen/seg_eval.log 2>&1
python experiments/archive_demo/select_default.py
say "MICROSAM_DONE"
