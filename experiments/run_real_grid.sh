#!/usr/bin/env bash
# T13 real-data training grid: 4 training sets x 3 strategies x 3 seeds, two GPU queues.
cd "$(dirname "$0")/.."
OUT=results/real/unet; mkdir -p $OUT/logs
INIT=results/v7/unet/unet_clean_base.pt     # simulation-trained, held-out images excluded
jobs=()
for cfg in ALL:UIT,CBMI,MITO LOO_UIT:CBMI,MITO LOO_CBMI:UIT,MITO LOO_MITO:UIT,CBMI; do
  name=${cfg%%:*}; sets=${cfg#*:}
  for st in finetune scratch joint; do for s in 0 1 2; do
    jobs+=("--train-sets $sets --strategy $st --seed $s --init $INIT --out $OUT/${name}_${st}_s$s.pt|${name}_${st}_s$s")
  done; done
done
queue () { gpu=$1; shift; for j in "$@"; do args=${j%%|*}; tag=${j##*|}
  [ -f $OUT/$tag.done ] && continue
  CUDA_VISIBLE_DEVICES=$gpu python experiments/train_real.py $args > $OUT/logs/$tag.log 2>&1 && touch $OUT/$tag.done
done; }
a=(); b=(); for i in "${!jobs[@]}"; do if (( i % 2 )); then b+=("${jobs[$i]}"); else a+=("${jobs[$i]}"); fi; done
queue 1 "${a[@]}" & queue 3 "${b[@]}" & wait
echo ALL_DONE
