#!/usr/bin/env bash
# T11.1: re-run every T5 output at the current commit (tag paper-v2).
# Same commands as run_paper_t5.sh, grouped into independent lanes that run
# concurrently (total_time_ms columns are therefore measured under load).
set -uo pipefail
cd "$(dirname "$0")/.."          # repo root (Nanograph/Nanograph)

ROOT=/mnt/nas1/nba055-2/idea_1
ORG=$ROOT/nmi_data/org
SEG=$ROOT/nmi_data/seg
OUT=results/paper
HASH=$(git rev-parse HEAD)
mkdir -p "$OUT"
echo "$HASH" > "$OUT/commit.txt"

stamp () { echo "$HASH" > "$1/commit.txt"; }
log () { echo "=== [$(date +%H:%M:%S)] $*"; }
rd () { local out=$1; shift; log "$out"; python run_dataset.py "$@" --outdir "$OUT/$out" && stamp "$OUT/$out"; }

lane_sted () {
  rd mito/sted --images "$ROOT/public_mito/sted_prep/images" --preset learned-replace --n-samples 4
}
lane_org_a () {
  [ "${SKIP_ORG_DEFAULT:-0}" = 1 ] || rd org_default --images "$ORG" --masks "$SEG" --n-samples 6
  rd org_classical --images "$ORG" --masks "$SEG" --preset classical --n-samples 6
}
lane_org_b () {
  rd org_replace --images "$ORG" --masks "$SEG" --preset learned-replace --n-samples 6
  if [ "${SKIP_ABLATIONS:-0}" != 1 ]; then
    log ablation_bg_residual
    python experiments/ablation_bg_residual.py --images "$ORG" --out "$OUT/ablation_bg_residual.csv"
  fi
}
lane_perturb_cross () {
  if [ "${SKIP_ABLATIONS:-0}" != 1 ]; then
    log seg_perturbation_ablation
    python experiments/seg_perturbation_ablation.py --images "$ORG" --masks "$SEG" \
      --out "$OUT/seg_perturbation_ablation.csv"
  fi
  for ds in cells3d_membrane cells3d_nuclei retina cell; do
    rd cross/$ds --images "$ROOT/datasets/$ds/images" --n-samples 4
  done
}
lane_crossgt_mito () {
  for ds in stare drive epfl_mito microtubules; do
    rd cross_gt/$ds --images "$ROOT/ext_datasets/prepared/$ds/images" \
      --masks "$ROOT/ext_datasets/prepared/$ds/labels" --preset curvilinear --n-samples 4
  done
  rd mito/temporal_clip --images "$ROOT/mito_aaron/images" --masks "$ROOT/mito_aaron/masks" --n-samples 4
  rd mito/temporal_clip_replace --images "$ROOT/mito_aaron/images" --masks "$ROOT/mito_aaron/masks" \
    --preset learned-replace --n-samples 4
  rd mito/mito_mip --images "$ROOT/public_mito/mito_mip_tiles/images" \
    --masks "$ROOT/public_mito/mito_mip_tiles/masks" --n-samples 4
  log heldout
  python experiments/write_heldout.py
}

mkdir -p "$OUT/t11_logs"
if [ "${SEQUENTIAL:-0}" = 1 ]; then
  # Timing re-run: one process at a time, no ablation/perturbation (no timing
  # macros; their outputs were bit-identical in the concurrent run).
  export SKIP_ABLATIONS=1
  for lane in ${LANES:-lane_org_a lane_org_b lane_perturb_cross lane_crossgt_mito lane_sted}; do
    $lane > "$OUT/t11_logs/seq_$lane.log" 2>&1
  done
else
  for lane in lane_sted lane_org_a lane_org_b lane_perturb_cross lane_crossgt_mito; do
    $lane > "$OUT/t11_logs/$lane.log" 2>&1 &
  done
  wait
fi
log "ALL T11.1 RUNS DONE"
