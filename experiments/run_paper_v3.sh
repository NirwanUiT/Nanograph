#!/usr/bin/env bash
# paper-v3: every paper output with the v7 method (graph.builder = 'branch',
# layered payload, one-diameter analysis rule), at one tagged commit.
#   phase 1  every run_dataset output, one process at a time (timing columns)
#   phase 2  downstream analysis (encode cache on GPU; measure single-threaded, timed)
#   phase 3  ablations, truth evaluations, real-data evaluations (no timing)
# Outputs overwrite results/paper/ (the paper-v1/v2 outputs stay in git history).
set -uo pipefail
cd "$(dirname "$0")/.."          # repo root (Nanograph/Nanograph)

ROOT=/mnt/nas1/nba055-2/idea_1
ORG=$ROOT/nmi_data/org
SEG=$ROOT/nmi_data/seg
REAL=$ROOT/real_mito/test_sets
OUT=results/paper
HASH=$(git rev-parse HEAD)
mkdir -p "$OUT"
echo "$HASH" > "$OUT/commit.txt"

stamp () { echo "$HASH" > "$1/commit.txt"; }
log () { echo "=== [$(date +%H:%M:%S)] $*"; }
rd () { local out=$1; shift; log "$out"; python run_dataset.py "$@" --outdir "$OUT/$out" && stamp "$OUT/$out"; }

# ---- phase 1: dataset runs (sequential) ------------------------------------
if [ "${SKIP_PHASE1:-0}" != 1 ]; then
rd org_default   --images "$ORG" --masks "$SEG" --n-samples 6
rd org_classical --images "$ORG" --masks "$SEG" --preset classical --n-samples 6
rd org_replace   --images "$ORG" --masks "$SEG" --preset learned-replace --n-samples 6
for ds in cells3d_membrane cells3d_nuclei retina cell; do
  rd cross/$ds --images "$ROOT/datasets/$ds/images" --n-samples 4
done
for ds in stare drive epfl_mito microtubules; do
  rd cross_gt/$ds --images "$ROOT/ext_datasets/prepared/$ds/images" \
    --masks "$ROOT/ext_datasets/prepared/$ds/labels" --preset curvilinear --n-samples 4
done
rd mito/temporal_clip         --images "$ROOT/mito_aaron/images" --masks "$ROOT/mito_aaron/masks" --n-samples 4
rd mito/temporal_clip_replace --images "$ROOT/mito_aaron/images" --masks "$ROOT/mito_aaron/masks" \
  --preset learned-replace --n-samples 4
rd mito/mito_mip --images "$ROOT/public_mito/mito_mip_tiles/images" \
  --masks "$ROOT/public_mito/mito_mip_tiles/masks" --n-samples 4
for d in UIT CBMI MITO HUMAN; do
  rd real/default/$d   --images "$REAL/$d/images" --masks "$REAL/$d/masks" --n-samples 2
  rd real/real-mito/$d --images "$REAL/$d/images" --masks "$REAL/$d/masks" --preset real-mito --n-samples 2
done
rd mito/sted --images "$ROOT/public_mito/sted_prep/images" --preset learned-replace --n-samples 4
fi

# ---- phase 2: downstream (T10/T11.2 protocol, v7 payloads) ------------------
log downstream
rm -rf "$OUT/downstream/cache" "$OUT/downstream/branch_lengths"
CUDA_VISIBLE_DEVICES=${DS_GPU:-1} OMP_NUM_THREADS=1 python experiments/downstream_morphometry.py \
  --stage encode --workers 4 --out "$OUT/downstream"
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python experiments/downstream_morphometry.py --stage measure --out "$OUT/downstream"
python experiments/downstream_morphometry.py --stage stats --out "$OUT/downstream"
stamp "$OUT/downstream"

# ---- phase 3: ablations and evaluations --------------------------------------
log ablation_bg_residual
python experiments/ablation_bg_residual.py --images "$ORG" --out "$OUT/ablation_bg_residual.csv"
log seg_perturbation_ablation
python experiments/seg_perturbation_ablation.py --images "$ORG" --masks "$SEG" \
  --out "$OUT/seg_perturbation_ablation.csv"
log heldout
python experiments/write_heldout.py
log sim_truth
CUDA_VISIBLE_DEVICES= python experiments/sim_truth_eval.py --dataset sim --workers 12 --out "$OUT/sim_truth"
stamp "$OUT/sim_truth"
log clip_truth
CUDA_VISIBLE_DEVICES= python experiments/sim_truth_eval.py --dataset clip --workers 12 --out "$OUT/clip_truth"
stamp "$OUT/clip_truth"
log real_downstream
CUDA_VISIBLE_DEVICES= WORKERS=12 OUT_DIR="$OUT/real/downstream" python experiments/real_downstream.py
stamp "$OUT/real/downstream"
log real_segmenters
CUDA_VISIBLE_DEVICES= WORKERS=12 OUT_DIR="$OUT/real/segmenters" python experiments/eval_real.py
stamp "$OUT/real/segmenters"

log "ALL PAPER-V3 RUNS DONE"
