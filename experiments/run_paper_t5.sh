#!/usr/bin/env bash
# T5: produce every paper result at the paper-v1 commit into results/paper/.
# No SAM anywhere. Each run_dataset output dir gets a commit.txt.
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

# --- organelle: default / classical / replace ------------------------------
log "org_default"
python run_dataset.py --images "$ORG" --masks "$SEG" --outdir "$OUT/org_default" \
  --n-samples 6 && stamp "$OUT/org_default"
log "org_classical"
python run_dataset.py --images "$ORG" --masks "$SEG" --outdir "$OUT/org_classical" \
  --preset classical --n-samples 6 && stamp "$OUT/org_classical"
log "org_replace"
python run_dataset.py --images "$ORG" --masks "$SEG" --outdir "$OUT/org_replace" \
  --preset learned-replace --n-samples 6 && stamp "$OUT/org_replace"

# --- ablations -------------------------------------------------------------
log "ablation_bg_residual"
python experiments/ablation_bg_residual.py --images "$ORG" \
  --out "$OUT/ablation_bg_residual.csv"
log "seg_perturbation_ablation"
python experiments/seg_perturbation_ablation.py --images "$ORG" --masks "$SEG" \
  --out "$OUT/seg_perturbation_ablation.csv"

# --- cross-modality (no GT) ------------------------------------------------
for ds in cells3d_membrane cells3d_nuclei retina cell; do
  log "cross/$ds"
  python run_dataset.py --images "$ROOT/datasets/$ds/images" \
    --outdir "$OUT/cross/$ds" --n-samples 4 && stamp "$OUT/cross/$ds"
done

# --- cross-GT (curvilinear preset, with masks) -----------------------------
for ds in stare drive epfl_mito microtubules; do
  log "cross_gt/$ds"
  python run_dataset.py --images "$ROOT/ext_datasets/prepared/$ds/images" \
    --masks "$ROOT/ext_datasets/prepared/$ds/labels" \
    --preset curvilinear --outdir "$OUT/cross_gt/$ds" --n-samples 4 \
    && stamp "$OUT/cross_gt/$ds"
done

# --- mitochondrial generalisation ------------------------------------------
log "mito/temporal_clip"
python run_dataset.py --images "$ROOT/mito_aaron/images" --masks "$ROOT/mito_aaron/masks" \
  --outdir "$OUT/mito/temporal_clip" --n-samples 4 && stamp "$OUT/mito/temporal_clip"
log "mito/temporal_clip_replace"
python run_dataset.py --images "$ROOT/mito_aaron/images" --masks "$ROOT/mito_aaron/masks" \
  --preset learned-replace --outdir "$OUT/mito/temporal_clip_replace" --n-samples 4 \
  && stamp "$OUT/mito/temporal_clip_replace"
log "mito/sted"
python run_dataset.py --images "$ROOT/public_mito/sted_prep/images" \
  --preset learned-replace --outdir "$OUT/mito/sted" --n-samples 4 && stamp "$OUT/mito/sted"
log "mito/mito_mip"
python run_dataset.py --images "$ROOT/public_mito/mito_mip_tiles/images" \
  --masks "$ROOT/public_mito/mito_mip_tiles/masks" \
  --outdir "$OUT/mito/mito_mip" --n-samples 4 && stamp "$OUT/mito/mito_mip"

# --- held-out list ---------------------------------------------------------
log "heldout"
python experiments/write_heldout.py

log "ALL T5 RUNS DONE"
