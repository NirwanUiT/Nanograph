#!/usr/bin/env bash
# Wait for the T5 driver to finish, then run T6 (polarity) and T7 (mito diag).
# Kept separate from run_paper_t5.sh so GPU work never overlaps.
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=results/paper
log () { echo "=== [$(date +%H:%M:%S)] $*"; }

# --- wait for T5 ---
log "waiting for run_paper_t5.sh to finish"
while pgrep -f run_paper_t5.sh >/dev/null 2>&1; do sleep 30; done
log "T5 finished; starting T6"

# --- T6: polarity ---
python experiments/eval_retrained.py --csv "$OUT/polarity.csv" 2>&1 | tail -8
log "T6 done"

# --- T7: mito diagnostics ---
python experiments/mito_diagnostics.py --out "$OUT/mito_diag" 2>&1 | tail -12
log "T7 done"

log "T5-T7 ALL COMPLETE"
