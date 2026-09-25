#!/usr/bin/env bash
# Run inside tmux: waits until paper_v3_watch.sh has finished, then runs the
# round-3 jobs that need no human input (T15-A inter-observer, STARE; DRIVE
# too if DRIVE_TEST points at the registered DRIVE test set).
set -uo pipefail
cd "$(dirname "$0")/.."
say () { echo "### [$(date '+%F %T')] $*"; }
say "waiting for the paper-v3 watcher"
until grep -q "ALL WATCHED WORK DONE" results/paper/watch.log 2>/dev/null; do sleep 120; done
say "T15-A inter-observer (vessels)"
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 python experiments/interobserver_vessels.py --workers 8 \
  --out results/paper/annotator > results/paper/annotator_vessels.log 2>&1; say "T15-A exit $?"
say "QUEUE DONE"
