#!/usr/bin/env bash
# tmux: after paper_v3_watch.sh (finish_v3 + T14) and queue_after_v3.sh (T15-A),
# rebuild numbers/figures/supplement/direction checks so T14/T15 outputs are included.
set -uo pipefail
cd "$(dirname "$0")/.."
until grep -q "ALL WATCHED WORK DONE" results/paper/watch.log 2>/dev/null && \
      grep -q "QUEUE DONE" results/paper/queue.log 2>/dev/null; do sleep 120; done
grep -q "refusing to build numbers" results/paper/finish_v3.log && { echo "finish_v3 refused (stale inputs): not rebuilding"; exit 1; }
python paper/make_numbers.py --runs results/paper --out paper
python paper/make_figures.py --runs results/paper --out paper/figures
python paper/make_supplement.py --runs results/paper --out paper
python paper/check_directions.py --numbers paper/numbers.tex --tex paper/nanograph_main.tex --out results/paper/DIRECTIONS.md
echo "FINAL_NUMBERS_DONE"
