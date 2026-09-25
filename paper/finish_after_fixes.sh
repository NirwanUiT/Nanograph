#!/usr/bin/env bash
# tmux: after the two in-run crashes were fixed and re-run (perturbation study,
# real segmenter evaluation), and after T14 / T15-A: build everything again.
set -uo pipefail
cd "$(dirname "$0")/.."
S=results/paper/commit.txt
fresh () { [ -e "$1" ] && [ "$1" -nt "$S" ]; }
until fresh results/paper/seg_perturbation_ablation.csv && fresh results/paper/real/segmenters/eval_summary.csv \
      && ! pgrep -f run_paper_v3_resume.sh > /dev/null; do sleep 60; done
until grep -q "ALL WATCHED WORK DONE" results/paper/watch.log && grep -q "QUEUE DONE" results/paper/queue.log; do sleep 60; done
echo "### [$(date '+%F %T')] all inputs fresh; building"
bash paper/finish_v3.sh > results/paper/finish_v3.log 2>&1; echo "finish_v3 exit $?"
tail -30 results/paper/finish_v3.log
echo FINISH_AFTER_FIXES_DONE
