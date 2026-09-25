#!/usr/bin/env bash
# Run inside tmux. Waits for the paper-v3 driver (PID $1) to exit; if it died
# before finishing, resumes it from the last step it had started; re-runs on
# CPU every step that lost images to CUDA out-of-memory (the GPUs are shared);
# then paper/finish_v3.sh and the T14 lossless baselines.
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=results/paper/v3_run.log
PID=${1:-}
say () { echo "### [$(date '+%F %T')] $*"; }

if [ -n "$PID" ]; then
  say "waiting for driver PID $PID"
  while kill -0 "$PID" 2>/dev/null; do sleep 60; done
fi
if ! grep -q "ALL PAPER-V3 RUNS DONE" "$LOG"; then
  last=$(grep -E '^=== \[[0-9:]+\] ' "$LOG" | grep -v ' skip \| ALL ' | tail -1 | sed -E 's/^=== \[[0-9:]+\] //')
  say "driver ended early; resuming from '$last'"
  FROM="$last" bash experiments/run_paper_v3_resume.sh >> "$LOG" 2>&1
fi

# steps whose section of the log shows CUDA OOM -> re-run on CPU
python - "$LOG" > results/paper/oom_steps.txt <<'PY'
import re, sys
step, bad = None, []
for ln in open(sys.argv[1], errors='replace'):
    m = re.match(r'^=== \[[0-9:]+\] (\S+)$', ln.rstrip())
    if m:
        step = m.group(1)
    elif 'out of memory' in ln and step and step not in bad:
        bad.append(step)
print('\n'.join(bad))
PY
for s in $(cat results/paper/oom_steps.txt); do
  say "re-running $s on CPU (CUDA OOM in the first pass)"
  echo "=== CPU re-run of $s (CUDA OOM in the first pass)" >> "$LOG"
  CUDA_VISIBLE_DEVICES= DS_GPU= ONLY="$s" bash experiments/run_paper_v3_resume.sh >> "$LOG" 2>&1
done

say "finish_v3"
bash paper/finish_v3.sh > results/paper/finish_v3.log 2>&1; say "finish_v3 exit $?"
say "T14 lossless baselines"
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= python experiments/lossless_baselines.py --workers 8 \
  --out results/paper/lossless > results/paper/lossless.log 2>&1; say "lossless exit $?"
say "ALL WATCHED WORK DONE"
