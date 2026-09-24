#!/usr/bin/env bash
# After experiments/run_paper_v3.sh: the real-data JPEG floor, then every
# number, table and figure of the paper from results/paper.
set -euo pipefail
cd "$(dirname "$0")/.."          # repo root (Nanograph/Nanograph)
grep -q "ALL PAPER-V3 RUNS DONE" results/paper/v3_run.log || { echo "paper-v3 run not finished"; exit 1; }

CUDA_VISIBLE_DEVICES= OUT_DIR=results/paper/real/downstream python experiments/real_jpeg_floor.py
cp results/paper/commit.txt results/paper/real/downstream/commit.txt
python paper/make_numbers.py --runs results/paper --out paper
python paper/make_figures.py --runs results/paper --out paper/figures
OUT_PATH=paper/figures/pipeline_overview.png python experiments/render_pipeline_diagram.py
python paper/make_fig_pipeline.py
python experiments/render_paper_panels.py --out paper/figures
python experiments/check_claims.py --runs results/paper --paper paper
echo "TBD macros used by the manuscript:"
python - <<'PY'
import re
t = open('paper/nanograph_main.tex').read()
n = open('paper/numbers.tex').read()
tbd = {m for m in re.findall(r'\\newcommand\{\\(\w+)\}\{\\tbd\}', n)}
used = sorted(m for m in tbd if re.search(r'\\' + m + r'(?![A-Za-z])', t))
print(len(used), used)
PY
