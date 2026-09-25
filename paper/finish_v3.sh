#!/usr/bin/env bash
# After experiments/run_paper_v3.sh: the real-data JPEG floor, then every
# number, table and figure of the paper from results/paper.
set -euo pipefail
cd "$(dirname "$0")/.."          # repo root (Nanograph/Nanograph)
grep -q "ALL PAPER-V3 RUNS DONE" results/paper/v3_run.log || { echo "paper-v3 run not finished"; exit 1; }

# every paper input must have been written by this run (a step that crashes leaves the old file behind)
START=results/paper/commit.txt          # written when the run started
stale=0
for f in org_default/metrics.csv org_classical/metrics.csv org_replace/metrics.csv \
         cross/cells3d_membrane/metrics.csv cross/cells3d_nuclei/metrics.csv cross/retina/metrics.csv cross/cell/metrics.csv \
         cross_gt/stare/metrics.csv cross_gt/drive/metrics.csv cross_gt/epfl_mito/metrics.csv cross_gt/microtubules/metrics.csv \
         mito/temporal_clip/metrics.csv mito/temporal_clip_replace/metrics.csv mito/mito_mip/metrics.csv mito/sted/metrics.csv \
         real/default/UIT/metrics.csv real/real-mito/UIT/metrics.csv real/default/CBMI/metrics.csv real/real-mito/CBMI/metrics.csv \
         real/default/MITO/metrics.csv real/real-mito/MITO/metrics.csv real/default/HUMAN/metrics.csv real/real-mito/HUMAN/metrics.csv \
         downstream/per_image.csv downstream/summary.csv downstream/cost_summary.csv downstream/branch_distances.csv \
         downstream/spur_fractions.csv ablation_bg_residual.csv seg_perturbation_ablation.csv \
         sim_truth/summary.csv sim_truth/calibrated.csv clip_truth/summary.csv \
         real/downstream/summary.csv real/segmenters/eval_summary.csv; do
  p=results/paper/$f; [ -e "$p" ] || p="$p.gz"
  if [ ! -e "$p" ] || [ "$p" -ot "$START" ]; then echo "STALE OR MISSING: results/paper/$f"; stale=1; fi
done
[ $stale = 0 ] || { echo "refusing to build numbers from stale inputs; see above"; exit 2; }

CUDA_VISIBLE_DEVICES= OUT_DIR=results/paper/real/downstream python experiments/real_jpeg_floor.py
cp results/paper/commit.txt results/paper/real/downstream/commit.txt
python paper/make_numbers.py --runs results/paper --out paper
python paper/make_figures.py --runs results/paper --out paper/figures
OUT_PATH=paper/figures/pipeline_overview.png python experiments/render_pipeline_diagram.py
python paper/make_fig_pipeline.py
python experiments/render_paper_panels.py --out paper/figures
python experiments/check_claims.py --runs results/paper --paper paper || echo "check_claims reported problems"
python paper/make_supplement.py --runs results/paper --out paper
python paper/check_directions.py --numbers paper/numbers.tex --tex paper/nanograph_main.tex \
  --out results/paper/DIRECTIONS.md || echo "direction checks: some FAIL, see results/paper/DIRECTIONS.md"
echo "TBD macros used by the manuscript:"
python - <<'PY'
import re
t = open('paper/nanograph_main.tex').read()
n = open('paper/numbers.tex').read()
tbd = {m for m in re.findall(r'\\newcommand\{\\(\w+)\}\{\\tbd\}', n)}
used = sorted(m for m in tbd if re.search(r'\\' + m + r'(?![A-Za-z])', t))
print(len(used), used)
PY
