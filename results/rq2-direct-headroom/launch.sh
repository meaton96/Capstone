#!/bin/bash
# rq2-direct-headroom (10-07): one-step rollout over MDD-TECT on B2 test seeds 0-39, first 1,800 s of the window,
# rule (15 H15 outcomes) vs direct (every job x machine, cap 60) options, scored in hindsight (real future) and in
# expectation (8 redrawn futures). 160 tasks; cheap cells first. run.py skips finished outputs, so rerunning resumes.
# Timing (smoke test, 300 s window): rule-expected4 68 s, direct-expected4 332 s per seed -> ~70 CPU-h in total.
# Usage (through the lab): bash results/rq2-direct-headroom/launch.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
D=results/rq2-direct-headroom
P=${WORKERS:-16}
for cell in "rule 0" "rule 8" "direct 0" "direct 8"; do
    set -- $cell
    for s in $(seq 0 39); do echo "--seed $s --variant $1 --futures $2"; done
done | xargs -P $P -L 1 .venv/bin/python $D/run.py >> $D/run.out 2>&1
.venv/bin/python $D/analyze.py > $D/analysis.out 2>&1
cat $D/analysis.out
