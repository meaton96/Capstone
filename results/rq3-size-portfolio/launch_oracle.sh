#!/bin/bash
# rq3-size-portfolio, step 3b (10-20, user): exact switching oracle over size-specific portfolios (m100 v5tect and
# block6, m50 block6), B2 test seeds 0-39, local lab run (W at a time), after the 200-machine runs. Finished seeds skip.
set -uo pipefail
W="${1:-16}"
cd "$(dirname "$0")/../.."
H=results/rq3-size-portfolio
mkdir -p $H/logs
.venv/bin/python $H/run.py list oracle > $H/tasks_oracle.txt
echo "$(wc -l < $H/tasks_oracle.txt) tasks, $W at a time ($(date -Is))"
xargs -P "$W" -L 1 bash -c '.venv/bin/python results/rq3-size-portfolio/run.py "$@" > "results/rq3-size-portfolio/logs/oracle_$2_$3_s$4.log" 2>&1; echo "oracle $2 $3 s$4 rc=$?"' _ < $H/tasks_oracle.txt
echo "all tasks ended ($(date -Is))"
