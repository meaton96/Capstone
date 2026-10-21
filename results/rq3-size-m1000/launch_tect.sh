#!/bin/bash
# rq3-size-m1000 (10-21): MDD-TECT at 1000 machines, test seeds 0-4, oracle simulator (sim_rules.py tect), local, W at a
# time. Added because the catalog found MDD-TECT the best pair at 200 machines (-19.0% vs MDD-PTECT), so it is the
# column's "best pair". The simulator matches evaluate.py at 1000 machines (mdd seed 0 check). Finished seeds skip.
set -uo pipefail
W="${1:-4}"
cd "$(dirname "$0")/../.."
H=results/rq3-size-m1000
mkdir -p $H/logs
for s in 0 1 2 3 4; do echo "tect $s"; done > $H/tasks_tect.txt
xargs -P "$W" -L 1 bash -c '.venv/bin/python results/rq3-size-m1000/sim_rules.py "$@" > "results/rq3-size-m1000/logs/sim_$1_s$2.log" 2>&1; echo "$1 s$2 rc=$?"' _ < $H/tasks_tect.txt
echo "all tasks ended ($(date -Is))"
