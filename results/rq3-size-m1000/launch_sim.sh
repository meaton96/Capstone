#!/bin/bash
# rq3-size-m1000 (10-20, user): MDD-PTECT and SRT-PTECT at 1000 machines, test seeds 0-4, in the oracle simulator
# (sim_rules.py), locally. First the check (mdd seed 0 = the probe's evaluate.py value), alone, which also measures
# peak memory; then the other 9 with at most W at a time, fewer if memory is short. Finished tasks are skipped.
set -uo pipefail
W="${1:-4}"
cd "$(dirname "$0")/../.."
H=results/rq3-size-m1000
PY=.venv/bin/python
mkdir -p $H/logs
$PY $H/sim_rules.py mdd 0 2>&1 | tee $H/logs/sim_mdd_s0.log
[[ -f $H/m1000/mdd_s0/DONE ]] || { echo "check failed: no DONE for mdd seed 0"; exit 1; }
RSS=$($PY -c "import json; print(json.load(open('$H/m1000/mdd_s0/run.json')).get('peak_rss_gb') or 8)")
AVAIL=$(awk '/MemAvailable/ {print $2 / 1048576}' /proc/meminfo)
WE=$($PY -c "print(max(1, min($W, int(0.6 * $AVAIL / max($RSS, 0.5)))))")
echo "mdd s0 peak ${RSS} GB, ${AVAIL} GB available -> $WE at a time ($(date -Is))"
{ for s in 1 2 3 4; do echo "mdd $s"; done; for s in 0 1 2 3 4; do echo "srt $s"; done; } > $H/tasks_sim.txt
xargs -P "$WE" -L 1 bash -c '.venv/bin/python results/rq3-size-m1000/sim_rules.py "$@" > "results/rq3-size-m1000/logs/sim_$1_s$2.log" 2>&1; echo "$1 s$2 rc=$?"' _ < $H/tasks_sim.txt
echo "all tasks ended ($(date -Is))"
