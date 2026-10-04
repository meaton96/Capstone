#!/bin/bash
# rq2-oracle-due-mix (10-03, user: train on a mix that varies c and load): Unity check of the training mix chosen
# from rq2-twin-scen, on held-out seeds 20-39. Per episode: c ~ U[1.75, 2.5], load profile base or utilhi (50:50);
# H15 oracle, tardiness, random warm-up, 5,400 s window, 6 x 900 s, machine failures as generated, 7 AGVs.
# Player linux_server_due/ (frozen 01:58 build; obs v3 not needed by the oracle). Python: frozen env_snapshot/
# (taken 10-03 23:55, includes the training-mix generator). Waits for rq2-oracle-due-csweep to finish first.
# Resume: rerun (seeds with result.json are skipped).
# Usage: nohup results/rq2-oracle-due-mix/run.sh > results/rq2-oracle-due-mix/run.out 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
HERE=results/rq2-oracle-due-mix
WORKERS="${WORKERS:-20}"
PAIRS="SRT-ECT,SRT-TECT,SRT-SRWT,SPT-ECT,SPT-TECT,SPT-SRWT,MDD-ECT,MDD-TECT,MDD-SRWT,EDD-ECT,EDD-TECT,EDD-SRWT,ATC-ECT,ATC-TECT,ATC-SRWT"
PARAMS='{"due_date_allowance_range": [1.75, 2.5], "load_mix": [{"name": "base"}, {"name": "utilhi", "utilization": [1.0, 1.8], "lull_utilization": [0.4, 0.7]}]}'
while pgrep -f "[r]q2-oracle-due-csweep/run.sh" > /dev/null; do sleep 120; done
echo "[$(date '+%m-%d %H:%M')] starting"
task() {
    local seed=$1 dir="$HERE/s$1"
    [[ -f "$dir/result.json" ]] && { echo "[$(date '+%m-%d %H:%M')] skip $dir"; return; }
    mkdir -p "$dir"
    if .venv/bin/python $HERE/env_snapshot/switch_oracle.py --unity-path linux_server_due/capstone.x86_64 --seed "$seed" \
            --scenario-generator randomized --random-warmup --episode-duration-seconds 5400 --segment-seconds 900 \
            --params "$PARAMS" --reward-spec $HERE/env_snapshot/config/rewards/tardiness.json \
            --pdr "$PAIRS" --base-worker-id $((2200 + 2 * seed)) --out "$dir" > "$dir/oracle.log" 2>&1; then
        echo "[$(date '+%m-%d %H:%M')] ok   $dir $(tail -n 1 "$dir/oracle.log" | cut -c1-120)"
    else
        echo "[$(date '+%m-%d %H:%M')] FAIL $dir (see $dir/oracle.log)"
    fi
}
export -f task; export HERE PAIRS PARAMS
seq 20 39 | xargs -P "$WORKERS" -I{} bash -c 'task {}'
echo "[$(date '+%m-%d %H:%M')] all done"
.venv/bin/python results/rq2-oracle-due/analyze.py "$HERE" > "$HERE/analysis.out" 2>&1
grep -E 'seeds finished|total_%|switch_%|oracle vs|oracle job rules' "$HERE/analysis.out"
