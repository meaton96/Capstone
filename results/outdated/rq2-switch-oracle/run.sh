#!/bin/bash
# rq2-switch-oracle: greedy rule-switching oracle on the 20 held-out instances (base-pdr12 / rq4-agvfail reg),
# scored on the flow_time reward (time in system of every job). env/switch_oracle.py, one process per seed,
# 6 segments of 900 s x 12 pairs = 72 episodes per seed, fresh player per stage (clean float clock).
# Player: linux_server/ (09-30 build). Plan: docs/WIP/RUNNING_EXPERIMENTS.md "rq2-switch-oracle".
# Resume: rerun; seeds with result.json are skipped. Usage: nohup results/rq2-switch-oracle/run.sh > results/rq2-switch-oracle/run.out 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=results/rq2-switch-oracle
WORKERS="${WORKERS:-16}"
SEGMENT="${SEGMENT:-900}"
run_one() {
    local seed=$1 dir="$OUT/s$1"
    [[ -f "$dir/result.json" ]] && return 0
    mkdir -p "$dir"
    if .venv/bin/python env/switch_oracle.py --unity-path linux_server/capstone.x86_64 --seed "$seed" \
            --scenario-generator randomized --episode-duration-seconds 5400 --segment-seconds "$SEGMENT" \
            --reward-spec env/config/rewards/flow_time.json --base-worker-id $(( 500 + seed * 3 )) \
            --out "$dir" > "$dir/oracle.log" 2>&1; then
        echo "[$(date '+%m-%d %H:%M')] ok   s$seed $(tail -n 1 "$dir/oracle.log")"
    else
        echo "[$(date '+%m-%d %H:%M')] FAIL s$seed (see $dir/oracle.log)"
    fi
}
export -f run_one
export OUT SEGMENT
echo "[$(date '+%m-%d %H:%M')] starting, $WORKERS workers, segment ${SEGMENT}s"
seq 0 19 | xargs -P "$WORKERS" -I{} bash -c 'run_one {}'
echo "[$(date '+%m-%d %H:%M')] all seeds finished"
