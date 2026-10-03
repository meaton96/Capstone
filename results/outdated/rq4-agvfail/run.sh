#!/bin/bash
# rq4-agvfail (was "AGVF", gap plan D4): AGV breakdown dose-response on the base-pdr12 instances (closes RQ4).
# Plan: docs/WIP/RUNNING_EXPERIMENTS.md, "rq4-agvfail". Player: linux_server/ (09-30 20:21 build with AGV breakdowns,
# Simulation.dll md5 5f78f6febcb8...); do not rebuild linux_server while this runs.
#   reg    12 pairs, breakdowns off (240 episodes). Must reproduce base-pdr12 (results/eval_pdr12_heads).
#   l8400  SPT-ECT, SRT-TECT, FIFO-SRWT with breakdowns at Weibull scale 8400 operating-s (default), 60 episodes
#   l3000  same, scale 3000 (60 episodes)
#   l1500  same, scale 1500 (60 episodes)
# Randomized generator, seeds 0-19, 5400 s from t = 0, layout D, 7 AGVs (the generator's defaults).
# One evaluate.py per (variant, seed), WORKERS at a time. A job with summary.csv is skipped, so rerunning resumes.
# By default it first waits for the twin-gap sweep (G1, linux_server_des) to finish; WAIT_FOR_G1=0 skips that.
# Usage (from anywhere): nohup results/rq4-agvfail/run.sh > results/rq4-agvfail/run.out 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=results/rq4-agvfail
WORKERS="${WORKERS:-16}"
PLAYER=linux_server/capstone.x86_64

if [[ "${WAIT_FOR_G1:-1}" == 1 ]]; then
    echo "[$(date '+%m-%d %H:%M')] waiting for G1 (launch_cell.sh / linux_server_des players) to finish"
    while pgrep -f launch_cell.sh >/dev/null || pgrep -f linux_server_des/BatchConfigs >/dev/null; do sleep 300; done
    echo "[$(date '+%m-%d %H:%M')] G1 done: $(find linux_server_des/Results/G1 -name results.csv | wc -l) / 400 cells"
fi

run_one() {  # index variant seed
    local idx=$1 v=$2 seed=$3 dir="$OUT/$2/s$3"
    [[ -f "$dir/summary.csv" ]] && return 0
    local args=(--pdr all)
    [[ "$v" != reg ]] && args=(--pdr SPT-ECT,SRT-TECT,FIFO-SRWT --agv-failures "{\"agvWeibullLambda\": ${v#l}}")
    mkdir -p "$dir"
    if .venv/bin/python env/evaluate.py --unity-path "$PLAYER" --no-graphics --seeds "$seed" "${args[@]}" \
            --scenario-generator randomized --episode-duration-seconds 5400 \
            --reward-spec env/config/rewards/flow_time.json \
            --base-worker-id $(( 700 + idx )) --out "$dir" > "$dir/eval.log" 2>&1; then
        echo "[$(date '+%m-%d %H:%M')] ok   $v s$seed"
    else
        echo "[$(date '+%m-%d %H:%M')] FAIL $v s$seed (see $dir/eval.log)"
    fi
}
export -f run_one
export OUT PLAYER

echo "[$(date '+%m-%d %H:%M')] starting, $WORKERS workers"
idx=0
for v in reg l8400 l3000 l1500; do
    for seed in $(seq 0 19); do echo "$idx $v $seed"; idx=$(( idx + 1 )); done
done | xargs -P "$WORKERS" -L 1 bash -c 'run_one "$@"' _

echo "[$(date '+%m-%d %H:%M')] all jobs finished"
.venv/bin/python "$OUT/check.py"
