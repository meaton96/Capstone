#!/bin/bash
# base-due-c2: one seed of the fixed-pair baseline on the tardiness objective: every RL head pair (evaluate.py --pdr all,
# 21 pairs), TWK due dates, random warm-up, 5,400 s window, one player per seed.
# usage: eval_task.sh OUT SEED WORKER_ID PARAMS_JSON ; skips a seed whose episodes.csv exists.
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=$1; SEED=$2; WID=$3; PARAMS=$4
PLAYER="${PLAYER:-linux_server_due/capstone.x86_64}"
dir="$OUT/s$SEED"
[[ -f "$dir/episodes.csv" ]] && { echo "[$(date '+%m-%d %H:%M')] skip $dir"; exit 0; }
mkdir -p "$dir"
if .venv/bin/python env/evaluate.py --unity-path "$PLAYER" --base-worker-id "$WID" \
        --reward-spec env/config/rewards/tardiness.json --scenario-generator randomized --random-warmup \
        --episode-duration-seconds 5400 --params "$PARAMS" --seeds "$SEED" --pdr all --no-graphics --out "$dir" \
        > "$dir/eval.log" 2>&1; then
    echo "[$(date '+%m-%d %H:%M')] ok   $dir"
else
    echo "[$(date '+%m-%d %H:%M')] FAIL $dir (see $dir/eval.log)"
fi
