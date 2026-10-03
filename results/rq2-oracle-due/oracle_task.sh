#!/bin/bash
# One rq2-oracle-due style task: env/switch_oracle.py with the H15 pairs (job SRT / SPT / MDD / EDD / ATC x machine
# ECT / TECT / SRWT), tardiness reward, random warm-up, 5,400 s window, 6 x 900 s stages.
# usage: oracle_task.sh OUT SEED WORKER_ID PARAMS_JSON [extra switch_oracle.py args, e.g. --agvs 4]
# Skips a seed whose result.json exists. PLAYER defaults to linux_server_due/ (frozen 10-03 01:58 due-date build).
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=$1; SEED=$2; WID=$3; PARAMS=$4; shift 4
PLAYER="${PLAYER:-linux_server_due/capstone.x86_64}"
PAIRS="SRT-ECT,SRT-TECT,SRT-SRWT,SPT-ECT,SPT-TECT,SPT-SRWT,MDD-ECT,MDD-TECT,MDD-SRWT,EDD-ECT,EDD-TECT,EDD-SRWT,ATC-ECT,ATC-TECT,ATC-SRWT"
dir="$OUT/s$SEED"
[[ -f "$dir/result.json" ]] && { echo "[$(date '+%m-%d %H:%M')] skip $dir"; exit 0; }
mkdir -p "$dir"
if .venv/bin/python env/switch_oracle.py --unity-path "$PLAYER" --seed "$SEED" --scenario-generator randomized \
        --random-warmup --episode-duration-seconds 5400 --segment-seconds 900 --params "$PARAMS" \
        --reward-spec env/config/rewards/tardiness.json --pdr "$PAIRS" --base-worker-id "$WID" --out "$dir" "$@" \
        > "$dir/oracle.log" 2>&1; then
    echo "[$(date '+%m-%d %H:%M')] ok   $dir $(tail -n 1 "$dir/oracle.log" | cut -c1-120)"
else
    echo "[$(date '+%m-%d %H:%M')] FAIL $dir (see $dir/oracle.log)"
fi
