#!/bin/bash
# rq2-oracle-due: Unity check of the twin due-date screen (results/rq2-twin-due), step 3 of the staged due-date plan
# (docs/Plans/PDR_RULE_SET_PLAN_2026-10-02.md §9.1). One env/switch_oracle.py per seed: the candidate head H15 (job
# SRT / SPT / MDD / EDD / ATC x machine ECT / TECT / SRWT) x 6 stages of 900 s, fresh player per stage, tardiness reward,
# TWK due dates at c = 2, random warm-up, 5,400 s window, randomized generator WITH its machine failures (2/3 of seeds).
# Go to training if the oracle beats the best-on-average fixed pair by >= 3% (median) with >= 60% of seeds above 2%.
# Player: the due-date build (default linux_server_dev/, see its BUILD_NOTES.md).
# Resume: rerun; seeds with result.json are skipped. Usage:
#   nohup results/rq2-oracle-due/run.sh > results/rq2-oracle-due/run.out 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=results/rq2-oracle-due
PLAYER="${PLAYER:-linux_server_dev/capstone.x86_64}"
WORKERS="${WORKERS:-8}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7}"
SEGMENT="${SEGMENT:-900}"
ALLOWANCE="${ALLOWANCE:-2.0}"
PAIRS="SRT-ECT,SRT-TECT,SRT-SRWT,SPT-ECT,SPT-TECT,SPT-SRWT,MDD-ECT,MDD-TECT,MDD-SRWT,EDD-ECT,EDD-TECT,EDD-SRWT,ATC-ECT,ATC-TECT,ATC-SRWT"
run_one() {  # seed
    local seed=$1 dir="$OUT/s$1"
    [[ -f "$dir/result.json" ]] && return 0
    mkdir -p "$dir"
    if .venv/bin/python env/switch_oracle.py --unity-path "$PLAYER" --seed "$seed" \
            --scenario-generator randomized --random-warmup --episode-duration-seconds 5400 \
            --segment-seconds "$SEGMENT" --params "{\"due_date_allowance\": $ALLOWANCE}" \
            --reward-spec env/config/rewards/tardiness.json --pdr "$PAIRS" \
            --base-worker-id $(( 1100 + seed * 2 )) --out "$dir" > "$dir/oracle.log" 2>&1; then
        echo "[$(date '+%m-%d %H:%M')] ok   s$seed $(tail -n 1 "$dir/oracle.log" | cut -c1-140)"
    else
        echo "[$(date '+%m-%d %H:%M')] FAIL s$seed (see $dir/oracle.log)"
    fi
}
export -f run_one
export OUT PLAYER SEGMENT ALLOWANCE PAIRS
echo "[$(date '+%m-%d %H:%M')] starting, $WORKERS workers, segment ${SEGMENT}s, c $ALLOWANCE, seeds: $SEEDS, player $PLAYER"
for seed in $SEEDS; do echo "$seed"; done | xargs -P "$WORKERS" -L 1 bash -c 'run_one "$@"' _
echo "[$(date '+%m-%d %H:%M')] all done"
.venv/bin/python results/rq2-oracle-due/analyze.py || true
