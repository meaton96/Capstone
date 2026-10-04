#!/bin/bash
# rq2-oracle-due-csweep (10-03, user asked): Unity switching oracle on tardiness at due-date allowance c = 1.5 / 2.5 / 3,
# seeds 0-7, same setup as rq2-oracle-due (c = 2): H15 pairs, random warm-up, 5,400 s window, 6 x 900 s stages,
# machine failures as generated, 7 AGVs. Player linux_server_due/ (frozen 01:58 build). Python: the frozen copy in
# env_snapshot/ (taken 10-03 22:30 from the main checkout with the clock budget removed), so edits to env/ during the
# run do not reach it. Each task skips itself when result.json exists: rerun the same command to resume.
# Usage: nohup results/rq2-oracle-due-csweep/run.sh > results/rq2-oracle-due-csweep/run.out 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
HERE=results/rq2-oracle-due-csweep
WORKERS="${WORKERS:-20}"
PAIRS="SRT-ECT,SRT-TECT,SRT-SRWT,SPT-ECT,SPT-TECT,SPT-SRWT,MDD-ECT,MDD-TECT,MDD-SRWT,EDD-ECT,EDD-TECT,EDD-SRWT,ATC-ECT,ATC-TECT,ATC-SRWT"
task() {  # c seed
    local c=$1 seed=$2 dir="results/rq2-oracle-due-c$1/s$2" wid=$((2100 + ${3}))
    [[ -f "$dir/result.json" ]] && { echo "[$(date '+%m-%d %H:%M')] skip $dir"; return; }
    mkdir -p "$dir"
    if .venv/bin/python $HERE/env_snapshot/switch_oracle.py --unity-path linux_server_due/capstone.x86_64 --seed "$seed" \
            --scenario-generator randomized --random-warmup --episode-duration-seconds 5400 --segment-seconds 900 \
            --params "{\"due_date_allowance\": $c}" --reward-spec $HERE/env_snapshot/config/rewards/tardiness.json \
            --pdr "$PAIRS" --base-worker-id "$wid" --out "$dir" > "$dir/oracle.log" 2>&1; then
        echo "[$(date '+%m-%d %H:%M')] ok   $dir $(tail -n 1 "$dir/oracle.log" | cut -c1-120)"
    else
        echo "[$(date '+%m-%d %H:%M')] FAIL $dir (see $dir/oracle.log)"
    fi
}
export -f task; export HERE PAIRS
i=0; for seed in 0 1 2 3 4 5 6 7; do for c in 1.5 2.5 3; do echo "$c $seed $((2*i))"; i=$((i+1)); done; done \
    | xargs -P "$WORKERS" -L1 bash -c 'task $0 $1 $2'
echo "[$(date '+%m-%d %H:%M')] all done"
for c in 1.5 2.5 3; do d=results/rq2-oracle-due-c$c
    .venv/bin/python results/rq2-oracle-due/analyze.py "$d" > "$d/analysis.out" 2>&1
    echo "== $d"; grep -E 'seeds finished|total_%|switch_%|oracle vs' "$d/analysis.out"
done
