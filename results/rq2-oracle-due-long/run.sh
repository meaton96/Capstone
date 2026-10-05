#!/bin/bash
# rq2-oracle-due-long (10-04, user: Unity check of long episodes / fleet blocks after the twin tests): Unity H15
# oracle on tardiness at the calibrated load (utilization 0.6-1.2 outside lulls, dev-load-calib), c ~ U[1.75, 2.5],
# random warm-up, machine failures as generated, seeds 0-7 per setting:
#   long-u:   one regime, 21,600 s window, 1,800 s slots (12 stages)
#   blocks-u: regime blocks of 5,400 s, profiles normal (2) : surge (1, util 1.0-1.6) : short (1, 3 AGVs on duty),
#             21,600 s window, 1,800 s slots
#   short-u:  one regime, 5,400 s window, 900 s slots (6 stages)
# Player linux_server_v4/ (fleet schedule, obs v3). Python: frozen env_snapshot/ (10-04 19:15). Waits for the
# rq2-oracle-due-mix launcher to exit. Resume: rerun (seeds with result.json are skipped).
# Usage: nohup results/rq2-oracle-due-long/run.sh > results/rq2-oracle-due-long/run.out 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
HERE=results/rq2-oracle-due-long
WORKERS="${WORKERS:-20}"
PAIRS="SRT-ECT,SRT-TECT,SRT-SRWT,SPT-ECT,SPT-TECT,SPT-SRWT,MDD-ECT,MDD-TECT,MDD-SRWT,EDD-ECT,EDD-TECT,EDD-SRWT,ATC-ECT,ATC-TECT,ATC-SRWT"
NORMAL='"utilization": [0.6, 1.2]'
P_SINGLE="{\"due_date_allowance_range\": [1.75, 2.5], $NORMAL, \"horizon_seconds\": 30600}"
P_SHORT="{\"due_date_allowance_range\": [1.75, 2.5], $NORMAL}"
P_BLOCKS="{\"due_date_allowance_range\": [1.75, 2.5], \"regime_block_seconds\": 5400, \"horizon_seconds\": 30600, \"load_mix\": [{\"name\": \"normal\", $NORMAL, \"weight\": 2}, {\"name\": \"surge\", \"utilization\": [1.0, 1.6]}, {\"name\": \"short\", $NORMAL, \"agv_count\": 3}]}"
while pgrep -f "[r]q2-oracle-due-mix/run.sh" > /dev/null; do sleep 120; done
echo "[$(date '+%m-%d %H:%M')] starting"
task() {  # setting seed worker
    local st=$1 seed=$2 wid=$3 dir="$HERE/$1/s$2" params window slot
    case "$st" in
        long-u)   params=$P_SINGLE; window=21600; slot=1800 ;;
        blocks-u) params=$P_BLOCKS; window=21600; slot=1800 ;;
        short-u)  params=$P_SHORT;  window=5400;  slot=900 ;;
    esac
    [[ -f "$dir/result.json" ]] && { echo "[$(date '+%m-%d %H:%M')] skip $dir"; return; }
    mkdir -p "$dir"
    if .venv/bin/python $HERE/env_snapshot/switch_oracle.py --unity-path linux_server_v4/capstone.x86_64 --seed "$seed" \
            --scenario-generator randomized --random-warmup --episode-duration-seconds "$window" --segment-seconds "$slot" \
            --params "$params" --reward-spec $HERE/env_snapshot/config/rewards/tardiness.json \
            --pdr "$PAIRS" --base-worker-id "$wid" --out "$dir" > "$dir/oracle.log" 2>&1; then
        echo "[$(date '+%m-%d %H:%M')] ok   $dir $(tail -n 1 "$dir/oracle.log" | cut -c1-120)"
    else
        echo "[$(date '+%m-%d %H:%M')] FAIL $dir (see $dir/oracle.log)"
    fi
}
export -f task; export HERE PAIRS P_SINGLE P_SHORT P_BLOCKS
i=0; for st in long-u blocks-u short-u; do for seed in 0 1 2 3 4 5 6 7; do echo "$st $seed $((2600 + 2 * i))"; i=$((i+1)); done; done \
    | xargs -P "$WORKERS" -L1 bash -c 'task $0 $1 $2'
echo "[$(date '+%m-%d %H:%M')] all done"
for st in long-u blocks-u short-u; do
    .venv/bin/python results/rq2-oracle-due/analyze.py "$HERE/$st" > "$HERE/$st/analysis.out" 2>&1
    echo "== $st"; grep -E 'seeds finished|total_%|switch_%|oracle job rules' "$HERE/$st/analysis.out"
done
