#!/bin/bash
# rq2-linked-price: on a linked 7-tile floor, does the best fixed machine rule change with fleet size and with uneven
# load between tiles? Fixed-rule step before any switching oracle on linked floors
# (docs/WIP/RUNNING_EXPERIMENTS.md, "rq2-linked-price").
#
# Floor: randomized generator x 7 tiles (105 machines), layout D, pooled AGVs. The generator always plans for 49 AGVs
# (7 per tile, its arrival cap), so every condition runs the same jobs; the 35-AGV conditions override the fleet.
# Machine rule (job rule SRT throughout):
#   l0 / l1 / l4  jobs may use any machine; TECT with travel price λ = 0 (plain TECT), 1, 4
#   loc           jobs stay in their home tile (jobScope "tile"), fleet still shared; plain TECT
# Release: bal = round-robin over tiles; skw = weighted 3,2,1,1,1,1,1 (the west tiles get 2.1x and 1.4x their share).
# Fleet: a49 (1 AGV per 2.1 machines, like the 15-machine floor) or a35 (1 per 3).
# One evaluate.py per (condition, seed), one episode each, from t = 0 for WINDOW sim-seconds; scored on the reward's
# quantity (episode return = time in system of every job). A job with summary.csv is skipped, so rerunning resumes.
# Player: linux_server_dev/ (10-02 15:13 build: TECT travel price, weighted release, cross-tile telemetry).
# Usage (from anywhere): nohup results/rq2-linked-price/run.sh > results/rq2-linked-price/run.out 2>&1 &
#   SEEDS="0 1" CONDS="a49-bal-l0 a49-bal-l4" WORKERS=8 WINDOW=5400 to limit.
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=results/rq2-linked-price
WORKERS="${WORKERS:-16}"
WINDOW="${WINDOW:-10800}"
SEEDS="${SEEDS:-0 1 2 3 4}"
PLAYER=linux_server_dev/capstone.x86_64
ALL_CONDS=""
for fleet in a49 a35; do for rel in bal skw; do for var in l0 l1 l4 loc; do
    ALL_CONDS+="$fleet-$rel-$var "
done; done; done
CONDS="${CONDS:-$ALL_CONDS}"

run_one() {  # index condition seed
    local idx=$1 cond=$2 seed=$3 dir="$OUT/$2/s$3"
    [[ -f "$dir/summary.csv" ]] && return 0
    local fleet=${cond%%-*} rest=${cond#*-}; local rel=${rest%%-*} var=${rest#*-}
    local scope=open price=0
    case "$var" in l0) price=0 ;; l1) price=1 ;; l4) price=4 ;; loc) scope=tile ;; *) echo "bad variant $var"; return 1 ;; esac
    local release='"release_rule": "roundRobin"'
    [[ "$rel" == skw ]] && release='"release_rule": "weighted", "release_weights": [3, 2, 1, 1, 1, 1, 1]'
    local params="{\"tiles\": 7, \"agv_count\": 49, \"agv_assignment\": \"pooled\", \"job_scope\": \"$scope\", $release, \"travel_price\": $price}"
    local agvs=(); [[ "$fleet" == a35 ]] && agvs=(--agvs 35)
    mkdir -p "$dir"
    if .venv/bin/python env/evaluate.py --unity-path "$PLAYER" --no-graphics --seeds "$seed" --pdr SRT-TECT \
            --scenario-generator randomized --episode-duration-seconds "$WINDOW" --params "$params" "${agvs[@]}" \
            --reward-spec env/config/rewards/flow_time.json \
            --base-worker-id $(( 4000 + idx )) --out "$dir" > "$dir/eval.log" 2>&1; then
        echo "[$(date '+%m-%d %H:%M')] ok   $cond s$seed"
    else
        echo "[$(date '+%m-%d %H:%M')] FAIL $cond s$seed (see $dir/eval.log)"
    fi
}
export -f run_one
export OUT PLAYER WINDOW

echo "[$(date '+%m-%d %H:%M')] start: $(wc -w <<< "$CONDS") conditions x seeds [$SEEDS], window ${WINDOW}s, $WORKERS workers"
i=0
for seed in $SEEDS; do for cond in $CONDS; do
    echo "$i $cond $seed"; i=$((i + 1))
done; done | xargs -P "$WORKERS" -L 1 bash -c 'run_one "$@"' _
echo "[$(date '+%m-%d %H:%M')] done: $(find "$OUT" -name summary.csv | wc -l) episodes"
.venv/bin/python "$OUT/analyze.py" || true
