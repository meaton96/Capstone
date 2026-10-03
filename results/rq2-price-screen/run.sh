#!/bin/bash
# rq2-price-screen: on the 15-machine floor, does TECT's travel price (λ) change the best fixed machine rule, and does
# the best λ change with how scarce AGVs are? Fixed-rule step before making priced TECT its own machine rule
# (docs/WIP/RUNNING_EXPERIMENTS.md, "rq2-price-screen"). Same regimes and episode setup as rq2-oracle-screen
# (randomized generator, random warm-up, 5,400 s agent window, layout D), so seeds 0-1 of SRT-ECT / SRT-TECT λ 0 must
# equal that oracle's stage-1 fixed returns.
#
# Regimes: base-w (7 AGVs), agv4 / agv3 (floor fleet only, jobs as for 7), amax1 (arrival cap at full AGV capacity).
# Variants (job rule SRT throughout): l0 = SRT-ECT and SRT-TECT with plain TECT; l1 / l2 / l4 = SRT-TECT with travel
# price λ = 1 / 2 / 4 (TECT scores max(travel, queue) + p + λ x travel). ECT ignores the price, so it runs once.
# One evaluate.py per (regime, variant, seed). A job with summary.csv is skipped, so rerunning resumes.
# Player: linux_server/ (10-02 16:13 build, strict FIFO, has the travel price).
# Usage: nohup results/rq2-price-screen/run.sh > results/rq2-price-screen/run.out 2>&1 &
#   SEEDS="0 1" REGIMES="agv3" VARIANTS="l0 l4" WORKERS=8 to limit.
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=results/rq2-price-screen
WORKERS="${WORKERS:-14}"
SEEDS="${SEEDS:-0 1 2 3 4}"
REGIMES="${REGIMES:-base-w agv4 agv3 amax1}"
VARIANTS="${VARIANTS:-l0 l1 l2 l4}"
PLAYER=linux_server/capstone.x86_64

run_one() {  # index regime variant seed
    local idx=$1 reg=$2 var=$3 seed=$4 dir="$OUT/$2/$3/s$4"
    [[ -f "$dir/summary.csv" ]] && return 0
    local price pdr=SRT-TECT
    case "$var" in l0) price=0; pdr=SRT-ECT,SRT-TECT ;; l1) price=1 ;; l2) price=2 ;; l4) price=4 ;;
        *) echo "bad variant $var"; return 1 ;; esac
    local params="{\"travel_price\": $price}" agvs=()
    case "$reg" in
        base-w) ;;
        agv4) agvs=(--agvs 4) ;;
        agv3) agvs=(--agvs 3) ;;
        amax1) params="{\"travel_price\": $price, \"agv_max_utilization\": 1.0}" ;;
        *) echo "bad regime $reg"; return 1 ;;
    esac
    mkdir -p "$dir"
    if .venv/bin/python env/evaluate.py --unity-path "$PLAYER" --no-graphics --seeds "$seed" --pdr "$pdr" \
            --scenario-generator randomized --random-warmup --episode-duration-seconds 5400 \
            --params "$params" "${agvs[@]}" --reward-spec env/config/rewards/flow_time.json \
            --base-worker-id $(( 6000 + idx * 2 )) --out "$dir" > "$dir/eval.log" 2>&1; then
        echo "[$(date '+%m-%d %H:%M')] ok   $reg $var s$seed"
    else
        echo "[$(date '+%m-%d %H:%M')] FAIL $reg $var s$seed (see $dir/eval.log)"
    fi
}
export -f run_one
export OUT PLAYER

echo "[$(date '+%m-%d %H:%M')] start: regimes [$REGIMES] x variants [$VARIANTS] x seeds [$SEEDS], $WORKERS workers"
i=0
for seed in $SEEDS; do for reg in $REGIMES; do for var in $VARIANTS; do
    echo "$i $reg $var $seed"; i=$((i + 1))
done; done; done | xargs -P "$WORKERS" -L 1 bash -c 'run_one "$@"' _
echo "[$(date '+%m-%d %H:%M')] done: $(find "$OUT" -name summary.csv | wc -l) jobs"
.venv/bin/python "$OUT/analyze.py" || true
