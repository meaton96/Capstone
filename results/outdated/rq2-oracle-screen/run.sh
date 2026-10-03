#!/bin/bash
# rq2-oracle-screen: switching-oracle headroom across regimes (phase 1 screen). Plan: docs/WIP/RUNNING_EXPERIMENTS.md
# "rq2-oracle-screen". One env/switch_oracle.py per (regime, seed): 12 pairs x SEGMENTS stages, fresh player per stage.
# Regimes in regimes.json (all with --random-warmup). Player linux_server/ (09-30 build).
# Resume: rerun; (regime, seed) with result.json are skipped. REGIMES="a b" limits the regimes.
# Usage: nohup results/rq2-oracle-screen/run.sh > results/rq2-oracle-screen/run.out 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=results/rq2-oracle-screen
WORKERS="${WORKERS:-16}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7}"
SEGMENT="${SEGMENT:-1800}"
REGIMES="${REGIMES:-$(.venv/bin/python -c "import json;print(' '.join(k for k in json.load(open('$OUT/regimes.json')) if not k.startswith('_')))")}"
run_one() {  # index regime seed
    local idx=$1 reg=$2 seed=$3 dir="$OUT/$2/s$3"
    [[ -f "$dir/result.json" ]] && return 0
    mkdir -p "$dir"
    local -a extra
    mapfile -t extra < <(.venv/bin/python -c "import json,sys;[print(a) for a in json.load(open('$OUT/regimes.json'))['$reg']['args']]")
    if .venv/bin/python env/switch_oracle.py --unity-path linux_server/capstone.x86_64 --seed "$seed" \
            --scenario-generator randomized --random-warmup --episode-duration-seconds 5400 \
            --segment-seconds "$SEGMENT" --reward-spec env/config/rewards/flow_time.json \
            --base-worker-id $(( 1000 + idx * 2 )) --out "$dir" "${extra[@]}" > "$dir/oracle.log" 2>&1; then
        echo "[$(date '+%m-%d %H:%M')] ok   $reg s$seed $(tail -n 1 "$dir/oracle.log" | cut -c1-120)"
    else
        echo "[$(date '+%m-%d %H:%M')] FAIL $reg s$seed (see $dir/oracle.log)"
    fi
}
export -f run_one
export OUT SEGMENT
echo "[$(date '+%m-%d %H:%M')] starting, $WORKERS workers, segment ${SEGMENT}s, seeds: $SEEDS"
idx=0
for seed in $SEEDS; do for reg in $REGIMES; do echo "$idx $reg $seed"; idx=$((idx + 1)); done; done \
    | xargs -P "$WORKERS" -L 1 bash -c 'run_one "$@"' _
echo "[$(date '+%m-%d %H:%M')] all done"
.venv/bin/python "$OUT/analyze.py" || true
