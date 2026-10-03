#!/bin/bash
# rq2-oracle-buffer: switching-oracle headroom with finite machine buffers. Plan: docs/WIP/FUTURE_EXPERIMENTS.md
# "rq2-oracle-buffer". One env/switch_oracle.py per (regime, seed): 12 pairs x SEGMENTS stages, fresh player per stage.
# Regimes in regimes.json (all with --random-warmup). PLAYER must be a build with machine buffers (source after
# 2026-10-02 22:00); an older player rejects the inputBufferCapacity key (exit 3).
# Before the full run, smoke-test one cell (see SMOKE below). Resume: rerun; (regime, seed) with result.json are
# skipped. REGIMES="a b" / SEEDS="0 1" limit the grid.
# Usage: PLAYER=linux_server nohup results/rq2-oracle-buffer/run.sh > results/rq2-oracle-buffer/run.out 2>&1 &
# Cluster: same regimes.json/tasks.txt with slurm/oracle.sbatch and --export=ALL,EXP=rq2-oracle-buffer,TASKS=...
#
# SMOKE (one fixed pair, two seeds, tightest buffers; expect output_blocked_machine_seconds > 0,
# buffer_wait_job_seconds > 0, deadlock 0 - if deadlocks are common, loosen the grid before the full run):
#   .venv/bin/python env/evaluate.py --unity-path $PLAYER/capstone.x86_64 --pdr SPT-ECT,SRT-TECT --seeds 0-1 \
#       --scenario-generator randomized --random-warmup --episode-duration-seconds 5400 \
#       --input-buffer 2 --output-buffer 1 --out results/dev-buffer-smoke
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=results/rq2-oracle-buffer
PLAYER="${PLAYER:?set PLAYER to a player folder built with machine buffers}"
if ! grep -aq "inputBufferCapacity" "$PLAYER/capstone_Data/Managed/Simulation.dll"; then
    echo "$PLAYER was built before machine buffers existed; rebuild first"; exit 2
fi
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
    if .venv/bin/python env/switch_oracle.py --unity-path "$PLAYER/capstone.x86_64" --seed "$seed" \
            --scenario-generator randomized --random-warmup --episode-duration-seconds 5400 \
            --segment-seconds "$SEGMENT" --reward-spec env/config/rewards/flow_time.json \
            --base-worker-id $(( 1000 + idx * 2 )) --out "$dir" "${extra[@]}" > "$dir/oracle.log" 2>&1; then
        echo "[$(date '+%m-%d %H:%M')] ok   $reg s$seed $(tail -n 1 "$dir/oracle.log" | cut -c1-120)"
    else
        echo "[$(date '+%m-%d %H:%M')] FAIL $reg s$seed (see $dir/oracle.log)"
    fi
}
export -f run_one
export OUT SEGMENT PLAYER
echo "[$(date '+%m-%d %H:%M')] starting, $WORKERS workers, segment ${SEGMENT}s, seeds: $SEEDS, player $PLAYER"
idx=0
for seed in $SEEDS; do for reg in $REGIMES; do echo "$idx $reg $seed"; idx=$((idx + 1)); done; done \
    | xargs -P "$WORKERS" -L 1 bash -c 'run_one "$@"' _
echo "[$(date '+%m-%d %H:%M')] all done"
.venv/bin/python "$OUT/analyze.py" || true
