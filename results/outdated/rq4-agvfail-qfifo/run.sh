#!/bin/bash
# rq4-agvfail-qfifo: the FIFO cells of rq4-agvfail (and base-pdr12) rerun on the strict-FIFO build. FIFO now ranks
# jobs by time in their current queue instead of time since shop arrival (2026-10-02). Plan:
# docs/WIP/RUNNING_EXPERIMENTS.md, "rq4-agvfail-qfifo". Player: linux_server/ (10-02 16:13 build).
#   reg      FIFO-ECT, FIFO-TECT, FIFO-SRWT, breakdowns off (60 episodes). With rq4-agvfail/reg's 9 other pairs this
#            is the 12-pair baseline (base-pdr12) on a clean clock: one player per seed, at most 12 episodes each.
#   l8400    FIFO-SRWT with AGV breakdowns at Weibull scale 8400 operating-s (20 episodes)
#   l3000    same, scale 3000 (20)
#   l1500    same, scale 1500 (20)
#   chk1500  SPT-ECT, SRT-TECT at scale 1500, seeds 0-1 (4): must equal rq4-agvfail/l1500 exactly, which shows the
#            breakdown path is unchanged on this build before old and new rows are put side by side.
# Randomized generator, seeds 0-19, 5400 s from t = 0, layout D, 7 AGVs. One evaluate.py per (variant, seed),
# WORKERS at a time. A job with summary.csv is skipped, so rerunning resumes.
# Usage (from anywhere): nohup results/rq4-agvfail-qfifo/run.sh > results/rq4-agvfail-qfifo/run.out 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=results/rq4-agvfail-qfifo
WORKERS="${WORKERS:-6}"
PLAYER=linux_server/capstone.x86_64

run_one() {  # index variant seed
    local idx=$1 v=$2 seed=$3 dir="$OUT/$2/s$3"
    [[ -f "$dir/summary.csv" ]] && return 0
    local args
    case "$v" in
        reg)     args=(--pdr FIFO-ECT,FIFO-TECT,FIFO-SRWT) ;;
        chk1500) args=(--pdr SPT-ECT,SRT-TECT --agv-failures '{"agvWeibullLambda": 1500}') ;;
        *)       args=(--pdr FIFO-SRWT --agv-failures "{\"agvWeibullLambda\": ${v#l}}") ;;
    esac
    mkdir -p "$dir"
    # --allow-unverified-player: slurm/oracle.sbatch in linux_server/ was edited after the 16:13 build (EXP variable),
    # so the folder no longer matches BUILD_MANIFEST.json there. The player files are unchanged; the mismatch is
    # recorded in each job's player_manifest.json. Drop the flag after the next build into linux_server/.
    if .venv/bin/python env/evaluate.py --unity-path "$PLAYER" --allow-unverified-player --no-graphics --seeds "$seed" "${args[@]}" \
            --scenario-generator randomized --episode-duration-seconds 5400 \
            --reward-spec env/config/rewards/flow_time.json \
            --base-worker-id $(( 2000 + idx )) --out "$dir" > "$dir/eval.log" 2>&1; then
        echo "[$(date '+%m-%d %H:%M')] ok   $v s$seed"
    else
        echo "[$(date '+%m-%d %H:%M')] FAIL $v s$seed (see $dir/eval.log)"
    fi
}
export -f run_one
export OUT PLAYER

echo "[$(date '+%m-%d %H:%M')] starting, $WORKERS workers"
idx=0
{
    for seed in 0 1; do echo "$idx chk1500 $seed"; idx=$(( idx + 1 )); done
    for v in reg l8400 l3000 l1500; do
        for seed in $(seq 0 19); do echo "$idx $v $seed"; idx=$(( idx + 1 )); done
    done
} | xargs -P "$WORKERS" -L 1 bash -c 'run_one "$@"' _

echo "[$(date '+%m-%d %H:%M')] all jobs finished"
