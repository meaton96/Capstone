#!/bin/bash
# eval-v5-final (10-14): pick the final PPO policy without the test seeds (Demo 1 slide 17). The 5 PPO seeds of the
# v5 stack (train-v5-nokl s0-s2, train-v5-final s3-s4; same frozen code, flags and 100k steps) are evaluated in the
# twin on the B2 FITTING seeds 2000-2159 (where rq2-rebase-v5 chose MDD-PTECT as the best fixed pair) and on the test
# seeds 0-39, next to MDD-PTECT and MDD-PECT. select_final.py picks the seed with the lowest pooled window tardiness on the
# fitting seeds (-> selected.txt) and reports all 5, their mean and the pick on the test seeds.
#   code: rq2-rebase-v5/code/env (6d13412a, the checkpoints' code); floor: dev-popart-smoke/floor7 (the training floor)
#   B2 settings of results/scripts/dev_b2_slot_train_eval.sh; slot length and discount horizon from the checkpoints
#   (900 s / 5,400 s), deterministic. One evaluate.py per chunk of 10 seeds; a chunk with DONE is skipped.
# Usage (lab): bash results/eval-v5-final/run.sh      env: WORKERS (default 10)
set -uo pipefail
cd "$(dirname "$0")/../.."
H=results/eval-v5-final
CODE=results/rq2-rebase-v5/code/env
export H CODE
export P='{"failure_probability": 0.0, "due_date_allowance_range": [1.75, 2.5], "regime_block_seconds": 5400, "horizon_seconds": 30600, "load_mix": [{"name": "normal", "utilization": [0.6, 1.2], "weight": 2}, {"name": "surge", "utilization": [1.0, 1.6]}, {"name": "short", "utilization": [0.6, 1.2], "agv_count": 2}]}'
for k in 0 1 2; do [[ -f results/train-v5-nokl_s$k/checkpoint.pt ]] || { echo "missing PPO s$k"; exit 1; }; done
for k in 3 4; do [[ -f results/train-v5-final_s$k/checkpoint.pt ]] || { echo "missing PPO s$k (pull train-v5-final)"; exit 1; }; done
chunk() {  # split first last
    local d="$H/$1/c$2" t0=$(date +%s)
    [[ -f $d/DONE ]] && return
    rm -rf "$d"; mkdir -p "$d"
    if OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 5 .venv/bin/python $CODE/evaluate.py \
            --twin results/dev-popart-smoke/floor7/des_floor.json --reward-spec $CODE/config/rewards/tardiness.json \
            --scenario-generator randomized --random-warmup --episode-duration-seconds 21600 --params "$P" \
            --seeds "$2-$3" --pdr MDD-PTECT,MDD-PECT --device cpu \
            --checkpoint results/train-v5-nokl_s0/checkpoint.pt --checkpoint results/train-v5-nokl_s1/checkpoint.pt \
            --checkpoint results/train-v5-nokl_s2/checkpoint.pt --checkpoint results/train-v5-final_s3/checkpoint.pt \
            --checkpoint results/train-v5-final_s4/checkpoint.pt --out "$d" > "$d/eval.out" 2>&1 \
       && [[ $(( $(wc -l < "$d/episodes.csv") - 1 )) -eq $(( 7 * ($3 - $2 + 1) )) ]]; then
        echo "$(( $(date +%s) - t0 ))" > "$d/DONE"; echo "[$(date '+%m-%d %H:%M')] ok   $1 $2-$3 ($(( $(date +%s) - t0 )) s)"
    else
        echo "[$(date '+%m-%d %H:%M')] FAIL $1 $2-$3 (see $d/eval.out)"
    fi
}
export -f chunk
{ for a in $(seq 2000 10 2150); do echo "fit $a $((a + 9))"; done
  for a in 0 10 20 30; do echo "test $a $((a + 9))"; done; } | xargs -P "${WORKERS:-10}" -L1 bash -c 'chunk $0 $1 $2'
echo "[$(date '+%m-%d %H:%M')] chunks done: $(ls $H/*/c*/DONE 2>/dev/null | wc -l)/20"
.venv/bin/python $H/select_final.py
