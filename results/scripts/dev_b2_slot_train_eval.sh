#!/bin/bash
# Train + evaluate a twin slot-action PPO variant on the B2 regime (2026-10-07, learning-stack fixes from
# docs/handoffs/HANDOFF_2026-10-07_learning_stack_fixes.md). Same setup as train-due-twin-slot-prior-kl (B2 regime,
# 900 s slots, 100k agent steps, entropy 0.01 -> 0.001, MDD-TECT prior at p = 0.8, KL 0.05) unless the extra flags
# override it; argparse keeps the last value, so extra flags win.
#
# Usage: bash results/scripts/dev_b2_slot_train_eval.sh NAME "SEEDS" WORKERS_PER_SEED [extra train.py flags...]
#   e.g. bash results/scripts/dev_b2_slot_train_eval.sh dev-prior-offset "0 1" 3 --prior-init offset
#
# - Code: env/ is copied once to results/NAME/code/env and every train / evaluate call runs from that copy, so later
#   edits in the checkout never reach a running or resumed experiment (memory: live-checkout runs pick up edits).
#   The copy's git head and diff hash go to results/NAME/code/STAMP.
# - Resumes: each seed resumes from its latest checkpoint; a finished seed (DONE_s<seed>) is skipped; evaluation is
#   skipped once results/NAME/eval/DONE exists.
# - Evaluation: twin, B2 seeds 0-39, deterministic, slot actions from the checkpoints, all 15 fixed pairs (the
#   per-instance hindsight best), MDD-TECT named as the best fixed pair (the reference of every B2 write-up), and the
#   switching oracle from rq2-twin-fleet (B2 tasks).
set -uo pipefail
cd "$(dirname "$0")/../.."
NAME="$1"; SEEDS="$2"; WORKERS="$3"; shift 3
EXTRA=("$@")
OUT="results/$NAME"
mkdir -p "$OUT"
if [ ! -d "$OUT/code/env" ]; then
    mkdir -p "$OUT/code"
    rsync -a --exclude __pycache__ --exclude tests --exclude '*.egg-info' env/ "$OUT/code/env/"
    { echo "snapshot $(date -Is)"; echo "git_head $(git rev-parse --short HEAD)";
      echo "diff_sha $(git diff HEAD -- env | sha256sum | cut -c1-12)"; } > "$OUT/code/STAMP"
fi
PY=.venv/bin/python
CODE="$OUT/code/env"
FLOOR=results/dev-popart-smoke/floor7/des_floor.json
PARAMS='{"failure_probability": 0.0, "due_date_allowance_range": [1.75, 2.5], "regime_block_seconds": 5400, "horizon_seconds": 30600, "load_mix": [{"name": "normal", "utilization": [0.6, 1.2], "weight": 2}, {"name": "surge", "utilization": [1.0, 1.6]}, {"name": "short", "utilization": [0.6, 1.2], "agv_count": 2}]}'

for SEED in $SEEDS; do
    RUN="results/${NAME}_s$SEED"
    [ -f "$OUT/DONE_s$SEED" ] && continue
    (
        RESUME=()
        LATEST="$(ls -t "$RUN"/checkpoint.pt "$RUN"/checkpoint_step*.pt 2>/dev/null | head -1 || true)"
        [[ -n "$LATEST" ]] && RESUME=(--resume-from "$LATEST")
        nice -n 5 $PY "$CODE/train.py" --twin "$FLOOR" --twin-workers "$WORKERS" \
            --scenario-generator randomized --random-warmup --episode-duration-seconds 21600 --params "$PARAMS" \
            --reward-spec env/config/rewards/tardiness.json --train-seed "$SEED" \
            --num-envs 16 --rollout-length 32 --batch-size 64 --total-timesteps 100000 --slot-seconds 900 \
            --discount-horizon-s 10800 --ent-coef 0.01 --ent-coef-final 0.001 \
            --prior-pair MDD-TECT --prior-prob 0.8 --prior-kl-coef 0.05 \
            --device cuda --torch-threads 2 --save-every 25 --stall-minutes 30 --notify-every-hours 2 \
            --results-dir results --run-id "${NAME}_s$SEED" "${RESUME[@]}" "${EXTRA[@]}" \
            > "$OUT/train_s$SEED.out" 2>&1 \
        && grep -q "Checkpoint saved to" "$OUT/train_s$SEED.out" && date -Is > "$OUT/DONE_s$SEED"
    ) &
done
wait
for SEED in $SEEDS; do
    [ -f "$OUT/DONE_s$SEED" ] || { echo "seed $SEED did not finish (see $OUT/train_s$SEED.out)"; exit 1; }
done

if [ ! -f "$OUT/eval/DONE" ]; then
    CK=()
    for SEED in $SEEDS; do CK+=(--checkpoint "results/${NAME}_s$SEED/checkpoint.pt"); done
    $PY "$CODE/evaluate.py" --twin "$FLOOR" --reward-spec env/config/rewards/tardiness.json \
        --scenario-generator randomized --random-warmup --episode-duration-seconds 21600 --params "$PARAMS" \
        --seeds 0-39 --pdr all --best-fixed MDD-TECT --oracle 'results/rq2-twin-fleet/tasks/B2_s*.json' \
        "${CK[@]}" --device cuda --decision-log --out "$OUT/eval" > "$OUT/eval.out" 2>&1 \
    && date -Is > "$OUT/eval/DONE"
fi
[ -f "$OUT/eval/DONE" ] || { echo "evaluation failed (see $OUT/eval.out)"; exit 1; }
$PY results/scripts/dev_b2_slot_analyze.py "$NAME" > "$OUT/analysis.out" 2>&1
cat "$OUT/analysis.out"
