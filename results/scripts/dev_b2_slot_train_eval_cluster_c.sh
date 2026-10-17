#!/bin/bash
# DUE-DATE VARIANT (2026-10-14, train-v5-loose / train-v5-cmix): identical to dev_b2_slot_train_eval_cluster.sh except
# that TRAINING draws c from the range in results/NAME/code/c_range (two numbers, e.g. "2.5 3.5") instead of
# B2's [1.75, 2.5]. The c range travels in a file because submit_dev_cluster.sh cannot pass a JSON argument through
# sbatch --wrap intact. The evaluation at the end stays on B2 itself (c 1.75-2.5, seeds 0-39), as in the original,
# unless results/NAME/code/train_only exists (10-15): then the job only trains, so one job per seed can run with a
# small CPU request (the 27-CPU 5-seed jobs waited two days for a node).
#
# CLUSTER COPY (10-07) of dev_b2_slot_train_eval.sh, run inside one GPU job by submit_dev_cluster.sh: PY is
# overridable (.venv-gpu), DEVICE overridable (cpu: CPU-only jobs, 10-07), no nice, no local analysis. Keep in step
# with the local script.
# Train + evaluate a twin slot-action PPO variant on the B2 regime (2026-10-07, learning-stack fixes from
# docs/handoffs/HANDOFF_2026-10-07_learning_stack_fixes.md). Defaults since 10-11 (RL heads v5 = {SRT, MDD, ATC} x
# {PECT, PTECT}, results/dev-pect-unity): the dev-nokl-free stack (B2 regime, 900 s slots, 100k agent steps, entropy
# 0.01 -> 0.001, NO action prior, NO KL, 5,400 s horizon, shared baseline groups of 4, rollout 128 / batch 256) unless
# the extra flags override it; argparse keeps the last value, so extra flags win. Before 10-11 the defaults were the
# H15 heads with an MDD-TECT prior at p = 0.8 and KL 0.05 (train-due-twin-slot-prior-kl): the dev-* runs of 10-07/08
# used those; rerunning them needs their own code snapshot (results/<name>/code/env) and the old defaults.
#
# Usage: bash results/scripts/dev_b2_slot_train_eval.sh NAME "SEEDS" WORKERS_PER_SEED [extra train.py flags...]
#   e.g. bash results/scripts/dev_b2_slot_train_eval.sh dev-heads-v5 "0 1 2" 2 --total-timesteps 300000
#
# - Code: env/ is copied once to results/NAME/code/env and every train / evaluate call runs from that copy, so later
#   edits in the checkout never reach a running or resumed experiment (memory: live-checkout runs pick up edits).
#   The copy's git head and diff hash go to results/NAME/code/STAMP.
# - Resumes: each seed resumes from its latest checkpoint; a finished seed (DONE_s<seed>) is skipped; evaluation is
#   skipped once results/NAME/eval/DONE exists.
# - Evaluation: twin, B2 seeds 0-39, deterministic, slot actions from the checkpoints, the 6 head pairs plus the old
#   H15 pairs (--pdr all,h15: per-instance hindsight best over both), MDD-PECT named as the best fixed pair (the B2
#   reference since 10-10, dev-pect-rule; MDD-TECT stays in the table), and the H15 switching oracle from
#   rq2-twin-fleet (B2 tasks).
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
PY="${PY:-.venv/bin/python}"
CODE="$OUT/code/env"
FLOOR=results/dev-popart-smoke/floor7/des_floor.json
PARAMS='{"failure_probability": 0.0, "due_date_allowance_range": [1.75, 2.5], "regime_block_seconds": 5400, "horizon_seconds": 30600, "load_mix": [{"name": "normal", "utilization": [0.6, 1.2], "weight": 2}, {"name": "surge", "utilization": [1.0, 1.6]}, {"name": "short", "utilization": [0.6, 1.2], "agv_count": 2}]}'
read -r C_LO C_HI < "$OUT/code/c_range" || { echo "missing $OUT/code/c_range"; exit 1; }
TRAIN_PARAMS="${PARAMS/\[1.75, 2.5\]/[$C_LO, $C_HI]}"
[[ "$TRAIN_PARAMS" == *"[$C_LO, $C_HI]"* ]] || { echo "c range substitution failed"; exit 1; }
echo "training c range [$C_LO, $C_HI]: $TRAIN_PARAMS"

for SEED in $SEEDS; do
    RUN="results/${NAME}_s$SEED"
    [ -f "$OUT/DONE_s$SEED" ] && continue
    (
        RESUME=()
        LATEST="$(ls -t "$RUN"/checkpoint.pt "$RUN"/checkpoint_step*.pt 2>/dev/null | head -1 || true)"
        [[ -n "$LATEST" ]] && RESUME=(--resume-from "$LATEST")
        $PY "$CODE/train.py" --twin "$FLOOR" --twin-workers "$WORKERS" \
            --scenario-generator randomized --random-warmup --episode-duration-seconds 21600 --params "$TRAIN_PARAMS" \
            --reward-spec env/config/rewards/tardiness.json --train-seed "$SEED" \
            --num-envs 16 --total-timesteps 100000 --slot-seconds 900 \
            --discount-horizon-s 5400 --ent-coef 0.01 --ent-coef-final 0.001 \
            --prior-kl-coef 0 --shared-baseline-group 4 --rollout-length 128 --batch-size 256 \
            --device "${DEVICE:-cuda}" --torch-threads 2 --save-every 25 --stall-minutes 30 --notify-every-hours 2 \
            --results-dir results --run-id "${NAME}_s$SEED" "${RESUME[@]}" "${EXTRA[@]}" \
            > "$OUT/train_s$SEED.out" 2>&1 \
        && grep -q "Checkpoint saved to" "$OUT/train_s$SEED.out" && date -Is > "$OUT/DONE_s$SEED"
    ) &
done
wait
for SEED in $SEEDS; do
    [ -f "$OUT/DONE_s$SEED" ] || { echo "seed $SEED did not finish (see $OUT/train_s$SEED.out)"; exit 1; }
done

if [ -f "$OUT/code/train_only" ]; then        # 10-15: per-seed jobs (small CPU requests) train only; their evaluation is a
    echo "train only (code/train_only): seeds $SEEDS done"   # separate experiment, so jobs never share eval/
    exit 0
fi
if [ ! -f "$OUT/eval/DONE" ]; then
    CK=()
    for SEED in $SEEDS; do CK+=(--checkpoint "results/${NAME}_s$SEED/checkpoint.pt"); done
    $PY "$CODE/evaluate.py" --twin "$FLOOR" --reward-spec env/config/rewards/tardiness.json \
        --scenario-generator randomized --random-warmup --episode-duration-seconds 21600 --params "$PARAMS" \
        --seeds 0-39 --pdr all,h15 --best-fixed MDD-PECT --oracle 'results/rq2-twin-fleet/tasks/B2_s*.json' \
        "${CK[@]}" --device "${DEVICE:-cuda}" --decision-log --out "$OUT/eval" > "$OUT/eval.out" 2>&1 \
    && date -Is > "$OUT/eval/DONE"
fi
[ -f "$OUT/eval/DONE" ] || { echo "evaluation failed (see $OUT/eval.out)"; exit 1; }
echo "eval done; analysis runs locally after the pull (dev_b2_slot_analyze.py needs pandas)"
