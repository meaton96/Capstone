#!/bin/bash
# FLOOR-SIZE VARIANT (2026-10-19, rq3-size-train): trains the final policy's stack (train-v5-cmix: frozen v5 code in
# results/NAME/code/env, c ~ U[c_range], no prior, no KL, shared baseline groups of 4, 900 s slots, 5,400 s horizon,
# 100k agent steps, entropy 0.01 -> 0.001) on the floor sizes in results/NAME/code/sizes (e.g. "50" or "15,50,100")
# through results/NAME/code/train_size.py. Train only (the evaluation is the cross-evaluation grid experiment).
# Run inside one cluster job by submit_dev_cluster.sh (TRAIN_SCRIPT=size_train_cluster.sh).
# Usage: bash results/scripts/size_train_cluster.sh NAME "SEEDS" WORKERS_PER_SEED [extra train.py flags...]
set -uo pipefail
cd "$(dirname "$0")/../.."
NAME="$1"; SEEDS="$2"; WORKERS="$3"; shift 3
EXTRA=("$@")
OUT="results/$NAME"
CODE="$OUT/code/env"
for f in "$CODE/train.py" "$OUT/code/train_size.py" "$OUT/code/sizes" "$OUT/code/c_range"; do
    [ -f "$f" ] || { echo "missing $f"; exit 1; }
done
PY="${PY:-.venv/bin/python}"
SIZES="$(tr -d ' \n' < "$OUT/code/sizes")"
read -r C_LO C_HI < "$OUT/code/c_range"
echo "training sizes $SIZES, c range [$C_LO, $C_HI]"
for SEED in $SEEDS; do
    RUN="results/${NAME}_s$SEED"
    [ -f "$OUT/DONE_s$SEED" ] && continue
    (
        RESUME=()
        LATEST="$(ls -t "$RUN"/checkpoint.pt "$RUN"/checkpoint_step*.pt 2>/dev/null | head -1 || true)"
        [[ -n "$LATEST" ]] && RESUME=(--resume-from "$LATEST")
        $PY "$OUT/code/train_size.py" --sizes "$SIZES" --c-range "$C_LO,$C_HI" --group 4 -- \
            --twin-workers "$WORKERS" --scenario-generator randomized --random-warmup --episode-duration-seconds 21600 \
            --reward-spec "$CODE/config/rewards/tardiness.json" --train-seed "$SEED" \
            --num-envs 16 --total-timesteps 100000 --slot-seconds 900 \
            --discount-horizon-s 5400 --ent-coef 0.01 --ent-coef-final 0.001 \
            --prior-kl-coef 0 --shared-baseline-group 4 --rollout-length 128 --batch-size 256 \
            --device "${DEVICE:-cpu}" --torch-threads 2 --save-every 25 --stall-minutes 60 --notify-every-hours 2 \
            --results-dir results --run-id "${NAME}_s$SEED" "${RESUME[@]}" "${EXTRA[@]}" \
            > "$OUT/train_s$SEED.out" 2>&1 \
        && grep -q "Checkpoint saved to" "$OUT/train_s$SEED.out" && date -Is > "$OUT/DONE_s$SEED"
    ) &
done
wait
for SEED in $SEEDS; do
    [ -f "$OUT/DONE_s$SEED" ] || { echo "seed $SEED did not finish (see $OUT/train_s$SEED.out)"; exit 1; }
done
echo "train only: seeds $SEEDS done"
