#!/bin/bash
# train-due-twin-slot-long (10-05, user: train longer after eval-due-twin-slot reached parity with MDD-TECT): slot actions
# (900 s) + 10,800 s horizon, B2 regime, to 400k agent steps. Seeds 0-2 continue from train-due-twin-slot_s<seed>'s 100k
# checkpoint (new run dirs; the entropy schedule decays over the new 400k total, so it rises again at the start);
# seeds 3-4 start fresh. 5 runs in parallel on the local GPU, 16 twin envs / 5 workers each. Each run resumes from its
# own latest checkpoint when relaunched.
set -uo pipefail
cd "$(dirname "$0")/../.."
PARAMS='{"failure_probability": 0.0, "due_date_allowance_range": [1.75, 2.5], "regime_block_seconds": 5400, "horizon_seconds": 30600, "load_mix": [{"name": "normal", "utilization": [0.6, 1.2], "weight": 2}, {"name": "surge", "utilization": [1.0, 1.6]}, {"name": "short", "utilization": [0.6, 1.2], "agv_count": 2}]}'
for SEED in 0 1 2 3 4; do
    RUN="results/train-due-twin-slot-long_s$SEED"
    LATEST="$(ls -t "$RUN"/checkpoint.pt "$RUN"/checkpoint_step*.pt 2>/dev/null | head -1 || true)"
    if [[ -z "$LATEST" && $SEED -le 2 ]]; then LATEST="results/train-due-twin-slot_s$SEED/checkpoint.pt"; fi
    RESUME=()
    [[ -n "$LATEST" ]] && RESUME=(--resume-from "$LATEST")
    nice -n 5 .venv/bin/python env/train.py --twin results/dev-popart-smoke/floor7/des_floor.json --twin-workers 5 \
        --scenario-generator randomized --random-warmup --episode-duration-seconds 21600 --params "$PARAMS" \
        --reward-spec env/config/rewards/tardiness.json --train-seed "$SEED" \
        --num-envs 16 --rollout-length 32 --batch-size 64 --total-timesteps 400000 --slot-seconds 900 \
        --discount-horizon-s 10800 --ent-coef 0.01 --ent-coef-final 0.001 --device cuda --torch-threads 2 \
        --save-every 50 --stall-minutes 30 --notify-every-hours 2 --results-dir results \
        --run-id "train-due-twin-slot-long_s$SEED" "${RESUME[@]}" > "results/train-due-twin-slot-long/train_s$SEED.out" 2>&1 &
done
wait
echo "[$(date '+%m-%d %H:%M')] all seeds done"
