#!/bin/bash
# train-due-twin-slot-ext (10-06): baseline seeds 3-4 of train-due-twin-slot (identical command, run dirs
# results/train-due-twin-slot_s3, _s4), so the action-prior comparison has 5 seeds per arm at 100k agent steps.
# Run on branch action-prior with no --prior-pair: the code path is the old one (tests: test_action_prior.py).
set -uo pipefail
cd "$(dirname "$0")/../.."
PARAMS='{"failure_probability": 0.0, "due_date_allowance_range": [1.75, 2.5], "regime_block_seconds": 5400, "horizon_seconds": 30600, "load_mix": [{"name": "normal", "utilization": [0.6, 1.2], "weight": 2}, {"name": "surge", "utilization": [1.0, 1.6]}, {"name": "short", "utilization": [0.6, 1.2], "agv_count": 2}]}'
for SEED in 3 4; do
    RUN="results/train-due-twin-slot_s$SEED"
    RESUME=()
    LATEST="$(ls -t "$RUN"/checkpoint.pt "$RUN"/checkpoint_step*.pt 2>/dev/null | head -1 || true)"
    [[ -n "$LATEST" ]] && RESUME=(--resume-from "$LATEST")
    nice -n 5 .venv/bin/python env/train.py --twin results/dev-popart-smoke/floor7/des_floor.json --twin-workers 5 \
        --scenario-generator randomized --random-warmup --episode-duration-seconds 21600 --params "$PARAMS" \
        --reward-spec env/config/rewards/tardiness.json --train-seed "$SEED" \
        --num-envs 16 --rollout-length 32 --batch-size 64 --total-timesteps 100000 --slot-seconds 900 --discount-horizon-s 10800 \
        --ent-coef 0.01 --ent-coef-final 0.001 \
        --device cuda --torch-threads 2 --save-every 25 \
        --stall-minutes 30 --notify-every-hours 2 --results-dir results --run-id "train-due-twin-slot_s$SEED" \
        "${RESUME[@]}" > "results/train-due-twin-slot-ext/train_s$SEED.out" 2>&1 &
done
wait
echo "[$(date '+%m-%d %H:%M')] all seeds done"
