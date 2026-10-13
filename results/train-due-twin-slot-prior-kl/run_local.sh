#!/bin/bash
# train-due-twin-slot-prior-kl (10-06, branch action-prior): train-due-twin-slot (slot actions 900 s, 10,800 s horizon,
# B2 regime, 100k agent steps, same entropy schedule) plus an action prior on MDD-TECT: the policy starts at p0 (0.8 on
# MDD and 0.8 on TECT per head) and the loss adds 0.05 * KL(pi || p0). Why: the 400k slot policies lose where MDD-TECT
# is right and ATC is not (results/eval-due-twin-slot-long/workup), and obs v3 already carries the congestion signal
# (results/dev-congestion-signal), so the fix tried is on the learning side. Seeds 0-4 in parallel on the local GPU;
# baseline seeds 0-2 are train-due-twin-slot, 3-4 train-due-twin-slot-ext. Each run resumes from its latest checkpoint.
set -uo pipefail
cd "$(dirname "$0")/../.."
PARAMS='{"failure_probability": 0.0, "due_date_allowance_range": [1.75, 2.5], "regime_block_seconds": 5400, "horizon_seconds": 30600, "load_mix": [{"name": "normal", "utilization": [0.6, 1.2], "weight": 2}, {"name": "surge", "utilization": [1.0, 1.6]}, {"name": "short", "utilization": [0.6, 1.2], "agv_count": 2}]}'
for SEED in 0 1 2 3 4; do
    RUN="results/train-due-twin-slot-prior-kl_s$SEED"
    RESUME=()
    LATEST="$(ls -t "$RUN"/checkpoint.pt "$RUN"/checkpoint_step*.pt 2>/dev/null | head -1 || true)"
    [[ -n "$LATEST" ]] && RESUME=(--resume-from "$LATEST")
    nice -n 5 .venv/bin/python env/train.py --twin results/dev-popart-smoke/floor7/des_floor.json --twin-workers 5 \
        --scenario-generator randomized --random-warmup --episode-duration-seconds 21600 --params "$PARAMS" \
        --reward-spec env/config/rewards/tardiness.json --train-seed "$SEED" \
        --num-envs 16 --rollout-length 32 --batch-size 64 --total-timesteps 100000 --slot-seconds 900 --discount-horizon-s 10800 \
        --ent-coef 0.01 --ent-coef-final 0.001 --prior-pair MDD-TECT --prior-prob 0.8 --prior-kl-coef 0.05 \
        --device cuda --torch-threads 2 --save-every 25 \
        --stall-minutes 30 --notify-every-hours 2 --results-dir results --run-id "train-due-twin-slot-prior-kl_s$SEED" \
        "${RESUME[@]}" > "results/train-due-twin-slot-prior-kl/train_s$SEED.out" 2>&1 &
done
wait
echo "[$(date '+%m-%d %H:%M')] all seeds done"
