#!/bin/bash
# train-due-twin, local variant (10-05: the cluster could not start it before 10-06 09:20; user chose the local GPU).
# 3 seeds in parallel on this machine's GPU, 16 twin envs each over 5 worker processes (the Unity rq2-oracle-due-long
# run shares the CPU), same regime and PPO settings as submit.sh. Each seed resumes from its latest checkpoint.
# Usage (through the lab): bash results/train-due-twin/run_local.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
PARAMS='{"failure_probability": 0.0, "due_date_allowance_range": [1.75, 2.5], "regime_block_seconds": 5400, "horizon_seconds": 30600, "load_mix": [{"name": "normal", "utilization": [0.6, 1.2], "weight": 2}, {"name": "surge", "utilization": [1.0, 1.6]}, {"name": "short", "utilization": [0.6, 1.2], "agv_count": 2}]}'
for SEED in 0 1 2; do
    RUN="results/train-due-twin_s$SEED"
    RESUME=()
    LATEST="$(ls -t "$RUN"/checkpoint.pt "$RUN"/checkpoint_step*.pt 2>/dev/null | head -1 || true)"
    [[ -n "$LATEST" ]] && RESUME=(--resume-from "$LATEST")
    nice -n 5 .venv/bin/python env/train.py --twin results/dev-popart-smoke/floor7/des_floor.json --twin-workers 5 \
        --scenario-generator randomized --random-warmup --episode-duration-seconds 21600 --params "$PARAMS" \
        --reward-spec env/config/rewards/tardiness.json --train-seed "$SEED" \
        --num-envs 16 --rollout-length 128 --batch-size 64 --total-timesteps 3000000 \
        --ent-coef 0.01 --ent-coef-final 0.001 --device cuda --torch-threads 2 --save-every 100 \
        --stall-minutes 30 --notify-every-hours 2 --results-dir results --run-id "train-due-twin_s$SEED" \
        "${RESUME[@]}" > "results/train-due-twin/train_s$SEED.out" 2>&1 &
done
wait
echo "[$(date '+%m-%d %H:%M')] all seeds done"
