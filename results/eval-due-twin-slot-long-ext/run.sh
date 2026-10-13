#!/bin/bash
# eval-due-twin-slot-long-ext (10-06): replication set for the bad-instance workup of the 400k slot policies. Seeds
# 1000-1199 of the B2 regime (outside the held-out 0-39 and training's >= 10,000; used only for this analysis, so a
# pattern found on seeds 0-39 can be checked out of sample). Best two checkpoints (s0, s3) + MDD-TECT and ATC-ECT,
# with decision logs. No oracle on these seeds. Writes DONE when finished.
# Usage (through the lab): bash results/eval-due-twin-slot-long-ext/run.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=results/eval-due-twin-slot-long-ext
P='{"failure_probability": 0.0, "due_date_allowance_range": [1.75, 2.5], "regime_block_seconds": 5400, "horizon_seconds": 30600, "load_mix": [{"name": "normal", "utilization": [0.6, 1.2], "weight": 2}, {"name": "surge", "utilization": [1.0, 1.6]}, {"name": "short", "utilization": [0.6, 1.2], "agv_count": 2}]}'
[ -f "$OUT/DONE" ] && { echo "already done"; exit 0; }
.venv/bin/python env/evaluate.py --twin results/dev-popart-smoke/floor7/des_floor.json \
    --reward-spec env/config/rewards/tardiness.json --scenario-generator randomized --random-warmup \
    --episode-duration-seconds 21600 --params "$P" --seeds 1000-1199 --pdr "MDD-TECT,ATC-ECT" \
    --checkpoint results/train-due-twin-slot-long_s0/checkpoint.pt --checkpoint results/train-due-twin-slot-long_s3/checkpoint.pt \
    --device cuda --slot-seconds 900 --decision-log --out "$OUT" && date -Is > "$OUT/DONE"
