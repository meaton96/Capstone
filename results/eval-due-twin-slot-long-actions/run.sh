#!/bin/bash
# eval-due-twin-slot-long-actions (10-06): eval-due-twin-slot-long rerun with --decision-log, so the bad-instance
# workup can see each 400k slot policy's pair per 900 s slot. Same seeds / regime / floor as eval-due-twin-slot-long
# (deterministic, so tardiness must match it exactly). Writes DONE when finished.
# Usage (through the lab): bash results/eval-due-twin-slot-long-actions/run.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=results/eval-due-twin-slot-long-actions
P='{"failure_probability": 0.0, "due_date_allowance_range": [1.75, 2.5], "regime_block_seconds": 5400, "horizon_seconds": 30600, "load_mix": [{"name": "normal", "utilization": [0.6, 1.2], "weight": 2}, {"name": "surge", "utilization": [1.0, 1.6]}, {"name": "short", "utilization": [0.6, 1.2], "agv_count": 2}]}'
[ -f "$OUT/DONE" ] && { echo "already done"; exit 0; }
CK=()
for s in 0 1 2 3 4; do CK+=(--checkpoint "results/train-due-twin-slot-long_s$s/checkpoint.pt"); done
.venv/bin/python env/evaluate.py --twin results/dev-popart-smoke/floor7/des_floor.json \
    --reward-spec env/config/rewards/tardiness.json --scenario-generator randomized --random-warmup \
    --episode-duration-seconds 21600 --params "$P" --seeds 0-39 --pdr "MDD-TECT" "${CK[@]}" \
    --device cuda --slot-seconds 900 --decision-log --out "$OUT" && date -Is > "$OUT/DONE"
