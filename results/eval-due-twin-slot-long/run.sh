#!/bin/bash
# eval-due-twin-slot-long (10-05): held-out evaluation of train-due-twin-slot-long (400k agent steps, 5 seeds) on the twin. The regime is rq2-twin-fleet's B2 (same
# generator parameters, seeds 0-39 held out from training's >= 10,000), whose task files already hold every H15 fixed
# pair and the switching oracle per seed (checked 10-05: evaluate.py --twin on this floor reproduces B2's fixed values
# exactly), so only the checkpoints run here, plus MDD-TECT / SRT-TECT as a consistency check. GPU inference.
# Usage (through the lab): bash results/eval-due-twin/run.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
P='{"failure_probability": 0.0, "due_date_allowance_range": [1.75, 2.5], "regime_block_seconds": 5400, "horizon_seconds": 30600, "load_mix": [{"name": "normal", "utilization": [0.6, 1.2], "weight": 2}, {"name": "surge", "utilization": [1.0, 1.6]}, {"name": "short", "utilization": [0.6, 1.2], "agv_count": 2}]}'
CK=()
for s in 0 1 2 3 4; do CK+=(--checkpoint "results/train-due-twin-slot-long_s$s/checkpoint.pt"); done
CK+=(--checkpoint "results/train-due-twin-slot-long_s3/checkpoint_init.pt")
.venv/bin/python env/evaluate.py --twin results/dev-popart-smoke/floor7/des_floor.json \
    --reward-spec env/config/rewards/tardiness.json --scenario-generator randomized --random-warmup \
    --episode-duration-seconds 21600 --params "$P" --seeds 0-39 --pdr "MDD-TECT,SRT-TECT" "${CK[@]}" \
    --device cuda --slot-seconds 900 --out results/eval-due-twin-slot-long
.venv/bin/python results/eval-due-twin-slot-long/analyze.py > results/eval-due-twin-slot-long/analysis.out 2>&1
cat results/eval-due-twin-slot-long/analysis.out
