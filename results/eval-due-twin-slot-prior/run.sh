#!/bin/bash
# eval-due-twin-slot-prior (10-06, branch action-prior): held-out twin evaluation (B2 seeds 0-39, deterministic, slot
# actions) of train-due-twin-slot-prior-kl s0-4 (+ its untrained init, which should play MDD-TECT exactly) and the
# baseline seeds 3-4 (train-due-twin-slot-ext); baseline seeds 0-2 were evaluated in ../eval-due-twin-slot (the twin is
# deterministic, so they are reused). MDD-TECT / SRT-TECT as the consistency check against rq2-twin-fleet B2.
# Decision logs on, for the slot choices. Usage (through the lab): bash results/eval-due-twin-slot-prior/run.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=results/eval-due-twin-slot-prior
P='{"failure_probability": 0.0, "due_date_allowance_range": [1.75, 2.5], "regime_block_seconds": 5400, "horizon_seconds": 30600, "load_mix": [{"name": "normal", "utilization": [0.6, 1.2], "weight": 2}, {"name": "surge", "utilization": [1.0, 1.6]}, {"name": "short", "utilization": [0.6, 1.2], "agv_count": 2}]}'
CK=()
for s in 0 1 2 3 4; do CK+=(--checkpoint "results/train-due-twin-slot-prior-kl_s$s/checkpoint.pt"); done
CK+=(--checkpoint "results/train-due-twin-slot-prior-kl_s0/checkpoint_init.pt")
for s in 3 4; do CK+=(--checkpoint "results/train-due-twin-slot_s$s/checkpoint.pt"); done
if [ ! -f "$OUT/DONE" ]; then
    .venv/bin/python env/evaluate.py --twin results/dev-popart-smoke/floor7/des_floor.json \
        --reward-spec env/config/rewards/tardiness.json --scenario-generator randomized --random-warmup \
        --episode-duration-seconds 21600 --params "$P" --seeds 0-39 --pdr "MDD-TECT,SRT-TECT" "${CK[@]}" \
        --device cuda --slot-seconds 900 --decision-log --out "$OUT" && date -Is > "$OUT/DONE"
fi
.venv/bin/python $OUT/analyze.py > $OUT/analysis.out 2>&1
cat $OUT/analysis.out
