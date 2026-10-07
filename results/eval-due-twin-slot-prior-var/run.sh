#!/bin/bash
# eval-due-twin-slot-prior-var (10-06, branch action-prior): held-out twin evaluation (B2 seeds 0-39, deterministic, slot
# actions) of the KL variants train-due-twin-slot-prior-kl01 (KL 0.01) and -kldecay (KL 0.05 -> 0), 5 seeds each.
# The other arms (baseline, KL 0.05) come from ../eval-due-twin-slot-prior and ../eval-due-twin-slot (deterministic, reused).
# MDD-TECT as the consistency check. Usage (through the lab): bash results/eval-due-twin-slot-prior-var/run.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=results/eval-due-twin-slot-prior-var
P='{"failure_probability": 0.0, "due_date_allowance_range": [1.75, 2.5], "regime_block_seconds": 5400, "horizon_seconds": 30600, "load_mix": [{"name": "normal", "utilization": [0.6, 1.2], "weight": 2}, {"name": "surge", "utilization": [1.0, 1.6]}, {"name": "short", "utilization": [0.6, 1.2], "agv_count": 2}]}'
CK=()
for v in kl01 kldecay; do for s in 0 1 2 3 4; do CK+=(--checkpoint "results/train-due-twin-slot-prior-${v}_s$s/checkpoint.pt"); done; done
if [ ! -f "$OUT/DONE" ]; then
    .venv/bin/python env/evaluate.py --twin results/dev-popart-smoke/floor7/des_floor.json \
        --reward-spec env/config/rewards/tardiness.json --scenario-generator randomized --random-warmup \
        --episode-duration-seconds 21600 --params "$P" --seeds 0-39 --pdr "MDD-TECT" "${CK[@]}" \
        --device cuda --slot-seconds 900 --decision-log --out "$OUT" && date -Is > "$OUT/DONE"
fi
.venv/bin/python $OUT/analyze.py > $OUT/analysis.out 2>&1
cat $OUT/analysis.out
