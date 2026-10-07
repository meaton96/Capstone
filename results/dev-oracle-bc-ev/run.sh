#!/bin/bash
# dev-oracle-bc-ev, local step (2026-10-07, handoff fix 7): behavior cloning on the expected-value oracle labels
# (../dev-oracle-bc/bc_train.py --labels ev and ev-safe, 3 BC seeds each), then the cloned full networks rolled out as
# slot policies on B2 test seeds 0-39 with the three-reference table (best fixed MDD-TECT, per-instance hindsight,
# oracle). Compare with ../dev-oracle-bc (single-future labels). Every step skips finished outputs.
# Usage (through the lab, after the cluster collection is pulled): bash results/dev-oracle-bc-ev/run.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
D=results/dev-oracle-bc-ev
P='{"failure_probability": 0.0, "due_date_allowance_range": [1.75, 2.5], "regime_block_seconds": 5400, "horizon_seconds": 30600, "load_mix": [{"name": "normal", "utilization": [0.6, 1.2], "weight": 2}, {"name": "surge", "utilization": [1.0, 1.6]}, {"name": "short", "utilization": [0.6, 1.2], "agv_count": 2}]}'
[ "$(ls $D/data/s*.npz 2>/dev/null | wc -l)" -ge 200 ] || { echo "collection incomplete"; exit 1; }
CK=()
for L in ev ev-safe; do
    [ -f $D/bc_scores_$L.csv ] || .venv/bin/python results/dev-oracle-bc/bc_train.py --data-dir $D/data --out-dir $D \
        --labels $L --tag "_$L" --bc-seeds 0,1,2 --device cuda > $D/bc_train_$L.out 2>&1 \
        || { echo "bc_train $L failed (see $D/bc_train_$L.out)"; exit 1; }
    for s in 0 1 2; do CK+=(--checkpoint "$D/bc_full_s${s}_$L.pt"); done
done
[ -f $D/rollout/DONE ] || { .venv/bin/python env/evaluate.py --twin results/dev-popart-smoke/floor7/des_floor.json \
    --reward-spec env/config/rewards/tardiness.json --scenario-generator randomized --random-warmup \
    --episode-duration-seconds 21600 --params "$P" --seeds 0-39 --pdr all --best-fixed MDD-TECT \
    --oracle 'results/rq2-twin-fleet/tasks/B2_s*.json' "${CK[@]}" --device cuda --slot-seconds 900 --decision-log \
    --out $D/rollout > $D/rollout.out 2>&1 && date -Is > $D/rollout/DONE; }
[ -f $D/rollout/DONE ] || { echo "rollout failed (see $D/rollout.out)"; exit 1; }
{ for L in ev ev-safe; do echo "== bc_train --labels $L (per slot, test seeds 0-39)"; sed -n '/^labels\|== test seeds/,$p' $D/bc_train_$L.out; echo; done
  echo "== rollout (three references)"; sed -n '/vs best/,/Change in mean cost/p' $D/rollout.out; } > $D/analysis.out
cat $D/analysis.out
