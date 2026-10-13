#!/bin/bash
# dev-oracle-bc (10-06, branch action-prior): can the policy network imitate the switching oracle from obs v3?
#   1. collect.py: oracle with all 15 tails per stage + full obs at each slot start, B2 seeds 0-39 (test) and 2000-2159
#      (train); ~12 CPU-min per seed (361 twin episodes)
#   2. bc_train.py: supervised imitation (full network vs scalars-only MLP), 3 BC seeds, GPU
#   3. evaluate.py: the cloned full networks run as slot policies on B2 0-39 (deterministic) vs MDD-TECT / the oracle
# Every step skips finished outputs, so a relaunch resumes. Usage (through the lab): bash results/dev-oracle-bc/run.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
D=results/dev-oracle-bc
P='{"failure_probability": 0.0, "due_date_allowance_range": [1.75, 2.5], "regime_block_seconds": 5400, "horizon_seconds": 30600, "load_mix": [{"name": "normal", "utilization": [0.6, 1.2], "weight": 2}, {"name": "surge", "utilization": [1.0, 1.6]}, {"name": "short", "utilization": [0.6, 1.2], "agv_count": 2}]}'
[ -f $D/DONE_collect ] || nice -n 10 .venv/bin/python $D/collect.py --seeds 0-39,2000-2159 --workers 16 >> $D/collect.out 2>&1
[ -f $D/DONE_collect ] || { echo "collection incomplete"; exit 1; }
[ -f $D/bc_scores.csv ] || .venv/bin/python $D/bc_train.py --bc-seeds 0,1,2 --device cuda > $D/bc_train.out 2>&1 || { echo "bc_train failed (see bc_train.out)"; exit 1; }
CK=(); for s in 0 1 2; do CK+=(--checkpoint "$D/bc_full_s$s.pt"); done
[ -f $D/rollout/DONE ] || { .venv/bin/python env/evaluate.py --twin results/dev-popart-smoke/floor7/des_floor.json \
    --reward-spec env/config/rewards/tardiness.json --scenario-generator randomized --random-warmup \
    --episode-duration-seconds 21600 --params "$P" --seeds 0-39 --pdr "MDD-TECT" "${CK[@]}" \
    --device cuda --slot-seconds 900 --decision-log --out $D/rollout && date -Is > $D/rollout/DONE; }
[ -f $D/rollout/DONE ] || { echo "rollout failed"; exit 1; }
.venv/bin/python $D/rollout/analyze.py > $D/rollout/analysis.out 2>&1
{ echo "== bc_train (per-slot imitation, test seeds 0-39)"; sed -n '/== test seeds/,$p' $D/bc_train.out; echo; echo "== rollout"; cat $D/rollout/analysis.out; } > $D/analysis.out
cat $D/analysis.out
