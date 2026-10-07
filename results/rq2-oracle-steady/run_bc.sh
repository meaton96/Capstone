#!/bin/bash
# rq2-oracle-steady, local step (10-07): behavior cloning on the S6 oracle data (../dev-oracle-bc/bc_train.py: full
# network vs scalars-only MLP, 3 BC seeds, GPU), then the cloned full networks rolled out as slot policies on the S6
# test seeds 0-39 (run.py --mode rollout, 12 processes), then the comparison with the fixed pairs, the observable
# policies and the oracle (../rq2-realizable/analyze.py). Every step skips finished outputs and stops on failure.
# Usage (through the lab, after rq2-oracle-steady): bash results/rq2-oracle-steady/run_bc.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
D=results/rq2-oracle-steady
mkdir -p $D/bc
[ -f $D/bc/bc_scores.csv ] || .venv/bin/python results/dev-oracle-bc/bc_train.py --data-dir $D/data --out-dir $D/bc \
    --bc-seeds 0,1,2 --device cuda > $D/bc/bc_train.out 2>&1 || { echo "bc_train failed (see $D/bc/bc_train.out)"; exit 1; }
CKS="$D/bc/bc_full_s0.pt,$D/bc/bc_full_s1.pt,$D/bc/bc_full_s2.pt"
seq 0 39 | xargs -P 12 -I{} .venv/bin/python $D/run.py --mode rollout --seed {} --setting S6 --checkpoints "$CKS" > $D/rollout.out 2>&1
[ "$(ls $D/rollout/s*.json 2>/dev/null | wc -l)" -eq 40 ] || { echo "rollout incomplete (see $D/rollout.out)"; exit 1; }
{ echo "== bc_train (S6, per slot, test seeds 0-39)"; sed -n '/== test seeds/,$p' $D/bc/bc_train.out; echo
  echo "== realizable vs hindsight (incl. the S6 cloned policies)"; .venv/bin/python results/rq2-realizable/analyze.py; } > $D/analysis.out 2>&1
cat $D/analysis.out
