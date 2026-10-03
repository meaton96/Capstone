#!/bin/bash
# 12 head-pair PDR baselines on held-out seeds 0-19 via evaluate.py (the path rnd02 checkpoints are scored on).
# 4 workers x 5 seeds, player linux_server_dev (two-branch build, 2026-09-26 17:20).
cd "$(dirname "$0")/../.."
OUT=results/eval_pdr12_heads
for i in 0 1 2 3; do
  lo=$(( i * 5 )); hi=$(( lo + 4 ))
  .venv/bin/python env/evaluate.py --unity-path linux_server_dev/capstone.x86_64 --no-graphics \
      --seeds "$lo-$hi" --pdr all --scenario-generator randomized --episode-duration-seconds 5400 \
      --reward-spec env/config/rewards/flow_time.json \
      --base-worker-id $(( 400 + i * 10 )) --out "$OUT/part$i" > "$OUT/part$i.log" 2>&1 &
done
wait
echo "all parts done"
