#!/bin/bash
# Seeds 15-19 split in two: queuing all 60 generated scenarios at once exceeded gRPC's 4 MB message cap.
cd "$(dirname "$0")/../.."
OUT=results/eval_pdr12_heads
for part in "3a 15-17" "3b 18-19"; do
  set -- $part
  .venv/bin/python env/evaluate.py --unity-path linux_server_dev/capstone.x86_64 --no-graphics \
      --seeds "$2" --pdr all --scenario-generator randomized --episode-duration-seconds 5400 \
      --reward-spec env/config/rewards/flow_time.json \
      --base-worker-id 430 --out "$OUT/part$1" > "$OUT/part$1.log" 2>&1
done
echo "part3 done"
