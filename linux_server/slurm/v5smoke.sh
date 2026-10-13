#!/bin/bash -l
set -euo pipefail
cd $HOME/capstone
unset PYTHONPATH
source .venv/bin/activate
export OMP_NUM_THREADS=1
O=results/dev-cluster-v5-smoke
mkdir -p $O
python env/evaluate.py --twin results/dev-popart-smoke/floor7/des_floor.json --reward-spec env/config/rewards/tardiness.json --scenario-generator randomized --random-warmup --episode-duration-seconds 21600 --params '{"failure_probability": 0.0, "due_date_allowance_range": [1.75, 2.5], "regime_block_seconds": 5400, "horizon_seconds": 30600, "load_mix": [{"name": "normal", "utilization": [0.6, 1.2], "weight": 2}, {"name": "surge", "utilization": [1.0, 1.6]}, {"name": "short", "utilization": [0.6, 1.2], "agv_count": 2}]}' --seeds 0 --pdr MDD-PECT,MDD-TECT --slot-seconds 900 --discount-horizon-s 10800 --device cpu --out $O/twin > $O/twin.out 2>&1
python env/evaluate.py --unity-path linux_server/capstone.x86_64 --no-graphics --reward-spec env/config/rewards/tardiness.json --scenario-generator randomized --random-warmup --episode-duration-seconds 21600 --params '{"failure_probability": 0.0, "due_date_allowance_range": [1.75, 2.5], "regime_block_seconds": 5400, "horizon_seconds": 30600, "load_mix": [{"name": "normal", "utilization": [0.6, 1.2], "weight": 2}, {"name": "surge", "utilization": [1.0, 1.6]}, {"name": "short", "utilization": [0.6, 1.2], "agv_count": 2}]}' --seeds 0 --pdr MDD-PECT,MDD-TECT --slot-seconds 900 --discount-horizon-s 10800 --device cpu --base-worker-id 7100 --out $O/unity > $O/unity.out 2>&1
echo done
