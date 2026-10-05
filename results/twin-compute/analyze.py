"""
@file analyze.py
@brief twin-compute: training throughput and compute cost of Unity vs the event-based twin, CPU vs GPU (2026-10-04).

@details Reads results/twin-compute/<config>/timing.json written by env/train.py (pull them from the cluster first:
rsync rit-research:capstone/results/twin-compute/ results/twin-compute/ --include '*/' --include 'timing.json'
--exclude '*'). Per config: steps per second in the training loop (env + inference + update, startup excluded),
the share of each part, startup time, and the resources a million training steps take: CPU core-hours (cores
requested x wall time) and GPU-hours. With --cpu-price and --gpu-price ($ per core-hour, $ per GPU-hour, e.g. a
cloud provider's on-demand list price) it also prints a cost per million steps and per 3M-step training run.

Caveats to report with the table: the cluster's nodes differ in CPU generation (host column); a run is 10 PPO updates,
so startup is excluded from steps/s but listed; batch 64 / 4 epochs is the production PPO setting, not tuned for GPU.

@par Usage
@code{.sh}
.venv/bin/python results/twin-compute/analyze.py [--cpu-price 0.04 --gpu-price 1.5]
@endcode
"""
import argparse
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
CPUS = {"unity-cpu-e6": 18, "unity-cpu-e12": 36, "unity-cpu-e24": 36, "unity-gpu-e12": 24, "unity-gpu-e24": 36,
        "twin-cpu-e16": 36, "twin-cpu-e32": 36, "twin-gpu-e16": 24, "twin-gpu-e32": 36}   # requested (submit_bench.sh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu-price", type=float, default=None, help="$ per CPU core-hour")
    ap.add_argument("--gpu-price", type=float, default=None, help="$ per GPU-hour")
    args = ap.parse_args()
    rows = []
    for f in sorted(HERE.glob("*/timing.json")):
        t = json.loads(f.read_text())
        name = f.parent.name
        loop = t["loop_s"]
        cores = CPUS.get(name, t["cpus"])
        hours_per_m = 1e6 / t["sps_loop"] / 3600 if t["sps_loop"] else float("nan")
        r = {"config": name, "host": t["host"].split(".")[0], "gpu": "yes" if t["gpu"] else "",
             "envs": t["num_envs"], "cores": cores, "steps/s": t["sps_loop"],
             "env_%": 100 * t["env_s"] / loop, "infer_%": 100 * t["infer_s"] / loop,
             "update_%": 100 * t["update_s"] / loop, "startup_s": t["startup_s"],
             "wall_h_per_1M": hours_per_m, "core_h_per_1M": cores * hours_per_m,
             "gpu_h_per_1M": hours_per_m if t["gpu"] else 0.0}
        if args.cpu_price is not None and args.gpu_price is not None:
            r["$_per_1M"] = r["core_h_per_1M"] * args.cpu_price + r["gpu_h_per_1M"] * args.gpu_price
            r["$_per_3M_run"] = 3 * r["$_per_1M"]
        r["days_per_3M_run"] = 3 * hours_per_m / 24
        rows.append(r)
    if not rows:
        print("no timing.json yet")
        return
    d = pd.DataFrame(rows).set_index("config")
    pd.set_option("display.width", 220)
    print(d.round(2).to_string())
    d.to_csv(HERE / "summary.csv", float_format="%.4f")


if __name__ == "__main__":
    main()
