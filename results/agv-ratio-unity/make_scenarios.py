"""
@file make_scenarios.py
@brief agv-ratio-unity (v5 design, 2026-10-14): the Unity batch scenarios for the fleet-knee check, the same instances
       as ../agv-ratio-twin-v5 (built exactly as its twin_point_v5.py builds them: randomized generator, machines per
       type = M / 5, agv_count = M bays, arrival cap off, failures off, 6 h of arrivals, fixed utilisation and mean
       op, TWK due dates c ~ U[1.75, 2.5]), and the fleets to run: the twin's knee5 under MDD-PTECT (mean over the
       3 seeds, rounded up to a fleet the twin ran) and the twin fleet nearest 1.5x it.
       Writes linux_server/BatchConfigs/Scenarios/agvratio5_m<M>_u<U>_op<O>_s<seed>.json and cells.txt
       ("M U OP SEED AGVS" per line, longest first).
@par Usage
@code{.sh}
.venv/bin/python results/agv-ratio-unity/make_scenarios.py
@endcode
"""
import json
import math
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "env"))
from scenarios.randomized import RandomizedParams, randomized_scenario  # noqa: E402

OUT = REPO / "linux_server" / "BatchConfigs" / "Scenarios"
TWIN = REPO / "results" / "agv-ratio-twin-v5"
LOADS = [(100, 0.85, 240), (100, 0.85, 480), (50, 0.85, 240), (50, 0.85, 480)]
SEEDS = [0, 1]


def fleets(m, u, op):
    k = pd.read_csv(TWIN / "knees.csv")
    k = k[(k.m == m) & (k.util == u) & (k.op == op) & (k.rule == "MDD_PTECT")]
    ran = sorted(pd.read_csv(TWIN / f"m{m}_u{u}_op{op}.csv").agv_count.unique())
    knee = min(n for n in ran if n >= math.ceil(k.knee5.mean()))
    big = min(ran, key=lambda n: (abs(n - 1.5 * knee), -n))
    if big <= knee:
        big = min(n for n in ran if n > knee)
    return int(knee), int(big)


def main():
    lines = []
    for m, u, op in LOADS:
        knee, big = fleets(m, u, op)
        for seed in SEEDS:
            p = RandomizedParams(machines_per_type=m // 5, agv_count=m, agv_max_utilization=10.0,
                                 failure_probability=0.0, horizon_seconds=21600.0, utilization=(u, u),
                                 op_mean_seconds=(float(op), float(op)), due_date_allowance_range=(1.75, 2.5))
            s = randomized_scenario(seed, p)
            s["name"] = f"agvratio5_m{m}_u{u}_op{op}_s{seed}"
            (OUT / f"{s['name']}.json").write_text(json.dumps(s))
            print(f"{s['name']}: {len(s['jobs'])} jobs, fleets {knee} (twin knee5) and {big}")
            lines += [f"{m} {u} {op} {seed} {big}", f"{m} {u} {op} {seed} {knee}"]
    (HERE / "cells.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
