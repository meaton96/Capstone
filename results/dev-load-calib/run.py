"""
@file run.py
@brief dev-load-calib: which load keeps a 6 h episode stationary (tardiness per hour levels off), and which fleet size
       makes transport bind in the twin? Fixed rules only, cheap (user asked 10-04: calibrate the load, then fleet blocks).

@details
Twin (DES-1k, kinematic, G1 floor, 7 bays), one regime per episode, 21,600 s agent window after a random warm-up,
c ~ U[1.75, 2.5], failures off, seeds 0-19, rules MDD-TECT and SRT-TECT. Configs vary the non-lull utilization range
and the fleet. Per run: tardiness per hour of the window (1000 job-s), share of exited jobs late, jobs in system at the
end, AGV busy fraction.

@par Usage
@code{.sh}
nice -n 19 .venv/bin/python results/dev-load-calib/run.py > results/dev-load-calib/run.out
@endcode
"""
import dataclasses
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "env"))

FLOOR = REPO / "linux_server_des/Results/G1/rnd_load_s0/D/agv7_SPT_ECT_s0/des_floor.json"
WINDOW = 21600.0
CONFIGS = {   # name: (utilization range outside lulls, fleet)
    "u0.70-1.40_a7": ((0.70, 1.40), 7),
    "u0.60-1.20_a7": ((0.60, 1.20), 7),
    "u0.55-1.10_a7": ((0.55, 1.10), 7),
    "u0.50-1.00_a7": ((0.50, 1.00), 7),
    "u0.60-1.20_a3": ((0.60, 1.20), 3),
    "u0.60-1.20_a2": ((0.60, 1.20), 2),
}
RULES = ("MDD_TECT", "SRT_TECT")
_floor = None


def floor():
    global _floor
    if _floor is None:
        from des_twin import Floor
        _floor = Floor.load(str(FLOOR))
    return _floor


def run(task):
    name, seed, rule = task
    from des_twin import TwinConfig, run_twin
    from des_twin.scenario import episode_settings, resolve_jobs
    from scenarios.randomized import DEFAULT_PARAMS, randomized_generator
    util, fleet = CONFIGS[name]
    params = dataclasses.replace(DEFAULT_PARAMS, failure_probability=0.0, due_date_allowance_range=(1.75, 2.5),
                                 utilization=util, horizon_seconds=30600.0)
    sc = randomized_generator(WINDOW, random_warmup=True, params=params)(seed)
    warmup, _, _ = episode_settings(sc)
    jobs = resolve_jobs(sc, floor())
    tw = run_twin(floor(), jobs, TwinConfig(rule=rule, transport="kinematic", agv_count=fleet, warmup_seconds=warmup,
                                            episode_duration_seconds=WINDOW, max_sim_seconds=200000.0))
    t0, t1 = warmup, tw.now
    per_h = [0.0] * 6
    late = exited = 0
    for j in tw.jobs.values():
        end = j.exit_time if j.exit_time is not None else t1
        for h in range(6):
            a = t0 + 3600 * h
            per_h[h] += max(0.0, min(end, a + 3600, t1) - max(j.due, a)) if j.due is not None else 0.0
        if j.exit_time is not None and t0 <= j.exit_time <= t1:
            exited += 1
            late += j.due is not None and j.exit_time > j.due
    open_end = sum(1 for j in tw.jobs.values() if j.arrival <= t1 and (j.exit_time is None or j.exit_time > t1))
    s = tw.summary()
    return {"config": name, "seed": seed, "rule": rule, "per_h": [x / 1000 for x in per_h],
            "late_share": late / max(exited, 1), "wip_end": open_end, "agv_busy": s.get("agv_busy_fraction", 0.0)}


def main():
    floor()
    tasks = [(c, s, r) for c in CONFIGS for s in range(20) for r in RULES]
    with Pool(16) as pool:
        rows = pool.map(run, tasks)
    (HERE / "runs.json").write_text(json.dumps(rows))
    print(f"{'config':16s} {'rule':9s} {'tard/h h1..h6 (1000 job-s, mean)':44s} h6/h1  late%  WIP_end  AGV busy")
    for c in CONFIGS:
        for r in RULES:
            g = [x for x in rows if x["config"] == c and x["rule"] == r]
            ph = np.mean([x["per_h"] for x in g], axis=0)
            print(f"{c:16s} {r:9s} {' '.join(f'{v:6.1f}' for v in ph):44s} {ph[5] / max(ph[0], 1e-9):5.2f} "
                  f"{100 * np.mean([x['late_share'] for x in g]):5.1f} {np.mean([x['wip_end'] for x in g]):7.1f} "
                  f"{np.mean([x['agv_busy'] for x in g]):8.2f}")


if __name__ == "__main__":
    main()
