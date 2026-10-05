"""@file blocks.py
@brief dev-load-calib, part 2: are the candidate regime-block mixes stationary over 6 h? Fixed MDD-TECT / SRT-TECT,
twin, blocks of 5,400 s, c ~ U[1.75, 2.5] per block, failures off, seeds 0-19. The fleet schedule comes from the
scenario's agvSchedule (profiles with agv_count). Usage: nice -n 19 .venv/bin/python results/dev-load-calib/blocks.py"""
import dataclasses
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run as base   # noqa: E402

NORMAL = {"utilization": (0.6, 1.2)}
MIXES = {
    "B0 normal only":           ({"name": "normal", **NORMAL},),
    "B1 normal2:surge1":        ({"name": "normal", **NORMAL, "weight": 2}, {"name": "surge", "utilization": (1.0, 1.6)}),
    "B2 normal2:surge1:short1": ({"name": "normal", **NORMAL, "weight": 2}, {"name": "surge", "utilization": (1.0, 1.6)},
                                 {"name": "short", **NORMAL, "agv_count": 2}),
    "B3 normal3:short1":        ({"name": "normal", **NORMAL, "weight": 3}, {"name": "short", **NORMAL, "agv_count": 2}),
}


def run(task):
    name, seed, rule = task
    from des_twin import TwinConfig, run_twin
    from des_twin.scenario import agv_schedule, episode_settings, resolve_jobs
    from scenarios.randomized import DEFAULT_PARAMS, randomized_generator
    params = dataclasses.replace(DEFAULT_PARAMS, failure_probability=0.0, due_date_allowance_range=(1.75, 2.5),
                                 regime_block_seconds=5400.0, horizon_seconds=30600.0, load_mix=MIXES[name])
    sc = randomized_generator(base.WINDOW, random_warmup=True, params=params)(seed)
    warmup, _, _ = episode_settings(sc)
    jobs = resolve_jobs(sc, base.floor())
    tw = run_twin(base.floor(), jobs, TwinConfig(rule=rule, transport="kinematic", warmup_seconds=warmup,
                                                 agv_schedule=agv_schedule(sc), episode_duration_seconds=base.WINDOW,
                                                 max_sim_seconds=200000.0))
    t0, t1 = warmup, tw.now
    per_h = [0.0] * 6
    for j in tw.jobs.values():
        if j.due is None:
            continue
        end = j.exit_time if j.exit_time is not None else t1
        for h in range(6):
            a = t0 + 3600 * h
            per_h[h] += max(0.0, min(end, a + 3600, t1) - max(j.due, a))
    return {"mix": name, "seed": seed, "rule": rule, "per_h": [x / 1000 for x in per_h],
            "profiles": [b["load_profile"] for b in sc["_meta"]["blocks"]]}


if __name__ == "__main__":
    base.floor()
    tasks = [(m, s, r) for m in MIXES for s in range(20) for r in base.RULES]
    with Pool(16) as pool:
        rows = pool.map(run, tasks)
    (Path(__file__).resolve().parent / "blocks_runs.json").write_text(json.dumps(rows))
    for m in MIXES:
        for r in base.RULES:
            ph = np.mean([x["per_h"] for x in rows if x["mix"] == m and x["rule"] == r], axis=0)
            print(f"{m:26s} {r:9s} " + " ".join(f"{v:6.1f}" for v in ph) + f"   h6/h3 {ph[5] / max(ph[2], 1e-9):4.2f}")
