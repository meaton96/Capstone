"""
@file agv5_scenarios.py
@brief agv-study-rerun (v5 design, 2026-10-14): the B2-family instances of the AGV study, with due dates, as Unity batch
       scenarios and as twin inputs. One builder for both, so the twin calibration and the Unity runs use the same jobs.

@details B2 (results/scripts/dev_b2_slot_train_eval.sh PARAMS) without the two parts that clash with a fleet sweep:
the 2-AGV short-fleet block (fleet size is the swept variable) and the random warm-up (batch runs start empty and
run until every job has finished, so tardiness is never censored). What is kept: 1.5 h load blocks drawn from
normal (utilization 0.6-1.2, weight 2) and surge (1.0-1.6), TWK due dates d = r + c * TWK with c ~ U[1.75, 2.5] per
instance, failures off, 6 h of arrivals, 15 machines (3 per type), layout D, releasePrevious, lane parking.

Regimes:
  b2m   machine-bound: B2's operation times (mean op drawn in 240-480 s) and B2's arrival cap at 7 AGVs, so at 7 AGVs
        this is a B2 instance (minus warm-up and short block)
  b2t   transport-bound: mean op OPS_T (calibrated in the twin, calib_twin.py) and the arrival cap off
        (agv_max_utilization 10), so the job list is the same at every fleet and AGVs are the bottleneck at small
        fleets
  b2tf  b2t with machine failures (failure_probability FAIL_P; Unity's own failure model)
  b2m105 / b2t105   105 machines = 7 tiles of D (tiling block in the scenario; job_scope tile or open)
@par Usage
@code{.sh}
.venv/bin/python results/agv-study-rerun/agv5_scenarios.py     # writes linux_server/BatchConfigs/Scenarios/agv5_*.json
@endcode
"""
import copy
import json
import sys
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "env"))
from scenarios.randomized import RandomizedParams, randomized_scenario  # noqa: E402

OUT = REPO / "linux_server" / "BatchConfigs" / "Scenarios"
LOAD_MIX = [{"name": "normal", "utilization": [0.6, 1.2], "weight": 2}, {"name": "surge", "utilization": [1.0, 1.6]}]
OPS_T = (90.0, 150.0)       # b2t mean-op range: twin knee 10-15 AGVs, AGVs 80% busy at 7 (calib_twin.out, 10-14)
FAIL_P = 1.0                # b2tf: share of instances with machine failures on (all; Weibull k 1.5, lambda 7200 s, repair ~620 s)


def params(regime, ops=None, tiles=1, job_scope="tile"):
    base = RandomizedParams(failure_probability=0.0, due_date_allowance_range=(1.75, 2.5), regime_block_seconds=5400.0,
                            horizon_seconds=21600.0, load_mix=copy.deepcopy(LOAD_MIX))
    if regime == "b2m":
        p = base
    elif regime in ("b2t", "b2tf"):
        p = replace(base, op_mean_seconds=tuple(ops or OPS_T), agv_max_utilization=10.0)
        if regime == "b2tf":
            p = replace(p, failure_probability=FAIL_P)
    else:
        raise ValueError(regime)
    if tiles > 1:
        p = replace(p, tiles=tiles, job_scope=job_scope, agv_assignment="pooled" if job_scope == "open" else "tile",
                    agv_count=7 * tiles)
    return p


def build(regime, seed, ops=None, tiles=1, job_scope="tile"):
    s = randomized_scenario(seed, params(regime, ops, tiles, job_scope))
    return s


def name(regime, seed, tiles=1, job_scope="tile"):
    t = "" if tiles == 1 else f"_t{tiles}{'open' if job_scope == 'open' else 'iso'}"
    return f"agv5_{regime}{t}_s{seed}"


# (regime, seeds, tiles, job_scope) written by main()
SETS = [("b2m", (0, 1), 1, "tile"), ("b2t", (0, 1), 1, "tile"), ("b2tf", (0, 1, 2), 1, "tile"),
        ("b2m", (0,), 7, "tile"), ("b2m", (0,), 7, "open"), ("b2t", (0,), 7, "tile"), ("b2t", (0,), 7, "open")]


def main():
    for regime, seeds, tiles, scope in SETS:
        for seed in seeds:
            s = build(regime, seed, tiles=tiles, job_scope=scope)
            s["name"] = name(regime, seed, tiles, scope)
            path = OUT / f"{s['name']}.json"
            path.write_text(json.dumps(s))
            due = sum(1 for j in s["jobs"] if j.get("dueDate") is not None)
            print(f"{path.relative_to(REPO)}: {len(s['jobs'])} jobs ({due} with due dates), "
                  f"{len(s['machineTypeLayout'])} machines, tiling {s.get('tiling')}, "
                  f"{path.stat().st_size / 2**20:.2f} MiB")


if __name__ == "__main__":
    main()
