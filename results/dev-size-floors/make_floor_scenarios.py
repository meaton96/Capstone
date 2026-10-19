"""
@file make_floor_scenarios.py
@brief dev-size-floors (2026-10-19, user: floor-size study): tiny scenarios whose only purpose is a des_floor.json export
       (-destrace) of an M-machine layout-D floor with M parking bays, as ../agv-ratio-scale/make_floor_scenarios.py,
       for the study's test sizes. Five jobs, no failures. Run with the frozen v5 code's generator.
"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "../rq2-rebase-v5/code/env")]
from scenarios.randomized import RandomizedParams, randomized_scenario   # noqa: E402

SIZES = [int(x) for x in (sys.argv[1:] or ["20", "30", "50", "75", "100", "150", "200", "1000"])]
for m in SIZES:
    assert m % 5 == 0, m
    p = RandomizedParams(machines_per_type=m // 5, agv_count=m, failure_probability=0.0)
    s = randomized_scenario(0, p)
    s["jobs"] = s["jobs"][:5]
    s["name"] = f"floor_m{m}"
    for k in ("_phases", "_meta", "stochastic"):
        s.pop(k, None)
    out = os.path.join(HERE, "floors", f"floor_m{m}.json")
    json.dump(s, open(out, "w"), indent=1)
    print(out, len(s["machineTypeLayout"]), s["agvCount"])
