"""
@file train_size.py
@brief rq3-size-train (2026-10-19, user: floor-size study): train the final policy's PPO stack (train-v5-cmix: frozen v5
       code, mixed due dates c ~ U[1.25, 3.5]) on one floor size or a mix of sizes, without editing the frozen code.
       Wraps <code>/train.py: sets --twin / --params per size and, for floors with more bays than AGVs on duty, the fleet
       wrapper of ../rq3-gen-v5/run_task.py (agvCount = bays; full-fleet blocks get the on-duty count; short blocks
       keep theirs). With several sizes, environment e gets size SIZES[(e // G) % len(SIZES)] (G = the shared-baseline
       group), so every baseline group shares one floor; observation rows are padded to the largest floor
       (--obs-max-machines / --obs-max-jobs), so all envs stack.
Sizes (as ../rq3-size-v5): 15 = B2 itself (floor7, 7 AGVs, B2 load, as the final policy trained); 50 / 100 = the
agv-ratio-scale floors, 26 / 52 AGVs on duty (short blocks 13 / 26), B2 load x 0.9, arrival cap off.
@par Usage (from the repo root; the code snapshot is the directory this file sits in)
@code{.sh}
python results/NAME/code/train_size.py --sizes 15,50,100 -- <train.py flags, without --twin / --params>
@endcode
"""
import argparse
import copy
import json
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent            # results/NAME/code (the run's snapshot: HERE/env)
CODE = HERE / "env"
REPO = Path.cwd()                                 # run from the repo root, as every results/ script
assert (REPO / "results").is_dir(), "run train_size.py from the repo root"
B2 = {"failure_probability": 0.0, "due_date_allowance_range": [1.75, 2.5], "regime_block_seconds": 5400,
      "horizon_seconds": 30600,
      "load_mix": [{"name": "normal", "utilization": [0.6, 1.2], "weight": 2},
                   {"name": "surge", "utilization": [1.0, 1.6]},
                   {"name": "short", "utilization": [0.6, 1.2], "agv_count": 2}]}
LOAD = 0.9


def size_params(per_type, fleet, short):
    p = copy.deepcopy(B2)
    p.update(machines_per_type=per_type, agv_count=fleet, agv_max_utilization=10.0)
    for prof in p["load_mix"]:
        prof["utilization"] = [round(u * LOAD, 4) for u in prof["utilization"]]
        if prof["name"] == "short":
            prof["agv_count"] = short
    return p


FLOORS = REPO / "results" / "agv-ratio-scale" / "floors"
SIZES = {15: (REPO / "results" / "dev-popart-smoke" / "floor7" / "des_floor.json", copy.deepcopy(B2), None),
         50: (FLOORS / "m50" / "des_floor.json", size_params(10, 26, 13), (26, 50)),
         100: (FLOORS / "m100" / "des_floor.json", size_params(20, 52, 26), (52, 100))}


def with_c(params, c_range):
    p = copy.deepcopy(params)
    p["due_date_allowance_range"] = list(c_range)
    return p


def fleet_wrap(gen, on_duty, bays):
    """rq3-gen-v5's fleet wrapper around one generator."""
    def g(seed):
        s = copy.deepcopy(gen(seed))
        full = int(s["agvCount"])
        sched = s.get("agvSchedule") or [{"start": 0, "agvCount": full}]
        s["agvSchedule"] = [{"start": e["start"],
                             "agvCount": on_duty if int(e["agvCount"]) == full else min(int(e["agvCount"]), on_duty)}
                            for e in sched]
        s["agvCount"] = bays
        return s
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", required=True, help="comma-separated, from %s" % sorted(SIZES))
    ap.add_argument("--c-range", default="1.25,3.5")
    ap.add_argument("--group", type=int, default=4, help="shared-baseline group (envs per floor block)")
    a, rest = ap.parse_known_args()
    if rest and rest[0] == "--":
        rest = rest[1:]
    sizes = [int(x) for x in a.sizes.split(",")]
    c_range = [float(x) for x in a.c_range.split(",")]
    assert all(s in SIZES for s in sizes), sizes
    assert not any(x in rest for x in ("--twin", "--params")), "train_size.py sets --twin / --params itself"
    sys.path.insert(0, str(CODE))
    import scenarios
    import env_wrappers.twin_env as te

    largest = max(sizes, key=lambda s: s)
    floor0, params0, fleet0 = SIZES[sizes[0]]
    argv = ["--twin", str(SIZES[largest][0]), "--params", json.dumps(with_c(SIZES[largest][1], c_range))]
    if len(sizes) == 1:
        argv = ["--twin", str(floor0), "--params", json.dumps(with_c(params0, c_range))]
        if fleet0 is not None:
            orig = scenarios.REGISTRY["randomized"]
            scenarios.REGISTRY["randomized"] = lambda *x, **k: fleet_wrap(orig(*x, **k), *fleet0)
    else:
        from config import obs_row_caps
        mrows, jrows = obs_row_caps(largest)
        argv += ["--obs-max-machines", str(mrows), "--obs-max-jobs", str(jrows)]
        orig = scenarios.REGISTRY["randomized"]
        init = te.TwinSchedulingEnv.__init__
        cache = {}

        def gen_for(size, duration, kwargs):
            key = (size, duration)
            if key not in cache:
                floor, params, fleet = SIZES[size]
                kw = dict(kwargs, params_overrides=with_c(params, c_range))
                g = orig(duration, **kw)
                cache[key] = (str(floor), fleet_wrap(g, *fleet) if fleet else g)
            return cache[key]

        captured = {}

        def registry_factory(duration, **kw):          # train.py builds one generator; remember its arguments
            captured["duration"], captured["kw"] = duration, {k: v for k, v in kw.items() if k != "params_overrides"}
            return orig(duration, **kw)
        scenarios.REGISTRY["randomized"] = registry_factory

        def patched(self, floor, *args, **kw):
            env_id = int(kw.get("env_id", 0))
            size = sizes[(env_id // a.group) % len(sizes)]
            if "duration" in captured and kw.get("scenario_generator") is not None:
                fpath, gen = gen_for(size, captured["duration"], captured["kw"])
                floor, kw["scenario_generator"] = fpath, gen
                self._train_size = size
            init(self, floor, *args, **kw)
        te.TwinSchedulingEnv.__init__ = patched
        print(f"train_size: sizes {sizes}, env e -> size sizes[(e // {a.group}) % {len(sizes)}], "
              f"obs rows {mrows} machines / {jrows} jobs", flush=True)
    sys.argv = [str(CODE / "train.py")] + argv + rest
    runpy.run_path(str(CODE / "train.py"), run_name="__main__")


if __name__ == "__main__":
    main()
