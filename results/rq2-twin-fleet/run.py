"""
@file run.py
@brief rq2-twin-fleet: in-episode switching headroom at a calibrated (stationary) load, with regime blocks that
       vary the load and the active AGV fleet (user asked 10-04: vary the fleet in episode; twin test first).

@details
Copy of ../rq2-twin-blocks/run.py (same H15 greedy oracle, tardiness, 900 s slots, random warm-up, failures off,
c ~ U[1.75, 2.5]) with the fleet schedule passed to the twin (scenario "agvSchedule" -> TwinConfig.agv_schedule) and
the load from ../dev-load-calib: normal = utilization 0.6-1.2 outside lulls (tardiness per hour levels off after
~3 h; the old 0.7-1.4 grows all 6 h). Settings, seeds 0-39:
  - single-cal: one regime (normal), 5,400 s window;
  - long-cal:   one regime (normal), 21,600 s window;
  - B0:         blocks of 5,400 s (c, op mean per block), normal load, 7 AGVs;
  - B3:         blocks, profiles normal (3) : short (1) = normal load with 2 AGVs on duty (transport binds in the twin
                only at 2; it has no zone blocking);
  - B2:         blocks, profiles normal (2) : surge (1, utilization 1.0-1.6) : short (1).
B3 vs B0 isolates fleet changes; long-cal vs single-cal the episode length at a stationary load.

@par Usage
@code{.sh}
nohup nice -n 19 .venv/bin/python results/rq2-twin-fleet/run.py --workers 16 > results/rq2-twin-fleet/run.out 2>&1 &
@endcode
"""
import argparse
import dataclasses
import hashlib
import importlib.util
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "env"))

_spec = importlib.util.spec_from_file_location("twin_due", REPO / "results/rq2-twin-due/run.py")
due = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(due)

PAIRS = due.PAIRSETS["H15"]
SEGMENT = 900.0
NORMAL = {"utilization": (0.6, 1.2)}
SURGE = {"name": "surge", "utilization": (1.0, 1.6)}
SHORT = {"name": "short", **NORMAL, "agv_count": 2}
BLOCKS = {"regime_block_seconds": 5400.0, "horizon_seconds": 30600.0}
SETTINGS = {
    "single-cal": {"window": 5400.0, "params": {**NORMAL}},
    "long-cal": {"window": 21600.0, "params": {**NORMAL, "horizon_seconds": 30600.0}},
    "B0": {"window": 21600.0, "params": {**BLOCKS, "load_mix": ({"name": "normal", **NORMAL},)}},
    "B3": {"window": 21600.0, "params": {**BLOCKS, "load_mix": ({"name": "normal", **NORMAL, "weight": 3}, SHORT)}},
    "B2": {"window": 21600.0, "params": {**BLOCKS, "load_mix": ({"name": "normal", **NORMAL, "weight": 2}, SURGE, SHORT)}},
}
_SCHED = {}     # agv schedule of the scenario being played (oracle() sets it; one task per process at a time)
STAMP = None


def scenario(setting, seed):
    from scenarios.randomized import DEFAULT_PARAMS, randomized_generator
    params = dataclasses.replace(DEFAULT_PARAMS, failure_probability=0.0, due_date_allowance_range=(1.75, 2.5),
                                 **SETTINGS[setting]["params"])
    return randomized_generator(SETTINGS[setting]["window"], random_warmup=True, params=params)(seed)


def play(jobs, warmup, warm_rule, window, prefix, tail, per_slot=False):
    """One episode: prefix[i] in slot i, then tail to the end of the window. Returns tardiness (and per slot)."""
    from des_twin import TwinConfig
    from des_twin.engine import Twin
    tw = Twin(due.floor(), jobs, TwinConfig(rule=warm_rule, transport="kinematic", warmup_seconds=warmup,
                                            episode_duration_seconds=window, max_sim_seconds=200000.0,
                                            agv_schedule=_SCHED.get("s", ())))
    halves = {name: tuple(name.split("-")) for name in set(prefix + [tail])}
    gen = tw.agent_decisions()
    t0 = None
    try:
        gen.send(None)
        while True:
            if t0 is None:
                t0 = tw.now
            k = int((tw.now - t0) // SEGMENT)
            gen.send(halves[prefix[k] if k < len(prefix) else tail])
    except StopIteration:
        pass
    t0 = warmup if t0 is None else t0
    t1 = tw.now
    n_slots = int(round(window / SEGMENT))
    tard, slots = 0.0, [0.0] * n_slots
    for j in tw.jobs.values():
        if j.due is None:
            continue
        end = j.exit_time if j.exit_time is not None else t1
        tard += due.overlap(j.due, end, t0, t1)
        if per_slot:
            for k in range(n_slots):
                a = t0 + k * SEGMENT
                slots[k] += due.overlap(j.due, end, a, min(a + SEGMENT, t1))
    out = {"tard": tard / 1000.0, "t0": t0, "t1": t1, "timed_out": tw.timed_out}
    if per_slot:
        out["slots"] = [x / 1000.0 for x in slots]
    return out


def oracle(task):
    setting, seed = task
    from des_twin.scenario import agv_schedule, episode_settings, resolve_jobs
    sc = scenario(setting, seed)
    _SCHED["s"] = agv_schedule(sc)
    warmup, _, warm_rule = episode_settings(sc)
    jobs = resolve_jobs(sc, due.floor())
    window = SETTINGS[setting]["window"]
    prefix, history, fixed = [], [], {}
    t_start = time.time()
    for stage in range(1, int(round(window / SEGMENT)) + 1):
        runs = {tail: play(jobs, warmup, warm_rule, window, prefix, tail, per_slot=(stage == 1)) for tail in PAIRS}
        if stage == 1:
            fixed = runs
        best = min(runs, key=lambda t: runs[t]["tard"])
        history.append({"stage": stage, "chosen": best, "tard": runs[best]["tard"]})
        prefix.append(best)
    best_fixed = min(fixed, key=lambda t: fixed[t]["tard"])
    t0 = fixed[best_fixed]["t0"]
    meta = sc["_meta"]
    # Regime of each slot: the block its midpoint falls in (blocks), or the episode's single regime.
    if "blocks" in meta:
        bl = meta["blocks"]
        slot_block = [min(int((t0 + (k + 0.5) * SEGMENT) // 5400.0), len(bl) - 1) for k in range(len(prefix))]
        regimes = [{"block": b["block"], "c": b["due_date_allowance"], "load": b["load_profile"],
                    "op_mean": b["op_mean"], "agvs": b["agv_count"]} for b in bl]
    else:
        slot_block = [0] * len(prefix)
        regimes = [{"block": 0, "c": meta["due_date_allowance"], "load": meta["load_profile"],
                    "op_mean": meta["op_mean_seconds"]}]
    return {"setting": setting, "seed": seed, "warmup": warmup, "jobs": len(jobs["jobs"]), "t0": t0,
            "schedule": prefix, "history": history, "best_fixed_pair": best_fixed,
            "best_fixed": fixed[best_fixed]["tard"], "oracle": history[-1]["tard"],
            "fixed": {k: {"tard": v["tard"], "slots": v["slots"]} for k, v in fixed.items()},
            "slot_block": slot_block, "regimes": regimes, "wall_s": round(time.time() - t_start, 1)}


def run_task(task):
    name = f"{task[0]}_s{task[1]}"
    out = HERE / "tasks" / f"{name}.json"
    if out.exists():
        old = json.loads(out.read_text()).get("code", {}).get("id")
        if old != STAMP["id"]:
            raise RuntimeError(f"{out.name} was run on code {old}, this launch is {STAMP['id']}")
        return name, "skip", 0.0
    res = oracle(task)
    res["code"] = STAMP
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(res))
    os.replace(tmp, out)
    gain = 100 * (res["best_fixed"] - res["oracle"]) / res["best_fixed"] if res["best_fixed"] > 0 else 0.0
    return name, f"best fixed {res['best_fixed_pair']} {res['best_fixed']:.3f} oracle {res['oracle']:.3f} ({gain:.1f}%)", res["wall_s"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("@par")[0])
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seeds", default="0-39")
    args = ap.parse_args()
    (HERE / "tasks").mkdir(exist_ok=True)
    global STAMP
    STAMP = due.code_stamp()
    STAMP["files"]["results/rq2-twin-fleet/run.py"] = hashlib.md5(Path(__file__).read_bytes()).hexdigest()[:12]
    STAMP["id"] = hashlib.md5(json.dumps(STAMP["files"], sort_keys=True).encode()).hexdigest()[:12]
    import des_twin.engine, des_twin.rules, des_twin.scenario, scenarios.randomized   # noqa: F401
    due.floor()
    print(f"code {STAMP['id']} at {STAMP['git_head']}{' (dirty)' if STAMP['dirty'] else ''}", flush=True)
    # Long tasks first within each seed; seed-major so partial results cover both settings.
    tasks = [(st, s) for s in due.parse_seeds(args.seeds) for st in ("B2", "B3", "B0", "long-cal", "single-cal")]
    t0 = time.time()
    with Pool(args.workers) as pool:
        for i, (name, msg, wall) in enumerate(pool.imap_unordered(run_task, tasks), 1):
            print(f"[{i}/{len(tasks)} {time.time() - t0:6.0f}s] {name}: {msg} ({wall}s)", flush=True)


if __name__ == "__main__":
    main()
