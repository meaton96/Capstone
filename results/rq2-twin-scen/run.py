"""
@file run.py
@brief rq2-twin-scen: which training regime gives the tardiness switching oracle the most headroom? Twin screen of
       the scenario levers before the Unity check (user asked 10-03 22:00: "set up our ideal scenario").

@details
Same oracle and objective as rq2-twin-due (greedy, 6 x 900 s stages, tardiness = late-WIP integral over the window,
warm-up windows, machine failures off), with the H15 pairs (job SRT / SPT / MDD / EDD / ATC x machine ECT / TECT /
SRWT, the RL heads since action schema v4). Levers, full grid:
  - c (TWK due-date allowance): 1.75, 2.0, 2.5   (twin rq2-twin-due: 1.5 halves the gain, >= 3 is sparse)
  - AGVs: 4, 5, 7                               (Unity agv4 screen: the only regime where switching itself added)
  - load: base (rnd_load defaults) or utilhi (utilization 1.0-1.8, lulls 0.4-0.7; = rq2-oracle-due-utilhi)
18 regimes x seeds 0-39. Imports the oracle machinery from ../rq2-twin-due/run.py (code-stamped with this file).

Writes tasks/<regime>_s<seed>.json (resumable; a launch refuses to add to tasks run on other code); analyze.py ranks
the regimes.

@par Usage
@code{.sh}
nohup nice -n 19 .venv/bin/python results/rq2-twin-scen/run.py --workers 16 > results/rq2-twin-scen/run.out 2>&1 &
@endcode
"""
import argparse
import dataclasses
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
ALLOWANCES = (1.75, 2.0, 2.5)
FLEETS = (4, 5, 7)
LOADS = {"base": {}, "utilhi": {"utilization": (1.0, 1.8), "lull_utilization": (0.4, 0.7)}}
STAMP = None


def regime_name(c, agvs, load):
    return f"c{c:g}-agv{agvs}-{load}"


def scenario(seed, load):
    from scenarios.randomized import DEFAULT_PARAMS, randomized_generator
    params = dataclasses.replace(DEFAULT_PARAMS, failure_probability=0.0, **LOADS[load])
    return randomized_generator(due.WINDOW, random_warmup=True, params=params)(seed)


def play(jobs, warmup, warm_rule, agvs, prefix, tail):
    """rq2-twin-due's play() with a fleet size: prefix[i] in segment i, then tail to the end of the window."""
    from des_twin import TwinConfig
    from des_twin.engine import Twin
    tw = Twin(due.floor(), jobs, TwinConfig(rule=warm_rule, transport="kinematic", warmup_seconds=warmup,
                                            episode_duration_seconds=due.WINDOW, max_sim_seconds=due.MAX_SIM,
                                            agv_count=agvs))
    halves = {name: tuple(name.split("-")) for name in set(prefix + [tail])}
    gen = tw.agent_decisions()
    t0 = None
    try:
        gen.send(None)
        while True:
            if t0 is None:
                t0 = tw.now
            seg = int((tw.now - t0) // due.SEGMENT)
            gen.send(halves[prefix[seg] if seg < len(prefix) else tail])
    except StopIteration:
        pass
    t0 = warmup if t0 is None else t0
    t1 = tw.now
    tis = tard = 0.0
    exited = late = 0
    for j in tw.jobs.values():
        end = j.exit_time if j.exit_time is not None else t1
        tis += due.overlap(j.arrival, end, t0, t1)
        if j.due is not None:
            tard += due.overlap(j.due, end, t0, t1)
        if j.exit_time is not None and t0 <= j.exit_time <= t1:
            exited += 1
            late += j.due is not None and j.exit_time > j.due
    return {"tis": tis / 1000.0, "tard": tard / 1000.0, "jobs_exited": exited, "late_exited": late,
            "timed_out": tw.timed_out}


def oracle(task):
    c, agvs, load, seed = task
    from des_twin.scenario import assign_due_dates, episode_settings, resolve_jobs
    sc = scenario(seed, load)
    warmup, _, warm_rule = episode_settings(sc)
    jobs = assign_due_dates(resolve_jobs(sc, due.floor()), c)
    prefix, history, fixed = [], [], {}
    t_start = time.time()
    for stage in range(1, int(due.WINDOW // due.SEGMENT) + 1):
        runs = {tail: play(jobs, warmup, warm_rule, agvs, prefix, tail) for tail in PAIRS}
        if stage == 1:
            fixed = runs
        best = min(runs, key=lambda t: runs[t]["tard"])
        history.append({"stage": stage, "chosen": best, "tard": runs[best]["tard"]})
        prefix.append(best)
    best_fixed = min(fixed, key=lambda t: fixed[t]["tard"])
    return {"regime": regime_name(c, agvs, load), "allowance": c, "agvs": agvs, "load": load, "seed": seed,
            "warmup": warmup, "jobs": len(jobs["jobs"]), "schedule": prefix, "history": history,
            "best_fixed_pair": best_fixed, "best_fixed": fixed[best_fixed]["tard"], "oracle": history[-1]["tard"],
            "fixed": fixed, "wall_s": round(time.time() - t_start, 1)}


def run_task(task):
    name = f"{regime_name(*task[:3])}_s{task[3]}"
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
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seeds", default="0-39")
    args = ap.parse_args()
    (HERE / "tasks").mkdir(exist_ok=True)
    global STAMP
    STAMP = due.code_stamp()
    STAMP["files"]["results/rq2-twin-scen/run.py"] = __import__("hashlib").md5(Path(__file__).read_bytes()).hexdigest()[:12]
    STAMP["id"] = __import__("hashlib").md5(json.dumps(STAMP["files"], sort_keys=True).encode()).hexdigest()[:12]
    import des_twin.engine, des_twin.rules, des_twin.scenario, scenarios.randomized   # noqa: F401
    due.floor()
    print(f"code {STAMP['id']} at {STAMP['git_head']}{' (dirty)' if STAMP['dirty'] else ''}", flush=True)
    # Seed-major, so a partial run already covers every regime on the first seeds.
    tasks = [(c, a, l, s) for s in due.parse_seeds(args.seeds) for c in ALLOWANCES for a in FLEETS for l in LOADS]
    t0 = time.time()
    with Pool(args.workers) as pool:
        for i, (name, msg, wall) in enumerate(pool.imap(run_task, tasks), 1):
            print(f"[{i}/{len(tasks)} {time.time() - t0:6.0f}s] {name}: {msg} ({wall}s)", flush=True)


if __name__ == "__main__":
    main()
