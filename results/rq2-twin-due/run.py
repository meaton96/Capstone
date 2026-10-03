"""
@file run.py
@brief rq2-twin-due: does a due-date objective (tardiness) give rule switching the headroom that time in system
       does not? Greedy switching oracle on the event-based twin (DES-1k), same design as env/switch_oracle.py.

@details
Instances: the randomized generator (rnd_load defaults) with machine failures off (the twin does not model them;
the jobs are identical to the default instances, only the failure block is dropped), seeds 0-19, 5,400 s agent
window. Two settings:
  - warm: random warm-up, the training distribution (= the oracle screen's mf-off regime);
  - t0:   from t = 0, no warm-up (= rq2-switch-oracle; used to calibrate the twin against Unity).
Floor: layout D, 7 AGVs, onTransport, exported by Unity (twin-gap G1 run). Transport: kinematic.

Due dates: TWK, d_i = r_i + c * work content (des_twin.scenario.assign_due_dates), c in ALLOWANCES. Calibrated as
Sels et al. 2012 do, by the share of jobs late under a flow rule: c = 2 leaves about half the jobs late under
SRT-ECT on these windows (flow / TWK median 1.9), c = 1.5 about three quarters, c = 3 about 15%, c = 6 none.

Objectives, each a window integral over [first agent decision, end of window], in units of 1000 s as the
flow_time reward (return = -value):
  - tis:  time in system of every job (the WIP integral; what the flow_time reward sums);
  - tard: tardiness of every job, finished or not (the late-WIP integral: open jobs past their due date, integrated).

Oracle per (setting, seed, objective, pair set, c): stage 1 runs every pair for the whole window; stage k keeps the
pairs chosen for segments 1..k-1 and tries every pair from segment k to the end; 6 segments of 900 s. Pair sets:
  - P12: the current heads, job {SPT, SRT, PTWINQ, FIFO} x machine {ECT, TECT, SRWT};
  - P30: P12's machine rules x job rules plus the due-date rules EDD, SLACK, CR, MDD, MOD, ATC.
Tasks: tis-P12 (c-independent), tard-P12 and tard-P30 at each c. Every candidate run records both objectives.

Writes tasks/<task>.json (resumable: existing files are skipped); analyze.py builds the tables.

@par Usage
@code{.sh}
.venv/bin/python results/rq2-twin-due/run.py [--workers 24] [--settings warm t0] [--seeds 0-19]
@endcode
"""
import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "env"))

FLOOR = REPO / "linux_server_des/Results/G1/rnd_load_s0/D/agv7_SPT_ECT_s0/des_floor.json"
WINDOW = 5400.0
SEGMENT = 900.0
MAX_SIM = 100000.0                      # MAX_EPISODE_SIM_SECONDS
ALLOWANCES = (1.5, 2.0, 3.0, 4.0, 6.0)   # share of jobs late under SRT-ECT (warm): about 77 / 46 / 15 / 5 / 0%
FLOW_JOB = ("SPT", "SRT", "PTWINQ", "FIFO")
DUE_JOB = ("EDD", "SLACK", "CR", "MDD", "MOD", "ATC")
MACHINE = ("ECT", "TECT", "SRWT")
P12 = [f"{j}-{m}" for j in FLOW_JOB for m in MACHINE]
P30 = [f"{j}-{m}" for j in FLOW_JOB + DUE_JOB for m in MACHINE]
# Candidate RL job head from the c = 2 results (best on average, per-seed winners, oracle choices by load):
# SRT (overload), SPT (lulls), MDD, EDD, ATC; with and without SRWT in the machine head.
HEAD = ("SRT", "SPT", "MDD", "EDD", "ATC")
PAIRSETS = {"P12": P12, "P30": P30,
            "H15": [f"{j}-{m}" for j in HEAD for m in MACHINE],
            "H10": [f"{j}-{m}" for j in HEAD for m in ("ECT", "TECT")]}

_floor = None
STAMP = None     # code stamp of the launch (code_stamp); set in main() before the pool forks, inherited by workers


def code_stamp():
    """Git HEAD, whether the twin / generator sources differ from it, and md5s of every source file the runs import.
    "id" changes whenever any of those files does. (Added 10-03 after c = 1.5 ran on a twin edited between launches.)"""
    files = sorted((REPO / "env/des_twin").glob("*.py")) + [REPO / "env/scenarios/randomized.py", Path(__file__).resolve()]
    md5 = {str(f.relative_to(REPO)): hashlib.md5(f.read_bytes()).hexdigest()[:12] for f in files}
    git = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True).stdout.strip()
    return {"git_head": git("rev-parse", "--short", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--", "env/des_twin", "env/scenarios")),
            "files": md5, "id": hashlib.md5(json.dumps(md5, sort_keys=True).encode()).hexdigest()[:12]}


def floor():
    global _floor
    if _floor is None:
        from des_twin import Floor
        _floor = Floor.load(str(FLOOR))
    return _floor


def scenario(setting, seed):
    from scenarios.randomized import DEFAULT_PARAMS, randomized_generator
    params = dataclasses.replace(DEFAULT_PARAMS, failure_probability=0.0)
    return randomized_generator(WINDOW, random_warmup=(setting == "warm"), params=params)(seed)


def overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def play(jobs, warmup, warm_rule, prefix, tail):
    """One episode: prefix[i] in segment i, then tail to the end of the window. Returns the run's metrics."""
    from des_twin import TwinConfig
    from des_twin.engine import Twin
    tw = Twin(floor(), jobs, TwinConfig(rule=warm_rule, transport="kinematic", warmup_seconds=warmup,
                                        episode_duration_seconds=WINDOW, max_sim_seconds=MAX_SIM))
    halves = {name: tuple(name.split("-")) for name in set(prefix + [tail])}
    gen = tw.agent_decisions()
    t0 = None
    try:
        gen.send(None)
        while True:
            if t0 is None:
                t0 = tw.now
            seg = int((tw.now - t0) // SEGMENT)
            gen.send(halves[prefix[seg] if seg < len(prefix) else tail])
    except StopIteration:
        pass
    t0 = warmup if t0 is None else t0
    t1 = tw.now
    tis = tard = 0.0
    exited = late = 0
    flow_sum = 0.0
    for j in tw.jobs.values():
        end = j.exit_time if j.exit_time is not None else t1
        tis += overlap(j.arrival, end, t0, t1)
        if j.due is not None:
            tard += overlap(j.due, end, t0, t1)
        if j.exit_time is not None and t0 <= j.exit_time <= t1:
            exited += 1
            flow_sum += j.exit_time - j.arrival
            if j.due is not None and j.exit_time > j.due:
                late += 1
    return {"tis": tis / 1000.0, "tard": tard / 1000.0, "t0": t0, "t1": t1, "jobs_exited": exited,
            "late_exited": late, "mean_flow_exited": flow_sum / exited if exited else 0.0,
            "decisions": tw.decisions, "timed_out": tw.timed_out}


def oracle(task):
    setting, seed, objective, pairset, c = task
    from des_twin.scenario import assign_due_dates, episode_settings, resolve_jobs
    sc = scenario(setting, seed)
    warmup, _, warm_rule = episode_settings(sc)
    jobs = resolve_jobs(sc, floor())
    if c is not None:
        jobs = assign_due_dates(jobs, c)
    pairs = PAIRSETS[pairset]
    n_seg = int(WINDOW // SEGMENT)
    prefix, rows, history = [], [], []
    t_start = time.time()
    for stage in range(1, n_seg + 1):
        stage_rows = []
        for tail in pairs:
            r = play(jobs, warmup, warm_rule, prefix, tail)
            r.update({"stage": stage, "prefix": "|".join(prefix), "tail": tail})
            stage_rows.append(r)
        best = min(stage_rows, key=lambda r: r[objective])
        history.append({"stage": stage, "chosen": best["tail"], objective: best[objective]})
        prefix.append(best["tail"])
        rows += stage_rows
    fixed = {r["tail"]: r for r in rows if r["stage"] == 1}
    best_fixed = min(fixed.values(), key=lambda r: r[objective])
    return {"setting": setting, "seed": seed, "objective": objective, "pairset": pairset, "allowance": c,
            "warmup": warmup, "warm_rule": warm_rule, "jobs": len(jobs["jobs"]), "schedule": prefix,
            "history": history, "best_fixed_pair": best_fixed["tail"], "best_fixed": best_fixed[objective],
            "oracle": history[-1][objective],
            "fixed": {k: {kk: v[kk] for kk in ("tis", "tard", "jobs_exited", "late_exited", "mean_flow_exited")}
                      for k, v in fixed.items()},
            "rows": rows, "wall_s": round(time.time() - t_start, 1)}


def task_name(task):
    setting, seed, objective, pairset, c = task
    return f"{setting}_s{seed}_{objective}_{pairset}" + ("" if c is None else f"_c{c:g}")


def run_task(task):
    out = HERE / "tasks" / f"{task_name(task)}.json"
    if out.exists():
        old = json.loads(out.read_text()).get("code", {}).get("id")
        if old != STAMP["id"]:
            raise RuntimeError(f"{out.name} was run on code {old}, this launch is {STAMP['id']}: move the old tasks "
                               "aside rather than mixing code versions")
        return task_name(task), "skip", 0.0
    res = oracle(task)
    res["code"] = STAMP
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(res))
    os.replace(tmp, out)
    return task_name(task), f"best fixed {res['best_fixed_pair']} {res['best_fixed']:.3f} oracle {res['oracle']:.3f}", res["wall_s"]


def parse_seeds(spec):
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("@par")[0])
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--settings", nargs="+", default=["warm", "t0"])
    ap.add_argument("--seeds", default="0-19")
    ap.add_argument("--extra", nargs="*", default=[], metavar="PAIRSET:C",
                    help="only these extra tard tasks, e.g. H15:2 H10:2 (default: the main grid)")
    args = ap.parse_args()
    (HERE / "tasks").mkdir(exist_ok=True)
    # Load everything the runs import in this process, so the forked workers all run exactly this code.
    global STAMP
    STAMP = code_stamp()
    import des_twin.engine, des_twin.rules, des_twin.scenario, scenarios.randomized   # noqa: F401
    floor()
    if code_stamp()["id"] != STAMP["id"]:
        raise SystemExit("source files changed while loading; relaunch")
    print(f"code {STAMP['id']} at {STAMP['git_head']}{' (dirty)' if STAMP['dirty'] else ''}", flush=True)
    tasks = []
    for setting in args.settings if args.extra else []:
        for seed in parse_seeds(args.seeds):
            for spec in args.extra:
                ps, c = spec.split(":")
                tasks.append((setting, seed, "tard", ps, float(c)))
    for setting in args.settings if not args.extra else []:
        for seed in parse_seeds(args.seeds):
            tasks.append((setting, seed, "tis", "P12", 2.0))   # due dates attached; flow rules ignore them
            for c in ALLOWANCES:
                tasks.append((setting, seed, "tard", "P12", c))
                tasks.append((setting, seed, "tard", "P30", c))
    tasks.sort(key=lambda t: -len(PAIRSETS[t[3]]))      # longest tasks first
    t0 = time.time()
    with Pool(args.workers) as pool:
        for i, (name, msg, wall) in enumerate(pool.imap_unordered(run_task, tasks), 1):
            print(f"[{i}/{len(tasks)} {time.time() - t0:6.0f}s] {name}: {msg} ({wall}s)", flush=True)
    if code_stamp()["id"] != STAMP["id"]:
        print("WARNING: source files changed on disk during the run. These results used the launch-time code "
              f"({STAMP['id']}); a relaunch would refuse to add to them.", flush=True)


if __name__ == "__main__":
    main()
