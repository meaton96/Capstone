"""
@file run.py
@brief base-late-share (2026-10-22, user: thesis RERUN-V5 item). The share of late jobs on B2 under the reference pair
       MDD-PTECT, at fixed due-date allowances c and over B2's own c ~ U[1.75, 2.5] (drawn per 1.5 h regime block). The
       thesis (Background, due dates) still says c = 2 "left about half of the jobs late under SRT-ECT", a calibration
       made on the earlier randomized instances (5,400 s windows; ../rq2-twin-due: SRT-ECT 77 / 46 / 15% late at
       c = 1.5 / 2 / 3). SRT-ECT and SPT-ECT (Sels et al. 2012's calibration rule) run next to MDD-PTECT for comparison.
Settings: B2 (c ~ U[1.75, 2.5] per block) and c fixed at 1.5, 1.75, 2, 2.25, 2.5, 3 (B2 otherwise unchanged).
Late share (as ../rq4-mfail-ccal): jobs due inside the window [t0, t1) that exit after their due date or are still open
at t1, over all jobs due in the window; pooled over seeds. Also recorded: window tardiness (1000 job-s), jobs that exit
in the window and how many of them were late.
Frozen code and simulator: ../rq2-rebase-v5 (6d13412a + sim.py), B2 through ../rq3-size-oracle/sizesim.py (condition
"base" = rq3-gen-v5's B2, checked: MDD-PTECT seed 0 = 95.842006). Seeds: test 0-39 and 1000-1039.
Output: out/<setting>.json {seed: {pair: {...}}}; one task per setting, finished settings are skipped.
@par Usage
@code{.sh}
python results/base-late-share/run.py c2
python results/base-late-share/run.py list
@endcode
"""
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = HERE.parent
SETTINGS = {"B2": None, "c1.5": 1.5, "c1.75": 1.75, "c2": 2.0, "c2.25": 2.25, "c2.5": 2.5, "c3": 3.0}
PAIRS = ["MDD-PTECT", "SRT-ECT", "SPT-ECT"]
SEEDS = list(range(40)) + list(range(1000, 1040))


def load(setting):
    spec = importlib.util.spec_from_file_location("rb_run", R / "rq2-rebase-v5" / "run.py")   # frozen code first
    rb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rb)
    spec = importlib.util.spec_from_file_location("sizesim", R / "rq3-size-oracle" / "sizesim.py")
    ss = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ss)
    gt = ss.grid_task()
    params, floor, layout, fleet = gt.rt.CONDITIONS["base"]
    params = dict(params)
    c = SETTINGS[setting]
    if c is not None:
        params["due_date_allowance_range"] = [c, c]
    name = f"late_{setting}"
    gt.rt.CONDITIONS[name] = (params, floor, layout, fleet)        # run_task's CONDITIONS is shared (sys.modules)
    ss.apply(rb.sim, name)
    return rb


def play(rb, ep, pair):
    from des_twin import TwinConfig
    from des_twin.engine import Twin
    sim = rb.sim
    tw = Twin(sim.sig.floor(), ep.jobs, TwinConfig(
        rule=ep.warm_rule, transport="kinematic", warmup_seconds=ep.warmup,
        episode_duration_seconds=ep.cfg["window"], max_sim_seconds=ep.warmup + ep.cfg["window"] + 200000.0,
        agv_schedule=ep.sched))
    gen = tw.agent_decisions()
    t0, act = None, tuple(pair.split("-"))
    try:
        gen.send(None)
        while True:
            if t0 is None:
                t0 = tw.now
            gen.send(act)
    except StopIteration:
        pass
    t0 = ep.warmup if t0 is None else t0
    t1 = tw.now
    jobs = [j for j in tw.jobs.values() if j.due is not None]
    due_in = [j for j in jobs if t0 <= j.due < t1]
    exited = [j for j in jobs if j.exit_time is not None and t0 <= j.exit_time <= t1]
    ov = sim.fleet.due.overlap
    return {"tard": sum(ov(j.due, j.exit_time if j.exit_time is not None else t1, t0, t1) for j in jobs) / 1000.0,
            "due_in": len(due_in), "late": sum(1 for j in due_in if j.exit_time is None or j.exit_time > j.due),
            "exited": len(exited), "exited_late": sum(1 for j in exited if j.exit_time > j.due)}


def run(setting, root=HERE, seeds=None):
    out = Path(root) / "out" / f"{setting}.json"
    if out.exists():
        return f"exists {out}"
    rb = load(setting)
    t = time.time()
    res = {}
    for s in (seeds or SEEDS):
        ep = rb.sim.Episode("B2", s)
        res[str(s)] = {p: play(rb, ep, p) for p in PAIRS}
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps({"setting": setting, "c": SETTINGS[setting], "results": res, "stamp": rb.stamp(),
                               "seconds": time.time() - t}))
    os.replace(tmp, out)
    sh = {p: sum(v[p]["late"] for v in res.values()) / max(1, sum(v[p]["due_in"] for v in res.values())) for p in PAIRS}
    return f"{setting}: late share " + ", ".join(f"{p} {100 * x:.1f}%" for p, x in sh.items()) + f" ({time.time() - t:.0f} s)"


if __name__ == "__main__":
    if sys.argv[1] == "list":
        print("\n".join(SETTINGS))
    else:
        print(run(sys.argv[1], *sys.argv[2:3]), flush=True)
