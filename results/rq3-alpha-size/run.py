"""
@file run.py
@brief rq3-alpha-size (2026-10-21, user: paper B's auto-tuned α gate). The best PECT / PTECT α as a function of floor
       size and machine-speed spread, and whether α* can be set from observable floor statistics. Extends
       ../rq3-alpha-curve (15 machines, speed spread only) to the floor-size grid's conditions (15-100 machines).
Cells: floor in {m15, m20, m30, m50, m75, m100} (../rq3-size-grid conditions via ../rq3-size-oracle/sizesim.py) x
speed spread s in {0, 0.5, 1.0}; plus held-out cells m30 / m75 at s 0.75 (never used for fitting). s > 0 adds a fixed
per-machine speed factor log-uniform in [1/(1+s), 1+s], capacity-matched per machine type and with B2-tight due dates,
exactly as ../rq2-hetero-screen (generalized here to any machines_per_type). s 0 is the grid's instance itself.
Configs per episode: MDD x {PECT, PTECT} x α in ALPHAS (quad, TwinConfig.ptime_alpha; α 0 = ECT / TECT exactly), each
held all episode, window tardiness split by regime block (as ../rq2-portfolio-screen).
Per instance it also records the observable floor statistics (stats()): speed spread σ (std of log speed factors),
mean travel time to a candidate machine and its spread across candidates, mean op time and its spread across
candidates, AGVs on duty per machine.
Frozen code and simulator: ../rq2-rebase-v5 (6d13412a + sim.py). Seeds: fitting 2000-2039, test 0-39.
Output: out/<cell>/s<seed>.json.
@par Usage
@code{.sh}
python results/rq3-alpha-size/run.py m50_s0.5 0
python results/rq3-alpha-size/run.py list
@endcode
"""
import importlib.util
import json
import math
import os
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = HERE.parent
FLOORS = ["m100", "m75", "m50", "m30", "m20", "m15"]
SPREADS = [0.0, 0.5, 1.0]
HELD_OUT = ["m30_s0.75", "m75_s0.75"]
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)
CONFIGS = [(f"MDD-{m}@{a:g}", f"MDD-{m}", a) for m in ("PECT", "PTECT") for a in ALPHAS]
FIT_SEEDS = list(range(2000, 2040))
TEST_SEEDS = list(range(40))


def cells():
    return [f"{f}_s{s:g}" for f in FLOORS for s in SPREADS] + HELD_OUT


def parse(cell):
    f, s = cell.split("_s")
    return f, float(s)


def load(cell):
    """Frozen simulator pointed at the cell: the floor-size condition plus speed spread s, capacity-matched."""
    floor, s = parse(cell)
    spec = importlib.util.spec_from_file_location("rb_run", R / "rq2-rebase-v5" / "run.py")   # frozen code first
    rb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rb)
    spec = importlib.util.spec_from_file_location("sizesim", R / "rq3-size-oracle" / "sizesim.py")
    ss = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ss)
    gt = ss.grid_task()
    params, fl, layout, fleet = gt.rt.CONDITIONS[floor]
    params = dict(params)
    if s > 0:
        params["machine_speed_spread"] = s
    gt.rt.CONDITIONS[cell] = (params, fl, layout, fleet)      # run_task's CONDITIONS is shared (sys.modules)
    ss.apply(rb.sim, cell)
    if s > 0:
        base = rb.sim.scenario
        rb.sim.scenario = lambda setting, seed: hetero(base(setting, seed), params, seed)
    return rb, params


def speed_factors(params, seed):
    """The generator's per-machine speed factors {type: [f per machine index]}, replayed from its own draws (the
    draw order of randomized_scenario: Random(seed), one op-mean draw, then k speed draws per type)."""
    from scenarios import randomized as RG
    p, _ = RG._episode_params(seed, RG.params_with_overrides(params))
    rng = random.Random(seed)
    RG._log_uniform(rng, *p.op_mean_seconds)
    s = 1.0 + p.machine_speed_spread
    return {t: [RG._log_uniform(rng, 1.0 / s, s) for _ in range(p.machines_per_type)] for t in RG.TYPES}


def hetero(sc, params, seed):
    """rq2-hetero-screen's capacity matching and due-date correction, for any machines_per_type."""
    fac = speed_factors(params, seed)
    scale = {t: sum(1.0 / f for f in fs) / len(fs) for t, fs in fac.items()}
    for job in sc["jobs"]:
        work = sum(min(op["duration"]) for op in job["operations"])
        work_ref = sum(min(d / fac[op["machineType"]][i] for i, d in enumerate(op["duration"]))
                       for op in job["operations"])
        for op in job["operations"]:
            op["duration"] = [round(d * scale[op["machineType"]], 2) for d in op["duration"]]
        if job.get("dueDate") is not None and work > 0:
            job["dueDate"] = round(job["arrivalTime"] + (job["dueDate"] - job["arrivalTime"]) * work_ref / work, 2)
    sc["_hetero"] = {"factors": fac, "scale": scale}
    return sc


def stats(rb, ep, params, seed):
    """Observable floor statistics of one instance (all knowable before or while running the floor)."""
    floor = rb.sim.sig.floor()
    speed = float(floor.agv["speed"])
    by_type = {}
    for m in floor.machines:
        by_type.setdefault(m.type, []).append(m.id)
    sources = [-1] + [m.id for m in floor.machines]
    tr_mean, tr_sd = [], []
    for a in sources:
        for t, ids in by_type.items():
            tt = [floor.estimate_path_length(a, b) / speed for b in ids if b != a]
            tt = [x for x in tt if math.isfinite(x)]
            if len(tt) > 1:
                m = sum(tt) / len(tt)
                tr_mean.append(m)
                tr_sd.append(math.sqrt(sum((x - m) ** 2 for x in tt) / len(tt)))
    p_mean, p_sd = [], []
    for j in ep.sc["jobs"]:
        for op in j["operations"]:
            d = [float(x) for x in op["duration"]]
            m = sum(d) / len(d)
            p_mean.append(m)
            p_sd.append(math.sqrt(sum((x - m) ** 2 for x in d) / len(d)))
    if params.get("machine_speed_spread"):
        logs = [math.log(f) for fs in speed_factors(params, seed).values() for f in fs]
        mu = sum(logs) / len(logs)
        sigma = math.sqrt(sum((x - mu) ** 2 for x in logs) / len(logs))
    else:
        sigma = 0.0
    sched = ep.sc.get("agvSchedule") or [{"agvCount": ep.sc["agvCount"]}]   # on-duty fleet per block
    full = max(int(e["agvCount"]) for e in sched)
    return {"machines": len(floor.machines), "sigma": sigma,
            "travel_mean_s": sum(tr_mean) / len(tr_mean), "travel_sd_s": sum(tr_sd) / len(tr_sd),
            "p_mean_s": sum(p_mean) / len(p_mean), "p_sd_s": sum(p_sd) / len(p_sd),
            "agvs_per_machine": full / len(floor.machines)}


def play(rb, ep, pair, alpha):
    """Window tardiness (1000 job-s) of a fixed pair at TwinConfig.ptime_alpha = alpha, total and per regime block."""
    from des_twin import TwinConfig
    from des_twin.engine import Twin
    sim = rb.sim
    tw = Twin(sim.sig.floor(), ep.jobs, TwinConfig(
        rule=ep.warm_rule, transport="kinematic", warmup_seconds=ep.warmup,
        episode_duration_seconds=ep.cfg["window"], max_sim_seconds=ep.warmup + ep.cfg["window"] + 200000.0,
        agv_schedule=ep.sched, ptime_alpha=alpha))
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
    ov = sim.fleet.due.overlap

    def tard(a, b):
        return sum(ov(j.due, j.exit_time if j.exit_time is not None else t1, a, b) for j in jobs) / 1000.0

    blocks = {}
    for b in ep.blocks:
        a, z = max(b["start"], t0), min(b["end"], t1)
        if z > a:
            blocks[str(b["block"])] = tard(a, z)
    return tard(t0, t1), blocks, (t0, t1)


def run(cell, seed, root=HERE, configs=None):
    out = Path(root) / "out" / cell / f"s{seed}.json"
    if out.exists():
        return f"exists {out}"
    rb, params = load(cell)
    t = time.time()
    ep = rb.sim.Episode("B2", seed)
    res, win = {}, None
    for name, pair, a in (configs or CONFIGS):
        total, blocks, win = play(rb, ep, pair, a)
        res[name] = {"total": total, "blocks": blocks}
    res["_blocks"] = [{k: b.get(k) for k in ("block", "start", "end", "load_profile", "agv_count")} for b in ep.blocks
                      if b["end"] > win[0] and b["start"] < win[1]]
    res["_window"] = list(win)
    res["_cell"] = cell
    res["_stats"] = stats(rb, ep, params, seed)
    res["_stamp"] = rb.stamp()
    res["_seconds"] = time.time() - t
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(res))
    os.replace(tmp, out)
    tot = {k: v["total"] for k, v in res.items() if not k.startswith("_")}
    best = min(tot, key=tot.get)
    return f"{cell} s{seed}: best {best} {tot[best]:.2f}, MDD-PTECT@3 {tot.get('MDD-PTECT@3', float('nan')):.2f}, {res['_seconds']:.0f} s"


def main(argv):
    if argv[0] == "list":
        for c in cells():
            for s in (TEST_SEEDS if c in HELD_OUT else FIT_SEEDS + TEST_SEEDS):
                print(c, s)
    else:
        print(run(argv[0], int(argv[1]), *argv[2:3]), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
