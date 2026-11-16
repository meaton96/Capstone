"""
@file run.py
@brief rq3-alpha-transport (2026-10-28, user: "design the size / transport term in the law, with the pair heads
       restrained above ~50 machines, with pre-declared criteria"): stage 1, the fixed-alpha curves the size / transport
       term of the alpha law is fitted on, in the frozen v5 simulator. Extends ../rq3-alpha-size (15-100 machines at
       0.52 AGVs per machine) to 150 / 200 machines and to the fleet ratios of ../rq3-size-headroom (0.35 / 0.80 AGVs
       per machine at 50 / 100 / 200 machines), all homogeneous (sigma 0).
Cells (m<M>_<fleet>): r052 = the size grid's condition (../rq3-size-grid: round(0.52 M) AGVs on duty, short blocks
half, B2 load x 0.9, arrival cap off; loaded as ../rq3-alpha-size does); r035 / r080 = the same floor with round(r M)
AGVs on duty (loaded as ../rq3-size-headroom does); m15_b2 = B2 itself (7 AGVs, full load).
Kinds of task:
  curve  MDD-PTECT at every alpha in ALPHAS (quad, TwinConfig.ptime_alpha; alpha 0 = TECT exactly), each held all
         episode: the fit cells ../rq3-alpha-size lacks (fitting seeds), the stage-2 test cells it lacks and the two
         held-out cells (test seeds);
  util   MDD-PTECT at alpha 3 only, for the AGV utilization of the cells whose curves ../rq3-alpha-size already holds
         (and the fitting seeds of the held-out cells, so every cell's floor statistics come from fitting seeds);
  law    (stage 2, after analyze.py has written law.json) MDD-PTECT at the fitted law's alpha for the cell, test seeds:
         out/<cell>/law/s<seed>.json.
Every task records the instance's floor statistics (../rq3-alpha-size run.stats: mean travel time to a candidate
machine, mean op time, AGVs on duty per machine) and the AGV utilization over the window under MDD-PTECT at alpha 3
(busy AGV-seconds / on-duty AGV-seconds, as ../rq3-size-headroom's descriptors).
Frozen code and simulator: ../rq2-rebase-v5 (6d13412a + sim.py). Seeds: fitting 2000-2039, test 0-39.
Output: out/<cell>/s<seed>.json.
@par Usage (from the repo root)
@code{.sh}
.venv/bin/python results/rq3-alpha-transport/run.py m200_r035 2000
.venv/bin/python results/rq3-alpha-transport/run.py list          # stage 1 tasks, longest first
.venv/bin/python results/rq3-alpha-transport/run.py list law      # stage 2's law tasks (needs law.json)
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
FIT_SEEDS = list(range(2000, 2040))
TEST_SEEDS = list(range(40))
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)
## @brief (cell, kind, seeds), longest first. Curves of m15-m100 at r052 on both seed sets are ../rq3-alpha-size's.
PLAN = [
    ("m200_r035", "curve", FIT_SEEDS),
    ("m200_r052", "curve", FIT_SEEDS + TEST_SEEDS),
    ("m200_r080", "curve", FIT_SEEDS),
    ("m150_r052", "curve", FIT_SEEDS),
    ("m150_r080", "curve", TEST_SEEDS),                 # held out
    ("m150_r080", "util", FIT_SEEDS),
    ("m100_r035", "curve", FIT_SEEDS + TEST_SEEDS),
    ("m100_r080", "curve", FIT_SEEDS + TEST_SEEDS),
    ("m75_r035", "curve", TEST_SEEDS),                  # held out
    ("m75_r035", "util", FIT_SEEDS),
    ("m50_r035", "curve", FIT_SEEDS),
    ("m50_r080", "curve", FIT_SEEDS),
    ("m100_r052", "util", FIT_SEEDS),
    ("m75_r052", "util", FIT_SEEDS),
    ("m50_r052", "util", FIT_SEEDS),
    ("m30_r052", "util", FIT_SEEDS),
    ("m20_r052", "util", FIT_SEEDS),
    ("m15_b2", "util", FIT_SEEDS),
]
HELD_OUT = ("m75_r035", "m150_r080")


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


AS = _module("alpha_size_run", R / "rq3-alpha-size" / "run.py")      # load, stats (no simulator at import)
RH = _module("size_headroom_run", R / "rq3-size-headroom" / "run.py")  # load(size, ratio), on_duty_seconds


def parse(cell):
    m, fleet = cell.split("_")
    return int(m[1:]), fleet


def load(cell):
    """The frozen simulator pointed at the cell (one call per process: both loaders wrap the scenario registry)."""
    size, fleet = parse(cell)
    if fleet in ("b2", "r052"):
        rb, _ = AS.load(f"m{size}_s0")
    else:
        rb, _ = RH.load(size, fleet)
    return rb


def play(rb, ep, alpha, util=False):
    """../rq3-alpha-size run.play for MDD-PTECT at @p alpha (window tardiness in 1000 job-s, total and per regime block,
    window), plus with @p util the AGV utilization over the window (busy / on-duty AGV-seconds)."""
    from des_twin import TwinConfig
    from des_twin.engine import Twin
    sim = rb.sim
    tw = Twin(sim.sig.floor(), ep.jobs, TwinConfig(
        rule=ep.warm_rule, transport="kinematic", warmup_seconds=ep.warmup,
        episode_duration_seconds=ep.cfg["window"], max_sim_seconds=ep.warmup + ep.cfg["window"] + 200000.0,
        agv_schedule=ep.sched, ptime_alpha=alpha))

    def idle_total():
        return sum(a.idle_time + ((tw.now - a.idle_since) if a.state == "idle" else 0.0) for a in tw.agvs)

    gen = tw.agent_decisions()
    t0, idle0, act = None, None, ("MDD", "PTECT")
    try:
        gen.send(None)
        while True:
            if t0 is None:
                t0, idle0 = tw.now, idle_total()
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
    u = None
    if util and idle0 is not None:
        n = len(tw.agvs)
        duty = RH.on_duty_seconds(ep.sched, n, t0, t1)
        u = (n * (t1 - t0) - (idle_total() - idle0)) / duty if duty > 0 else None
    return tard(t0, t1), blocks, (t0, t1), u


def law_alpha(cell, root=HERE):
    """The fitted law's alpha for @p cell (law.json, written by analyze.py; a homogeneous floor, so 3 x h(x))."""
    return float(json.loads((Path(root) / "law.json").read_text())["cells"][cell]["alpha"])


def run(cell, seed, kind, root=HERE):
    out = Path(root) / "out" / cell / ("law" if kind == "law" else "") / f"s{seed}.json"
    if out.exists():
        return f"exists {out}"
    alphas = {"curve": ALPHAS, "util": (3.0,), "law": None}[kind]
    if kind == "law":
        alphas = (law_alpha(cell, root),)
    rb = load(cell)
    t = time.time()
    ep = rb.sim.Episode("B2", seed)
    res, win, util = {}, None, None
    for a in alphas:
        total, blocks, win, u = play(rb, ep, a, util=(a == 3.0))
        res[f"MDD-PTECT@{a:g}"] = {"total": total, "blocks": blocks}
        util = u if a == 3.0 else util
    res["_blocks"] = [{k: b.get(k) for k in ("block", "start", "end", "load_profile", "agv_count")} for b in ep.blocks
                      if b["end"] > win[0] and b["start"] < win[1]]
    res["_window"] = list(win)
    res["_cell"], res["_kind"] = cell, kind
    res["_stats"] = dict(AS.stats(rb, ep, {}, seed), agv_util=util)
    res["_stamp"] = rb.stamp()
    res["_seconds"] = time.time() - t
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(res))
    os.replace(tmp, out)
    tot = {k: v["total"] for k, v in res.items() if not k.startswith("_")}
    best = min(tot, key=tot.get)
    return (f"{cell} s{seed} ({kind}): best {best} {tot[best]:.2f}, MDD-PTECT@3 {tot.get('MDD-PTECT@3', float('nan')):.2f}, "
            f"AGV util {util if util is None else round(util, 3)}, {res['_seconds']:.0f} s")


## @brief Stage 2's law runs (MDD-PTECT at the fitted law's alpha, test seeds; ../eval-alpha-transport-size): its six
##        policy-test cells and the two held-out cells, longest first.
LAW_CELLS = ("m200_r052", "m100_r035", "m150_r080", "m100_r052", "m100_r080", "m75_r035", "m75_r052", "m50_r052")


def tasks(kind=None):
    if kind == "law":
        return [(cell, s, "law") for cell in LAW_CELLS for s in TEST_SEEDS]
    return [(cell, s, kind) for cell, kind, seeds in PLAN for s in seeds]


def main(argv):
    if argv[0] == "list":
        for cell, s, kind in tasks(argv[1] if len(argv) > 1 else None):
            print(cell, s, kind)
    else:
        print(run(argv[0], int(argv[1]), argv[2] if len(argv) > 2 else "curve", *argv[3:4]), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
