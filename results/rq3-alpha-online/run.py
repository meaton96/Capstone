"""
@file run.py
@brief rq3-alpha-online (2026-10-29, user: "design the online alpha experiment (h from the on-duty fleet per regime,
       plus the spread law) with pre-declared criteria, and set it up in the lab"): stage 1, the per-regime alpha
       surfaces the online law is fitted on, in the frozen v5 simulator.
The static law of ../rq3-alpha-transport sets one alpha per floor from x = tau / a, a the full fleet per machine. The
online law reads a from the AGVs on duty at each decision, so x doubles (or more) in short-fleet blocks. To fit and
score it, each instance is played at every pair (alpha_full, alpha_short) of ALPHAS: MDD-PTECT (quad,
TwinConfig.ptime_alpha) at alpha_full while the whole fleet is on duty (normal and surge blocks) and at alpha_short
while fewer AGVs are (short-fleet blocks). The pair's alpha is set before every agent decision from the on-duty count
at that moment (the engine's own _on_duty rule), which is what the policy harness of stage 2 does; the diagonal
(a, a) is ../rq3-alpha-transport's fixed-alpha run exactly (checked).
Cells: ../rq3-alpha-transport's (b2 = B2, r052 = the size grid, r035 / r080 = round(0.35 / 0.80 M) AGVs on duty,
short blocks half) and ../rq3-alpha-size's speed-spread cells (m<M>_s<s>, grid fleet, capacity-matched spread).
Kinds of task:
  surf   the 49 pairs on one instance: the fit cells on fitting seeds, the held-out cells on test seeds;
  stats  the floor statistics only (no simulation), held-out cells on fitting seeds (the law reads them from there);
  law    (stage 2, after analyze.py has written law_online.json) MDD-PTECT under the laws of law_online.json, test
         seeds: out/<cell>/law/s<seed>.json.
Frozen code and simulator: ../rq2-rebase-v5 (6d13412a + sim.py). Seeds: fitting 2000-2039, test 0-39.
Output: out/<cell>/s<seed>.json (surf, stats), out/<cell>/law/s<seed>.json.
@par Usage (from the repo root)
@code{.sh}
.venv/bin/python results/rq3-alpha-online/run.py m100_r052 2000
.venv/bin/python results/rq3-alpha-online/run.py list          # stage 1 tasks, longest first
.venv/bin/python results/rq3-alpha-online/run.py list law      # stage 2's law tasks (needs law_online.json)
@endcode
"""
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = HERE.parent
FIT_SEEDS = list(range(2000, 2040))
TEST_SEEDS = list(range(40))
## @brief The alpha grid of both axes (../rq3-alpha-transport's less 0.75 and 1.5, to keep 49 pairs).
ALPHAS = (0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0)
PAIRS = [(af, ash) for af in ALPHAS for ash in ALPHAS]
## @brief ../rq3-alpha-transport's 14 fit cells less m200_r035 (x 0.53 full, 1.07 short: alpha 0 under every
##        candidate law with x1 < 0.53, its fixed-alpha curve falls monotonically to alpha 0, and it would cost a
##        third of the run).
FIT = ["m15_b2", "m20_r052", "m30_r052", "m50_r052", "m75_r052", "m100_r052", "m150_r052", "m200_r052",
       "m50_r035", "m50_r080", "m100_r035", "m100_r080", "m200_r080"]
## @brief Never fitted on: ../rq3-alpha-transport's held-out size / fleet pairs, and speed-spread floors (the
##        spread law's factor times h).
HELD_OUT = ["m75_r035", "m150_r080", "m15_s1", "m50_s1", "m100_s0.5", "m100_s1"]
## @brief Seconds per episode under a full local load (../rq3-alpha-transport, ../rq3-alpha-size timings), for the
##        longest-first order.
EP_SECONDS = {"m200_r052": 66, "m200_r080": 49, "m150_r052": 30, "m150_r080": 13, "m100_r035": 17, "m100_r052": 20,
              "m100_s0.5": 20, "m100_s1": 20, "m100_r080": 15, "m75_r052": 12, "m75_r035": 5, "m50_r052": 7.5,
              "m50_s1": 7.5, "m50_r035": 6, "m50_r080": 6, "m30_r052": 4, "m20_r052": 3, "m15_b2": 1.6,
              "m15_s1": 1.6}


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


AS = _module("alpha_size_run", R / "rq3-alpha-size" / "run.py")              # spread loader, stats
AT = _module("alpha_transport_run", R / "rq3-alpha-transport" / "run.py")    # b2 / r0xx loader


def spread_cell(cell):
    return "_s" in cell


def load(cell):
    """(frozen simulator pointed at the cell, generator params): one call per process (the loaders wrap the scenario
    registry)."""
    if spread_cell(cell):
        return AS.load(cell)
    return AT.load(cell), {}


def on_duty(tw):
    """AGVs that may take a new task now: the engine's _on_duty rule (the last agv_schedule entry at or before now)."""
    n = len(tw.agvs)
    for t, k in tw.cfg.agv_schedule:
        if t > tw.now:
            break
        n = k
    return n


def play(rb, ep, alpha_of):
    """MDD-PTECT with alpha = alpha_of(on-duty AGVs, full fleet) set before every agent decision. Returns the window
    tardiness (1000 job-s, total and per regime block), the window, and the decisions taken short of the full fleet.
    The full fleet is the largest on-duty count of the schedule (the size grid's fleet wrapper keeps spare AGVs
    parked, so len(agvs) can be larger)."""
    from des_twin import TwinConfig
    from des_twin.engine import Twin
    sim = rb.sim
    full = max(int(k) for _, k in ep.sched) if ep.sched else None     # None: no schedule, the whole fleet
    tw = Twin(sim.sig.floor(), ep.jobs, TwinConfig(
        rule=ep.warm_rule, transport="kinematic", warmup_seconds=ep.warmup,
        episode_duration_seconds=ep.cfg["window"], max_sim_seconds=ep.warmup + ep.cfg["window"] + 200000.0,
        agv_schedule=ep.sched, ptime_alpha=alpha_of(full, full)))
    gen = tw.agent_decisions()
    t0, act, n_dec, n_short = None, ("MDD", "PTECT"), 0, 0
    try:
        gen.send(None)
        while True:
            if t0 is None:
                t0 = tw.now
            k = on_duty(tw)
            tw.cfg.ptime_alpha = alpha_of(k, full)
            n_dec += 1
            n_short += full is not None and k < full
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
    return tard(t0, t1), blocks, (t0, t1), (n_dec, n_short)


def pair_alpha(af, ash):
    """alpha(on-duty k, full fleet) of a pair; full None = an instance without a fleet schedule (whole fleet)."""
    return lambda k, full: af if (full is None or k >= full) else ash


def floor_stats(rb, ep, params, seed):
    """../rq3-alpha-size run.stats plus the short-fleet AGVs per machine (the smallest on-duty count of the
    schedule)."""
    st = AS.stats(rb, ep, params, seed)
    counts = [int(k) for _, k in ep.sched] if ep.sched else []
    st["agvs_per_machine_short"] = (min(counts) / st["machines"]) if counts else st["agvs_per_machine"]
    st["tau"] = st["travel_mean_s"] / st["p_mean_s"]
    return st


# ── stage 2's law runs ─────────────────────────────────────────────────────────────────────────────────────────────

def hinge(x, x0, x1):
    return min(1.0, max(0.0, (x1 - x) / (x1 - x0)))


def law_alpha_fn(law, cell_stats, sigma):
    """alpha(on-duty k, full) of one law of law_online.json for a floor: min(4, max(0, 3 - 6 sigma) x h(x_k)),
    x_k = tau / (k / M), h the law's hinge for that fleet state ('full' / 'short' knots; a static law uses the full
    fleet's x in every block). Rounded to 3 decimals, as the policy harness does."""
    spread = max(0.0, 3.0 - 6.0 * sigma)
    tau, m = cell_stats["tau"], cell_stats["machines"]

    def f(k, full):
        if full is None:                    # no fleet schedule: the whole fleet all episode (the full-fleet x)
            return round(min(4.0, spread * hinge(cell_stats["x_full"], *law["knots"]["full"])), 3)
        regime = "full" if k >= full else "short"
        knots = law["knots"][regime] if law["form"] != "static" else law["knots"]["full"]
        kk = full if law["form"] == "static" else k
        if law.get("short_alpha") is not None and regime == "short":
            return float(law["short_alpha"])
        return round(min(4.0, spread * hinge(tau / (kk / m), *knots)), 3)
    return f


def run(cell, seed, kind=None, root=HERE):
    kind = kind or ("stats" if (cell in HELD_OUT and seed in FIT_SEEDS) else "surf")
    out = Path(root) / "out" / cell / ("law" if kind == "law" else "") / f"s{seed}.json"
    if out.exists():
        return f"exists {out}"
    rb, params = load(cell)
    t = time.time()
    ep = rb.sim.Episode("B2", seed)
    st = floor_stats(rb, ep, params, seed)
    res = {}
    win = None
    if kind == "surf":
        for af, ash in PAIRS:
            total, blocks, win, dec = play(rb, ep, pair_alpha(af, ash))
            res[f"{af:g}|{ash:g}"] = {"total": total, "blocks": blocks, "decisions": dec}
    elif kind == "law":
        spec = json.loads((Path(root) / "law_online.json").read_text())
        cs = spec["cells"][cell]
        for name, law in spec["laws"].items():
            total, blocks, win, dec = play(rb, ep, law_alpha_fn(law, cs, st["sigma"]))
            f = law_alpha_fn(law, cs, st["sigma"])
            counts = [int(k) for _, k in ep.sched] or [None]
            full = max(counts) if counts[0] is not None else None
            res[name] = {"total": total, "blocks": blocks, "decisions": dec,
                         "alpha_full": f(full, full) if full is not None else f(1, None),
                         "alpha_short": f(min(counts), full) if full is not None else None}
    if win is not None:
        res["_blocks"] = [{k: b.get(k) for k in ("block", "start", "end", "load_profile", "agv_count")}
                          for b in ep.blocks if b["end"] > win[0] and b["start"] < win[1]]
        res["_window"] = list(win)
    res["_sched"] = [list(e) for e in ep.sched]
    res["_cell"], res["_kind"], res["_seed"] = cell, kind, seed
    res["_stats"] = st
    res["_stamp"] = rb.stamp()
    res["_seconds"] = time.time() - t
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(res))
    os.replace(tmp, out)
    if kind == "surf":
        tot = {k: v["total"] for k, v in res.items() if not k.startswith("_")}
        best = min(tot, key=tot.get)
        return (f"{cell} s{seed} (surf): best {best} {tot[best]:.2f}, diag 3|3 {tot['3|3']:.2f}, 0|0 {tot['0|0']:.2f}, "
                f"{res['_seconds']:.0f} s")
    if kind == "law":
        return f"{cell} s{seed} (law): " + ", ".join(f"{k} {v['total']:.2f}" for k, v in res.items()
                                                      if not k.startswith("_")) + f", {res['_seconds']:.0f} s"
    return f"{cell} s{seed} (stats): tau {st['tau']:.3f}, a {st['agvs_per_machine']:.3f} / {st['agvs_per_machine_short']:.3f}"


def tasks(kind=None):
    if kind == "law":
        spec = json.loads((HERE / "law_online.json").read_text())
        cells = sorted(spec["law_cells"], key=lambda c: -EP_SECONDS.get(c, 1))
        return [(c, s, "law") for c in cells for s in TEST_SEEDS]
    t = [(c, s, "surf") for c in FIT for s in FIT_SEEDS] + [(c, s, "surf") for c in HELD_OUT for s in TEST_SEEDS]
    t.sort(key=lambda x: -EP_SECONDS[x[0]])
    return t + [(c, s, "stats") for c in HELD_OUT for s in FIT_SEEDS]


def main(argv):
    if argv[0] == "list":
        for cell, s, kind in tasks(argv[1] if len(argv) > 1 else None):
            print(cell, s, kind)
    else:
        print(run(argv[0], int(argv[1]), argv[2] if len(argv) > 2 else None, *argv[3:4]), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
