"""
@file run.py
@brief rq2-replan-axes (2026-10-29, plan §5b, paper C's main contribution): what sets the best re-plan interval Δ* of the
       twin planner? One task = (variant, seed) -> out/<variant>/s<seed>.json: planner v2 (fork the running twin, roll
       each of the 6 v5 pairs out to the window end, keep the best) re-planning every Δ, against MDD-PTECT, on a B2
       instance put through one time-scale arm. The gain curve over Δ per condition gives Δ* (analyze.py).

Axis 1, time scale (this folder's first run). Three arms, each a one-at-a-time change of B2 by a factor k:
  all   every duration x k: operations, arrivals, due dates, regime blocks, the AGV schedule, warm-up and window, and
        transport (AGV speed / k, turn speed / k, handshake x k, pre-dispatch lead x k, and the tick dt x k, so the
        tick lattice scales too). A change of units: the run must be the k=1 run in other units (check C0).
  proc  as `all` but transport stays physical (speed, turn rate, handshake, lead, dt unchanged): processing and the
        clocks tied to it scale, the transport share τ / p̄ falls as 1 / k. The user's "scale processing but not
        transport, to break the invariance".
  reg   only the regime blocks change: B2 regenerated with regime_block_seconds = 5400 k (load surges and short-fleet
        blocks k times as long); processing, transport and the window unchanged. Separates p̄ from the regime clock,
        which `proc` scales together.
Δ = s x D1, with D1 the B2-equivalent interval (D1_GRID) and s = k for all / proc (the clock of operations scales),
s = 1 for reg. proc-k1 = reg-k1 = B2 (rq2-replan's exact-d<D1> where the grids overlap: check C1).

Axis 6, floor size and fleet ratio (README §13, approved 10-30, done 11-04). Two arms on the floor-size grid's
conditions (../rq3-size-grid: M bays, M / 5 machines per type, B2 load x 0.9, arrival cap off; built here as
../rq3-size-headroom builds them, with this folder's frozen code):
  size  M machines, round(0.52 M) AGVs on duty (short-fleet blocks half): variant size-m<M>-d<D1>. m15 is B2 (proc-k1).
  fleet M = 50, round(r M) AGVs on duty (short blocks half): variant fleet-r<r>-d<D1>. r 0.52 is size-m50.
Operations keep B2's clock, so Δ = D1 in both arms.

Axis 2, load x due-date allowance at two regime block lengths (README §15, approved 11-04). B2's generator on B2's floor
with every target utilization (normal, surge, short-fleet and lull) x ℓ (the AGV arrival cap kept), each block's c drawn
from [c - 0.375, c + 0.375] (B2's width; c 2.125 is B2's [1.75, 2.5]) and regime blocks of 5,400 s (B2) or 1,350 s:
  ax2   variant ax2-l<ℓ>-c<c>-b<block>[-<mix>]-d<D1>, ℓ 0.7 / 1 / 1.3 x c 1.5 / 2.125 / 3 x block 5400 / 1350, plus the
        regime-mix cells at ℓ 1, c 2.125 without B2's surge blocks (-nosurge) or short-fleet blocks (-noshort). The two
        centres (ℓ 1, c 2.125, B2's mix) are axis 1's proc-k1 (5,400 s) and reg-k0.25 (1,350 s) and are not run.
Δ = D1. The jobs of a (ℓ, block, mix, seed) are the same at every c; the fixed-pair task records their hash (check C3).

Variant names: <arm>-k<k>-d<D1>, e.g. proc-k4-d300 (Δ = 1,200 s); <arm>-k<k>-fixed is the level's fixed-pair task (the
6 v5 pairs held all window: the plan-once baseline of the gain, README §3, and the instance descriptors). Forecast: exact (the planner forks the world; with
no failures and no uncertainty trace nothing is hidden from it). Later axes (disruptions, forecast quality) need the
event-driven planner model of README §6.

Frozen code: code/env = `git archive` of env/ at 09f53cbb (code/STAMP), first on sys.path; the module asserts that
nothing under test is imported from the repo's live env/.
@par Usage
@code{.sh}
.venv/bin/python results/rq2-replan-axes/run.py proc-k4-d300 0
.venv/bin/python results/rq2-replan-axes/run.py list axis1       # or axis1-ext (README §11), axis6 / size / fleet (§13), axis2 (§15)
.venv/bin/python results/rq2-replan-axes/run.py describe proc-k4 0     # instance descriptors only (MDD-PTECT run)
.venv/bin/python results/rq2-replan-axes/run.py ax2-l1.3-c1.5-b1350-d300 0
.venv/bin/python results/rq2-replan-axes/run.py describe ax2-l1-c2.125-b5400-nosurge 0
@endcode
"""
import copy
import dataclasses
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CODE = HERE / "code"
MAIN = HERE.parents[1]
sys.path.insert(0, str(CODE / "env"))
import config                                          # noqa: E402,F401
import des_twin                                        # noqa: E402
import des_twin.branch                                 # noqa: E402
import des_twin.engine                                 # noqa: E402
import des_twin.planner                                # noqa: E402
import des_twin.scenario                               # noqa: E402
import scenarios.randomized                            # noqa: E402

_spec = importlib.util.spec_from_file_location("sim", MAIN / "results/rq2-oracle-steady/sim.py")
sim = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sim)                          # B2 instances (rq2-twin-fleet's generator settings)
while str(CODE / "env") in sys.path:                   # sim.py's import chain put the live env/ first; undo that
    sys.path.remove(str(CODE / "env"))
sys.path.insert(0, str(CODE / "env"))


def live_env_modules():
    return [n for n, m in list(sys.modules.items())
            if getattr(m, "__file__", None) and str(m.__file__).startswith(str(MAIN / "env") + "/")]


assert not live_env_modules(), f"modules imported from the repo's live env/: {live_env_modules()}"
assert str(CODE) in des_twin.__file__, des_twin.__file__

from des_twin import TwinConfig                        # noqa: E402
from des_twin import branch                            # noqa: E402
from des_twin.engine import Twin                       # noqa: E402
from des_twin.floor import Floor                       # noqa: E402
from des_twin.planner import _play_tail, belief_transform, fixed_rule, window_tardiness   # noqa: E402
from des_twin.scenario import _f32, agv_schedule, episode_settings, resolve_jobs         # noqa: E402

PAIRS = [f"{j}-{m}" for j in ("SRT", "MDD", "ATC") for m in ("PECT", "PTECT")]   # the 6 v5 pairs
REF = "MDD-PTECT"
SEEDS = list(range(20))                                # B2 test seeds (rq2-replan used 0-39)
SEEDS_EXT = list(range(20, 40))                        # rule 5 (README §6): an arm whose reading S is indeterminate
D1_GRID = (120, 180, 300, 450, 600, 900, 1800, 3600)   # B2-equivalent intervals (s); Δ / p̄ ≈ D1 / 360 at k = 1
D1_EXTRA = (60, 7200)                                  # added per level only if Δ* is censored at a grid end (README §4)
LEVELS = {"all": (0.125, 10.0, 32.0), "proc": (0.5, 1.0, 2.0, 4.0, 16.0), "reg": (0.25, 0.5, 2.0, 4.0),
          "size": (20, 30, 50, 100), "fleet": (0.35, 0.8)}   # axis 6 (README §13): m15 = proc-k1, fleet r 0.52 = size-m50
SEEDS6 = list(range(40))                               # axis 6: 40 seeds from the start (user, 10-30)
D1_GRID6 = (60,) + D1_GRID                             # axis 6: D1 60 s in the grid from the start (user, 10-29)
# single-core seconds of one re-plan over the full window, from the 10-29 dry run (README §13; B2 = 9.7 s): orders
# the task lists costliest first, nothing else
C_W = {("size", 20): 11.6, ("size", 30): 17.6, ("size", 50): 30.2, ("size", 100): 58.6, ("fleet", 0.35): 29.0,
       ("fleet", 0.8): 36.3}
CHECK_D1, CHECK_SEEDS = (300, 900), range(5)           # C0: the `all` arm on a few seeds and intervals
BLOCK_S = 5400.0                                       # B2's regime block
FLOOR_PATH = Path(sim.sig.FLOOR)
WINDOW = sim.SETTINGS["B2"]["window"]
# Axis 6 (README §13, draft): the floor-size grid's exports and generator settings (../rq3-size-grid/task.py,
# ../rq3-size-v5/task.py size_params, ../rq3-gen-v5/run_task.py B2 and fleet_wrapper)
SIZE_FLOORS = {20: "dev-size-floors/floors/m20", 30: "dev-size-floors/floors/m30", 50: "agv-ratio-scale/floors/m50",
               75: "dev-size-floors/floors/m75", 100: "agv-ratio-scale/floors/m100",
               150: "dev-size-floors/floors/m150", 200: "dev-size-floors/floors/m200"}
SIZE_LOAD = 0.9                                        # B2's utilization ranges x 0.9 above 15 machines
SIZE_RATIO = 0.52                                      # AGVs on duty per machine in the size arm (the grid's)
FLEET_M = 50                                           # floor of the fleet arm
B2_PARAMS = {"failure_probability": 0.0, "due_date_allowance_range": [1.75, 2.5], "regime_block_seconds": 5400,
             "horizon_seconds": 30600,
             "load_mix": [{"name": "normal", "utilization": [0.6, 1.2], "weight": 2},
                          {"name": "surge", "utilization": [1.0, 1.6]},
                          {"name": "short", "utilization": [0.6, 1.2], "agv_count": 2}]}
PREFIX = {"size": "m", "fleet": "r"}                   # level letter in variant names (k otherwise)
# Axis 2 (README §15, approved by the user 11-04 as proposed): cells (ℓ, c, block, mix)
AX2_LOADS, AX2_CS, AX2_BLOCKS, AX2_MIXES = (0.7, 1.0, 1.3), (1.5, 2.125, 3.0), (5400, 1350), ("nosurge", "noshort")
AX2_C_HALF = 0.375                                     # half-width of each block's c range (B2's [1.75, 2.5])
AX2_DROP = {"b2": None, "nosurge": "surge", "noshort": "short"}   # the load profile a mix leaves out of B2's
AX2_CENTRE = {5400: ("proc", 1.0), 1350: ("reg", 0.25)}           # ℓ 1, c 2.125, B2's mix: axis 1's levels, reused
LEVELS["ax2"] = tuple([(ell, c, b, "b2") for b in AX2_BLOCKS for ell in AX2_LOADS for c in AX2_CS
                       if (ell, c) != (1.0, 2.125)] + [(1.0, 2.125, b, m) for b in AX2_BLOCKS for m in AX2_MIXES])
# c_W of the 11-04 dry run (dryrun2/, idle box: B2 7.3 s there) on the 9.7 s basis above; ordering only
C_W.update({("ax2", k): round(9.7 / 7.3 * ({0.7: 5.0, 1.0: 7.4, 1.3: 8.7}[k[0]] if k[3] == "b2" else
                                         {"nosurge": 6.5, "noshort": 7.0}[k[3]]), 1) for k in LEVELS["ax2"]})


def fmt_k(k):
    return f"{k:g}"


def ax2_level(k):
    """Level name of an axis 2 cell (ℓ, c, block, mix): ax2-l<ℓ>-c<c>-b<block>[-<mix>]."""
    ell, c, b, mix = k
    return f"ax2-l{fmt_k(ell)}-c{fmt_k(c)}-b{int(b)}" + ("" if mix == "b2" else f"-{mix}")


def variant(arm, k, d1):
    if arm == "ax2":
        return f"{ax2_level(k)}-d{int(d1)}"
    return f"{arm}-{PREFIX.get(arm, 'k')}{fmt_k(k)}-d{int(d1)}"


def parse(v):
    """(arm, k, D1); D1 None for the fixed-pair task <arm>-k<k>-fixed. Axis 2: k = (ℓ, c, block, mix)."""
    if v.startswith("ax2-"):
        p = v.split("-")
        k = (float(p[1][1:]), float(p[2][1:]), int(p[3][1:]), p[4] if len(p) == 6 else "b2")
        return "ax2", k, (None if p[-1] == "fixed" else float(p[-1][1:]))
    arm, k, d = v.split("-")
    return arm, float(k[1:]), (None if d == "fixed" else float(d[1:]))


def task_sets():
    """Task lists for the lab: check (C0, the `all` arm), proc, reg (k = 1 lives in proc only); proc and reg include
    their fixed-pair tasks. axis1-ext (README §11): rule 5, both arms on seeds 20-39 (grid and fixed-pair tasks), and
    rule 6, D1 60 s at every proc / reg level on seeds 0-39 (the best gain sat at the finest grid point everywhere)."""
    out = {"check": [(variant("all", k, d), s) for k in LEVELS["all"] for d in CHECK_D1 for s in CHECK_SEEDS]}
    for arm in ("proc", "reg"):
        out[arm] = [(variant(arm, k, d), s) for k in LEVELS[arm] for d in D1_GRID for s in SEEDS]
        out[arm] += [(f"{arm}-k{fmt_k(k)}-fixed", s) for k in LEVELS[arm] for s in SEEDS]
    out["axis1"] = out["check"] + out["proc"] + out["reg"]
    arms = [(arm, k) for arm in ("proc", "reg") for k in LEVELS[arm]]
    out["axis1-ext"] = ([(variant(a, k, d), s) for a, k in arms for d in D1_GRID for s in SEEDS_EXT]
                        + [(f"{a}-k{fmt_k(k)}-fixed", s) for a, k in arms for s in SEEDS_EXT]
                        + [(variant(a, k, D1_EXTRA[0]), s) for a, k in arms for s in SEEDS + SEEDS_EXT])
    for arm in ("size", "fleet"):                       # axis 6 (README §13, approved 10-30)
        out[arm] = [(variant(arm, k, d), s) for k in LEVELS[arm] for d in D1_GRID6 for s in SEEDS6]
        out[arm] += [(f"{arm}-{PREFIX[arm]}{fmt_k(k)}-fixed", s) for k in LEVELS[arm] for s in SEEDS6]
    out["axis6"] = out["size"] + out["fleet"]
    out["axis2"] = ([(variant("ax2", k, d), s) for k in LEVELS["ax2"] for d in D1_GRID6 for s in SEEDS6]   # §15
                    + [(f"{ax2_level(k)}-fixed", s) for k in LEVELS["ax2"] for s in SEEDS6])
    return out


def est_cost(v, seed):
    """Relative single-core cost of a task: (n + 1) / 2 full-window re-plans at n = window / Δ re-plans, times the
    level's re-plan cost (README §13); a fixed-pair task is about one full-window re-plan."""
    arm, k, d1 = parse(v)
    c = C_W.get((arm, k), 9.7)
    return c if d1 is None else (WINDOW / d1 + 1) / 2 * c


# ── Instances ─────────────────────────────────────────────────────────────────────────────────────────────────

_FLOORS = {}


def floor_for(kt):
    """The exported 7-AGV layout-D floor with every transport duration x kt: AGV speed and turn rate / kt, handshake and
    pre-dispatch lead x kt, and the tick dt x kt (so a kinematic leg takes the same ticks; exact for kt = 2^n)."""
    if kt not in _FLOORS:
        if kt == 1.0:
            _FLOORS[kt] = sim.sig.floor()
        else:
            d = json.loads(FLOOR_PATH.read_text())
            d["fixed_dt"] = float(d["fixed_dt"]) * kt
            d["pre_dispatch_lead"] = float(d["pre_dispatch_lead"]) * kt
            a = d["agv"]
            a["speed"], a["turn_speed"], a["handshake"] = a["speed"] / kt, a["turn_speed"] / kt, a["handshake"] * kt
            _FLOORS[kt] = Floor(d)
    return _FLOORS[kt]


def scale_jobs(jobs, k):
    """des_jobs/1 with arrivals, due dates and every eligible duration x k (float32 as resolve_jobs writes them)."""
    out = dict(jobs)
    out["jobs"] = [dict(j, arrival=_f32(j["arrival"] * k), due=None if j["due"] is None else _f32(j["due"] * k),
                        ops=[dict(op, eligible=[[mid, _f32(d * k)] for mid, d in op["eligible"]]) for op in j["ops"]])
                   for j in jobs["jobs"]]
    return out


def scenario_reg(seed, k):
    """B2 with regime blocks of 5400 k s (rq2-twin-fleet's B2 parameters otherwise); k = 1 is B2 itself."""
    if k == 1.0:
        return sim.scenario("B2", seed)
    from scenarios.randomized import DEFAULT_PARAMS, randomized_generator
    fl = sim.fleet
    params = dataclasses.replace(DEFAULT_PARAMS, failure_probability=0.0, due_date_allowance_range=(1.75, 2.5),
                                 **{**fl.SETTINGS["B2"]["params"], "regime_block_seconds": BLOCK_S * k})
    return randomized_generator(fl.SETTINGS["B2"]["window"], random_warmup=True, params=params)(seed)


def scenario_ax2(seed, k):
    """Axis 2 cell k = (ℓ, c, block, mix) (README §15): B2's generator (rq2-twin-fleet's B2 parameters) with every target
    utilization x ℓ (lull included; the AGV arrival cap kept), each block's c from [c - 0.375, c + 0.375], regime blocks
    of `block` s, and B2's load mix without the profile `mix` drops. (1, 2.125, 5400, b2) is B2 itself."""
    from scenarios.randomized import DEFAULT_PARAMS, randomized_generator
    ell, c, block, mix = k
    fl = sim.fleet
    b2 = fl.SETTINGS["B2"]["params"]
    lull = tuple(round(u * ell, 4) for u in DEFAULT_PARAMS.lull_utilization)
    profiles = tuple(dict(p, utilization=tuple(round(u * ell, 4) for u in p["utilization"]), lull_utilization=lull)
                     for p in b2["load_mix"] if p["name"] != AX2_DROP[mix])
    params = dataclasses.replace(DEFAULT_PARAMS, failure_probability=0.0,
                                 due_date_allowance_range=(round(c - AX2_C_HALF, 4), round(c + AX2_C_HALF, 4)),
                                 **{**b2, "regime_block_seconds": float(block), "load_mix": profiles})
    return randomized_generator(fl.SETTINGS["B2"]["window"], random_warmup=True, params=params)(seed)


def jobs_hash(jobs):
    """Hash of a job set's arrivals and eligible machines / durations (the c levels of an axis 2 cell share it)."""
    return hashlib.md5(json.dumps([(j["arrival"], [op["eligible"] for op in j["ops"]]) for j in jobs["jobs"]])
                       .encode()).hexdigest()[:12]


def size_fleet(m, r):
    """AGVs on duty in full-fleet and in short-fleet blocks (../rq3-size-headroom fleet())."""
    on = round(r * m)
    return on, round(on / 2)


def scenario_size(m, r, seed):
    """The floor-size grid's instance on m machines with round(r m) AGVs on duty: B2 with m / 5 machines per type and
    utilization x 0.9 (rq3-size-v5 size_params), through rq3-gen-v5's fleet wrapper (agvCount = m bays; full-fleet blocks
    get the on-duty fleet, short blocks keep theirs). The job stream does not depend on r."""
    from scenarios import REGISTRY
    on, short = size_fleet(m, r)
    p = copy.deepcopy(B2_PARAMS)
    p.update(machines_per_type=m // 5, agv_count=on, agv_max_utilization=10.0)
    for prof in p["load_mix"]:
        prof["utilization"] = [round(u * SIZE_LOAD, 4) for u in prof["utilization"]]
        if prof["name"] == "short":
            prof["agv_count"] = short
    gen = REGISTRY["randomized"](WINDOW, random_warmup=True, warmup_dispatching_rule=None, agv_move_speed=None,
                                 agv_handshake_duration=None, params_overrides=p)
    s = copy.deepcopy(gen(seed))
    full = int(s["agvCount"])
    sched = s.get("agvSchedule") or [{"start": 0, "agvCount": full}]
    s["agvSchedule"] = [{"start": e["start"], "agvCount": on if int(e["agvCount"]) == full else min(int(e["agvCount"]), on)}
                        for e in sched]
    s["agvCount"] = m
    return s


def on_duty_seconds(sched, n_agv, now):
    """AGV-seconds on duty in [0, now] under a TwinConfig.agv_schedule ((start, count) pairs; all n_agv before the
    first entry), as ../rq3-size-headroom on_duty_seconds."""
    pts = [(-1e18, n_agv)] + [(float(t), int(k)) for t, k in sched] + [(1e18, None)]
    return sum((min(now, t1) - max(0.0, t0)) * k for (t0, k), (t1, _) in zip(pts, pts[1:]) if min(now, t1) > max(0.0, t0))


def size_floor(m):
    if m not in _FLOORS:
        _FLOORS[m] = Floor.load(str(MAIN / "results" / SIZE_FLOORS[m] / "des_floor.json"))
    return _FLOORS[m]


class AxEpisode:
    """One B2 instance under one time-scale arm and factor k, or (axis 6) one floor-size grid instance: arm size with
    k = machines, arm fleet with k = AGVs on duty per machine at FLEET_M machines, or (axis 2) one B2-floor instance of
    cell k = (ℓ, c, block, mix)."""

    def __init__(self, arm, k, seed):
        if arm not in ("all", "proc", "reg", "size", "fleet", "ax2"):
            raise ValueError(arm)
        self.arm, self.k, self.seed = arm, k, seed
        if arm in ("size", "fleet"):
            m, r = (int(k), SIZE_RATIO) if arm == "size" else (FLEET_M, k)
            sc, base = scenario_size(m, r, seed), size_floor(m)
        elif arm == "ax2":
            sc, base = scenario_ax2(seed, k), sim.sig.floor()
        else:
            sc, base = (scenario_reg(seed, k) if arm == "reg" else sim.scenario("B2", seed)), sim.sig.floor()
        self.sc = sc
        jobs = resolve_jobs(sc, base)
        warmup, _, self.warm_rule = episode_settings(sc)
        sched = agv_schedule(sc)
        self.scale = k if arm in ("all", "proc") else 1.0      # clock of operations, arrivals, due dates, regimes
        kt = k if arm == "all" else 1.0                        # clock of transport
        if self.scale != 1.0:
            jobs = scale_jobs(jobs, self.scale)
            warmup *= self.scale
            sched = tuple((t * self.scale, n) for t, n in sched)
        self.jobs, self.warmup, self.window, self.sched = jobs, warmup, WINDOW * self.scale, sched
        self.floor = base if arm in ("size", "fleet") else floor_for(kt)
        self.blocks = [dict(b, start=b["start"] * self.scale, end=b["end"] * self.scale)
                       for b in sc["_meta"].get("blocks", [])]

    def twin(self, rule=None):
        cfg = TwinConfig(rule=rule or self.warm_rule, transport="kinematic", warmup_seconds=self.warmup,
                         episode_duration_seconds=self.window,
                         max_sim_seconds=self.warmup + self.window + 200000.0 * self.scale, agv_schedule=self.sched)
        return Twin(self.floor, self.jobs, cfg)

    def fixed(self, pair):
        """Window tardiness (1000 job-s) of `pair` at every decision, and the twin after the run."""
        tw = self.twin()
        return fixed_rule(tw, pair, self.warmup), tw

    def describe(self, tw):
        """Time scales of the instance and of a finished run (for the dimensionless fit, README §5)."""
        t0, t1 = self.warmup, self.warmup + self.window
        win = [j for j in self.jobs["jobs"] if t0 <= j["arrival"] < t1]
        ops = [op for j in win for op in j["ops"]]
        p_mean = float(np.mean([np.mean([d for _, d in op["eligible"]]) for op in ops]))
        p_fast = float(np.mean([min(d for _, d in op["eligible"]) for op in ops]))
        rows = [r for j in tw.jobs.values() for r in j.op_rows if r.get("routed") is not None and t0 <= r["routed"] < t1]
        trans = [r["queued"] - r["routed"] for r in rows if r.get("queued") is not None]
        proc = [r["end"] - r["start"] for r in rows if r.get("end") is not None and r.get("start") is not None]
        now = tw.now
        idle = sum(a.idle_time + (max(0.0, now - a.idle_since) if tw._status(a) == "idle" else 0.0) for a in tw.agvs)
        trips = sum(a.trips for a in tw.agvs)
        busy_m = sum(m.busy for m in tw.machines)
        blocks = [b for b in self.blocks if b["end"] > t0 and b["start"] < t1]
        return {"p_mean_plan": p_mean, "p_fast_plan": p_fast, "jobs_window": len(win), "ops_window": len(ops),
                "proc_realized_mean": float(np.mean(proc)) if proc else None,
                "transport_mean": float(np.mean(trans)) if trans else None,
                "agv_busy_share": 1.0 - idle / max(1e-9, len(tw.agvs) * now),
                "agv_busy_per_trip": (len(tw.agvs) * now - idle) / max(1, trips),
                # busy AGV-seconds over on-duty AGV-seconds (agv_busy_share counts parked off-duty AGVs as idle; on
                # the size floors every machine has a bay and only round(r M) AGVs are on duty; axis 6, README §13)
                "agv_busy_on_duty": (len(tw.agvs) * now - idle) / max(1e-9, on_duty_seconds(self.sched, len(tw.agvs), now)),
                "machine_busy_share": busy_m / max(1e-9, len(tw.machines) * now),
                "block_seconds": float(blocks[0]["end"] - blocks[0]["start"]) if blocks else None,
                "blocks_in_window": len(blocks),
                "short_fleet_share": float(sum(min(b["end"], t1) - max(b["start"], t0) for b in blocks
                                               if b.get("load_profile") == "short") / self.window) if blocks else None}


# ── Planner (rq2-replan's plan_triggered: planner v2, belief on, no ensemble) ─────────────────────────────────────

def plan_triggered(tw, pairs, slot, warmup, window, plan_source=None, trigger=None):
    """des_twin.planner.plan_episode with a re-plan trigger (rq2-replan/run.py, unchanged): at boundary k > 0 the forks
    run only if trigger(t_last_plan, t_plan) is true, otherwise the current pair is kept. trigger None = plan at every
    boundary (= plan_episode). Kept for the comparator arms (periodic vs event-triggered vs hybrid, README §7)."""
    n = int(round(window / slot))
    gen = tw.agent_decisions()
    schedule, planned, plan_s, replans = [], [], [], []
    t0, k_next, t_last = None, 0, None
    tw.pause_at(warmup)
    try:
        y = gen.send(None)
        while True:
            if y.kind == "pause":
                t_plan = warmup + k_next * slot
                if k_next == 0 or trigger is None or trigger(t_last, t_plan):
                    fut = plan_source(t_plan, tw) if plan_source is not None else None
                    prefix = list(schedule)

                    def child(pair, prefix=prefix, fut=fut):
                        belief_transform(tw, fut)
                        try:
                            return _play_tail(tw, gen, prefix, pair, t0, slot, warmup)
                        except StopIteration:   # no agent decision left before the window ends: the end state
                            return window_tardiness(tw, warmup if t0 is None else t0)
                    t_start = time.time()
                    res = branch.fork_map(child, list(pairs), max_parallel=1)
                    plan_s.append(time.time() - t_start)
                    i = int(np.argmin(res))
                    schedule.append(pairs[i])
                    planned.append(float(res[i]))
                    replans.append(k_next)
                    t_last = t_plan
                else:
                    schedule.append(schedule[-1])
                k_next += 1
                if k_next < n:
                    tw.pause_at(warmup + k_next * slot)
                y = gen.send(None)
            else:
                if t0 is None:
                    t0 = tw.now
                kk = int((tw.now - t0) // slot)
                p = schedule[min(kk, len(schedule) - 1)]
                y = gen.send(tuple(p.split("-")))
    except StopIteration:
        pass
    return {"schedule": schedule, "planned": planned, "replan_slots": replans,
            "real": window_tardiness(tw, warmup if t0 is None else t0), "plan_seconds": plan_s, "t0": t0,
            "decisions": tw.decisions, "replans": len(replans)}


# ── Tasks ─────────────────────────────────────────────────────────────────────────────────────────────────────

def _stamp():
    h = hashlib.md5()
    for f in sorted((CODE / "env" / "des_twin").glob("*.py")):
        h.update(f.read_bytes())
    return {"code": (CODE / "STAMP").read_text().splitlines()[1], "des_twin_md5": h.hexdigest()[:12],
            "run_py_md5": hashlib.md5(Path(__file__).read_bytes()).hexdigest()[:12], "host": os.uname().nodename}


def _write(out, obj):
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, out)


def run(v, seed, root=HERE):
    out = Path(root) / "out" / v / f"s{seed}.json"
    if out.exists():
        return f"exists {out}"
    arm, k, d1 = parse(v)
    t = time.time()
    if d1 is None:
        obj = {"variant": v, "arm": arm, "k": k, **fixed_pairs(arm, k, seed), "seconds": None, "stamp": _stamp()}
        obj["seconds"] = time.time() - t
        assert not live_env_modules(), live_env_modules()
        _write(out, obj)
        return f"{v} s{seed}: best fixed {obj['best_pair']} {obj['best']:.6f}, {REF} {obj['fixed'][REF]:.6f}"
    e = AxEpisode(arm, k, seed)
    slot = d1 * e.scale
    res = plan_triggered(e.twin(), PAIRS, slot, e.warmup, e.window)
    ref, tw_ref = e.fixed(REF)
    sched = res["schedule"]
    obj = {"variant": v, "arm": arm, "k": k, "d1": d1, "slot_seconds": slot, "scale": e.scale, "seed": seed,
           "real": res["real"], "mdd_ptect": ref, "schedule": sched, "replans": res["replans"],
           "switches": sum(p != q for p, q in zip(sched, sched[1:])), "decisions": res["decisions"],
           "plan_seconds": float(sum(res["plan_seconds"])), "seconds": time.time() - t,
           "warmup": e.warmup, "window": e.window, "describe": e.describe(tw_ref), "stamp": _stamp()}
    assert not live_env_modules(), live_env_modules()
    _write(out, obj)
    return (f"{v} s{seed}: real {res['real']:.6f} vs {REF} {ref:.6f} ({100 * (res['real'] / ref - 1):+.2f}%), "
            f"{res['replans']} re-plans, {res['plan_seconds'] and sum(res['plan_seconds']):.0f} s planning, "
            f"{obj['seconds']:.0f} s")


def fixed_pairs(arm, k, seed):
    """The 6 v5 pairs each held all window (with the exact forecast the planner's single plan at the warm-up picks the
    best of them: the plan-once baseline) and the instance descriptors under MDD-PTECT."""
    e = AxEpisode(arm, k, seed)
    fixed, desc = {}, None
    for p in PAIRS:
        fixed[p], tw = e.fixed(p)
        if p == REF:
            desc = e.describe(tw)
    best = min(fixed, key=fixed.get)
    return {"seed": seed, "fixed": fixed, "best_pair": best, "best": fixed[best], "describe": desc,
            **({"jobs_hash": jobs_hash(e.jobs)} if arm == "ax2" else {})}


def describe(level, seed):
    """Descriptors of one (arm, k) instance under MDD-PTECT and the 6 fixed pairs (no planner, nothing written)."""
    arm, k, _ = parse(f"{level}-fixed")
    return {"level": level, **fixed_pairs(arm, k, seed)}


def main(argv):
    if argv[0] == "list":
        for v, s in sorted(task_sets()[argv[1] if len(argv) > 1 else "axis1"], key=lambda t: -est_cost(*t)):
            print(v, s)                                # costliest first (stable within equal cost)
    elif argv[0] == "describe":
        print(json.dumps(describe(argv[1], int(argv[2]))))
    else:
        print(run(argv[0], int(argv[1]), *argv[2:3]), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
