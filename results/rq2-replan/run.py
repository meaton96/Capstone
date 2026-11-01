"""
@file run.py
@brief rq2-replan (2026-10-21, user after the advisor meeting: "why 5 minutes? why 15?"). When should the twin
       planner re-plan? Planner v2 (des_twin.planner: fork the running twin, roll each v5 pair out to the window end,
       keep the best, hold-to-end tails) on B2 test seeds 0-39, with the uncertainty screen's harness and frozen code
       (../rq4-uncert-screen/uc.py, code/ = ba26f3f5; equal to v1 / rq2-perturbed-v5 where they overlap, checked).
Three parts, one task = (variant, seed) -> out/<variant>/s<seed>.json:
  1a granularity, exact forecast: interval Δ in {60, 120, 180, 300, 600, 900, 1200, 1800, 3600} s. With the exact
     future the planner is the greedy switching oracle at decision interval Δ.
  1b forecast error x interval: published schedule with arrival jitter σ 1 / 10 / 30 min, 10% churn (+ jitter 10 min) or
     duration errors 10% (rq2-perturbed-v5's Forecast, copied here), Δ in {120, 300, 900, 1800} s.
  1c event-triggered re-planning: pairs may switch every 300 s, but the plan is recomputed at a boundary only if an event
     happened since the last re-plan (otherwise the current pair is kept, i.e. the last plan's hold-to-end choice):
       ev2 / ev10: an arrival off its published time by more than 2 / 10 min, an unannounced arrival (churn), or a
                   published job overdue by more than the threshold (late or cancelled);
       hyb10:      ev10 or 30 min since the last re-plan.
     Levels exact, jit10, jit30, churn10. The first boundary always plans.
Variant names: <level>-d<Δ> (1a / 1b) and <level>-d300-<trigger> (1c).
@par Usage
@code{.sh}
python results/rq2-replan/run.py exact-d300 0
python results/rq2-replan/run.py list
@endcode
"""
import copy
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
R = HERE.parent
_spec = importlib.util.spec_from_file_location("uc", R / "rq4-uncert-screen" / "uc.py")
uc = importlib.util.module_from_spec(_spec)
sys.modules["uc"] = uc
_spec.loader.exec_module(uc)                          # frozen code first on sys.path; asserts no live env/ imports
from des_twin import branch                            # noqa: E402
from des_twin.planner import _play_tail, belief_transform, window_tardiness   # noqa: E402

sim = uc.sim
PAIRS = list(uc.V5)
SEEDS = list(range(40))
LEVELS = {                                             # rq2-perturbed-v5 / rq2-oracle-perturbed, unchanged
    "exact": {},
    "jit1": {"jitter": 60.0}, "jit5": {"jitter": 300.0},
    "jit10": {"jitter": 600.0}, "jit30": {"jitter": 1800.0}, "jit60": {"jitter": 3600.0},
    "proc02": {"proc": 0.02}, "proc10": {"proc": 0.10}, "proc25": {"proc": 0.25},
    "churn10": {"churn": 0.10, "jitter": 600.0}, "churn25": {"churn": 0.25, "jitter": 600.0},
    "redrawn": {"redrawn": True},
}
OTHER_SEED = 70000
TRIGGERS = {"ev2": (120.0, None), "ev10": (600.0, None), "hyb10": (600.0, 1800.0)}
D1A = (60, 120, 180, 300, 600, 900, 1200, 1800, 3600)
L1B, D1B = ("jit1", "jit10", "jit30", "churn10", "proc10"), (120, 300, 900, 1800)
L1C = ("exact", "jit10", "jit30", "churn10")


def variants():
    v = [f"exact-d{d}" for d in D1A]
    v += [f"{lv}-d{d}" for lv in L1B for d in D1B]
    v += [f"{lv}-d300-{t}" for lv in L1C for t in TRIGGERS]
    return v


def parse(variant):
    parts = variant.split("-")
    return parts[0], float(parts[1][1:]), (parts[2] if len(parts) > 2 else None)


class Forecast:
    """rq2-perturbed-v5's published schedule (copied verbatim, sim = uc.sim): per-job errors drawn once per seed and
    level, applied to the jobs not yet arrived."""

    def __init__(self, ep, level, seed):
        self.ep, self.cfg = ep, LEVELS[level]
        rng = np.random.default_rng([seed, sorted(LEVELS).index(level)])
        jobs = self.source = ep.sc["jobs"]
        self.real_arrival = {int(j["id"]): j["arrivalTime"] for j in ep.sc["jobs"]}
        self.shift = rng.normal(0.0, self.cfg.get("jitter", 0.0), len(jobs)) if self.cfg.get("jitter") else \
            np.zeros(len(jobs))
        self.proc = [[float(np.exp(rng.normal(0.0, self.cfg["proc"]))) for _ in j["operations"]] for j in jobs] \
            if self.cfg.get("proc") else None
        churn = self.cfg.get("churn", 0.0)
        self.missing = rng.random(len(jobs)) < churn
        self.other = None
        if churn or self.cfg.get("redrawn"):
            other = sim.scenario(ep.setting, OTHER_SEED + seed)["jobs"]
            if churn:
                pick = rng.random(len(other)) < churn * len(jobs) / max(len(other), 1)
                other = [j for j, keep in zip(other, pick) if keep]
            self.other = [dict(copy.deepcopy(j), id=int(j["id"]) + 100000) for j in other]

    def _perturbed(self, i, j, t):
        j = copy.deepcopy(j)
        if self.shift[i]:
            new = max(j["arrivalTime"] + self.shift[i], t + 1.0)
            if j.get("dueDate") is not None:
                j["dueDate"] += new - j["arrivalTime"]
            j["arrivalTime"] = new
        if self.proc is not None:
            for op, f in zip(j["operations"], self.proc[i]):
                op["duration"] = [d * f for d in op["duration"]]
        return j

    def _arrived(self, j, t):
        real = self.real_arrival.get(int(j["id"]))
        return real <= t if real is not None else j["arrivalTime"] <= t

    def episode(self, t):
        past = [j for j in self.ep.sc["jobs"] if j["arrivalTime"] <= t]
        future = []
        if self.cfg.get("redrawn"):
            future = [j for j in self.other if j["arrivalTime"] > t + 1.0]
        else:
            for i, j in enumerate(self.source):
                if self._arrived(j, t) or self.missing[i]:
                    continue
                future.append(self._perturbed(i, j, t))
            if self.other is not None:
                future += [j for j in self.other if j["arrivalTime"] > t + 1.0]
        if not self.cfg:
            return self.ep
        from des_twin.scenario import resolve_jobs
        sc = dict(self.ep.sc)
        sc["jobs"] = past + sorted(future, key=lambda j: j["arrivalTime"])
        f = copy.copy(self.ep)
        f.sc, f.jobs = sc, resolve_jobs(sc, sim.sig.floor())
        return f

    def source_fn(self):
        """plan_source for planner v2 (dev-twin-uncertainty/check_planner.py's forecast_source)."""
        def src(t, tw):
            arrived = {j.id for j in tw.order}
            return [jd for jd in self.episode(t).jobs["jobs"] if int(jd["id"]) not in arrived]
        return src

    def events(self, t_from, t_to, theta):
        """Number of events in (t_from, t_to] a controller would see: a real arrival off its published time by more than
        theta or unannounced (missing from the plan); a published job (real or phantom) overdue by more than theta."""
        n = 0
        for i, j in enumerate(self.source):
            real = j["arrivalTime"]
            if self.missing[i]:
                n += t_from < real <= t_to
                continue
            pub = real + float(self.shift[i])
            if t_from < real <= t_to and abs(real - pub) > theta:
                n += 1
            elif real > pub + theta and t_from < pub + theta <= t_to:
                n += 1                                   # still not here theta after its published time
        for j in self.other or []:
            n += t_from < j["arrivalTime"] + theta <= t_to   # a cancelled (phantom) job noticed theta after its time
        return n


def plan_triggered(tw, pairs, slot, warmup, window, plan_source, trigger):
    """des_twin.planner.plan_episode (planner v2, belief on, no ensemble) with a re-plan trigger: at boundary k > 0 the
    forks run only if trigger(t_last_plan, t_plan) is true; otherwise the current pair is kept (the last plan's
    hold-to-end choice). trigger None = plan at every boundary (= plan_episode)."""
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
                        except StopIteration:   # no agent decision left before the window ends (1-2 min slots,
                            # last boundary): the fork ran to the end; its tail is the end state, the same for all pairs
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


def run(variant, seed, root=HERE):
    out = Path(root) / "out" / variant / f"s{seed}.json"
    if out.exists():
        return f"exists {out}"
    level, slot, trig = parse(variant)
    t = time.time()
    e = uc.UEpisode(seed, None)
    fc = Forecast(e.ep, level, seed)
    src = None if not LEVELS[level] else fc.source_fn()
    trigger = None
    if trig:
        theta, floor_s = TRIGGERS[trig]
        trigger = lambda a, b: fc.events(a, b, theta) > 0 or (floor_s is not None and b - a >= floor_s - 1e-6)
    res = plan_triggered(e.twin(), PAIRS, slot, e.warmup, e.window, src, trigger)
    ref = e.play("MDD-PTECT")[0]
    sched = res["schedule"]
    obj = {"variant": variant, "level": level, "slot_seconds": slot, "trigger": trig, "seed": seed,
           "real": res["real"], "mdd_ptect": ref, "schedule": sched, "replan_slots": res["replan_slots"],
           "replans": res["replans"], "switches": sum(p != q for p, q in zip(sched, sched[1:])),
           "planned_last": res["planned"][-1] if res["planned"] else None,
           "plan_seconds": float(sum(res["plan_seconds"])), "seconds": time.time() - t,
           "stamp": (uc.CODE / "STAMP").read_text()[:200]}
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, out)
    return (f"{variant} s{seed}: real {res['real']:.3f} vs MDD-PTECT {ref:.3f} ({100 * (res['real'] / ref - 1):+.1f}%), "
            f"{res['replans']} re-plans, {obj['seconds']:.0f} s")


def main(argv):
    if argv[0] == "list":
        for v in variants():
            for s in SEEDS:
                print(v, s)
    else:
        print(run(argv[0], int(argv[1]), *argv[2:3]), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
