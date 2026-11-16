"""
@file run7.py
@brief rq2-replan-axes, axis 7 (README §19, drafted 11-10): run.py's planner with a compute latency T_c, a shifted
       re-plan lattice, a rollout horizon H and parallel forks. run.py is left unchanged (md5 1f09ef96, the file the
       axis 2 outputs stamp); this module imports its instances, episodes and stamps.

Latency model. Plan k >= 1 is computed at t_k = warm-up + k Δ (+ offset) from the state at t_k and acts from
a_k = t0 + k Δ + T_c, t0 being the first agent decision (run.py's planner has a_k = t0 + k Δ). The forks know it: each
plays the pair in force until a_k, then its candidate. Plan 0 acts at t0, as if computed during the warm-up.
Real time: one plan's compute ends before the next one starts, T_c <= Δ (asserted), so no two plans compute at once.
The switch lattice is anchored at t0 and the re-plan lattice at the warm-up, as in run.py. When the first decision
comes late (t0 - warm-up > Δ - T_c; 68-122 s on a few seeds at D1 60 s), plan k acts after plan k + 1 has started.
That is not overlapping compute: plan k's pair and the time it acts are known when plan k + 1 forks, and its forks
play the pairs in force until then. Asserted instead: a plan never acts before it is computed,
(t0 - warm-up) + T_c >= offset. (11-13: the guard of 11-10 required T_c + (t0 - warm-up) < Δ + offset and stopped 9 7b
tasks at T_c = 0, which run.py runs. The fix changes only that check: a run the old guard let through is unchanged.)
  offset   moves re-plans k >= 1 to t_k + offset. Check L0: latency T_c at offset 0 equals latency T_c at offset T_c,
           i.e. the zero-latency planner on a lattice shifted by T_c, which is what latency is under the exact forecast.
  horizon  each fork stops at t_k + H and scores the window tardiness accrued by then (None: to the window end, run.py's
           hold-to-end tail).
  par      forks at once (branch.fork_map max_parallel; 1 = run.py's serial planner). Changes only the wall time.
With latency 0, offset 0, horizon None and par 1 this is run.py's plan_triggered with trigger None (check L1).

Variants: a run.py planner variant plus options joined by '+': +lat<T_c s>, +off<s>, +h<H s>, +par<P>, e.g.
proc-k1-d900+lat300 or size-m50-d1800+lat600+off600. They never collide with run.py's variants. Outputs go to
<root>/out7/<variant>/s<seed>.json (root: this folder by default), never to run.py's out/ (11-10; the 11-10 dry run
wrote <root>/out/ under root dryrun7/).

7b, the horizon arm (README §19, approved 11-10): floors m15 (B2 = proc k 1), m50, m100 x H 900 / 1,800 / 3,600 s x
D1 60 / 300 s x seeds 0-39 = 720 tasks (`list horizon`); H 7,200 s at a floor where H 3,600 s fails H1 (`list
horizon-ext <floor> ...`, 80 tasks per floor; analyze7.py extension names the floors). Serial forks, no latency.
@par Usage
@code{.sh}
.venv/bin/python results/rq2-replan-axes/run7.py proc-k1-d900+lat300 0 results/rq2-replan-axes/dryrun7
.venv/bin/python results/rq2-replan-axes/run7.py list horizon [--todo] [--root DIR]   # 7b tasks, costliest first
.venv/bin/python results/rq2-replan-axes/run7.py list horizon-ext m100 [--todo] [--root DIR]
.venv/bin/python results/rq2-replan-axes/run7.py time size-m100 0 [P ...]      # T_c of single re-plans
.venv/bin/python results/rq2-replan-axes/run7.py compare <a.json> <b.json>     # the fields a regression compares
@endcode
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run as R                                                    # noqa: E402  (frozen code/env first on the path)
from des_twin import branch                                        # noqa: E402
from des_twin.planner import belief_transform, window_tardiness   # noqa: E402

assert str(R.CODE) in branch.__file__, branch.__file__
OPTS = {"lat": "latency", "off": "offset", "h": "horizon", "par": "par"}
SAME = ("real", "mdd_ptect", "schedule", "replans", "switches", "decisions", "describe", "slot_seconds", "warmup",
        "window", "d1", "scale")
OUT_DIR = "out7"                                       # under root; run.py's out/ stays axes 1, 2 and 6
# 7b, the horizon arm (README §19, approved by the user 11-10 as proposed)
FLOORS7 = {"m15": "proc-k1", "m50": "size-m50", "m100": "size-m100"}   # floor -> run.py level (m15 is B2)
H7, H7_EXT, D1_7, SEEDS7 = (900, 1800, 3600), 7200, (60, 300), list(range(40))


def horizon_tasks(floors=tuple(FLOORS7), hs=H7):
    return [(f"{FLOORS7[f]}-d{d}+h{h}", s) for f in floors for h in hs for d in D1_7 for s in SEEDS7]


def rollout_cw(d1, h, window=R.WINDOW):
    """Rollout volume of one task in full-window re-plans: Σ_k min(H, window end − t_k) / window (6 forks each)."""
    n = int(round(window / d1))
    return sum(min(h if h is not None else window, window - k * d1) for k in range(n)) / window


def est7(v):
    """Ordering only: rollout volume x the level's re-plan cost (run.py's C_W, single-core seconds)."""
    base, o = parse7(v)
    arm, k, d1 = R.parse(base)
    return rollout_cw(d1, o["horizon"]) * R.C_W.get((arm, k), 9.7)


def parse7(v):
    """(run.py variant, options): options latency / offset (s), horizon (s or None), par (int)."""
    base, *opts = v.split("+")
    o = {"latency": 0.0, "offset": 0.0, "horizon": None, "par": 1}
    for s in opts:
        key = next(k for k in sorted(OPTS, key=len, reverse=True) if s.startswith(k))
        o[OPTS[key]] = int(s[len(key):]) if key == "par" else float(s[len(key):])
    return base, o


def _tail(tw, gen, prefix, pair, t0, index, warmup, stop):
    """In a forked child paused at a re-plan: the pairs in force (prefix) until the plan acts, then `pair`; to the end
    of the window, or to the horizon's pause when stop is set. run.py's _play_tail when index is its slot index."""
    y = gen.send(None)
    try:
        while True:
            if y.kind == "pause":
                if stop:
                    break
                y = gen.send(None)
                continue
            if t0 is None:
                t0 = tw.now
            kk = index(tw.now, t0)
            p = prefix[kk] if kk < len(prefix) else pair
            y = gen.send(tuple(p.split("-")))
    except StopIteration:
        pass
    return window_tardiness(tw, warmup if t0 is None else t0)


def plan_latency(tw, pairs, slot, warmup, window, latency=0.0, offset=0.0, horizon=None, par=1):
    """Planner v2 with latency (module docstring). Returns run.py's plan_triggered fields plus the per-plan wall
    times and nominal plan times."""
    n = int(round(window / slot))
    if not 0.0 <= offset < slot:
        raise ValueError(f"offset {offset} outside [0, Δ {slot})")
    if latency > slot:               # one plan's compute must end by the time the next one starts (11-13 guard)
        raise ValueError(f"not real time: T_c {latency} > Δ {slot}")

    def index(now, t0):
        if not latency:
            return int((now - t0) // slot)                       # run.py / planner v2
        return max(0, int((now - t0 - latency) // slot))         # plan k acts at t0 + k Δ + T_c

    gen = tw.agent_decisions()
    schedule, planned, plan_s, plan_t = [], [], [], []
    t0, k_next = None, 0
    tw.pause_at(warmup)
    try:
        y = gen.send(None)
        while True:
            if y.kind == "pause":
                t_plan = warmup + k_next * slot + (offset if k_next else 0.0)
                prefix = list(schedule)
                stop = horizon is not None and t_plan + horizon < warmup + window

                def child(pair, prefix=prefix, t0c=t0, t_end=t_plan + (horizon or 0.0), stop=stop):
                    belief_transform(tw, None)
                    if stop:
                        tw.pause_at(t_end)
                    try:
                        return _tail(tw, gen, prefix, pair, t0c, index, warmup, stop)
                    except StopIteration:   # no agent decision left before the window ends: the end state
                        return window_tardiness(tw, warmup if t0c is None else t0c)
                t_start = time.time()
                res = branch.fork_map(child, list(pairs), max_parallel=par)
                plan_s.append(time.time() - t_start)
                plan_t.append(t_plan)
                i = int(np.argmin(res))
                schedule.append(pairs[i])
                planned.append(float(res[i]))
                k_next += 1
                if k_next < n:
                    tw.pause_at(warmup + k_next * slot + offset)
                y = gen.send(None)
            else:
                if t0 is None:
                    t0 = tw.now
                    # causality: plan k >= 1 acts at t0 + k Δ + T_c, no earlier than it is computed (warm-up + k Δ +
                    # offset). A late first decision only delays when plans act (as in run.py), never this.
                    if (t0 - warmup) + latency < offset:
                        raise ValueError(f"a plan would act before it is computed: first decision {t0 - warmup:.2f} s "
                                         f"+ T_c {latency} < offset {offset}")
                kk = index(tw.now, t0)
                p = schedule[min(kk, len(schedule) - 1)]
                y = gen.send(tuple(p.split("-")))
    except StopIteration:
        pass
    return {"schedule": schedule, "planned": planned, "real": window_tardiness(tw, warmup if t0 is None else t0),
            "plan_seconds": plan_s, "plan_times": plan_t, "t0": t0, "decisions": tw.decisions,
            "replans": len(schedule)}


def _stamp():
    return {**R._stamp(), "run7_py_md5": hashlib.md5(Path(__file__).read_bytes()).hexdigest()[:12]}


def out_path(v, seed, root=HERE):
    return Path(root) / OUT_DIR / v / f"s{seed}.json"


def run(v, seed, root=HERE):
    out = out_path(v, seed, root)
    if out.exists():
        return f"exists {out}"
    base, o = parse7(v)
    arm, k, d1 = R.parse(base)
    if d1 is None:
        raise ValueError(f"{v}: run7.py runs planner variants only (fixed-pair tasks are run.py's)")
    t = time.time()
    e = R.AxEpisode(arm, k, seed)
    slot = d1 * e.scale
    res = plan_latency(e.twin(), R.PAIRS, slot, e.warmup, e.window, latency=o["latency"] * e.scale,
                       offset=o["offset"] * e.scale, horizon=None if o["horizon"] is None else o["horizon"] * e.scale,
                       par=o["par"])
    ref, tw_ref = e.fixed(R.REF)
    sched = res["schedule"]
    obj = {"variant": v, "base": base, "options": o, "arm": arm, "k": k, "d1": d1, "slot_seconds": slot,
           "scale": e.scale, "seed": seed, "real": res["real"], "mdd_ptect": ref, "schedule": sched,
           "replans": res["replans"], "switches": sum(p != q for p, q in zip(sched, sched[1:])),
           "decisions": res["decisions"], "planned": res["planned"], "plan_times": res["plan_times"],
           "plan_seconds_each": res["plan_seconds"], "plan_seconds": float(sum(res["plan_seconds"])), "t0": res["t0"],
           "seconds": time.time() - t, "warmup": e.warmup, "window": e.window, "describe": e.describe(tw_ref),
           "stamp": _stamp()}
    assert not R.live_env_modules(), R.live_env_modules()
    R._write(out, obj)
    rel = f"{100 * (res['real'] / ref - 1):+.2f}%" if ref else "n/a (MDD-PTECT 0)"   # §16: all-zero seeds
    return (f"{v} s{seed}: real {res['real']:.6f} vs {R.REF} {ref:.6f} ({rel}), {res['replans']} re-plans, "
            f"{obj['plan_seconds']:.0f} s planning, {obj['seconds']:.0f} s")


def time_plans(level, seed, pars=(1, 6), points=(0.0, 0.5), horizons=(None, 3600.0)):
    """Wall time of single re-plans (the 6 forks, hold-to-end or horizon H) at fractions `points` of the window, the
    world advancing under MDD-PTECT in between; the fork results must not depend on par. Writes nothing."""
    arm, k, _ = R.parse(f"{level}-fixed")
    e = R.AxEpisode(arm, k, seed)
    tw = e.twin()
    gen = tw.agent_decisions()
    times = [e.warmup + f * e.window for f in points]
    t0, rows, i = None, [], 0
    tw.pause_at(times[0])
    y = gen.send(None)
    while True:
        if y.kind == "pause":
            row = {"level": level, "seed": seed, "at": points[i], "remaining_s": e.warmup + e.window - times[i]}
            ref = None
            for h in horizons:
                stop = h is not None and times[i] + h < e.warmup + e.window
                for p in pars:
                    def child(pair, t0c=t0, t_end=times[i] + (h or 0.0), stop=stop):
                        belief_transform(tw, None)
                        if stop:
                            tw.pause_at(t_end)
                        try:
                            return _tail(tw, gen, [], pair, t0c, lambda now, t: 0, e.warmup, stop)
                        except StopIteration:
                            return window_tardiness(tw, e.warmup if t0c is None else t0c)
                    ts = time.time()
                    res = branch.fork_map(child, list(R.PAIRS), max_parallel=p)
                    row[f"h{'full' if h is None else int(h)}_par{p}_s"] = round(time.time() - ts, 2)
                    if p == pars[0]:
                        ref = res
                    else:
                        row.setdefault("par_equal", True)
                        row["par_equal"] &= res == ref
            rows.append(row)
            print(json.dumps(row), flush=True)
            i += 1
            if i == len(times):
                break
            tw.pause_at(times[i])
            y = gen.send(None)
        else:
            if t0 is None:
                t0 = tw.now
            y = gen.send(tuple(R.REF.split("-")))
    return rows


def compare(a, b):
    """The fields a regression must reproduce exactly (SAME), as {field: (a, b)} for those that differ. Descriptors
    are compared on the keys both have (axis 1's outputs predate agv_busy_on_duty)."""
    A, B = json.loads(Path(a).read_text()), json.loads(Path(b).read_text())
    keys = set(A.get("describe") or {}) & set(B.get("describe") or {})
    for X in (A, B):
        X["describe"] = {k: X["describe"][k] for k in sorted(keys)} if X.get("describe") else X.get("describe")
    return {f: (A.get(f), B.get(f)) for f in SAME if A.get(f) != B.get(f)}


def main(argv):
    if argv[0] == "list":
        args, root, todo = list(argv[1:]), HERE, False
        if "--root" in args:
            i = args.index("--root")
            root = Path(args[i + 1])
            del args[i:i + 2]
        if "--todo" in args:
            todo = True
            args.remove("--todo")
        if args[0] == "horizon":
            tasks = horizon_tasks()
        elif args[0] == "horizon-ext":
            bad = [f for f in args[1:] if f not in FLOORS7]
            if bad:
                raise SystemExit(f"unknown floor(s) {bad}; expected {list(FLOORS7)}")
            tasks = horizon_tasks(tuple(args[1:]), (H7_EXT,)) if args[1:] else []
        else:
            raise SystemExit(f"unknown task set {args[0]!r}")
        for v, s in sorted(tasks, key=lambda t: -est7(t[0])):   # costliest first (stable within equal cost)
            if not (todo and out_path(v, s, root).exists()):
                print(v, s, root)
    elif argv[0] == "time":
        time_plans(argv[1], int(argv[2]), *([tuple(int(p) for p in argv[3:])] if len(argv) > 3 else []))
    elif argv[0] == "compare":
        diff = compare(argv[1], argv[2])
        print("identical" if not diff else json.dumps({f: [str(x)[:120] for x in d] for f, d in diff.items()}))
    else:
        print(run(argv[0], int(argv[1]), *argv[2:3]), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
