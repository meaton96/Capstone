"""
@file run.py
@brief dev-congestion-signal (2026-10-06): does the realized transport backlog at a slot boundary say which rule
       family to play from there on, beyond what obs v3's global scalars and flags already carry?

@details
For each B2 seed with an oracle schedule (rq2-twin-fleet tasks, seeds 0-39) the oracle's slot schedule is replayed in
the twin. At the first decision of every 900 s slot k it records
  - the exact obs v3 global scalars (18) and event flags (6) the policy's MLP sees (des_twin ObservationBuilder), and
  - candidate congestion features read from the twin state: jobs waiting for an AGV (needs routing / waiting
    pickup), per on-duty AGV, their mean / max wait, jobs in transit, AGV idle share over the last slot, queued jobs
    and queued work per machine.
Targets: from the same prefix (oracle slots 0..k-1), the episode is played to the end with MDD-TECT, ATC-TECT and
ATC-ECT held from slot k on; d = tardiness(ATC-x tail) - tardiness(MDD-TECT tail), in % of the MDD-TECT tail
(< 0: ATC better from here on). 1 + 24 x 3 twin episodes per seed. Sanity: the full oracle replay must reproduce the
task file's oracle tardiness.

Second pass (--mode deviation, added 10-06 after a first look): the hold-to-the-end target mostly measures the future
regimes, which no state feature can know. So per slot k it also plays the oracle schedule with slots k..k+L-1 replaced
by MDD-TECT or ATC-TECT (L = 1 slot and 4 slots = 1 h) and back to the oracle after: dev{L} = tardiness(ATC-TECT
deviation) - tardiness(MDD-TECT deviation), in % of the oracle's tardiness (the local, decision-relevant difference).
2 x 2 x 24 twin episodes per seed, written to dev/s<seed>.csv.

Writes rows/s<seed>.csv (one row per slot); analyze.py does the comparison. Skips seeds already written.

@par Usage (through the lab)
@code{.sh}
nice -n 10 .venv/bin/python results/dev-congestion-signal/run.py --workers 12
@endcode
"""
import argparse
import importlib.util
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "env"))
FLOOR = REPO / "results/dev-popart-smoke/floor7/des_floor.json"
TASKS = REPO / "results/rq2-twin-fleet/tasks"
SEGMENT, WINDOW, N_SLOTS = 900.0, 21600.0, 24
TAILS = ("MDD-TECT", "ATC-TECT", "ATC-ECT")
OUT = HERE / "rows"
DEV = HERE / "dev"
DEV_LENGTHS = (1, 4)

_spec = importlib.util.spec_from_file_location("fleet", REPO / "results/rq2-twin-fleet/run.py")
fleet = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fleet)
_floor = None


def floor():
    global _floor
    if _floor is None:
        from des_twin import Floor
        _floor = Floor.load(str(FLOOR))
    return _floor


def congestion(tw, prev_idle, prev_now):
    """Candidate congestion features from the twin state at a decision."""
    from des_twin.engine import IN_TRANSIT, NEEDS_ROUTING, QUEUED, WAITING_PICKUP
    now = tw.now
    live = list(tw._live())
    wait = [now - j.since for j in live if j.state in (NEEDS_ROUTING, WAITING_PICKUP)]
    on_duty = [a for a in tw.agvs if tw._on_duty(a)]
    n_duty = max(1, len(on_duty))
    busy = sum(tw._status(a) != "idle" for a in on_duty)
    idle_tot = sum(a.idle_time + ((now - a.idle_since) if a.state == "idle" else 0.0) for a in tw.agvs)
    dt = now - prev_now if prev_now is not None else None
    queued = [j for j in live if j.state == QUEUED]
    qwork = sum(j.proc(j.location) for j in queued if j.location >= 0)
    n_m = len(tw.machines)
    f = {
        "c_wait_n": len(wait),
        "c_wait_per_agv": len(wait) / n_duty,
        "c_wait_mean": float(np.mean(wait)) if wait else 0.0,
        "c_wait_max": float(np.max(wait)) if wait else 0.0,
        "c_wait_sum_per_agv": float(np.sum(wait)) / n_duty,
        "c_transit_n": sum(j.state == IN_TRANSIT for j in live),
        "c_agv_on_duty": len(on_duty),
        "c_agv_busy_share": busy / n_duty,
        "c_agv_idle_share_last": (None if dt is None or dt <= 0 else
                                  max(0.0, min(1.0, (idle_tot - prev_idle) / (dt * len(tw.agvs))))),
        "c_queued_per_machine": len(queued) / n_m,
        "c_queued_work_per_machine": qwork / n_m,
        "c_wip": len(live),
    }
    return f, idle_tot


def play(jobs, warmup, warm_rule, sched, prefix, tail, capture=None):
    """One episode: prefix[i] in slot i, then tail. capture(tw, dec, k) is called at each slot's first decision."""
    from des_twin import TwinConfig
    from des_twin.engine import Twin
    tw = Twin(floor(), jobs, TwinConfig(rule=warm_rule, transport="kinematic", warmup_seconds=warmup,
                                        episode_duration_seconds=WINDOW, max_sim_seconds=200000.0, agv_schedule=sched))
    halves = {name: tuple(name.split("-")) for name in set(prefix + [tail])}
    gen = tw.agent_decisions()
    t0, seen = None, set()
    try:
        dec = gen.send(None)
        while True:
            if t0 is None:
                t0 = tw.now
            k = int((tw.now - t0) // SEGMENT)
            if capture is not None and k not in seen and k < N_SLOTS:
                seen.add(k)
                capture(tw, dec, k)
            dec = gen.send(halves[prefix[k] if k < len(prefix) else tail])
    except StopIteration:
        pass
    t0 = warmup if t0 is None else t0
    t1 = tw.now
    tard = sum(fleet.due.overlap(j.due, j.exit_time if j.exit_time is not None else t1, t0, t1)
               for j in tw.jobs.values() if j.due is not None)
    return tard / 1000.0


def run_seed(seed):
    import pandas as pd      # imported here: the cluster .venv has no pandas, and other scripts import this module
    from des_twin.observation import ObservationBuilder
    from des_twin.scenario import agv_schedule, episode_settings, resolve_jobs
    out = OUT / f"s{seed}.csv"
    if out.exists():
        return seed, "skip"
    task = json.loads((TASKS / f"B2_s{seed}.json").read_text())
    sc = fleet.scenario("B2", seed)
    sched = agv_schedule(sc)
    warmup, _, warm_rule = episode_settings(sc)
    jobs = resolve_jobs(sc, floor())
    schedule = list(task["schedule"])
    builder = ObservationBuilder(floor(), 105, 1792)
    rows, state = {}, {"idle": 0.0, "now": None}

    def capture(tw, dec, k):
        obs = builder.build(tw, dec)
        c, idle = congestion(tw, state["idle"], state["now"])
        state["idle"], state["now"] = idle, tw.now
        r = {"seed": seed, "slot": k, "sim_time": tw.now, "oracle_pair": schedule[k],
             "regime": task["regimes"][task["slot_block"][k]]["load"],
             **{f"s{i}": float(v) for i, v in enumerate(obs["global_scalars"])},
             **{f"f{i}": float(v) for i, v in enumerate(obs["event_flags"])}, **c}
        rows[k] = r

    replay = play(jobs, warmup, warm_rule, sched, schedule, schedule[-1], capture=capture)
    check = abs(replay - task["oracle"])
    for k in range(N_SLOTS):
        tails = {t: play(jobs, warmup, warm_rule, sched, schedule[:k], t) for t in TAILS}
        for t in TAILS:
            rows[k][f"tail_{t}"] = tails[t]
        for t in TAILS[1:]:
            rows[k][f"d_{t}"] = 100.0 * (tails[t] - tails["MDD-TECT"]) / max(tails["MDD-TECT"], 1e-9)
        rows[k]["oracle_replay_diff"] = check
    OUT.mkdir(exist_ok=True)
    pd.DataFrame([rows[k] for k in range(N_SLOTS)]).to_csv(out, index=False)
    return seed, f"oracle replay diff {check:.2e}"


def run_seed_dev(seed):
    import pandas as pd
    from des_twin.scenario import agv_schedule, episode_settings, resolve_jobs
    out = DEV / f"s{seed}.csv"
    if out.exists():
        return seed, "skip"
    task = json.loads((TASKS / f"B2_s{seed}.json").read_text())
    sc = fleet.scenario("B2", seed)
    sched = agv_schedule(sc)
    warmup, _, warm_rule = episode_settings(sc)
    jobs = resolve_jobs(sc, floor())
    schedule = list(task["schedule"])
    rows = []
    for k in range(N_SLOTS):
        r = {"seed": seed, "slot": k}
        for L in DEV_LENGTHS:
            t = {}
            for pair in ("MDD-TECT", "ATC-TECT"):
                s2 = schedule[:k] + [pair] * min(L, N_SLOTS - k) + schedule[k + L:]
                t[pair] = play(jobs, warmup, warm_rule, sched, s2, s2[-1])
            r[f"dev{L}_MDD-TECT"], r[f"dev{L}_ATC-TECT"] = t["MDD-TECT"], t["ATC-TECT"]
            r[f"dev{L}"] = 100.0 * (t["ATC-TECT"] - t["MDD-TECT"]) / max(task["oracle"], 1e-9)
        rows.append(r)
    DEV.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    return seed, "deviation done"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seeds", type=str, default="0-39")
    ap.add_argument("--mode", choices=("hold", "deviation"), default="hold")
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.seeds.split("-"))
    fn, out, done = (run_seed, OUT, "DONE") if a.mode == "hold" else (run_seed_dev, DEV, "DONE_dev")
    with Pool(a.workers) as pool:
        for seed, msg in pool.imap_unordered(fn, range(lo, hi + 1)):
            print(f"s{seed}: {msg}", flush=True)
    n = len(list(out.glob("s*.csv")))
    print(f"{n} seeds written ({a.mode})")
    if n == hi - lo + 1:
        (HERE / done).write_text("done\n")


if __name__ == "__main__":
    main()
