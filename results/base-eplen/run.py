"""
@file run.py
@brief base-eplen (2026-10-06): do instances stop having "their own" best rule when episodes get longer? All 15 H15
       fixed pairs on the twin, B2 regime mix, at agent windows of 6 / 24 / 96 / 200 h (about 300 / 1,200 / 4,800 /
       10,000 jobs) and regime blocks of 1.5 h (B2) or 6 h, seeds 0-39.

Question (user + advisor, 10-06): each 6 h episode seems to have a preferred rule; the advisor suggested episodes of
10,000s of jobs. dev-congestion-signal found the local preference follows the realized arrivals of each 1.5 h block
(instance x block explains 64% of its variance, regime type 6%), i.e. sampling noise that longer episodes and longer
blocks should average out. Measured per (block, length): how often an instance's best pair is the overall best, and
the gain of picking the best pair per instance (in hindsight) over the overall best; per 6 h chunk inside long
episodes, the same at the chunk level on a warm floor.

The generator is rq2-twin-fleet's B2 with horizon_seconds = window + 9,000 s (B2: 21,600 + 9,000), so window 6 h /
block 1.5 h reproduces B2 exactly (checked against rq2-twin-fleet's fixed pairs in analyze.py). One call = one (seed,
window, block); writes w<window>_b<block>/s<seed>.json; skips existing outputs. Cluster: twin_array.sbatch.
@par Usage
@code{.sh}
python results/base-eplen/run.py --seed 0 --window 86400 --block 5400
@endcode
"""
import argparse
import dataclasses
import importlib.util
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "env"))
_spec = importlib.util.spec_from_file_location("sig", REPO / "results/dev-congestion-signal/run.py")
sig = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sig)
fleet = sig.fleet
PAIRS = list(fleet.PAIRS)
CHUNK = 21600.0


def scenario(seed, window, block, mix="B2"):
    """mix B2: normal 0.6-1.2 (2) : surge 1.0-1.6 (1) : short fleet (1), as rq2-twin-fleet B2; B0: normal only (the
    calibrated load that levels off, rq2-twin-fleet B0), the stable control for long windows (added 10-06 after a 24 h
    test showed B2's tardiness per 6 h chunk rising: 155, 94, 540, 808)."""
    from scenarios.randomized import DEFAULT_PARAMS, randomized_generator
    params = dataclasses.replace(DEFAULT_PARAMS, failure_probability=0.0, due_date_allowance_range=(1.75, 2.5),
                                 regime_block_seconds=block, horizon_seconds=window + 9000.0,
                                 load_mix=fleet.SETTINGS[mix]["params"]["load_mix"])
    return randomized_generator(window, random_warmup=True, params=params)(seed)


def play(jobs, warmup, warm_rule, sched, window, pair):
    """One episode with one pair held; window tardiness in total and per 6 h chunk (overlap of each job's lateness)."""
    from des_twin import TwinConfig
    from des_twin.engine import Twin
    tw = Twin(sig.floor(), jobs, TwinConfig(rule=warm_rule, transport="kinematic", warmup_seconds=warmup,
                                            episode_duration_seconds=window, max_sim_seconds=warmup + window + 200000.0,
                                            agv_schedule=sched))
    halves = tuple(pair.split("-"))
    gen = tw.agent_decisions()
    t0 = None
    try:
        gen.send(None)
        while True:
            if t0 is None:
                t0 = tw.now
            gen.send(halves)
    except StopIteration:
        pass
    t0 = warmup if t0 is None else t0
    t1 = tw.now
    n = max(1, int(round(window / CHUNK)))
    chunks = [0.0] * n
    tard, exited = 0.0, 0
    for j in tw.jobs.values():
        if j.exit_time is not None and t0 <= j.exit_time <= t1:
            exited += 1
        if j.due is None:
            continue
        end = j.exit_time if j.exit_time is not None else t1
        tard += fleet.due.overlap(j.due, end, t0, t1)
        for k in range(n):
            a = t0 + k * CHUNK
            chunks[k] += fleet.due.overlap(j.due, end, a, min(a + CHUNK, t1))
    return {"tard": tard / 1000.0, "chunks": [c / 1000.0 for c in chunks], "jobs_exited": exited,
            "t0": t0, "t1": t1, "timed_out": bool(tw.timed_out)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--window", type=float, required=True)
    ap.add_argument("--block", type=float, default=5400.0)
    ap.add_argument("--mix", choices=("B2", "B0"), default="B2")
    a = ap.parse_args()
    out = HERE / (f"w{int(a.window)}_b{int(a.block)}" + ("" if a.mix == "B2" else f"_{a.mix}")) / f"s{a.seed}.json"
    if out.exists():
        print("exists", out)
        return
    from des_twin.scenario import agv_schedule, episode_settings, resolve_jobs
    sc = scenario(a.seed, a.window, a.block, a.mix)
    sched = agv_schedule(sc)
    warmup, _, warm_rule = episode_settings(sc)
    jobs = resolve_jobs(sc, sig.floor())
    t = time.time()
    res = {p: play(jobs, warmup, warm_rule, sched, a.window, p) for p in PAIRS}
    blocks = sc["_meta"]["blocks"]
    share = {prof: sum(max(0.0, min(warmup + a.window, b["end"]) - max(warmup, b["start"])) for b in blocks
                       if b["load_profile"] == prof) / a.window for prof in ("normal", "surge", "short")}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"seed": a.seed, "window": a.window, "block": a.block, "mix": a.mix, "warmup": warmup,
                               "jobs_arrived_window": sum(warmup <= j["arrivalTime"] < warmup + a.window for j in sc["jobs"]),
                               "regime_share": share, "fixed": res, "wall_s": time.time() - t}))
    best = min(res, key=lambda p: res[p]["tard"])
    print(f"s{a.seed} w{a.window:.0f} b{a.block:.0f}: best {best} {res[best]['tard']:.1f}, MDD-TECT {res['MDD-TECT']['tard']:.1f}, "
          f"{res['MDD-TECT']['jobs_exited']} jobs out, {time.time() - t:.0f} s")


if __name__ == "__main__":
    main()
