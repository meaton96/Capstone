"""
@file collect_ev.py
@brief dev-oracle-bc-ev (2026-10-07, handoff fix 7): expected-value oracle labels for behavior cloning. As
       ../dev-oracle-bc/collect.py, but at every slot the 15 pairs' hold-to-end tails are also scored on M redrawn
       futures, and the schedule follows the pair with the best mean over the realized and redrawn futures.

Why: on one future, the hindsight-best slot choice predicts its sign on other futures only about half the time
(docs/experiments/review_1007/CREDIT_ASSIGNMENT_TRACE_1007.md), so cloning the single-future argmax clones noise and
overstates the realizable headroom. The mean over futures is the expected-value target; with per-future tails stored,
bc_train.py can also require a switch to clear its uncertainty (--labels ev-safe).

A redrawn future keeps the instance's jobs that arrived by the slot's first decision and splices in the later jobs of
another generator seed (env_wrappers.paired_slot_env.splice_future, as docs/experiments/review_1007/scripts/branch.py).
Every spliced run is checked to reach that decision at the same sim time and decision count as the realized run.

Writes data/s<seed>.npz with collect.py's keys (obs along the EV schedule, tails = the realized future's tails along
it, schedule, pairs, seed, oracle = the EV schedule's realized tardiness) plus tails_fut (slots, M, 15), future_seeds
(slots, M), slot_time (slots,) and oracle_realized (collect.py's single-future oracle is not recomputed: for seeds it
already has, its data/s<seed>.npz holds it). Skips seeds already written.

@par Usage
@code{.sh}
nice -n 10 .venv/bin/python results/dev-oracle-bc-ev/collect_ev.py --seeds 0-39,2000-2159 --futures 4 --workers 16
@endcode
"""
import argparse
import importlib.util
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "env"))
_spec = importlib.util.spec_from_file_location("collect", REPO / "results/dev-oracle-bc/collect.py")
collect = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(collect)
sig, fleet, PAIRS, N_SLOTS, CAPS = collect.sig, collect.fleet, collect.PAIRS, collect.N_SLOTS, collect.CAPS
OUT = HERE / "data"
## @brief Future seeds are drawn above the RL training seeds' floor (>= 10,000) and are never a test seed (0-39) or
##        a BC training seed (2000-2159).
FUTURE_LOW, FUTURE_HIGH = 100_000, 1 << 24


def _play(args):
    """One hold-to-end run; also returns (time, decision count) at slot k's first decision, for the state check."""
    jobs, warmup, warm_rule, sched, prefix, pair, k = args
    seen = {}

    def capture(tw, dec, kk):
        if kk == k:
            seen["state"] = (tw.now, tw.decisions)

    tard = sig.play(jobs, warmup, warm_rule, sched, prefix, pair, capture=capture)
    return tard, seen.get("state")


def run_seed(seed, futures, pool, slots=N_SLOTS, out_dir=OUT):
    from des_twin.observation import ObservationBuilder
    from des_twin.scenario import agv_schedule, episode_settings, resolve_jobs
    from env_wrappers.paired_slot_env import splice_future
    out = Path(out_dir) / f"s{seed}.npz"
    if out.exists():
        return "skip"
    sc = fleet.scenario("B2", seed)
    sched = agv_schedule(sc)
    warmup, _, warm_rule = episode_settings(sc)
    floor = sig.floor()
    jobs = resolve_jobs(sc, floor)
    tails = np.zeros((slots, len(PAIRS)))
    tails_fut = np.zeros((slots, futures, len(PAIRS)))
    fseeds = np.zeros((slots, futures), dtype=np.int64)
    slot_time = np.zeros(slots)
    prefix, mismatches = [], 0
    for k in range(slots):
        real = pool.map(_play, [(jobs, warmup, warm_rule, sched, prefix, p, k) for p in PAIRS])
        tails[k] = [t for t, _ in real]
        state = real[0][1]
        slot_time[k] = state[0]
        rng = np.random.default_rng([seed, k, 4049])
        fseeds[k] = rng.integers(FUTURE_LOW, FUTURE_HIGH, futures)
        worlds = [resolve_jobs(splice_future(sc, fleet.scenario("B2", int(fs)), state[0]), floor) for fs in fseeds[k]]
        runs = pool.map(_play, [(w, warmup, warm_rule, sched, prefix, p, k) for w in worlds for p in PAIRS])
        mismatches += sum(st != state for _, st in runs)
        tails_fut[k] = np.array([t for t, _ in runs]).reshape(futures, len(PAIRS))
        ev = (tails[k] + tails_fut[k].sum(0)) / (futures + 1)
        prefix.append(PAIRS[int(np.argmin(ev))])
    oracle = float(tails[-1][PAIRS.index(prefix[-1])])
    builder = ObservationBuilder(floor, *CAPS)
    obs = {}

    def capture(tw, dec, kk):
        o = builder.build(tw, dec)
        o["action_mask"] = np.ones_like(o["action_mask"])
        obs[kk] = o

    replay = sig.play(jobs, warmup, warm_rule, sched, prefix, prefix[-1], capture=capture)
    keys = ("factory_grid", "machine_table", "job_table", "global_scalars", "event_flags", "action_mask")
    old = REPO / "results/dev-oracle-bc/data" / f"s{seed}.npz"
    realized = float(np.load(old)["oracle"]) if old.exists() else float("nan")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **{kk: np.stack([obs[i][kk] for i in range(slots)]).astype(
        np.float16 if kk in ("factory_grid", "job_table") else np.float32) for kk in keys},
        tails=tails, tails_fut=tails_fut, future_seeds=fseeds, slot_time=slot_time, schedule=np.array(prefix),
        pairs=np.array(PAIRS), seed=seed, oracle=oracle, oracle_realized=realized)
    agree = float(np.mean(tails.argmin(1) == np.array([PAIRS.index(p) for p in prefix])))
    return (f"EV schedule realized tardiness {oracle:.3f} (single-future oracle {realized:.3f}), replay diff "
            f"{abs(replay - oracle):.1e}, EV choice = realized-future best on {100 * agree:.0f}% of slots, "
            f"state mismatches {mismatches}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-39,2000-2159")
    ap.add_argument("--futures", type=int, default=4)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--slots", type=int, default=N_SLOTS, help="stages to label (fewer for a quick check)")
    ap.add_argument("--out", default=str(OUT), help="npz folder (another one for a quick check)")
    a = ap.parse_args()
    seeds = collect.parse_seeds(a.seeds)
    with Pool(a.workers) as pool:
        for seed in seeds:
            print(f"s{seed}: {run_seed(seed, a.futures, pool, a.slots, a.out)}", flush=True)
    have = {int(p.stem[1:]) for p in Path(a.out).glob("s*.npz")}
    print(f"{len(have & set(seeds))}/{len(seeds)} seeds written")
    if set(seeds) <= have and a.slots == N_SLOTS and Path(a.out) == OUT:
        (HERE / "DONE_collect").write_text("done\n")


if __name__ == "__main__":
    main()
