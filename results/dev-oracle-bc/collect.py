"""
@file collect.py
@brief dev-oracle-bc step 1: oracle labels with every pair's outcome, and the full observation at each slot start.

Per B2 seed: the H15 greedy switching oracle of rq2-twin-fleet (same pairs, 900 s slots, hold-to-end tails), but every
stage keeps all 15 tails' window tardiness (rq2-twin-fleet stored only the winner), so the labels can be soft. The
oracle schedule is then replayed and, at the first decision of each slot, the full obs v3 is saved exactly as the slot
policy sees it (all-ones action mask; row caps 15 machines / 256 jobs, as the twin training runs used).

Seeds 0-39 (held out, rq2-twin-fleet has their oracle): the recomputed schedule and tardiness are checked against the
task files. Training seeds 2000-2159 (outside 0-39 and RL training's >= 10,000).

Writes data/s<seed>.npz (obs streams float16 / float32, tails (24, 15), schedule) and skips seeds already written; DONE
when every requested seed is there.

@par Usage (through the lab: run.sh)
@code{.sh}
nice -n 10 .venv/bin/python results/dev-oracle-bc/collect.py --seeds 0-39,2000-2159 --workers 16
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
_spec = importlib.util.spec_from_file_location("sig", REPO / "results/dev-congestion-signal/run.py")
sig = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sig)
fleet = sig.fleet
PAIRS = list(fleet.PAIRS)
N_SLOTS = sig.N_SLOTS
CAPS = (15, 256)
OUT = HERE / "data"


def parse_seeds(text):
    out = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    return out


def run_seed(seed):
    from des_twin.observation import ObservationBuilder
    from des_twin.scenario import agv_schedule, episode_settings, resolve_jobs
    out = OUT / f"s{seed}.npz"
    if out.exists():
        return seed, "skip"
    sc = fleet.scenario("B2", seed)
    sched = agv_schedule(sc)
    warmup, _, warm_rule = episode_settings(sc)
    jobs = resolve_jobs(sc, sig.floor())
    prefix, tails = [], np.zeros((N_SLOTS, len(PAIRS)))
    for k in range(N_SLOTS):
        for i, p in enumerate(PAIRS):
            tails[k, i] = sig.play(jobs, warmup, warm_rule, sched, prefix, p)
        prefix.append(PAIRS[int(np.argmin(tails[k]))])   # min() picks the first of ties, as the oracle does
    oracle = float(tails[-1].min())
    builder = ObservationBuilder(sig.floor(), *CAPS)
    obs = {}

    def capture(tw, dec, k):
        o = builder.build(tw, dec)
        o["action_mask"] = np.ones_like(o["action_mask"])
        obs[k] = o

    replay = sig.play(jobs, warmup, warm_rule, sched, prefix, prefix[-1], capture=capture)
    msg = f"oracle {oracle:.3f} replay diff {abs(replay - oracle):.1e}"
    task = fleet.HERE / "tasks" / f"B2_s{seed}.json"
    if task.exists():
        t = json.loads(task.read_text())
        msg += f"; vs rq2-twin-fleet: schedule {'same' if t['schedule'] == prefix else 'DIFFERENT'}, " \
               f"oracle diff {abs(t['oracle'] - oracle):.1e}"
    keys = ("factory_grid", "machine_table", "job_table", "global_scalars", "event_flags", "action_mask")
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(out, **{k: np.stack([obs[i][k] for i in range(N_SLOTS)]).astype(
        np.float16 if k in ("factory_grid", "job_table") else np.float32) for k in keys},
        tails=tails, schedule=np.array(prefix), pairs=np.array(PAIRS), seed=seed, oracle=oracle)
    return seed, msg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-39,2000-2159")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    seeds = parse_seeds(a.seeds)
    with Pool(a.workers) as pool:
        for seed, msg in pool.imap_unordered(run_seed, seeds):
            print(f"s{seed}: {msg}", flush=True)
    have = {int(p.stem[1:]) for p in OUT.glob("s*.npz")}
    print(f"{len(have & set(seeds))}/{len(seeds)} seeds written")
    if set(seeds) <= have:
        (HERE / "DONE_collect").write_text("done\n")


if __name__ == "__main__":
    main()
