"""
@file run.py
@brief rq2-oracle-slotlen (2026-10-06): switching headroom vs decision-slot length. The H15 greedy oracle of
       rq2-twin-fleet B2 (hold-to-end tails, twin, tardiness, 6 h agent window), with slots of 1,800 / 3,600 / 5,400 /
       10,800 s instead of 900 s, on seeds 0-39 (900 s: rq2-twin-fleet, recomputed with every tail by dev-oracle-bc).

Question (user, 10-06): would a longer decision interval make a difference? Longer slots mean fewer, larger-effect
decisions (easier credit assignment) but slower reaction; if the oracle keeps most of its gain at 1 h slots, longer
slots are nearly free for the policy.

One call = one (seed, slot length); writes seg<G>/s<seed>.json with every stage's 15 tails, the schedule, the oracle
and the fixed pairs (stage 1). Skips existing outputs. Runs on the cluster (twin_array.sbatch, one CPU per task).
@par Usage
@code{.sh}
python results/rq2-oracle-slotlen/run.py --seed 0 --segment 3600
@endcode
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "env"))
_spec = importlib.util.spec_from_file_location("sig", REPO / "results/dev-congestion-signal/run.py")
sig = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sig)
PAIRS = list(sig.fleet.PAIRS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--segment", type=float, required=True)
    a = ap.parse_args()
    out = HERE / f"seg{int(a.segment)}" / f"s{a.seed}.json"
    if out.exists():
        print("exists", out)
        return
    from des_twin.scenario import agv_schedule, episode_settings, resolve_jobs
    sig.SEGMENT = float(a.segment)                     # play() reads the slot length from the module
    n = int(round(sig.WINDOW / a.segment))
    sc = sig.fleet.scenario("B2", a.seed)
    sched = agv_schedule(sc)
    warmup, _, warm_rule = episode_settings(sc)
    jobs = resolve_jobs(sc, sig.floor())
    t0 = time.time()
    prefix, tails = [], []
    for k in range(n):
        row = [sig.play(jobs, warmup, warm_rule, sched, prefix, p) for p in PAIRS]
        tails.append(row)
        prefix.append(PAIRS[int(np.argmin(row))])
    fixed = dict(zip(PAIRS, tails[0]))
    best_fixed = min(fixed, key=fixed.get)
    res = {"seed": a.seed, "segment": a.segment, "stages": n, "pairs": PAIRS, "tails": tails, "schedule": prefix,
           "oracle": min(tails[-1]), "fixed": fixed, "best_fixed_pair": best_fixed, "best_fixed": fixed[best_fixed],
           "wall_s": time.time() - t0}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res))
    print(f"s{a.seed} seg {a.segment:.0f}: best fixed {best_fixed} {fixed[best_fixed]:.2f}, oracle {res['oracle']:.2f} "
          f"({100 * (res['oracle'] / fixed[best_fixed] - 1):+.1f}%), {res['wall_s']:.0f} s")


if __name__ == "__main__":
    main()
