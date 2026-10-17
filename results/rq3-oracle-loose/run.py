"""
@file run.py
@brief rq3-oracle-loose (2026-10-14): the switching-oracle ceiling at loose due dates (c ~ U[2.5, 3.5] per 1.5 h
       block, rq3-gen-v5's `due-loose`), B2 otherwise. Same greedy oracle as ../rq2-rebase-v5/run.py (6 v5 pairs, 900 s
       slots, hold-to-end tails) and the same frozen code (rq2-rebase-v5/code/env + its frozen sim.py); only the
       instance changes. The instance comes from evaluate.py's own generator call (scenarios.REGISTRY["randomized"]
       with params_overrides, as evaluate.py --params builds it), so the oracle sees exactly the instances
       rq3-gen-v5 / eval-v5-cmix score the policies on. `check` replays MDD-PTECT at B2's own c on seed 0 and must
       give the known 95.842 (train-v5-nokl/eval), which shows the generator path equals rq2-twin-fleet's.
       Output out/s<seed>.json: per-slot tails, the oracle schedule, the oracle and every fixed pair's tardiness.
@par Usage
@code{.sh}
python results/rq3-oracle-loose/run.py check
python results/rq3-oracle-loose/run.py oracle 0
python results/rq3-oracle-loose/run.py list > results/rq3-oracle-loose/tasks.txt
@endcode
"""
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RB = HERE.parent / "rq2-rebase-v5"
_spec = importlib.util.spec_from_file_location("rb_run", RB / "run.py")    # sets up the frozen code path + sim
rb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rb)
sim = rb.sim
V5 = rb.V5
SEEDS = list(range(40))
LOAD_MIX = [{"name": "normal", "utilization": [0.6, 1.2], "weight": 2}, {"name": "surge", "utilization": [1.0, 1.6]},
            {"name": "short", "utilization": [0.6, 1.2], "agv_count": 2}]


def params(c):
    return {"failure_probability": 0.0, "due_date_allowance_range": list(c), "regime_block_seconds": 5400,
            "horizon_seconds": 30600, "load_mix": LOAD_MIX}


def use_generator(c):
    from scenarios import REGISTRY
    gen = REGISTRY["randomized"](21600.0, random_warmup=True, warmup_dispatching_rule=None, agv_move_speed=None,
                                 agv_handshake_duration=None, params_overrides=params(c))
    sim.scenario = lambda setting, seed: gen(seed)          # sim.Episode looks scenario() up at call time


def check():
    use_generator((1.75, 2.5))
    tard = sim.Episode("B2", 0).play(["MDD-PTECT"])[0]
    ok = abs(tard - 95.842007) < 1e-3
    print(f"check: MDD-PTECT on B2 seed 0 via evaluate's generator = {tard:.6f} (known 95.842007) -> "
          f"{'OK' if ok else 'MISMATCH'}")
    if not ok:
        raise SystemExit(1)


def oracle(seed):
    out = HERE / "out" / f"s{seed}.json"
    if out.exists():
        return f"exists {out}"
    use_generator((2.5, 3.5))
    t = time.time()
    ep = sim.Episode("B2", seed)
    n = sim.n_slots("B2")
    prefix, tails = [], np.zeros((n, len(V5)))
    for k in range(n):
        for i, p in enumerate(V5):
            tails[k, i] = ep.play(prefix + [p])[0]
        prefix.append(V5[int(np.argmin(tails[k]))])
    fixed = dict(zip(V5, tails[0].tolist()))
    res = {"seed": seed, "c": [2.5, 3.5], "oracle": float(tails[-1].min()), "schedule": prefix, "fixed": fixed,
           "tails": tails.tolist(), "warmup": ep.warmup, "stamp": rb.stamp(), "seconds": time.time() - t}
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(res))
    os.replace(tmp, out)
    best = min(fixed, key=fixed.get)
    return (f"oracle s{seed}: best fixed {best} {fixed[best]:.2f}, oracle {res['oracle']:.2f} "
            f"({100 * (res['oracle'] / fixed[best] - 1):+.1f}%), {res['seconds']:.0f} s")


def main(argv):
    if argv[0] == "list":
        for s in SEEDS:
            print("oracle", s)
        return
    if argv[0] == "check":
        check()
        return
    print(oracle(int(argv[1])), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
