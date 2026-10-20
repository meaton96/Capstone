"""
@file run.py
@brief rq3-size-oracle (2026-10-20, user: switching headroom at larger floors): the v5 switching-oracle ceiling at 50
       and 100 machines, the training sizes of train-size-m50 / -m100 (the check CLAUDE.md asks for before training on
       a regime, done late: those runs were queued 10-19). Same greedy oracle as ../rq2-rebase-v5 and
       ../rq3-oracle-loose (6 v5 pairs, 900 s slots, hold-to-end tails; the frozen code and sim.py of rq2-rebase-v5),
       on the floor-size grid's conditions through sizesim.py. B2 test seeds 0-39. 200 machines added 10-20 (user),
       run after the 50 / 100-machine oracle and the catalog screen (launch_m200.sh).
       Output out/<cond>/s<seed>.json: per-slot tails, the oracle schedule, the oracle and every fixed pair's window
       tardiness (1000 job-s), seconds.
@par Usage
@code{.sh}
python results/rq3-size-oracle/run.py check m50          # MDD-PTECT via the oracle's simulator = the grid's, seeds 0-2
python results/rq3-size-oracle/run.py oracle m100 7
python results/rq3-size-oracle/run.py list [m200 ...]     # task lines (default m100 m50, m100 first)
@endcode
"""
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = HERE.parent
CONDS = ["m100", "m50"]
SEEDS = list(range(40))


def load(cond):
    spec = importlib.util.spec_from_file_location("rb_run", R / "rq2-rebase-v5" / "run.py")   # frozen code first
    rb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rb)
    spec = importlib.util.spec_from_file_location("sizesim", HERE / "sizesim.py")
    ss = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ss)
    ss.apply(rb.sim, cond)
    return rb, ss


def check(cond):
    rb, ss = load(cond)
    bad = 0
    for s in (0, 1, 2):
        v = rb.sim.Episode("B2", s).play(["MDD-PTECT"])[0]
        g = ss.grid_value(cond, s)
        ok = g is not None and abs(v - g) <= 1e-6 * max(1.0, abs(g)) + 1e-4
        bad += not ok
        print(f"check {cond} s{s}: MDD-PTECT oracle simulator {v:.6f}, grid {g} -> {'OK' if ok else 'MISMATCH'}")
    if bad:
        raise SystemExit(1)


def oracle(cond, seed):
    out = HERE / "out" / cond / f"s{seed}.json"
    if out.exists():
        return f"exists {out}"
    rb, _ = load(cond)
    sim, V5 = rb.sim, rb.V5
    import numpy as np
    t = time.time()
    ep = sim.Episode("B2", seed)
    n = sim.n_slots("B2")
    prefix, tails = [], np.zeros((n, len(V5)))
    for k in range(n):
        for i, p in enumerate(V5):
            tails[k, i] = ep.play(prefix + [p])[0]
        prefix.append(V5[int(np.argmin(tails[k]))])
    fixed = dict(zip(V5, tails[0].tolist()))
    res = {"cond": cond, "seed": seed, "oracle": float(tails[-1].min()), "schedule": prefix, "fixed": fixed,
           "tails": tails.tolist(), "warmup": ep.warmup, "stamp": rb.stamp(), "seconds": time.time() - t}
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(res))
    os.replace(tmp, out)
    return (f"oracle {cond} s{seed}: MDD-PTECT {fixed['MDD-PTECT']:.2f}, oracle {res['oracle']:.2f} "
            f"({100 * (res['oracle'] / fixed['MDD-PTECT'] - 1):+.1f}%), {res['seconds']:.0f} s")


def main(argv):
    if argv[0] == "list":
        for c in argv[1:] or CONDS:
            for s in SEEDS:
                print("oracle", c, s)
    elif argv[0] == "check":
        check(argv[1])
    else:
        print(oracle(argv[1], int(argv[2])), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
