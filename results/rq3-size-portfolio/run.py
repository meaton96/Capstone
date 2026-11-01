"""
@file run.py
@brief rq3-size-portfolio (2026-10-20, user: step 3 of the size rule sweep). Exact runs of what the catalog screen
       (../rq3-size-catalog) found by proxy, in the oracle's frozen simulator on the floor-size grid's conditions
       (../rq3-size-oracle/sizesim.py; frozen v5 code and sim.py of rq2-rebase-v5), B2 test seeds 0-39.
  map <cond> <seed>      the observable regime map chosen on fitting seeds over all 84 pairs (MAPS: one pair per regime,
                         normal / surge / short fleet, applied at each 900 s slot's first decision by the regime then,
                         as rq2-rebase-v5's map policies). At 15-50 machines it is the v5 map; at 100 machines it is
                         {short fleet -> MDD-TECT, otherwise MDD-PTECT}. -> out/<cond>/map/s<seed>.json
  oracle <cond> <portfolio> <seed>
                         rq2-rebase-v5's greedy switching oracle (900 s slots, hold-to-end tails) over a portfolio
                         (PORTFOLIOS): at m100 "v5tect" (the 6 v5 pairs + MDD-TECT) and "block6" (the catalog's greedy
                         per-block K = 6 set), at m50 "block6". -> out/<cond>/<portfolio>/s<seed>.json
The maps and portfolios are frozen here as chosen by rq3-size-catalog on fitting seeds 2000-2039 (checked against its
<cond>/portfolios.json at import). `check`: at m15 the map must equal rq2-rebase-v5's exact value for the same map.
@par Usage
@code{.sh}
python results/rq3-size-portfolio/run.py check
python results/rq3-size-portfolio/run.py map m100 0
python results/rq3-size-portfolio/run.py oracle m100 v5tect 0
python results/rq3-size-portfolio/run.py list map | list oracle
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
SEEDS = list(range(40))
V5 = [f"{j}-{m}" for j in ("SRT", "MDD", "ATC") for m in ("PECT", "PTECT")]
MAPS = {   # regime -> pair, fitted over all 84 pairs on fitting seeds (rq3-size-catalog "full" fit map)
    "m15": {"normal": "ATC-PTECT", "surge": "MDD-PTECT", "short": "MDD-PECT"},
    "m30": {"normal": "ATC-PECT", "surge": "ATC-PECT", "short": "MDD-PECT"},
    "m50": {"normal": "ATC-PTECT", "surge": "MDD-PTECT", "short": "MDD-PECT"},
    "m100": {"normal": "MDD-PTECT", "surge": "MDD-PTECT", "short": "MDD-TECT"},
}
PORTFOLIOS = {
    ("m100", "v5tect"): V5 + ["MDD-TECT"],
    ("m100", "block6"): ["MDD-PTECT", "MDD-TECT", "ATC-PTECT", "FIFO-PTECT", "SLACK-TECT", "MOD-PTECT"],
    ("m50", "block6"): ["MDD-PTECT", "ATC-PTECT", "CR-TECT", "MDD-PECT", "ATC-PECT", "SLACK-TECT"],
}
for _c, _m in MAPS.items():                                     # frozen choices = the catalog's
    _p = json.loads((R / "rq3-size-catalog" / _c / "portfolios.json").read_text())
    assert _p["full"]["fit"]["map"] == _m, (_c, _p["full"]["fit"]["map"])
for (_c, _n), _s in PORTFOLIOS.items():
    if _n == "block6":
        _p = json.loads((R / "rq3-size-catalog" / _c / "portfolios.json").read_text())
        assert _p["block6"]["pairs"] == _s, (_c, _p["block6"]["pairs"])


def load(cond):
    spec = importlib.util.spec_from_file_location("rb_run", R / "rq2-rebase-v5" / "run.py")   # frozen code first
    rb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rb)
    spec = importlib.util.spec_from_file_location("sizesim", R / "rq3-size-oracle" / "sizesim.py")
    ss = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ss)
    ss.apply(rb.sim, cond)
    return rb


def write(out, obj):
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, out)


def map_policy(cond):
    m = MAPS[cond]
    return lambda tw, dec, k, ep: m[ep.regime_at(tw.now)]


def run_map(cond, seed, root=HERE):
    out = Path(root) / "out" / cond / "map" / f"s{seed}.json"
    if out.exists():
        return f"exists {out}"
    rb = load(cond)
    t = time.time()
    ep = rb.sim.Episode("B2", seed)
    tard, played = ep.play(map_policy(cond))
    write(out, {"cond": cond, "seed": seed, "map": MAPS[cond], "tard": tard, "played": played,
                "stamp": rb.stamp(), "seconds": time.time() - t})
    return f"map {cond} s{seed}: {tard:.3f} k job-s, {time.time() - t:.0f} s"


def run_oracle(cond, name, seed, root=HERE):
    out = Path(root) / "out" / cond / name / f"s{seed}.json"
    if out.exists():
        return f"exists {out}"
    pairs = PORTFOLIOS[(cond, name)]
    rb = load(cond)
    import numpy as np
    t = time.time()
    ep = rb.sim.Episode("B2", seed)
    n = rb.sim.n_slots("B2")
    prefix, tails = [], np.zeros((n, len(pairs)))
    for k in range(n):
        for i, p in enumerate(pairs):
            tails[k, i] = ep.play(prefix + [p])[0]
        prefix.append(pairs[int(np.argmin(tails[k]))])
    write(out, {"cond": cond, "portfolio": name, "pairs": pairs, "seed": seed, "oracle": float(tails[-1].min()),
                "schedule": prefix, "fixed": dict(zip(pairs, tails[0].tolist())), "tails": tails.tolist(),
                "warmup": ep.warmup, "stamp": rb.stamp(), "seconds": time.time() - t})
    return f"oracle {cond}/{name} s{seed}: {float(tails[-1].min()):.3f} k job-s, {time.time() - t:.0f} s"


def check():
    """m15 map = rq2-rebase-v5's exact value for the same map (its key orders regimes normal / surge / short)."""
    m = MAPS["m15"]
    key = "map:" + "/".join(m[r] for r in ("normal", "surge", "short"))
    bad = 0
    for s in (0, 1, 2):
        ref = json.loads((R / "rq2-rebase-v5" / "out" / "policies" / f"s{s}.json").read_text())["tard"][key]
        rb = load("m15") if s == 0 else rb                       # sizesim.apply once per process
        v = rb.sim.Episode("B2", s).play(map_policy("m15"))[0]
        ok = abs(v - ref) < 1e-6
        bad += not ok
        print(f"check m15 s{s}: map {v:.6f} vs rq2-rebase-v5 {key} {ref:.6f} -> {'OK' if ok else 'MISMATCH'}")
    if bad:
        raise SystemExit(1)


def main(argv):
    if argv[0] == "check":
        check()
    elif argv[0] == "list":
        if argv[1] == "map":
            for c in ("m100", "m50", "m30", "m15"):
                for s in SEEDS:
                    print("map", c, s)
        else:
            for c, n in PORTFOLIOS:
                for s in SEEDS:
                    print("oracle", c, n, s)
    elif argv[0] == "map":
        print(run_map(argv[1], int(argv[2]), *argv[3:4]), flush=True)
    else:
        print(run_oracle(argv[1], argv[2], int(argv[3]), *argv[4:5]), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
