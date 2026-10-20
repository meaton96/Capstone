"""
@file run.py
@brief rq3-size-catalog (2026-10-20, user: do the rules that give switching headroom change with floor size?): the
       84-pair catalog screen of ../rq2-portfolio-screen (12 job rules x 7 machine rules, each held as a fixed pair,
       window tardiness split by 1.5 h regime block) at 30, 50 and 100 machines, on the floor-size grid's conditions
       through ../rq3-size-oracle/sizesim.py (frozen v5 code and sim.py of rq2-rebase-v5; MDD-PTECT = the grid's).
Seeds: test 0-39 (reporting) and fitting 2000-2039 (portfolio selection for the step-3 oracle). 15 machines = the
original screen (its fitting seeds 2000-2039 are used for comparability).
Output <cond>/out/s<seed>.json in rq2-portfolio-screen's format (so its select_portfolios functions read it).
@par Usage
@code{.sh}
python results/rq3-size-catalog/run.py run m50 0 [out_root]
python results/rq3-size-catalog/run.py list              # task lines, m100 first
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
CONDS = ["m100", "m50", "m30"]
SEEDS = list(range(40)) + list(range(2000, 2040))


def load(cond):
    spec = importlib.util.spec_from_file_location("ps_run", R / "rq2-portfolio-screen" / "run.py")  # frozen code first
    ps = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ps)
    spec = importlib.util.spec_from_file_location("sizesim", R / "rq3-size-oracle" / "sizesim.py")
    ss = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ss)
    ss.apply(ps.sim, cond)
    return ps, ss


def run(cond, seed, root=HERE):
    out = Path(root) / cond / "out" / f"s{seed}.json"
    if out.exists():
        return f"exists {out}"
    ps, ss = load(cond)
    t = time.time()
    ep = ps.sim.Episode("B2", seed)
    res, win = {}, None
    for p in ps.POOL:
        total, blocks, win = ps.play_blocks(ep, p)
        res[p] = {"total": total, "blocks": blocks}
    g = ss.grid_value(cond, seed)
    if g is not None and abs(res["MDD-PTECT"]["total"] - g) > 1e-6 * max(1.0, abs(g)) + 1e-4:
        raise SystemExit(f"{cond} s{seed}: MDD-PTECT {res['MDD-PTECT']['total']:.6f} != grid {g:.6f}")
    res["_blocks"] = [{k: b.get(k) for k in ("block", "start", "end", "load_profile", "agv_count")} for b in ep.blocks
                      if b["end"] > win[0] and b["start"] < win[1]]
    res["_window"] = list(win)
    res["_cond"] = cond
    res["_grid_check"] = g
    res["_stamp"] = ps.rebase.stamp()
    res["_seconds"] = time.time() - t
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(res))
    os.replace(tmp, out)
    fx = {p: res[p]["total"] for p in ps.POOL}
    best = min(fx, key=fx.get)
    return (f"{cond} s{seed}: best {best} {fx[best]:.2f}, MDD-PTECT {fx['MDD-PTECT']:.2f} (grid {g}), "
            f"{time.time() - t:.0f} s")


def main(argv):
    if argv[0] == "list":
        for c in CONDS:
            for s in SEEDS:
                print("run", c, s)
    else:
        print(run(argv[1], int(argv[2]), *(argv[3:4])), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
