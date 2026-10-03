#!/usr/bin/env python3
"""G: gap between the Unity floor and its event-based twin (env/des_twin), over a sweep run with -destrace.

  python results/scripts/des_twin_gap.py linux_server_des/Results/G1 [--out results/des_twin] [--workers 16]

For every Unity run under the sweep directory (<scenario>/<layout>/agv<N>_<RULE>_s<seed>/) the twin is run with
each transport model on the same floor and jobs:

  instant    DES-0: no vehicles, moves take no time (the usual DRL-DFJSP model);
  geometric  DES-1g: the same AGVs, path length / speed, no interaction;
  kinematic  DES-1k: Unity's own free-flow motion, no interaction. The docking runs (V1) show this reproduces Unity
             exactly when AGVs cannot meet, so Unity - kinematic is the effect of AGVs sharing the floor.

Outputs (in --out):
  G1_runs.csv     one row per Unity run: Unity and twin metrics, the gaps, leg-level trip inflation;
  G1_cells.csv    per scenario x layout x fleet: mean gaps over rules, Unity deadlocks / censored jobs;
  G1_ranks.csv    per scenario x layout x fleet: Kendall tau between Unity's rule ranking (mean flow) and each twin's,
                  and whether the best rule agrees;
  G1_fleet.png    mean flow time against fleet size, Unity vs the three twins, one panel per scenario x layout.
"""
import argparse
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from functools import partial

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "env"))

TRANSPORTS = ("instant", "geometric", "kinematic")


## @brief Strict queue FIFO since this player build (linux_server/, 2026-10-02 16:13); env/des_twin/rules.py changed with
##        it. A Unity FIFO run older than this used FIFO by time since shop arrival, so it can't be paired with the twin.
STRICT_FIFO_SINCE = datetime(2026, 10, 2, 16, 13).timestamp()


def old_fifo_runs(runs):
    """Unity runs with a FIFO rule whose results predate the strict-FIFO build."""
    return [r for r in runs if "FIFO" in os.path.basename(r).upper()
            and os.path.getmtime(os.path.join(r, "results.csv")) < STRICT_FIFO_SINCE]


def one_run(run_dir, twin_out="twin"):
    from des_twin.legs import compare_legs
    from des_twin.run import run_dir as twin_run
    try:
        rows = twin_run(run_dir, TRANSPORTS, out=twin_out)
    except Exception as ex:                      # a bad run must not stop the sweep analysis
        return {"run": run_dir, "error": f"{type(ex).__name__}: {ex}"}
    by = {r["transport"]: r for r in rows}
    u = rows[0]
    parts = run_dir.rstrip("/").split("/")
    out = {"run": run_dir, "scenario": parts[-3], "layout": parts[-2], "agvs": u["agv_count"] if u["agv_count"] else
           by["kinematic"]["agv_count"], "rule": u["rule"], "seed": u["seed"],
           "unity_flow": u["unity_mean_flow_time"], "unity_p95": u["unity_p95_flow_time"],
           "unity_makespan": u["unity_makespan"], "unity_transport_wait": u["unity_mean_transport_wait"],
           "unity_censored": u["unity_jobs_censored"], "unity_deadlock": u["unity_deadlock"], "jobs": u["jobs"]}
    for tr in TRANSPORTS:
        r = by[tr]
        out[f"{tr}_flow"] = r["mean_flow_time"]
        out[f"{tr}_p95"] = r["p95_flow_time"]
        out[f"{tr}_makespan"] = r["makespan"]
        out[f"{tr}_transport_wait"] = r["mean_transport_wait"]
        out[f"gap_{tr}"] = (u["unity_mean_flow_time"] - r["mean_flow_time"]) / r["mean_flow_time"] \
            if r["mean_flow_time"] else float("nan")
    legs = compare_legs(run_dir)
    if legs:
        u_s = sum(r["unity_s"] for r in legs)
        f_s = sum(r["freeflow_s"] for r in legs)
        out.update(legs=len(legs), legs_delayed=sum(r["zone_wait_s"] > 0 for r in legs) / len(legs),
                   trip_inflation=u_s / f_s if f_s else float("nan"),
                   zone_wait_per_leg=sum(r["zone_wait_s"] for r in legs) / len(legs),
                   p95_leg_delay=pd.Series([r["delay_s"] for r in legs]).quantile(0.95))
    return out


def kendall_tau(a, b):
    """Kendall tau-b between two score lists (lower = better), ties counted as neither."""
    n, c, d, ta, tb = len(a), 0, 0, 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            x, y = a[i] - a[j], b[i] - b[j]
            if x == 0 and y == 0:
                continue
            if x == 0:
                ta += 1
            elif y == 0:
                tb += 1
            elif (x > 0) == (y > 0):
                c += 1
            else:
                d += 1
    denom = ((c + d + ta) * (c + d + tb)) ** 0.5
    return (c - d) / denom if denom else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweep_dir")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "des_twin"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--tag", default="G1")
    ap.add_argument("--twin-out", default=None,
                    help="per-run folder for the twin's outputs (default twin_<tag>; never overwrites another tag's)")
    ap.add_argument("--allow-old-fifo", action="store_true",
                    help="pair FIFO Unity runs from before the strict-FIFO build with today's (strict-FIFO) twin")
    a = ap.parse_args()
    runs = sorted(os.path.dirname(p) for p in glob.glob(os.path.join(a.sweep_dir, "*", "*", "*", "results.csv"))
                  if os.path.exists(os.path.join(os.path.dirname(p), "des_floor.json")))
    print(f"{len(runs)} Unity runs")
    stale = old_fifo_runs(runs)
    if stale and not a.allow_old_fifo:
        sys.exit(f"{len(stale)} FIFO runs (e.g. {stale[0]}) predate the strict-FIFO build of 2026-10-02 16:13, but the "
                 "twin now uses strict queue FIFO. Rerun them on the new build (twin-gap-qfifo), or pass "
                 "--allow-old-fifo to pair them anyway.")
    twin_out = a.twin_out or f"twin_{a.tag}"
    with ProcessPoolExecutor(a.workers) as pool:
        rows = list(pool.map(partial(one_run, twin_out=twin_out), runs, chunksize=2))
    os.makedirs(a.out, exist_ok=True)
    df = pd.DataFrame(rows)
    if "error" in df:
        bad = df[df["error"].notna()]
        for _, r in bad.iterrows():
            print("ERROR", r["run"], r["error"])
        df = df[df["error"].isna()].drop(columns="error")
    df.to_csv(os.path.join(a.out, f"{a.tag}_runs.csv"), index=False)

    ok = df[(df.unity_deadlock == 0) & (df.unity_censored == 0)]
    key = ["scenario", "layout", "agvs"]
    cells = df.groupby(key).agg(runs=("rule", "size"), unity_deadlocks=("unity_deadlock", "sum"),
                                unity_censored_runs=("unity_censored", lambda s: int((s > 0).sum()))).reset_index()
    gaps = ok.groupby(key).agg(**{f"gap_{t}_mean": (f"gap_{t}", "mean") for t in TRANSPORTS},
                               **{f"gap_{t}_max": (f"gap_{t}", "max") for t in TRANSPORTS},
                               unity_flow=("unity_flow", "mean"), kinematic_flow=("kinematic_flow", "mean"),
                               geometric_flow=("geometric_flow", "mean"), instant_flow=("instant_flow", "mean"),
                               trip_inflation=("trip_inflation", "mean"), legs_delayed=("legs_delayed", "mean"),
                               zone_wait_per_leg=("zone_wait_per_leg", "mean")).reset_index()
    cells = cells.merge(gaps, on=key, how="left")
    cells.to_csv(os.path.join(a.out, f"{a.tag}_cells.csv"), index=False)

    ranks = []
    for k, g in ok.groupby(key):
        if len(g) < 3:
            continue
        g = g.sort_values("rule")
        def best(col):   # every rule tied (to 0.01 s) for the lowest mean flow; several rules often tie exactly
            return set(g.loc[g[col] <= g[col].min() + 0.01, "rule"])
        ub = best("unity_flow")
        row = dict(zip(key, k), rules=len(g), unity_best="|".join(sorted(ub)))
        for t in TRANSPORTS:
            tb = best(f"{t}_flow")
            row[f"tau_{t}"] = kendall_tau(list(g.unity_flow), list(g[f"{t}_flow"]))
            row[f"best_{t}"] = "|".join(sorted(tb))
            row[f"best_agrees_{t}"] = bool(ub & tb)
        ranks.append(row)
    pd.DataFrame(ranks).to_csv(os.path.join(a.out, f"{a.tag}_ranks.csv"), index=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        panels = sorted(set(zip(cells.scenario, cells.layout)))
        fig, axes = plt.subplots(len(panels), 1, figsize=(7, 2.8 * len(panels)), squeeze=False)
        for ax, (s, l) in zip(axes[:, 0], panels):
            c = cells[(cells.scenario == s) & (cells.layout == l)].sort_values("agvs")
            for col, lab, sty in (("unity_flow", "Unity (physical)", "-o"), ("kinematic_flow", "DES-1k free-flow", "--s"),
                                  ("geometric_flow", "DES-1g distance/speed", ":^"), ("instant_flow", "DES-0 instant", "-.")):
                ax.plot(c.agvs, c[col], sty, label=lab)
            for _, r in c[c.unity_deadlocks > 0].iterrows():
                ax.annotate(f"{int(r.unity_deadlocks)} deadlock", (r.agvs, r.kinematic_flow), fontsize=7)
            ax.set_title(f"{s}, layout {l}", fontsize=9)
            ax.set_xlabel("AGVs")
            ax.set_ylabel("mean flow time (s)")
            ax.set_yscale("log")
        axes[0, 0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, f"{a.tag}_fleet.png"), dpi=130)
    except Exception as ex:
        print("plot skipped:", ex)
    print(cells.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
