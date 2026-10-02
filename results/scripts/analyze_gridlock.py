"""
Classify AGV-traffic gridlock in batch results and summarise it by AGV count / rule.

Works on any directory tree containing per-run `results.csv` + `agv_performance.csv`
pairs (run_gridlock_sweep.sh cells, or the older Results/mfsweep/* layout). Works on
builds from before AND after the 2026-09-19 watchdog change; columns added by that
change (first_stall_sim_time, orphan_predispatch_released) are used when present.

A run is GRIDLOCKED (terminal) if either:
  * deadlock_detected == 1            (watchdog fired; post-change builds)
  * jobs_censored > 0                 (episode ended with jobs unfinished, e.g. hit the time cap)
A run is STALLED-BUT-RECOVERED if it finished normally yet an AGV needed >= 1 zone-stall
recovery (sat 180 s on one reservation). On the original build every stall was terminal, so
"stall" and "gridlock" coincided; with -releasepreviouszone they do not, so they are reported
separately (columns `gridlock` and `recovered`).
A run has the ORPHAN signature if an AGV was >50 % idle for the episode with < 20 stall
recoveries -- an AGV parked at a pickup dock (MovingToPrePickup counts as idle) rather than
waiting on a zone. Pre-fix, every gridlocked run had exactly one.

Usage: python analyze_gridlock.py <root_dir> [--csv out.csv]
"""
import argparse
import glob
import os
import sys

import pandas as pd


def load_runs(root):
    frames = []
    for res_path in sorted(glob.glob(os.path.join(root, "**", "results.csv"), recursive=True)):
        d = os.path.dirname(res_path)
        agv_path = os.path.join(d, "agv_performance.csv")
        if not os.path.exists(agv_path):
            continue
        res = pd.read_csv(res_path)
        agv = pd.read_csv(agv_path)
        res["cell"] = os.path.relpath(d, root)
        agv["cell"] = os.path.relpath(d, root)
        frames.append((res, agv))
    return frames


def summarise(frames):
    rows = []
    for res, agv in frames:
        for _, r in res.iterrows():
            a = agv[(agv["cell"] == r["cell"]) & (agv["rule"] == r["rule"]) & (agv["seed"] == r["seed"])
                    & (agv["instance"] == r["instance"])]
            makespan = float(r["makespan"])
            idle_frac = a["time_idle"] / max(makespan, 1e-9)
            orphan = int(((idle_frac > 0.5) & (a["stall_recovery_count"] < 20)).sum())
            stalls = int(a["stall_recovery_count"].sum())
            wd = int(r.get("deadlock_detected", 0)) == 1
            censored = int(r.get("jobs_censored", 0)) > 0
            rows.append({
                "cell": r["cell"], "rule": r["rule"], "seed": int(r["seed"]),
                "agv": int(r["agvCount"]), "failures": int(r.get("episode_failures", 0)),
                "makespan": makespan, "jobs_censored": int(r.get("jobs_censored", 0)),
                "watchdog": wd, "stall_recoveries": stalls, "orphan_agvs": orphan,
                "first_stall": float(r.get("first_stall_sim_time", -1.0)),
                "orphans_released": int(r.get("orphan_predispatch_released", 0)),
                "gridlock": bool(wd or censored),
                "recovered": bool(stalls >= 1 and not (wd or censored)),
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--csv")
    args = ap.parse_args()

    frames = load_runs(args.root)
    if not frames:
        sys.exit(f"no results.csv + agv_performance.csv pairs under {args.root}")
    df = summarise(frames)
    if args.csv:
        df.to_csv(args.csv, index=False)

    print(f"{len(df)} runs, {int(df.gridlock.sum())} gridlocked ({df.gridlock.mean():.0%}), "
          f"{int(df.recovered.sum())} stalled-but-recovered\n")
    print("Gridlock rate by AGV count x rule (gridlocked/runs):")
    piv = df.groupby(["agv", "rule"]).gridlock.agg(lambda s: f"{int(s.sum())}/{len(s)}").unstack("rule")
    print(piv.to_string(), "\n")
    print("Gridlock rate by AGV count:")
    by = df.groupby("agv").agg(runs=("gridlock", "size"), gridlocked=("gridlock", "sum"),
                               rate=("gridlock", "mean"), recovered=("recovered", "sum"), median_first_stall=("first_stall", lambda s: s[s >= 0].median()),
                               orphan_runs=("orphan_agvs", lambda s: int((s > 0).sum())),
                               orphans_released=("orphans_released", "sum"))
    print(by.to_string(float_format=lambda x: f"{x:.2f}"), "\n")
    print("Orphan signature vs gridlock (rows = gridlock, cols = orphan AGVs in run):")
    print(pd.crosstab(df.gridlock, df.orphan_agvs).to_string())


if __name__ == "__main__":
    main()
