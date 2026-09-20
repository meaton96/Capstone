"""
AGV-AGV physical-overlap summary by traffic protocol / layout / fleet size.

Reads every results.csv under <root> (needs the columns added 2026-09-20:
release_previous_zone, split_spines, agv_collision_events, ...) plus the sibling
agv_collisions.csv (one row per overlap event) when present. Config is taken from the CSV
columns, not from directory names, so it works on any sweep layout.

  overlap_ev/run   contiguous body overlaps per run between AGVs that are not both parked
  traffic_ev/run   the same, minus events where either AGV was in the Parking_Alcove
                   (needs agv_collisions.csv; parking overlaps are a parking-bay design issue)
  clear_ev/run     centre distance < 2 x NavMesh radius (2.36): zone-design margin violated
  min_dist         smallest centre distance seen in any run (0 = two AGVs at the same point)
  gridlock         terminal gridlocks (watchdog / unfinished jobs); such runs end early, so their
                   event counts are truncated and not comparable to completed runs

Usage: python analyze_collisions.py <root> [--csv out.csv] [--failures on|off]
"""
import argparse
import glob
import os

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--csv")
    ap.add_argument("--failures", choices=["on", "off"])
    a = ap.parse_args()

    runs, ev = [], []
    for f in sorted(glob.glob(os.path.join(a.root, "**", "results.csv"), recursive=True)):
        r = pd.read_csv(f)
        if "agv_collision_events" not in r.columns:
            continue
        d = os.path.dirname(f)
        r["cell"] = os.path.relpath(d, a.root)
        runs.append(r)
        cf = os.path.join(d, "agv_collisions.csv")
        if os.path.exists(cf):
            c = pd.read_csv(cf)
            c["cell"] = r["cell"].iloc[0]
            ev.append(c)
    if not runs:
        raise SystemExit("no results.csv with collision columns found")
    r = pd.concat(runs, ignore_index=True)
    if a.failures:
        r = r[(r.episode_failures > 0) == (a.failures == "on")]
    r["gridlock"] = (r.deadlock_detected == 1) | (r.jobs_censored > 0)
    r["cfg"] = ("rel=" + r.release_previous_zone.astype(str) + " split=" + r.split_spines.astype(str))

    if ev:
        e = pd.concat(ev, ignore_index=True)
        e["parking"] = e.zone_a.str.startswith("Parking") | e.zone_b.str.startswith("Parking")
        t = (e[~e.parking].groupby(["cell", "rule", "seed"]).size().rename("traffic_events").reset_index())
        r = r.merge(t, on=["cell", "rule", "seed"], how="left")
        r["traffic_events"] = r["traffic_events"].fillna(0)
    else:
        r["traffic_events"] = float("nan")

    g = r.groupby(["cfg", "agvCount"]).agg(
        runs=("gridlock", "size"), gridlock=("gridlock", "sum"),
        overlap_ev=("agv_collision_events", "mean"), traffic_ev=("traffic_events", "mean"),
        overlap_s=("agv_collision_pair_seconds", "mean"), clear_ev=("agv_clearance_events", "mean"),
        min_dist=("agv_min_centre_distance", "min"), flow=("mean_flow_time", "mean"))
    pd.set_option("display.width", 200)
    print(g.round(1).to_string())
    if a.csv:
        g.to_csv(a.csv)


if __name__ == "__main__":
    main()
