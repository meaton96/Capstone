#!/usr/bin/env python3
"""Per-tile breakdown of tiled-floor runs (docs/features/TILED_LAYOUT_SCOPE.md), plus the phase-1 confinement check.

For each run folder (one holding results.csv): a job's tile is the tile of the machines its routing decisions chose
(decision_log.csv, machine // machines_per_tile). Confinement check: every job's routed machines lie in one tile.
AGV i serves tile i // (agvs / tiles); machine m is in tile m // machines_per_tile. Zone traffic per tile comes from
the T{t}_ zone-name prefix in segment_congestion.csv.

Linked floors (job_scope open / agv_assignment pooled, scope section 10) are not confined by design: instead of the
confinement check this reports how many machine-to-machine moves cross tiles, how far (in tiles), and the seam
bridges' traffic. A job's per-tile row there is the tile of its first routed machine.

Usage: python3 results/scripts/tile_split.py <run folder or experiment folder> [...]
"""
import argparse
import csv
import os
import statistics as st
from collections import defaultdict


def rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else float("nan")


def analyse(run):
    res = rows(os.path.join(run, "results.csv"))[0]
    tiles = int(res.get("tiles") or 1)
    mpt = int(res.get("machines_per_tile") or res["machines"])
    agvs = int(res["agvCount"])
    apt = agvs // tiles

    linked = res.get("job_scope", "tile") == "open" or res.get("agv_assignment", "tile") == "pooled"
    job_tiles = defaultdict(set)
    job_path = defaultdict(list)   # routed machine tiles in decision order
    for r in rows(os.path.join(run, "decision_log.csv")):
        if r["decision_type"] == "Routing":
            job_tiles[int(r["subject_id"])].add(int(r["chosen_id"]) // mpt)
            job_path[int(r["subject_id"])].append(int(r["chosen_id"]) // mpt)
    leaked = {j: t for j, t in job_tiles.items() if len(t) > 1}
    hops = [abs(b - a) for p in job_path.values() for a, b in zip(p, p[1:])]

    flow = defaultdict(list)
    unfinished = defaultdict(int)
    for r in rows(os.path.join(run, "job_completions.csv")):
        path = job_path.get(int(r["job_id"]))
        t = path[0] if path else -1
        if r["completed"] == "1":
            flow[t].append(float(r["flow_time"]))
        else:
            unfinished[t] += 1

    ms = float(res["makespan"])
    busy = defaultdict(list)
    for r in rows(os.path.join(run, "agv_performance.csv")):
        work = sum(float(r[k]) for k in ("time_traveling", "time_loading", "time_unloading", "time_waiting_route"))
        busy[int(r["agv_id"]) // apt].append(work / ms)
    mutil = defaultdict(list)
    for r in rows(os.path.join(run, "machine_utilization.csv")):
        mutil[int(r["machine_id"]) // mpt].append(float(r["utilization_rate"]))
    block = defaultdict(float)
    seam_trav, seam_block = 0, 0.0
    for r in rows(os.path.join(run, "segment_congestion.csv")):
        name = r["zone_name"]
        if name.startswith("Seam"):
            seam_trav += int(float(r.get("traversal_count") or 0))
            seam_block += float(r["total_block_time"])
            continue
        t = int(name[1:name.index("_")]) if tiles > 1 and name.startswith("T") else 0
        block[t] += float(r["total_block_time"])

    print(f"\n{run}")
    print(f"  {tiles} tile(s) x {mpt} machines, {apt} AGVs each, release={res.get('release_rule', '-')}, "
          f"makespan={ms:.0f}, mean flow={res['mean_flow_time']}, collisions={res['agv_collision_events']}, "
          f"deadlock={res['deadlock_detected']}, floor={res['floor_width']} x {res['floor_depth']}")
    if linked:
        cross = [h for h in hops if h > 0]
        print(f"  linked ({res.get('job_scope')}/{res.get('agv_assignment')}): {len(job_tiles)} routed jobs, "
              f"{len(leaked)} used more than one tile; machine-to-machine moves {len(hops)}, "
              f"{len(cross)} cross tiles ({100 * len(cross) / max(1, len(hops)):.0f}%), "
              f"mean {st.mean(cross) if cross else 0:.1f} tiles per crossing; "
              f"seam bridges {seam_trav} traversals, {seam_block:.0f} s blocked")
    else:
        print(f"  confinement: {len(job_tiles)} routed jobs, {len(leaked)} routed to more than one tile"
              + (f"  <-- FAIL e.g. {list(leaked.items())[:3]}" if leaked else "  (OK)"))
    print("  tile  jobs  unfinished  mean_flow  p95_flow  agv_busy  machine_util  zone_block_s")
    for t in range(tiles):
        f = flow[t]
        print(f"  {t:4d}  {len(f):4d}  {unfinished[t]:10d}  {st.mean(f) if f else float('nan'):9.1f}  "
              f"{pct(f, 0.95):8.1f}  {st.mean(busy[t]) if busy[t] else float('nan'):8.2f}  "
              f"{st.mean(mutil[t]) if mutil[t] else float('nan'):12.2f}  {block[t]:12.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    for p in ap.parse_args().paths:
        runs = sorted(r for r, _, fs in os.walk(p) if "results.csv" in fs)
        if not runs:
            print(f"{p}: no results.csv found")
        for run in runs:
            analyse(run)


if __name__ == "__main__":
    main()
