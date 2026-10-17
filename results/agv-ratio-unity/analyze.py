"""
@file analyze.py
@brief agv-ratio-unity (v5 design, 2026-10-14): does Unity's congestion move the twin's free-flow fleet knee at 50 / 100 machines?
       For every Unity run (knee5 and ~1.5x knee, util 0.85, op 240 / 480 s, SRT_TECT, seeds 0 / 1): Unity's mean time
       in system against the twin sweep's value at the same fleet, seed and rule (../agv-ratio-twin-v5/*.csv, MDD-PTECT; same
       instance, run to empty, so mean flow = mean time in system) and against DES-1k on the run's own export
       (des_twin_gap.py, agv-ratio-unity_runs.csv), and the knee test: is Unity at the twin knee within 5% of Unity at
       1.5x the knee (on mean time in system and on total tardiness)? Also AGV busy / route-wait fractions, stalls, teleports to parking, collisions.
@par Usage
@code{.sh}
.venv/bin/python results/agv-ratio-unity/analyze.py     # -> analysis.out
@endcode
"""
import io
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RUNS = REPO / "linux_server" / "Results" / "agv-ratio-unity"
TWIN = REPO / "results" / "agv-ratio-twin-v5"
SCEN = re.compile(r"agvratio5_m(\d+)_u([\d.]+)_op(\d+)_s(\d+)$")
CELL = re.compile(r"agv(\d+)_(.+)_s(\d+)$")


def unity():
    rows = []
    for f in sorted(RUNS.glob("agvratio5_*/D/agv*/results.csv")):
        d = f.parent
        sm, cm = SCEN.match(d.parent.parent.name), CELL.match(d.name)
        if not (sm and cm):
            continue
        r = pd.read_csv(f, dtype={"config_hash": str, "instance_hash": str}).iloc[-1]
        a = pd.read_csv(d / "agv_performance.csv")
        busy = a.time_traveling + a.time_loading + a.time_unloading
        tot = busy + a.time_idle + a.time_waiting_route
        rows.append({"m": int(sm.group(1)), "util": float(sm.group(2)), "op": int(sm.group(3)), "seed": int(sm.group(4)),
                     "agvs": int(cm.group(1)), "rule": cm.group(2), "unity_tis": r.mean_flow_time_penalized,
                     "unity_tard_k": r.jobs_with_due_date * r.mean_tardiness / 1000.0,
                     "censored": r.jobs_censored, "deadlock": r.deadlock_detected, "collisions": r.agv_collision_events,
                     "agv_busy": float((busy / tot).mean()), "route_wait": float((a.time_waiting_route / tot).mean()),
                     "stalls": int(a.stall_recovery_count.sum()), "snaps": int(a.stall_snap_count.sum()),
                     "s_per_trip": float((tot.sum() - a.time_idle.sum()) / max(1, a.total_trips.sum()))})
    return pd.DataFrame(rows)


def main():
    pd.set_option("display.width", 220)
    u = unity()
    if u.empty:
        print("no Unity runs yet")
        return
    tw = []
    for f in TWIN.glob("m*_u*_op*.csv"):
        x = pd.read_csv(f)
        m = re.match(r"m(\d+)_u([\d.]+)_op(\d+)", f.stem)
        x["m"], x["util"], x["op"] = int(m.group(1)), float(m.group(2)), int(m.group(3))
        tw.append(x)
    tw = pd.concat(tw, ignore_index=True).rename(columns={"agv_count": "agvs"})
    tw["twin_tard_k"] = tw.tardiness_sum / 1000.0
    tw = tw[["m", "util", "op", "seed", "agvs", "rule", "mean_tis", "twin_tard_k", "agv_busy_fraction"]]
    d = u.merge(tw, on=["m", "util", "op", "seed", "agvs", "rule"], how="left")
    d["premium_%"] = 100 * (d.unity_tis / d.mean_tis - 1)
    d["tard_premium_%"] = 100 * (d.unity_tard_k / d.twin_tard_k - 1)
    g = HERE / "agv-ratio-unity_runs.csv"
    if g.exists():
        k = pd.read_csv(g)
        k["seed"] = k.seed.astype(int)
        k = k.rename(columns={"gap_kinematic": "gap_des1k_%"})
        sm = k.scenario.str.extract(r"agvratio5_m(\d+)_u([\d.]+)_op(\d+)_s(\d+)")
        k["m"], k["op"] = sm[0].astype(int), sm[2].astype(int)
        d = d.merge(k[["m", "op", "agvs", "rule", "seed", "gap_des1k_%", "trip_inflation", "zone_wait_per_leg"]],
                    on=["m", "op", "agvs", "rule", "seed"], how="left")
    print("== agv-ratio-unity: Unity vs the twin's free-flow fleet sweep (same instance, fleet, seed, rule)")
    print("   premium_% = Unity mean time in system / twin sweep's - 1; gap_des1k_% = vs DES-1k on the run's own export")
    print(d.sort_values(["m", "op", "seed", "agvs"]).to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print("\n== knee test: Unity at the twin's knee vs Unity at ~1.5x the knee (within 5% = the twin knee holds in Unity)")
    rows = []
    for (m, op, seed), x in d.groupby(["m", "op", "seed"]):
        if len(x) < 2:
            continue
        x = x.sort_values("agvs")
        lo, hi = x.iloc[0], x.iloc[-1]
        rows.append({"m": m, "op": op, "seed": seed, "knee": lo.agvs, "big": hi.agvs,
                     "unity_knee_vs_big_%": 100 * (lo.unity_tis / hi.unity_tis - 1),
                     "twin_knee_vs_big_%": 100 * (lo.mean_tis / hi.mean_tis - 1),
                     "premium_at_knee_%": lo["premium_%"], "premium_at_big_%": hi["premium_%"],
                     "unity_tard_knee_vs_big_%": 100 * (lo.unity_tard_k / hi.unity_tard_k - 1),
                     "twin_tard_knee_vs_big_%": 100 * (lo.twin_tard_k / hi.twin_tard_k - 1)})
    if rows:
        r = pd.DataFrame(rows)
        r["knee_holds_in_unity"] = r["unity_knee_vs_big_%"] <= 5.0
        print(r.to_string(index=False, float_format=lambda v: f"{v:.3g}"))


if __name__ == "__main__":
    buf = io.StringIO()
    with redirect_stdout(buf):
        main()
    (HERE / "analysis.out").write_text(buf.getvalue())
    sys.stdout.write(buf.getvalue())
