"""
@file analyze.py
@brief agv-study-rerun (v5 design, 2026-10-14): tables for the AGV physical study on the v5 rules, from the Unity runs
       of the agv-v5-* sweeps (linux_server/Results/agv-v5-*).

@details One row per Unity run (results.csv + agv_performance.csv). Objective: total tardiness (mean tardiness x jobs
with a due date; every run goes until the floor is empty, so it is uncensored unless the run deadlocked or timed
out, which is flagged). Also mean flow time (penalized), censored jobs, deadlock, collision / clearance /
parking-overlap events, closest approach, AGV busy and route-wait fractions, stall recoveries and stall snaps
(teleports to parking: the "physical" caveat). Regime = the scenario's family: b2m (machine-bound), b2t (transport-
bound), b2tf (b2t + machine failures), b2m / b2t on 105 machines (t7iso / t7open). Tables:
  1. protocol x fleet on D / J (agv-v5-hold vs agv-v5-rel, same cells), knee per protocol;
  2. layouts D / J / C / G x fleet (release) and the knee per layout; all 15 layouts at 7 / 15 AGVs vs D;
  3. machine failures on: release vs hold;
  4. 105 machines: isolated vs linked tiles;
  5. safety / liveness per sweep, with the stall-snap share;
  6. the travel term: MDD-PECT vs MDD-PTECT tardiness by regime and fleet;
  7. Unity vs the free-flow twin (DES-1k) on the single-tile sweeps (twin/<sweep>_runs.csv, des_twin_gap.py).
Knee = the smallest fleet within 5% of the best fleet's total tardiness (and, as a check, mean flow time), per
(sweep, regime, layout, rule), on the mean over instance seeds. Output: analysis.out, runs.csv.
@par Usage
@code{.sh}
.venv/bin/python results/agv-study-rerun/analyze.py
@endcode
"""
import io
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RES = REPO / "linux_server" / "Results"
SWEEPS = ["agv-v5-rel", "agv-v5-hold", "agv-v5-layouts", "agv-v5-fail-rel", "agv-v5-fail-hold", "agv-v5-tiles",
          "agv-v5-linked"]
HASH_STR = {"config_hash": str, "instance_hash": str}
CELL = re.compile(r"agv(\d+)_(.+)_s(\d+)$")
SCEN = re.compile(r"agv5_(b2[a-z]+)(?:_t7(iso|open))?_s(\d+)$")


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def runs():
    rows = []
    for sweep in SWEEPS:
        for res in sorted((RES / sweep).glob("*/*/agv*/results.csv")):
            d = res.parent
            cm, sm = CELL.match(d.name), SCEN.match(d.parent.parent.name)
            if not (cm and sm):
                continue
            try:
                r = pd.read_csv(res, dtype=str).iloc[-1]
            except Exception as exc:
                print(f"unreadable {res}: {exc}")
                continue
            ndue, mtard = f(r.get("jobs_with_due_date")), f(r.get("mean_tardiness"))
            row = {"sweep": sweep, "regime": sm.group(1) + ("" if sm.group(2) is None else f"-105{sm.group(2)}"),
                   "inst": int(sm.group(3)), "layout": d.parent.name, "agvs": int(cm.group(1)), "rule": cm.group(2),
                   "protocol": r.get("reservation_protocol"), "jobs": f(r.get("jobs")),
                   "tard_k": ndue * mtard / 1000.0, "pct_tardy": f(r.get("pct_tardy")),
                   "flow_pen": f(r.get("mean_flow_time_penalized")), "makespan": f(r.get("makespan")),
                   "censored": f(r.get("jobs_censored")), "deadlock": str(r.get("deadlock_detected")).lower() == "true",
                   "failures": f(r.get("episode_failures")), "collisions": f(r.get("agv_collision_events")),
                   "clearance": f(r.get("agv_clearance_events")), "parking_overlap": f(r.get("agv_parking_overlap_events")),
                   "min_dist": f(r.get("agv_min_centre_distance")), "cross_tile_moves": f(r.get("cross_tile_moves")),
                   "routed_moves": f(r.get("routed_moves"))}
            row["incomplete"] = bool(row["deadlock"] or (row["censored"] or 0) > 0)
            p = d / "agv_performance.csv"
            if p.exists():
                a = pd.read_csv(p)
                busy = a.time_traveling + a.time_loading + a.time_unloading
                tot = busy + a.time_idle + a.time_waiting_route
                row.update(agv_busy=float((busy / tot).mean()), route_wait=float((a.time_waiting_route / tot).mean()),
                           stalls=int(a.stall_recovery_count.sum()), snaps=int(a.stall_snap_count.sum()))
            rows.append(row)
    return pd.DataFrame(rows)


def knees(d, keys, metric):
    out = []
    for k, g in d.groupby(keys):
        s = g.groupby("agvs")[metric].mean().sort_index()
        if len(s) < 3:
            continue
        out.append(dict(zip(keys, k), knee5=int(s.index[s <= 1.05 * s.min()].min()), best_fleet=int(s.idxmin()),
                        best=round(float(s.min()), 1), fleets=len(s)))
    return pd.DataFrame(out)


def fl(v):
    return f"{v:.3g}" if isinstance(v, float) else str(v)


def main():
    pd.set_option("display.width", 240)
    pd.set_option("display.max_rows", 500)
    d = runs()
    if d.empty:
        print("no runs yet")
        return
    d.to_csv(HERE / "runs.csv", index=False)
    print("== agv-study-rerun (v5): Unity runs per sweep x regime; tard_k = total tardiness (1000 job-s), run to empty")
    print(d.groupby(["sweep", "regime"]).size().unstack(fill_value=0).to_string())

    print("\n== 5. safety and liveness per sweep (sums over runs; snaps = stall recoveries that teleported to parking)")
    s = d.groupby("sweep").agg(runs=("tard_k", "size"), deadlocks=("deadlock", "sum"),
                               incomplete_runs=("incomplete", "sum"), collisions=("collisions", "sum"),
                               min_dist=("min_dist", "min"), parking_overlap=("parking_overlap", "sum"),
                               runs_with_stall=("stalls", lambda x: int((x > 0).sum())), stalls=("stalls", "sum"),
                               snaps=("snaps", "sum"))
    s["snap_share"] = s.snaps / s.stalls.where(s.stalls > 0)
    print(s.to_string(float_format=lambda v: f"{v:.3g}"))

    pj = d[d.sweep.isin(["agv-v5-rel", "agv-v5-hold"]) & d.layout.isin(["D", "J"])].copy()
    if len(pj):
        pj["proto"] = np.where(pj.sweep == "agv-v5-rel", "release", "hold")
        print("\n== 1. protocol x fleet on D / J, MDD-PTECT, mean over instance seeds: total tardiness (1000 job-s), "
              "deadlocks, stalls, snaps")
        x = pj[pj.rule == "MDD_PTECT"]
        print(x.pivot_table(index=["regime", "layout", "agvs"], columns="proto",
                            values=["tard_k", "deadlock", "stalls", "snaps", "collisions"], aggfunc="mean")
              .round(1).to_string())
        print("\n-- knee (5%) per protocol: on total tardiness | on mean flow time")
        k1 = knees(pj, ["proto", "regime", "layout", "rule"], "tard_k")
        k2 = knees(pj, ["proto", "regime", "layout", "rule"], "flow_pen").rename(columns={"knee5": "knee5_flow"})
        print(k1.merge(k2[["proto", "regime", "layout", "rule", "knee5_flow"]], how="left")
              .to_string(index=False, float_format=fl))

    four = d[(d.sweep == "agv-v5-rel")]
    if len(four):
        print("\n== 2a. layouts D / J / C / G x fleet (release), MDD-PTECT: total tardiness (1000 job-s), mean over seeds")
        print(four[four.rule == "MDD_PTECT"].pivot_table(index=["regime", "agvs"], columns="layout", values="tard_k")
              .round(1).to_string())
        print("\n-- knee (5%, total tardiness) per layout")
        print(knees(four, ["regime", "layout", "rule"], "tard_k").to_string(index=False, float_format=fl))
    fif = d[d.sweep.isin(["agv-v5-rel", "agv-v5-layouts"]) & (d.rule == "MDD_PTECT") & d.agvs.isin([7, 15])
            & (d.inst == 0)]
    if len(fif):
        print("\n== 2b. all 15 layouts at 7 / 15 AGVs, MDD-PTECT, instance seed 0: total tardiness (1000 job-s) and % vs D")
        p = fif.pivot_table(index="layout", columns=["regime", "agvs"], values="tard_k")
        if "D" in p.index:
            print(pd.concat({"tard_k": p.round(1), "% vs D": (100 * (p / p.loc["D"] - 1)).round(1)}, axis=1).to_string())
        else:
            print(p.round(1).to_string())

    fa = d[d.sweep.isin(["agv-v5-fail-rel", "agv-v5-fail-hold"])]
    if len(fa):
        print("\n== 3. machine failures on (b2tf, seeds 0-2, D, MDD-PTECT): mean over seeds")
        print(fa.pivot_table(index="agvs", columns="sweep", values=["tard_k", "failures", "deadlock", "collisions",
                                                                     "stalls", "snaps"], aggfunc="mean").round(1).to_string())
        nf = d[(d.sweep == "agv-v5-rel") & (d.regime == "b2t") & (d.layout == "D") & (d.rule == "MDD_PTECT")]
        if len(nf):
            a = fa[fa.sweep == "agv-v5-fail-rel"].groupby(["inst", "agvs"]).tard_k.mean()
            b = nf.groupby(["inst", "agvs"]).tard_k.mean()
            both = pd.concat([a.rename("failures"), b.rename("no failures")], axis=1).dropna()
            if len(both):
                both["cost_%"] = 100 * (both["failures"] / both["no failures"] - 1)
                print("-- cost of machine failures under release (same jobs, seeds 0-1):")
                print(both.round(1).to_string())

    tl = d[d.sweep.isin(["agv-v5-tiles", "agv-v5-linked"])].copy()
    if len(tl):
        tl["per_tile"] = tl.agvs / 7
        tl["cross_share"] = tl.cross_tile_moves / tl.routed_moves.where(tl.routed_moves > 0)
        print("\n== 4. 105 machines (7 tiles of D): isolated vs linked")
        print(tl.pivot_table(index=["regime", "rule", "per_tile"], columns="sweep",
                             values=["tard_k", "flow_pen", "cross_share", "collisions", "deadlock", "snaps"])
              .round(2).to_string())

    r2 = d[d.sweep.isin(["agv-v5-rel", "agv-v5-hold"])]
    if len(r2) and r2.rule.nunique() == 2:
        t = r2.pivot_table(index=["sweep", "regime", "layout", "inst", "agvs"], columns="rule", values="tard_k").dropna()
        t["PECT_vs_PTECT_%"] = 100 * (t.MDD_PECT / t.MDD_PTECT - 1)
        print("\n== 6. the travel term: MDD-PECT vs MDD-PTECT total tardiness, % (> 0: the travel term helps), "
              "mean over layouts and seeds")
        print(t.groupby(["sweep", "regime", "agvs"])["PECT_vs_PTECT_%"].mean().unstack("agvs").round(1).to_string())

    tw = []
    for sweep in ["agv-v5-rel", "agv-v5-hold", "agv-v5-layouts"]:   # failures-on runs are not twin-paired
        g = HERE / "twin" / f"{sweep}_runs.csv"
        if g.exists():
            x = pd.read_csv(g)
            x["sweep"] = sweep
            tw.append(x)
    if tw:
        x = pd.concat(tw, ignore_index=True)
        x["regime"] = x.scenario.str.extract(r"agv5_(b2[a-z]+)_s")[0]
        x["fleet_band"] = pd.cut(x.agvs, [0, 3, 7, 15, 30], labels=["2-3", "5-7", "10-15", "20-25"])
        print("\n== 7. Unity vs the free-flow twin (DES-1k): mean |gap| of mean flow time, %")
        print(x.assign(abs_gap=x.gap_kinematic.abs()).pivot_table(index=["sweep", "regime", "layout"],
                                                                  columns="fleet_band", values="abs_gap",
                                                                  aggfunc="mean", observed=False).round(1).to_string())


if __name__ == "__main__":
    buf = io.StringIO()
    with redirect_stdout(buf):
        main()
    (HERE / "analysis.out").write_text(buf.getvalue())
    sys.stdout.write(buf.getvalue())
