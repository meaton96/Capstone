"""
@file calib_twin.py
@brief agv-study-rerun (v5 design, 2026-10-14): pick the transport-bound regime's operation times in the twin
       (DES-1k, free flow). For candidate mean-op ranges, run the b2t instance (agv5_scenarios.py) at fleets 2-25 under
       MDD-PTECT and report the knee (smallest fleet within 5% of the best mean time in system) and total tardiness.
       Target: a twin knee of about 7-10 AGVs on 15 machines, like the transport-bound compound scenario of the old
       study, so small fleets are AGV-limited and Unity congestion can matter at 10-25. b2m is printed for reference.
@par Usage
@code{.sh}
.venv/bin/python results/agv-study-rerun/calib_twin.py [--workers 6]
@endcode
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "env"))
sys.path.insert(0, str(HERE))
FLOOR = REPO / "linux_server/Results/outdated/G1/_mfsweep_control/D/agv25_FIFO_ECT_s42/des_floor.json"   # 25 bays
FLEETS = [2, 3, 5, 7, 10, 15, 20, 25]
CANDIDATES = {"b2m": None, "b2t 60-120": (60.0, 120.0), "b2t 90-150": (90.0, 150.0), "b2t 120-240": (120.0, 240.0)}


def point(args):
    label, ops, seed, agvs, rule = args
    from des_twin.engine import Twin, TwinConfig
    from des_twin.floor import Floor
    from des_twin.scenario import resolve_jobs
    import agv5_scenarios as sc
    regime = "b2m" if ops is None else "b2t"
    scen = sc.build(regime, seed, ops=ops)
    floor = Floor.load(FLOOR)
    tw = Twin(floor, resolve_jobs(scen, floor), TwinConfig(rule=rule, transport="kinematic", agv_count=agvs,
                                                            max_sim_seconds=21600 * 6))
    for _ in tw._events():
        raise RuntimeError("agent decision in a fixed-rule run")
    m, s = tw.metrics(), tw.summary()
    return {"label": label, "seed": seed, "agvs": agvs, "rule": rule, "jobs": m["jobs_total"], "exited": m["jobs_exited"],
            "mean_tis": m["time_in_system_sum"] / max(1, m["jobs_total"]), "tard_k": m["tardiness_sum"] / 1000.0,
            "agv_busy": s.get("agv_busy_fraction"), "machine_util": s.get("machine_util_mean"), "trips": s.get("agv_trips")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    import pandas as pd
    tasks = [(lab, ops, seed, n, rule) for lab, ops in CANDIDATES.items() for seed in (0, 1) for n in FLEETS
             for rule in ("MDD_PTECT", "MDD_PECT")]
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(point, tasks))
    d = pd.DataFrame(rows)
    d.to_csv(HERE / "calib_twin.csv", index=False)
    pd.set_option("display.width", 220)
    print("== mean time in system (s) by fleet, mean over seeds; MDD-PTECT")
    x = d[d.rule == "MDD_PTECT"]
    print(x.pivot_table(index="label", columns="agvs", values="mean_tis").round(0).to_string())
    print("\n== total tardiness (1000 job-s) by fleet, mean over seeds; MDD-PTECT")
    print(x.pivot_table(index="label", columns="agvs", values="tard_k").round(1).to_string())
    print("\n== AGV busy fraction by fleet; MDD-PTECT")
    print(x.pivot_table(index="label", columns="agvs", values="agv_busy").round(2).to_string())
    print("\n== knee5 (mean time in system) per label x rule x seed, jobs, machine util at 25 AGVs")
    out = []
    for (lab, rule, seed), g in d.groupby(["label", "rule", "seed"]):
        g = g.set_index("agvs").sort_index()
        best = g.mean_tis.min()
        out.append({"label": lab, "rule": rule, "seed": seed, "knee5": int(g.index[g.mean_tis <= 1.05 * best].min()),
                    "knee5_tard": int(g.index[g.tard_k <= 1.05 * g.tard_k.min()].min()), "jobs": int(g.jobs.iloc[0]),
                    "mach_util@25": round(float(g.machine_util.iloc[-1]), 2),
                    "PECT_vs_PTECT_%@3": None})
    k = pd.DataFrame(out)
    print(k.drop(columns="PECT_vs_PTECT_%@3").to_string(index=False))
    print("\n== MDD-PECT vs MDD-PTECT, total tardiness % by fleet (mean over seeds; > 0: the travel term helps)")
    t = d.pivot_table(index=["label", "agvs"], columns="rule", values="tard_k")
    print((100 * (t.MDD_PECT / t.MDD_PTECT - 1)).unstack("agvs").round(1).to_string())


if __name__ == "__main__":
    main()
