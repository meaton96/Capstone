#!/usr/bin/env python3
"""E1 phase split: how machine failures and the rule change flow time inside each phase kind of the compound
scenarios. Reads the per-run table written by analyze_load_sweep.py (results/E1_runs.csv).

Pinned phases (bottleneck, burst, starvation) send every job to one Weld machine, so with failures on their flow time
partly measures repair time, not dispatch quality; quiet has no Weld op; standoff / speed_trap choose within a 2-machine
pool. Usage: python3 results/scripts/phase_split.py [--csv results/E1_runs.csv] [--protocol releasePrevious] [--min-agv 7]
"""
import argparse
import pandas as pd

PH = ["flow_quiet", "flow_quiet_2b", "flow_bottleneck", "flow_standoff", "flow_speed_trap", "flow_burst", "flow_starvation"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/E1_runs.csv")
    ap.add_argument("--protocol", default="releasePrevious")
    ap.add_argument("--min-agv", type=int, default=7, help="skip AGV-limited fleets")
    a = ap.parse_args()
    pd.set_option("display.width", 220)
    d = pd.read_csv(a.csv)
    d["base"] = d.scenario.str.replace("_fail", "")
    d["fail"] = d.scenario.str.endswith("_fail")
    r = d[(d.protocol == a.protocol) & d.base.str.startswith("compound") & (d.agv >= a.min_agv) & (d.unfinished == 0)]
    g = r.groupby(["base", "fail"])[PH].mean().round(0)
    print("mean flow by phase kind (failures off / on):\n", g.T)
    print("\nfailures-on / failures-off:\n", (g.xs(True, level="fail") / g.xs(False, level="fail")).round(2).T)
    for base in ("compound_scenario", "compound_scenario_v2"):
        rows = {}
        for f in (False, True):
            x = r[(r.base == base) & (r.fail == f)].groupby("rule")[PH].mean()
            rows["fail off" if not f else "fail on"] = ((x.loc["LRT_MMUR"] / x.loc["SPT_SRWT"] - 1) * 100).round(1)
        print(f"\n{base}: LRT_MMUR vs SPT_SRWT (%)\n", pd.DataFrame(rows))
    f = r[r.fail]
    cv = f.groupby(["base", "layout", "agv", "rule"])[PH].agg(lambda s: s.std() / s.mean()).groupby("base").mean().round(2)
    print("\nmean coefficient of variation across seeds (failures on):\n", cv.T)


if __name__ == "__main__":
    main()
