"""Analyse the rq3-flex (FLEX_p<p>_m<m>) machine-flexibility sweep (docs/experiments/rq3-flex_findings_1002.md).

Grid: the 12 head rule pairs x rnd_flex_s0-8 x flexibility p in {0, 0.15, 0.3, 0.5} x secondary-time multiplier
m in {1.0, 1.25} (p = 0 once), layout D, 7 AGVs, releasePrevious, lane parking. 756 batch-runner cells, one run each.

Every cell runs its instance until the last job exits (the script checks jobs_censored == 0), so a cell's mean flow
time is the mean time in system of every job: the quantity the flow_time reward integrates, not the censored total
flow of a 5,400 s window (docs/experiments/HEADROOM_METRIC_CORRECTION_1002.md). All rankings and gaps use it.
Instances differ in job count (129-248), so gaps are per-seed ratios averaged over seeds.

    .venv/bin/python results/scripts/analyze_flex.py [--results linux_server/Results] [--out results/rq3-flex]

Writes <out>/cells_all.csv (one row per cell: results.csv columns plus flow components and routing statistics)
and <out>/gaps.csv (mean gap to the per-seed best per rule and group), and prints the tables the findings use.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[2]
GROUPS = [(0.0, 1.0), (0.15, 1.0), (0.3, 1.0), (0.5, 1.0), (0.15, 1.25), (0.3, 1.25), (0.5, 1.25)]
JOB_RULES = ["SPT", "SRT", "PTWINQ", "FIFO"]
MACHINE_RULES = ["ECT", "TECT", "SRWT"]
COMPONENTS = ["time_needs_routing", "time_waiting_pickup", "time_in_transit", "time_queued", "time_processing"]


def group_name(p, m):
    return f"p{p:g} m{m:g}"


def cell_stats(cell):
    """Flow components, processing paid above the fastest eligible machine, and secondary-machine share."""
    jc = pd.read_csv(cell / "job_completions.csv")
    ops = pd.read_csv(cell / "job_operations.csv")
    dl = pd.read_csv(cell / "decision_log.csv")
    mu = pd.read_csv(cell / "machine_utilization.csv")
    out = {f"mean_{c}": jc[c].mean() for c in COMPONENTS}
    out["mean_eligible_per_op"] = ops["eligible_machine_count"].mean()
    out["proc_over_min"] = ops["realized_proc_time"].sum() / ops["min_proc_time"].sum() - 1
    out["work_content"] = ops["realized_proc_time"].sum()
    out["mean_utilization"] = mu["utilization_rate"].mean()
    out["utilization_sd"] = mu["utilization_rate"].std()

    # Which machine ran each op: the last Dispatch of that job inside the op's queue-to-finish window. Routing rows
    # cannot be used one-to-one (failures re-route queued jobs, so there are more Routing rows than ops).
    types = mu.set_index("machine_id")["machine_type"]
    disp = dl[dl["decision_type"] == "Dispatch"][["sim_time", "subject_id", "chosen_id"]]
    disp = disp.rename(columns={"subject_id": "machine_id", "chosen_id": "job_id"})
    merged = ops[["job_id", "op_index", "machine_type_required", "queue_entry_time", "proc_end_time"]].merge(
        disp, on="job_id")
    merged = merged[(merged["sim_time"] >= merged["queue_entry_time"] - 0.11)
                    & (merged["sim_time"] <= merged["proc_end_time"] + 0.11)]
    last = merged.sort_values("sim_time").groupby(["job_id", "op_index"]).tail(1)
    secondary = last["machine_id"].map(types) != last["machine_type_required"]
    out["ops_matched"] = len(last) / len(ops)
    out["secondary_share"] = secondary.mean()

    # Would ECT and SRWT pick the same machine in the routing states this run visited? Routing rows log each
    # candidate's job time (stat_a) and queued work (stat_b); ECT = argmin(a + b), SRWT = argmin(b), first index on
    # ties as in DispatchingEngine. TECT needs travel times, which are not logged.
    routing = dl[(dl["decision_type"] == "Routing") & (dl["candidate_count"] > 1)]
    agree = ect_chosen = 0
    for a, b, ids, chosen in zip(routing["candidate_stat_a"], routing["candidate_stat_b"],
                                 routing["candidate_ids"], routing["chosen_id"]):
        a = np.array(str(a).split("|"), float)
        b = np.array(str(b).split("|"), float)
        ect = int(np.argmin(a + b))
        agree += ect == int(np.argmin(b))
        ect_chosen += int(str(ids).split("|")[ect]) == chosen
    out["ect_srwt_agree"] = agree / len(routing)
    out["ect_is_chosen"] = ect_chosen / len(routing)
    return out


def load(results_dir):
    rows = []
    for p, m in GROUPS:
        exp = results_dir / f"FLEX_p{p:g}_m{'1.0' if m == 1.0 else f'{m:g}'}"
        files = sorted(exp.glob("rnd_flex_s*/D/*/results.csv"))
        if not files:
            raise SystemExit(f"no results under {exp}")
        for f in files:
            r = pd.read_csv(f).iloc[0].to_dict()
            r.update(cell_stats(f.parent))
            rows.append(r)
    df = pd.DataFrame(rows)
    df["p"] = df["machine_flexibility"].round(2)
    df["m"] = df["secondary_time_multiplier"].round(2)
    df["group"] = [group_name(p, m) for p, m in zip(df["p"], df["m"])]
    df["job_rule"] = df["rule"].str.split("_").str[0]
    df["machine_rule"] = df["rule"].str.split("_").str[1]
    df["best"] = df.groupby(["group", "seed"])["mean_flow_time"].transform("min")
    df["gap"] = df["mean_flow_time"] / df["best"] - 1
    return df


def check(df):
    print("== Integrity")
    print(df.groupby("group", sort=False).size().to_string())
    problems = {
        "jobs censored": int(df["jobs_censored"].sum()),
        "deadlocks": int(df["deadlock_detected"].sum()),
        "AGV collisions": int(df["agv_collision_events"].sum()),
        "parking overlaps": int(df["agv_parking_overlap_events"].sum()),
        "duplicate cells": int(df.duplicated(["group", "instance", "rule"]).sum()),
    }
    print(", ".join(f"{k} {v}" for k, v in problems.items()))
    for col in ("agvCount", "layout_id", "reservation_protocol", "parking_method", "routing_trigger", "tiles"):
        print(f"{col}: {sorted(df[col].astype(str).unique())}")
    print(f"ops matched to a machine: min {df['ops_matched'].min():.3f}")
    ect = df[df["machine_rule"] == "ECT"]
    print(f"ECT recomputed from the logged routing stats = the logged choice: min {ect['ect_is_chosen'].min():.3f} "
          f"over the {len(ect)} ECT cells")
    g = df.groupby("group", sort=False)
    print(pd.DataFrame({
        "capabilities/machine": g["mean_capabilities_per_machine"].mean(),
        "eligible/op": g["mean_eligible_per_op"].mean(),
        "machine failures": g["episode_failures"].mean(),
        "makespan": g["makespan"].mean(),
    }).round(2).to_string())


def paired(df, a, b, by):
    """Per seed: mean over `by` values of flow(a) / flow(b) - 1. a, b map a `by` value to a rule name."""
    piv = df.pivot_table(index="seed", columns="rule", values="mean_flow_time")
    ratios = [piv[a(v)] / piv[b(v)] - 1 for v in by]
    return pd.concat(ratios, axis=1).mean(axis=1)


def summarize(x):
    p = wilcoxon(x).pvalue if (x != 0).any() else 1.0
    return f"{100 * x.mean():+6.1f}% [{100 * x.min():+.0f}, {100 * x.max():+.0f}] {int((x > 0).sum())}/{len(x)} p={p:.3f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=REPO / "linux_server" / "Results")
    ap.add_argument("--out", type=Path, default=REPO / "results" / "rq3-flex")
    args = ap.parse_args()
    pd.set_option("display.width", 200)

    df = load(args.results)
    check(df)
    order = [group_name(p, m) for p, m in GROUPS]
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "cells_all.csv", index=False)

    print("\n== Mean flow time (s, time in system of every job), mean over 9 seeds")
    flow = df.pivot_table(index="rule", columns="group", values="mean_flow_time")[order]
    print(flow.round(0).sort_values(order[0]).to_string())

    print("\n== Gap to the per-seed best of the 12 pairs (%, mean over seeds); [seeds won]")
    gaps = df.pivot_table(index="rule", columns="group", values="gap")[order] * 100
    gaps.to_csv(args.out / "gaps.csv", float_format="%.2f")
    wins = df[df["gap"] < 1e-9].groupby(["rule", "group"]).size().unstack(fill_value=0)
    wins = wins.reindex(index=gaps.index, columns=order, fill_value=0)
    table = gaps.round(1).astype(str) + wins.map(lambda w: f" [{w}]" if w else "")
    print(table.loc[gaps.mean(axis=1).sort_values().index].to_string())
    best = gaps.idxmin()
    print("\nper group: best pair on average (its gap = static headroom), worst pair's gap, per-seed best vs p0")
    p0_best = df[df["group"] == order[0]].groupby("seed")["best"].first()
    for grp in order:
        seeds = df[df["group"] == grp].groupby("seed")["best"].first()
        print(f"  {grp:12s} best {best[grp]:12s} {gaps[grp].min():4.1f}%   worst {gaps[grp].idxmax():12s} "
              f"{gaps[grp].max():5.1f}%   per-seed best vs p0 {100 * (seeds / p0_best - 1).mean():+6.1f}%")

    print("\n== Change vs p0 on the same (rule, seed), mean over seeds (%)")
    base = df[df["group"] == order[0]].set_index(["rule", "seed"])["mean_flow_time"]
    df["vs_p0"] = df["mean_flow_time"] / df.set_index(["rule", "seed"]).index.map(base) - 1
    vs = df.pivot_table(index="rule", columns="group", values="vs_p0")[order[1:]] * 100
    vs = vs.loc[gaps.mean(axis=1).sort_values().index]
    vs.loc["all 12 pairs"] = vs.mean()
    print(vs.round(1).to_string())

    print("\n== Head contrasts per group: mean over seeds of flow(A)/flow(B)-1, pooled over the other head"
          "\n   (mean [min, max] over the 9 seeds, seeds with A worse, two-sided Wilcoxon p; n = 9, smallest p 0.004)")
    contrasts = [
        ("SRT vs SPT (machine ECT/TECT)", lambda v: f"SRT_{v}", lambda v: f"SPT_{v}", ["ECT", "TECT"]),
        ("PTWINQ vs SPT (ECT/TECT)", lambda v: f"PTWINQ_{v}", lambda v: f"SPT_{v}", ["ECT", "TECT"]),
        ("FIFO vs SPT (ECT/TECT)", lambda v: f"FIFO_{v}", lambda v: f"SPT_{v}", ["ECT", "TECT"]),
        ("TECT vs ECT (all job rules)", lambda v: f"{v}_TECT", lambda v: f"{v}_ECT", JOB_RULES),
        ("SRWT vs ECT (all job rules)", lambda v: f"{v}_SRWT", lambda v: f"{v}_ECT", JOB_RULES),
    ]
    for name, a, b, by in contrasts:
        print(f"  {name}")
        for grp in order:
            print(f"    {grp:12s} {summarize(paired(df[df['group'] == grp], a, b, by))}")

    print("\n== Lever size per group: mean spread (worst/best - 1) across one head with the other fixed (%)")
    for grp in order:
        g = df[df["group"] == grp]
        jl = g.groupby(["seed", "machine_rule"])["mean_flow_time"].agg(lambda s: s.max() / s.min() - 1)
        ml = g.groupby(["seed", "job_rule"])["mean_flow_time"].agg(lambda s: s.max() / s.min() - 1)
        ml_ex = g[g["machine_rule"] != "SRWT"].groupby(["seed", "job_rule"])["mean_flow_time"].agg(
            lambda s: s.max() / s.min() - 1)
        jl_ex = g[(g["job_rule"] != "FIFO") & (g["machine_rule"] != "SRWT")].groupby(
            ["seed", "machine_rule"])["mean_flow_time"].agg(lambda s: s.max() / s.min() - 1)
        print(f"  {grp:12s} job head {100 * jl.mean():5.1f} (without FIFO, ECT/TECT only {100 * jl_ex.mean():4.1f})"
              f"   machine head {100 * ml.mean():5.1f} (ECT vs TECT only {100 * ml_ex.mean():4.1f})")

    print("\n== Routing: processing paid above the fastest eligible machine (%) / share of ops on a secondary machine (%)")
    proc = df.pivot_table(index="machine_rule", columns="group", values="proc_over_min")[order] * 100
    sec = df.pivot_table(index="machine_rule", columns="group", values="secondary_share")[order] * 100
    print((proc.round(1).astype(str) + " / " + sec.round(0).astype(int).astype(str)).to_string())
    agree = df[df["machine_rule"] == "ECT"].groupby("group")["ect_srwt_agree"].mean()[order] * 100
    print("ECT and SRWT pick the same machine (% of routing decisions with 2+ candidates, in the ECT runs' states): "
          + "  ".join(f"{grp} {v:.0f}" for grp, v in agree.items()))

    print("\n== Where the time goes: mean per job (s) for the best pairs and SRWT, by group")
    for rule in ("SPT_ECT", "SRT_ECT", "SPT_SRWT"):
        sub = df[df["rule"] == rule].groupby("group")[[f"mean_{c}" for c in COMPONENTS] + ["mean_flow_time"]]
        t = sub.mean().loc[order].round(0)
        t.columns = ["routing", "pickup", "transit", "queued", "processing", "flow"]
        print(f"  {rule}\n{t.to_string()}")

    print("\n== Tail: p95 flow time (s), mean over seeds")
    print(df.pivot_table(index="rule", columns="group", values="p95_flow_time")[order].round(0)
          .loc[["SPT_ECT", "SPT_TECT", "SRT_ECT", "SRT_TECT", "FIFO_ECT", "SPT_SRWT"]].to_string())

    print("\n== Machine utilization (mean over machines / SD across machines), selected rules")
    for rule in ("SPT_ECT", "SRT_ECT", "SPT_SRWT"):
        g = df[df["rule"] == rule].groupby("group")
        print(f"  {rule:9s} " + "  ".join(f"{grp}: {g['mean_utilization'].mean()[grp]:.2f}/{g['utilization_sd'].mean()[grp]:.2f}"
                                          for grp in order))

    print(f"\nwrote {args.out / 'cells_all.csv'} and {args.out / 'gaps.csv'}")


if __name__ == "__main__":
    main()
