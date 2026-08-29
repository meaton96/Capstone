#!/usr/bin/env python3
"""
DFJSP figure generation: machine utilization vs AGV workload.

Built around four CSVs (headers must match):
  agv_performance.csv, job_operations.csv, machine_utilization.csv, results.csv

Central question this script is built to answer:
  Why does adding AGVs not move makespan?
The figures test whether transport is a *slack* resource (AGVs over-provisioned,
machines binding) vs. a *binding* one.

Usage:
  python dfjsp_figures.py --data-dir . --out-dir figures
  python dfjsp_figures.py --data-dir . --instance MK04 --rule SPT   # optional filters
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- run identity -----------------------------------------------------------
# After de-duplicating the double-logged rows, (instance, rule, seed) uniquely
# identifies a run in this dataset, so we key on that. timestamp is NOT used as
# a key here because entity logs and the results row for the same run can carry
# slightly different timestamps (one orphan run in generated_poisson_02).
RUN_KEY = ["instance", "rule", "seed"]

AGV_TIME_COLS = [
    "time_idle", "time_waiting_route", "time_traveling",
    "time_loading", "time_unloading",
]
AGV_TIME_LABELS = {
    "time_idle": "idle",
    "time_waiting_route": "waiting (route/contention)",
    "time_traveling": "traveling",
    "time_loading": "loading",
    "time_unloading": "unloading",
}

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 150, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
})


# ---- io ---------------------------------------------------------------------
def load(data_dir, name):
    path = os.path.join(data_dir, name)
    if not os.path.exists(path):
        sys.exit(f"[error] missing file: {path}")
    df = pd.read_csv(path)
    n0 = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    dropped = n0 - len(df)
    note = f"  (dropped {dropped} exact-duplicate rows)" if dropped else ""
    print(f"  loaded {name:28s} rows={len(df):6d} cols={len(df.columns)}{note}")
    return df


def need(df, cols, where):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        sys.exit(f"[error] {where}: missing columns {missing}")


def attach_agv_count(df, results):
    """Bring agvCount onto a per-run frame by joining to results.csv on RUN_KEY.
    Warns if RUN_KEY is not unique in results (means the join is ambiguous)."""
    key = [c for c in RUN_KEY if c in results.columns and c in df.columns]
    if "agvCount" not in results.columns:
        return df
    dup = results.duplicated(key).sum()
    if dup:
        print(f"  [warn] results.csv has {dup} rows sharing a run key {key}; "
              f"agvCount join may be ambiguous. Consider unique timestamps or "
              f"an explicit agvCount column in every CSV.")
    rc = results[key + ["agvCount"]].drop_duplicates(key)
    out = df.merge(rc, on=key, how="left")
    if "agvCount" in out and out["agvCount"].isna().any():
        n = int(out["agvCount"].isna().sum())
        print(f"  [warn] {n} rows did not match a run in results.csv (agvCount NaN).")
    return out


def apply_filters(frames, instance, rule):
    out = {}
    for k, df in frames.items():
        if instance and "instance" in df.columns:
            df = df[df["instance"] == instance]
        if rule and "rule" in df.columns:
            df = df[df["rule"] == rule]
        out[k] = df
    return out


# ---- figures ----------------------------------------------------------------
def fig_proc_vs_travel(ops, out):
    """The explicit ask: per-operation processing time vs travel time."""
    need(ops, ["mean_proc_time", "travel_time"], "job_operations")
    p = ops["mean_proc_time"].astype(float)
    t = ops["travel_time"].astype(float)
    ratio = t / (t + p).replace(0, np.nan)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))

    # A: overlaid distributions
    bins = np.histogram_bin_edges(np.concatenate([p.dropna(), t.dropna()]), bins=40)
    ax[0].hist(p, bins=bins, alpha=0.6, label="processing", color="#3b6ea5")
    ax[0].hist(t, bins=bins, alpha=0.6, label="travel", color="#c0563f")
    ax[0].axvline(p.median(), color="#3b6ea5", ls="--", lw=1)
    ax[0].axvline(t.median(), color="#c0563f", ls="--", lw=1)
    ax[0].set(xlabel="time (sim units)", ylabel="operation count",
              title="Processing vs travel time")
    ax[0].legend()

    # B: scatter with y=x reference
    dyn = ops["is_dynamic"].astype(bool) if "is_dynamic" in ops else pd.Series(False, index=ops.index)
    ax[1].scatter(p[~dyn], t[~dyn], s=10, alpha=0.4, color="#3b6ea5", label="static job")
    ax[1].scatter(p[dyn], t[dyn], s=10, alpha=0.4, color="#c0563f", label="dynamic job")
    lim = max(p.max(), t.max())
    ax[1].plot([0, lim], [0, lim], color="k", lw=0.8, ls=":", label="travel = proc")
    ax[1].set(xlabel="mean processing time", ylabel="travel time",
              title="Per-operation: proc vs travel")
    ax[1].legend()

    # C: handling ratio
    ax[2].hist(ratio.dropna(), bins=40, color="#6a8d3f")
    med = ratio.median()
    ax[2].axvline(med, color="k", ls="--", lw=1, label=f"median {med:.2f}")
    ax[2].set(xlabel="travel / (travel + proc)", ylabel="operation count",
              title="Transport share of operation budget")
    ax[2].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(out, "01_proc_vs_travel.png"))
    plt.close(fig)
    return dict(median_proc=float(p.median()), median_travel=float(t.median()),
                median_handling_ratio=float(med))


def fig_agv_budget(agv, out):
    """Where AGV time goes, grouped by fleet size. The over-provisioning tell."""
    need(agv, AGV_TIME_COLS + ["makespan"], "agv_performance")
    df = agv.copy()
    df["agvCount"] = df["agvCount"].fillna(
        df.groupby(RUN_KEY)["agv_id"].transform("nunique"))

    # composition: fraction of each AGV's tracked time
    tracked = df[AGV_TIME_COLS].sum(axis=1).replace(0, np.nan)
    comp = df[AGV_TIME_COLS].div(tracked, axis=0)
    comp["agvCount"] = df["agvCount"]
    by_n = comp.groupby("agvCount")[AGV_TIME_COLS].mean()

    # idle as fraction of makespan (the absolute over-provisioning signal)
    idle_frac = (df["time_idle"] / df["makespan"]).groupby(df["agvCount"]).mean()

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    bottom = np.zeros(len(by_n))
    palette = ["#9aa0a6", "#c0563f", "#3b6ea5", "#6a8d3f", "#b08a3e"]
    for col, color in zip(AGV_TIME_COLS, palette):
        ax[0].bar(by_n.index.astype(str), by_n[col], bottom=bottom,
                  label=AGV_TIME_LABELS[col], color=color)
        bottom += by_n[col].values
    ax[0].set(xlabel="instance (AGV count, not a sweep)", ylabel="share of AGV tracked time",
              title="AGV time composition by fleet size")
    ax[0].legend(fontsize=8, loc="lower center", ncol=3)

    ax[1].plot(idle_frac.index, idle_frac.values, "o-", color="#9aa0a6")
    ax[1].set(xlabel="instance (AGV count, not a sweep)", ylabel="mean AGV idle / makespan",
              title="AGV idle fraction vs fleet size", ylim=(0, 1))
    fig.tight_layout()
    fig.savefig(os.path.join(out, "02_agv_time_budget.png"))
    plt.close(fig)
    return dict(idle_frac_by_n=idle_frac.round(3).to_dict())


def fig_system_load(results, agv, mach, out):
    """Replaces the confounded makespan-vs-AGV plot. agvCount here is just an
    instance label (no sweep), so instead we ask the real question: is the
    system capacity-bound or arrival-starved? Shows resource utilization (low =
    not the bottleneck) and makespan vs the arrival horizon (high = makespan is
    set by when jobs arrive, not by capacity)."""
    # per-instance order by mean fleet size (= instance scale)
    order = (results.groupby("instance")["agvCount"].mean().sort_values().index.tolist())

    mu = mach.groupby("instance")["utilization_rate"].mean().reindex(order)
    a = agv.copy()
    a["busy_frac"] = 1 - (a["time_idle"] / a["makespan"])
    au = a.groupby("instance")["busy_frac"].mean().reindex(order)

    r = results.copy()
    r["arr_frac"] = r["last_arrival_sim_time"] / r["makespan"]
    mk = r.groupby("instance")["makespan"].mean().reindex(order)
    arr = r.groupby("instance")["last_arrival_sim_time"].mean().reindex(order)
    arr_frac = r.groupby("instance")["arr_frac"].mean().reindex(order)

    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
    x = np.arange(len(order))
    w = 0.38
    ax[0].bar(x - w/2, mu.values, w, label="mean machine utilization", color="#3b6ea5")
    ax[0].bar(x + w/2, au.values, w, label="mean AGV busy fraction", color="#b08a3e")
    ax[0].axhline(0.9, color="k", ls="--", lw=0.8, label="90% (saturation)")
    ax[0].set(xticks=x, ylabel="fraction of time busy", ylim=(0, 1),
              title="Are machines or AGVs the bottleneck?")
    ax[0].set_xticklabels(order, rotation=20, ha="right")
    ax[0].legend(fontsize=8)

    # makespan vs arrival horizon: the tail beyond last arrival is small ->
    # makespan is arrival-driven
    ax[1].bar(x, mk.values, 0.6, label="makespan", color="#d9d2c5")
    ax[1].bar(x, arr.values, 0.6, label="last job arrival time", color="#c0563f")
    for xi, f in zip(x, arr_frac.values):
        ax[1].text(xi, arr.values[list(order).index(order[xi])], f"{f:.0%}",
                   ha="center", va="bottom", fontsize=8)
    ax[1].set(xticks=x, ylabel="sim time",
              title="Makespan vs arrival horizon (label = arrival/makespan)")
    ax[1].set_xticklabels(order, rotation=20, ha="right")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "03_system_load.png"))
    plt.close(fig)
    return dict(machine_util_by_instance=mu.round(3).to_dict(),
                agv_busy_by_instance=au.round(3).to_dict(),
                arrival_frac_by_instance=arr_frac.round(2).to_dict())


def fig_fleet_scaling(agv, out):
    """Per-AGV busy fraction (~1/N if work fixed) and total fleet transport work."""
    need(agv, AGV_TIME_COLS + ["makespan"], "agv_performance")
    df = agv.copy()
    df["agvCount"] = df["agvCount"].fillna(
        df.groupby(RUN_KEY)["agv_id"].transform("nunique"))
    busy_cols = ["time_traveling", "time_loading", "time_unloading", "time_waiting_route"]
    df["busy"] = df[busy_cols].sum(axis=1)

    per_agv_busy = (df["busy"] / df["makespan"]).groupby(df["agvCount"]).mean()
    # total transport work per run, then averaged across runs of same fleet size
    fleet_work = (df.groupby(RUN_KEY + ["agvCount"])["busy"].sum()
                    .reset_index().groupby("agvCount")["busy"].mean())

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    ax[0].plot(per_agv_busy.index, per_agv_busy.values, "o-", color="#3b6ea5", label="observed")
    n0 = per_agv_busy.index.min()
    ideal = per_agv_busy.loc[n0] * n0 / per_agv_busy.index.values
    ax[0].plot(per_agv_busy.index, ideal, "k:", lw=1, label="1/N (fixed work)")
    ax[0].set(xlabel="instance (AGV count, not a sweep)", ylabel="per-AGV busy / makespan",
              title="Per-AGV utilization vs fleet size")
    ax[0].legend(fontsize=8)

    ax[1].plot(fleet_work.index, fleet_work.values, "s-", color="#6a8d3f")
    ax[1].set(xlabel="instance (AGV count, not a sweep)", ylabel="total fleet transport time",
              title="Total transport work vs fleet size")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "04_fleet_scaling.png"))
    plt.close(fig)
    return dict(per_agv_busy_by_n=per_agv_busy.round(3).to_dict())


def fig_machine_util(mach, out):
    """Bottleneck hunt, PER INSTANCE. Aggregating machine_id across instances is
    meaningless (different machines, different counts), so one panel per instance."""
    need(mach, ["machine_id", "utilization_rate", "instance"], "machine_utilization")
    insts = (mach.groupby("instance")["agvCount"].mean().sort_values().index.tolist()
             if "agvCount" in mach else sorted(mach["instance"].unique()))
    n = len(insts)
    ncol = min(2, n); nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(7 * ncol, 3.4 * nrow), squeeze=False)
    tops = {}
    for ax, inst in zip(axes.ravel(), insts):
        sub = (mach[mach["instance"] == inst]
               .groupby("machine_id")["utilization_rate"].mean()
               .sort_values(ascending=False))
        colors = ["#c0563f" if u >= 0.9 else "#3b6ea5" for u in sub.values]
        ax.bar(sub.index.astype(str), sub.values, color=colors)
        ax.axhline(0.9, color="k", ls="--", lw=0.8)
        ax.set(title=f"{inst}  (busiest {sub.idxmax()} @ {sub.max():.0%})",
               ylabel="mean utilization", ylim=(0, 1))
        ax.tick_params(axis="x", rotation=90, labelsize=7)
        tops[inst] = round(float(sub.max()), 3)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    fig.suptitle("Machine utilization per instance (red = >90%)", y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "05_machine_utilization.png"), bbox_inches="tight")
    plt.close(fig)
    return dict(busiest_machine_util_by_instance=tops)


def fig_contention(agv, out):
    """Is there ANY transport contention to relieve? waiting + congestion + reroute."""
    df = agv.copy()
    df["agvCount"] = df["agvCount"].fillna(
        df.groupby(RUN_KEY)["agv_id"].transform("nunique"))
    metrics = {}
    if "time_waiting_route" in df and "makespan" in df:
        metrics["wait_frac"] = (df["time_waiting_route"] / df["makespan"]).groupby(df["agvCount"]).mean()
    if "congestion_fraction" in df:
        metrics["congestion"] = df.groupby("agvCount")["congestion_fraction"].mean()
    if "reroute_count" in df:
        metrics["reroute/trip"] = (df["reroute_count"] /
                                   df.get("total_trips", pd.Series(1, index=df.index)).replace(0, np.nan)
                                   ).groupby(df["agvCount"]).mean()
    if not metrics:
        return {}
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for name, s in metrics.items():
        ax.plot(s.index, s.values, "o-", label=name)
    ax.set(xlabel="instance (AGV count, not a sweep)", ylabel="contention metric (low = no contention)",
           title="Transport contention vs fleet size")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "06_contention.png"))
    plt.close(fig)
    return {k: round(float(v.mean()), 4) for k, v in metrics.items()}


# ---- main -------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=".")
    ap.add_argument("--out-dir", default="figures")
    ap.add_argument("--instance", default=None, help="optional single-instance filter, e.g. MK04")
    ap.add_argument("--rule", default=None, help="optional single-rule filter")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("[load]")
    agv = load(args.data_dir, "agv_performance.csv")
    ops = load(args.data_dir, "job_operations.csv")
    mach = load(args.data_dir, "machine_utilization.csv")
    results = load(args.data_dir, "results.csv")

    # attach agvCount where it's needed
    agv = attach_agv_count(agv, results)
    mach = attach_agv_count(mach, results)
    ops = attach_agv_count(ops, results)

    frames = apply_filters(
        {"agv": agv, "ops": ops, "mach": mach, "results": results},
        args.instance, args.rule)
    agv, ops, mach, results = frames["agv"], frames["ops"], frames["mach"], frames["results"]

    print("[figures]")
    summary = {}
    summary["proc_vs_travel"] = fig_proc_vs_travel(ops, args.out_dir)
    summary["agv_budget"] = fig_agv_budget(agv, args.out_dir)
    summary["system_load"] = fig_system_load(results, agv, mach, args.out_dir)
    summary["fleet_scaling"] = fig_fleet_scaling(agv, args.out_dir)
    summary["machine_util"] = fig_machine_util(mach, args.out_dir)
    summary["contention"] = fig_contention(agv, args.out_dir)

    print("\n[diagnostic summary]")
    mt = summary["proc_vs_travel"]
    sl = summary["system_load"]
    print(f"  transport share of operation budget (median): {mt['median_handling_ratio']:.1%}")
    print(f"  median proc={mt['median_proc']:.1f}  median travel={mt['median_travel']:.1f}")
    print(f"  busiest-machine util by instance: {summary['machine_util']['busiest_machine_util_by_instance']}")
    print(f"  mean machine util by instance:    {sl['machine_util_by_instance']}")
    print(f"  mean AGV busy frac by instance:   {sl['agv_busy_by_instance']}")
    print(f"  last-arrival/makespan by instance:{sl['arrival_frac_by_instance']}")
    print(f"  AGV idle/makespan by instance:    {summary['agv_budget']['idle_frac_by_n']}")
    print(f"  contention (mean): {summary['contention']}")
    mx_mach = max(sl['machine_util_by_instance'].values())
    mx_agv = max(sl['agv_busy_by_instance'].values())
    if mx_mach < 0.5 and mx_agv < 0.5:
        print("\n  >> Neither machines nor AGVs exceed 50% utilization in any instance.")
        print("  >> Makespan is arrival-starved, not capacity-bound: it is set by the")
        print("  >> Poisson arrival schedule. Adding AGVs/machines cannot reduce it.")
    print(f"\n  figures written to: {os.path.abspath(args.out_dir)}/")


if __name__ == "__main__":
    main()