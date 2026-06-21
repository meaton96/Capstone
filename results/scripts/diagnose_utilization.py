"""
diagnose_utilization.py — Answer "why won't the machines saturate?" from
machine_utilization.csv.

The core hypothesis: aggregate utilization is low not because the floor is
under-loaded, but because processing times are imbalanced across machine TYPES.
A slow type (e.g. Assemble) pins near 100% and gates throughput while fast types
(e.g. Inspect) sit idle, dragging the mean down. More AGVs or more jobs can't fix
a type-imbalance bottleneck.

IMPORTANT — whole-episode caveat:
    utilization_rate here is computed over the FULL episode, including the
    post-arrival drain tail (low-occupancy by construction in a finite run).
    So ABSOLUTE utilization is biased LOW for every machine and must NOT be read
    as "the floor is underutilized." What survives the bias is the RELATIVE
    picture across types: the drain tail depresses fast types more than slow
    types, so an imbalance seen here is a *lower bound* on the true steady-state
    imbalance. This script leans on relative/ranked findings and flags absolute
    ones as unreliable.

Usage:
    python diagnose_utilization.py <machine_utilization.csv> [--out <dir>]
                                   [--by-instance] [--rule <name>]

Produces (in --out, default `plots_utilization`):
    util_by_type.png             per-type utilization (box over machines/seeds/rules)
    util_imbalance_heatmap.png   instance x type mean utilization
    bottleneck_summary.csv       per (instance, type) util + imbalance ratio
    console report               the saturation answer, stated directionally
"""

import argparse
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)

plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# Expected schema (from the logger):
# timestamp,instance,rule,seed,makespan,machine_id,machine_type,ops_completed,
# time_processing,time_operational,utilization_rate,idle_time,idle_rate,
# availability_rate,failure_count,total_repair_time
REQUIRED = ["instance", "machine_type", "utilization_rate"]


def load(csv_path: str, rule: str | None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise SystemExit(f"missing required columns: {missing}\nhave: {list(df.columns)}")

    for c in ("machine_type", "instance", "rule"):
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    if rule and "rule" in df.columns:
        df = df[df["rule"] == rule].copy()
        if df.empty:
            raise SystemExit(f"no rows for rule '{rule}'")

    # utilization_rate may be a fraction (0..1) or a percent (0..100); normalise to %.
    u = df["utilization_rate"].astype(float)
    if u.max() <= 1.5:  # looks like a fraction
        df["util_pct"] = u * 100.0
    else:
        df["util_pct"] = u

    # Recompute utilization from components when present — a cross-check on the
    # logged rate, and lets us see how much the drain tail inflates time_operational.
    if {"time_processing", "time_operational"}.issubset(df.columns):
        denom = df["time_operational"].replace(0, np.nan)
        df["util_recomputed_pct"] = (df["time_processing"] / denom * 100.0)
    return df


def report(df: pd.DataFrame) -> pd.DataFrame:
    # Per-type utilization across all machines/seeds/rules in scope.
    by_type = (df.groupby("machine_type")["util_pct"]
               .agg(mean="mean", median="median", std="std", n="size")
               .sort_values("mean", ascending=False))

    print("\n" + "=" * 64)
    print("BOTTLENECK DIAGNOSIS — per machine type (whole-episode util)")
    print("=" * 64)
    print("NOTE: absolute values are biased LOW by the finite-run drain tail.")
    print("      Read the RANKING and the RATIO, not the absolute levels.\n")
    for t, r in by_type.iterrows():
        bar = "#" * int(round(r["mean"] / 2))
        print(f"  {t:<10} {r['mean']:5.1f}% mean  {bar}")
    print()

    hi, lo = by_type["mean"].iloc[0], by_type["mean"].iloc[-1]
    ratio = hi / lo if lo > 0 else float("inf")
    busiest, idlest = by_type.index[0], by_type.index[-1]

    print(f"Busiest type : {busiest} ({hi:.1f}%)")
    print(f"Idlest type  : {idlest} ({lo:.1f}%)")
    print(f"Imbalance    : {ratio:.1f}x  (busiest / idlest mean utilization)")
    print()

    # The verdict, stated in terms the whole-episode bias can support.
    print("-" * 64)
    if ratio >= 2.0:
        print("VERDICT: type IMBALANCE is the saturation ceiling.")
        print(f"  '{busiest}' is pinned far above '{idlest}'. Aggregate utilization")
        print("  is gated by the slow type — adding AGVs or jobs cannot raise it,")
        print("  which matches the AGV-count and parking sweeps showing no effect.")
        print(f"  Because the drain tail depresses fast types MORE, the true")
        print(f"  steady-state imbalance is AT LEAST {ratio:.1f}x.")
        print("  Lever to test: flatten ProcTimeParams (all types ~equal mu) and")
        print("  re-run; aggregate utilization should rise if this verdict holds.")
    else:
        print("VERDICT: utilization is balanced across types.")
        print("  The low aggregate is NOT type imbalance — suspect genuine")
        print("  under-load (finite run / drain tail) or a transport ceiling.")
        print("  Next: window utilization to the run interior, or check")
        print("  agv_performance time_waiting_route / segment block_events.")
    print("-" * 64)

    # Cross-check logged vs recomputed, if available — surfaces drain-tail inflation.
    if "util_recomputed_pct" in df.columns:
        m_log = df["util_pct"].mean()
        m_rec = df["util_recomputed_pct"].mean()
        print(f"\nLogged util mean {m_log:.1f}% vs recomputed "
              f"(time_processing/time_operational) {m_rec:.1f}%.")
        if abs(m_log - m_rec) > 5:
            print("  Large gap → logged rate likely uses a different denominator")
            print("  (e.g. makespan, not time_operational). Worth confirming.")

    return by_type


def per_instance_type(df: pd.DataFrame) -> pd.DataFrame:
    grid = (df.groupby(["instance", "machine_type"])["util_pct"]
            .mean().unstack("machine_type"))
    grid["__imbalance__"] = grid.max(axis=1) / grid.replace(0, np.nan).min(axis=1)
    return grid


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_by_type(df: pd.DataFrame, out_dir: str) -> None:
    order = (df.groupby("machine_type")["util_pct"].mean()
             .sort_values(ascending=False).index.tolist())
    fig, ax = plt.subplots(figsize=(max(7, len(order) * 1.3), 5))
    sns.boxplot(data=df, x="machine_type", y="util_pct", order=order,
                color="#4c72b0", fliersize=2, ax=ax)
    sns.stripplot(data=df, x="machine_type", y="util_pct", order=order,
                  color="black", alpha=0.25, size=3, ax=ax)
    ax.set_xlabel("Machine type")
    ax.set_ylabel("Utilization %  (whole-episode; biased low)")
    ax.set_title("Per-type utilization — imbalance = bottleneck\n"
                 "spread = machines × seeds × rules")
    ax.tick_params(axis="x", rotation=20)
    _save(fig, out_dir, "util_by_type.png")


def plot_imbalance_heatmap(grid: pd.DataFrame, out_dir: str) -> None:
    g = grid.drop(columns="__imbalance__", errors="ignore")
    g = g.reindex(sorted(g.index, key=str))
    fig, ax = plt.subplots(figsize=(max(7, g.shape[1] * 1.1),
                                    max(3.5, g.shape[0] * 0.7)))
    sns.heatmap(g, annot=True, fmt=".0f", cmap="RdYlGn", vmin=0, vmax=100,
                linewidths=0.5, linecolor="white",
                cbar_kws={"label": "Utilization %"}, ax=ax)
    ax.set_title("Utilization by instance × type\n"
                 "red row-cells = pinned bottleneck, green = idle/slack")
    ax.set_xlabel("Machine type")
    ax.set_ylabel("Instance")
    _save(fig, out_dir, "util_imbalance_heatmap.png")


def _save(fig, out_dir, name):
    p = os.path.join(out_dir, name)
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p}")


def main():
    ap = argparse.ArgumentParser(description="Diagnose machine saturation/imbalance.")
    ap.add_argument("csv", help="machine_utilization.csv")
    ap.add_argument("--out", default="plots_utilization")
    ap.add_argument("--rule", default=None, help="restrict to one dispatching rule")
    ap.add_argument("--by-instance", action="store_true",
                    help="also print the per-instance × type table")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = load(args.csv, args.rule)

    print(f"\nRows: {len(df)}  |  instances: {df['instance'].nunique()}  |  "
          f"types: {sorted(df['machine_type'].unique())}")
    if "rule" in df.columns and not args.rule:
        print(f"Rules pooled: {sorted(df['rule'].unique())} "
              f"(use --rule to isolate one)")

    report(df)
    grid = per_instance_type(df)

    if args.by_instance:
        print("\nPer-instance utilization by type (mean %, last col = imbalance x):")
        with pd.option_context("display.width", 120,
                               "display.float_format", lambda v: f"{v:6.1f}"):
            print(grid)

    plot_by_type(df, args.out)
    plot_imbalance_heatmap(grid, args.out)

    out_csv = os.path.join(args.out, "bottleneck_summary.csv")
    grid.to_csv(out_csv)
    print(f"  Saved: {out_csv}")
    print("\nDone.")


if __name__ == "__main__":
    main()
