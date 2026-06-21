"""
analyze_agv.py — Decompose AGV time to answer "machines are ~90% idle and it's
not type-imbalance, so where does the time go?"

Two complementary views:

  (A) FLEET-TIME DECOMPOSITION — of each AGV's wall-clock, what fraction is
      traveling / waiting-for-zone / idle / loading / unloading. This says what
      the VEHICLES do. High traveling+waiting => transport-bound (the thesis-
      confirming outcome). High idle => AGVs are starved of work (a dispatch or
      fleet-size problem, not a transport one).

  (B) TRANSPORT-vs-PROCESSING share of makespan — per AGV, time_traveling as a
      fraction of makespan, and the active (non-idle) share. Connects the fleet
      view back to "travel time + production time ~ makespan".

Usage:
    python analyze_agv.py <agv_performance.csv> [--out <dir>] [--rule <name>]

Schema expected:
    timestamp,instance,rule,seed,makespan,agv_id,total_trips,mean_trip_duration,
    time_idle,time_waiting_route,time_traveling,time_loading,time_unloading,
    total_path_length,reroute_count,congestion_fraction
"""

import argparse
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# Time buckets that partition an AGV's wall-clock, in stacked-plot order.
BUCKETS = ["time_traveling", "time_waiting_route", "time_loading",
           "time_unloading", "time_idle"]
BUCKET_LABELS = {
    "time_traveling":     "Traveling",
    "time_waiting_route": "Waiting for zone",
    "time_loading":       "Loading",
    "time_unloading":     "Unloading",
    "time_idle":          "Idle (at parking)",
}
BUCKET_COLORS = {
    "time_traveling":     "#2a6f97",   # transport — the thesis bucket
    "time_waiting_route": "#e76f51",   # blocked by zone contention
    "time_loading":       "#83c5be",
    "time_unloading":     "#a8dadc",
    "time_idle":          "#cccccc",
}
REQUIRED = ["instance", "makespan"] + BUCKETS


def load(csv_path: str, rule: str | None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    miss = [c for c in REQUIRED if c not in df.columns]
    if miss:
        raise SystemExit(f"missing columns: {miss}\nhave: {list(df.columns)}")
    for c in ("instance", "rule", "machine_type"):
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    if rule and "rule" in df.columns:
        df = df[df["rule"] == rule].copy()
        if df.empty:
            raise SystemExit(f"no rows for rule '{rule}'")

    # Active = everything except idle. Accounted = sum of buckets; compare to
    # makespan to see how well the buckets partition wall-clock.
    df["time_active"] = df[[b for b in BUCKETS if b != "time_idle"]].sum(axis=1)
    df["time_accounted"] = df[BUCKETS].sum(axis=1)
    df["transport_share"] = df["time_traveling"] / df["makespan"].replace(0, np.nan)
    df["wait_share"] = df["time_waiting_route"] / df["makespan"].replace(0, np.nan)
    df["idle_share"] = df["time_idle"] / df["makespan"].replace(0, np.nan)
    df["active_share"] = df["time_active"] / df["makespan"].replace(0, np.nan)
    # Partition coverage: accounted / makespan. ~1.0 means buckets tile wall-clock.
    df["coverage"] = df["time_accounted"] / df["makespan"].replace(0, np.nan)
    return df


def report(df: pd.DataFrame) -> None:
    print("\n" + "=" * 66)
    print("AGV TIME DECOMPOSITION")
    print("=" * 66)

    # Fleet-mean fraction of accounted time in each bucket (per-AGV partition).
    frac = (df[BUCKETS].div(df["time_accounted"].replace(0, np.nan), axis=0))
    mean_frac = frac.mean().reindex(BUCKETS)
    print("\nShare of AGV wall-time (fleet mean, buckets partition each AGV):")
    for b in BUCKETS:
        pct = mean_frac[b] * 100
        bar = "#" * int(round(pct / 2))
        print(f"  {BUCKET_LABELS[b]:<18} {pct:5.1f}%  {bar}")

    travel = mean_frac["time_traveling"] * 100
    wait = mean_frac["time_waiting_route"] * 100
    idle = mean_frac["time_idle"] * 100
    transport_total = travel + wait

    print("\n" + "-" * 66)
    print(f"Transport-related (traveling + waiting-for-zone): {transport_total:.1f}%")
    print(f"Idle at parking:                                  {idle:.1f}%")
    print("-" * 66)

    # The verdict — which of the three explanations for 90%-idle machines holds.
    if idle >= 55:
        print("VERDICT: AGVs are mostly IDLE → the fleet is NOT the bottleneck.")
        print("  Machines wait but vehicles are free, so jobs aren't being moved")
        print("  fast enough for a non-transport reason: too few decisions/")
        print("  dispatch lag, or jobs genuinely sparse (finite run). Adding AGVs")
        print("  won't help. Look at decision cadence / job availability, not layout.")
    elif transport_total >= 45:
        print("VERDICT: AGV time is TRANSPORT-DOMINATED → transport-bound floor.")
        print("  The machine idle time is going into moving jobs and waiting on")
        print("  one-way zone contention. THIS is the spatial/travel-time effect")
        print("  the thesis argues a DES cannot capture. Layout, parking, fleet")
        print("  size, and zone contention are now the live levers.")
        if wait >= 15:
            print(f"  Note: waiting-for-zone alone is {wait:.0f}% — zone contention is")
            print("  a material share, so the one-way traffic model is biting.")
    else:
        print("VERDICT: mixed — neither idle nor transport clearly dominates.")
        print("  Travel is real but vehicles also sit idle. Likely load-dependent;")
        print("  re-check under the higher-load configs before concluding.")
    print("-" * 66)

    # Coverage sanity — do the buckets actually tile wall-clock?
    cov = df["coverage"].mean()
    print(f"\nBucket coverage of makespan (mean accounted/makespan): {cov:.2f}")
    if cov < 0.85:
        print("  < 0.85 → buckets DON'T tile wall-clock. Likely the AGV spends")
        print("  time in a state not logged (e.g. aligning/handshaking at a dock,")
        print("  or pre-wait staging counted elsewhere). The missing slice means")
        print("  'idle' here understates true non-productive time — interpret the")
        print("  transport share as a lower bound.")
    elif cov > 1.15:
        print("  > 1.15 → buckets exceed wall-clock: states overlap (double-counted")
        print("  seconds). Treat per-bucket fractions as approximate.")
    else:
        print("  ~1.0 → buckets cleanly partition AGV wall-clock; shares are solid.")

    # Per-instance transport vs idle, since load varies the picture.
    pi = (df.groupby("instance")
          .agg(travel_pct=("transport_share", lambda s: s.mean() * 100),
               wait_pct=("wait_share", lambda s: s.mean() * 100),
               idle_pct=("idle_share", lambda s: s.mean() * 100),
               n_agv=("agv_id", "nunique") if "agv_id" in df.columns
                       else ("makespan", "size"))
          .reset_index())
    print("\nPer-instance (share of makespan, fleet-mean):")
    print(f"  {'instance':<24} {'travel':>7} {'zone-wait':>9} {'idle':>7}")
    for _, r in pi.sort_values("instance").iterrows():
        print(f"  {r['instance']:<24} {r['travel_pct']:6.1f}% {r['wait_pct']:8.1f}% "
              f"{r['idle_pct']:6.1f}%")


# ── Plot 1 — stacked time composition per instance ────────────────────────────

def plot_stacked(df: pd.DataFrame, out_dir: str) -> None:
    # Fleet-mean SECONDS per bucket, per instance, normalised to % of accounted.
    g = df.groupby("instance")[BUCKETS].mean()
    g = g.reindex(sorted(g.index, key=str))
    frac = g.div(g.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(max(8, len(frac) * 1.4), 5.5))
    bottom = np.zeros(len(frac))
    x = np.arange(len(frac))
    for b in BUCKETS:
        ax.bar(x, frac[b].values, 0.62, bottom=bottom,
               label=BUCKET_LABELS[b], color=BUCKET_COLORS[b])
        bottom += frac[b].values
    ax.set_xticks(x)
    ax.set_xticklabels(frac.index, rotation=25, ha="right")
    ax.set_ylabel("% of AGV wall-time")
    ax.set_ylim(0, 100)
    ax.set_title("Where AGV time goes, per instance\n"
                 "blue = transport (thesis bucket), red = zone-contention wait")
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    _save(fig, out_dir, "agv_time_composition.png")


# ── Plot 2 — transport share of makespan vs problem size ──────────────────────

def plot_transport_share(df: pd.DataFrame, out_dir: str) -> None:
    pi = (df.groupby("instance")
          .agg(travel=("transport_share", "mean"),
               wait=("wait_share", "mean"))
          .reindex(sorted(df["instance"].unique(), key=str)))
    pi = pi * 100

    fig, ax = plt.subplots(figsize=(max(8, len(pi) * 1.4), 5))
    x = np.arange(len(pi))
    ax.bar(x - 0.2, pi["travel"], 0.4, label="Traveling", color=BUCKET_COLORS["time_traveling"])
    ax.bar(x + 0.2, pi["wait"], 0.4, label="Waiting for zone",
           color=BUCKET_COLORS["time_waiting_route"])
    ax.set_xticks(x)
    ax.set_xticklabels(pi.index, rotation=25, ha="right")
    ax.set_ylabel("% of makespan (per-AGV, fleet-mean)")
    ax.set_title("Transport share of makespan by instance\n"
                 "how much wall-clock each AGV spends moving / blocked")
    ax.legend(fontsize=8)
    _save(fig, out_dir, "agv_transport_share.png")


def _save(fig, out_dir, name):
    p = os.path.join(out_dir, name)
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p}")


def main():
    ap = argparse.ArgumentParser(description="Decompose AGV time / transport share.")
    ap.add_argument("csv", help="agv_performance.csv")
    ap.add_argument("--out", default="plots_agv")
    ap.add_argument("--rule", default=None, help="restrict to one dispatching rule")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = load(args.csv, args.rule)

    print(f"\nRows: {len(df)}  |  instances: {df['instance'].nunique()}")
    if "rule" in df.columns and not args.rule:
        print(f"Rules pooled: {sorted(df['rule'].unique())} (use --rule to isolate)")

    report(df)
    plot_stacked(df, args.out)
    plot_transport_share(df, args.out)

    out_csv = os.path.join(args.out, "agv_decomposition.csv")
    (df.groupby("instance")[BUCKETS + ["transport_share", "idle_share", "coverage"]]
       .mean().to_csv(out_csv))
    print(f"  Saved: {out_csv}")


    print("\nDone.")


if __name__ == "__main__":
    main()
