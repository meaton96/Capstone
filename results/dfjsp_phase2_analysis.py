"""
DFJSP Phase-2 Analysis Suite
=============================
Covers all four CSVs from the updated logging schema:
  --machine   machine_utilization.csv   (one row per machine per episode)
  --results   results.csv               (one row per episode)
  --agv        agv_performance.csv       (one row per AGV per episode)
  --segments   segment_congestion.csv    (one row per zone per episode)

All four are optional; figures that need a missing file are skipped with a message.
Regime is auto-detected from the stochastic_tag column in results.csv, or from
instance-name suffixes when only machine/agv/segment data is passed.

Output figures (deterministic = green, stochastic = red convention throughout):

  Machine (01–10)   — existing utilization / cascade diagnostics  [from old script]
  AGV      (11–17)  — fleet time-budget, congestion, trip analysis
  Segment  (18–22)  — zone-level hotspot identification and heatmaps
  Combined (23–25)  — cross-domain cascade chain and PDR stability

Usage:
  python dfjsp_phase2_analysis.py \\
      --machine  det/machine_utilization.csv  stoch/machine_utilization.csv \\
      --results  det/results.csv              stoch/results.csv             \\
      --agv      det/agv_performance.csv      stoch/agv_performance.csv     \\
      --segments det/segment_congestion.csv   stoch/segment_congestion.csv  \\
      --out figs/

Or pass a single already-merged file per domain:
  python dfjsp_phase2_analysis.py \\
      --machine  combined_machine.csv \\
      --results  combined_results.csv \\
      --agv      combined_agv.csv     \\
      --segments combined_segments.csv \\
      --out figs/
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

mpl.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})

REGIME_COLORS  = {"deterministic": "#2E8B57", "stochastic_low": "#DC143C"}
REGIME_PALETTE = [REGIME_COLORS["deterministic"], REGIME_COLORS["stochastic_low"]]

# ─────────────────────────────────────────────────────────────────────────────
#  Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _instance_order(col) -> list:
    vals = pd.Series(col).dropna().unique().tolist()
    def _key(x):
        m = re.search(r"(\d+)$", str(x))
        return (int(m.group(1)) if m else 0, str(x))
    return sorted(vals, key=_key)


def _strip_regime_suffix(df: pd.DataFrame) -> pd.DataFrame:
    """Strip stochastic-tag suffixes from instance names so regimes join cleanly."""
    if "regime" not in df.columns:
        return df
    det = set(df.loc[df["regime"] == "deterministic", "instance"].astype(str).unique())
    sto = set(df.loc[df["regime"] == "stochastic_low", "instance"].astype(str).unique())
    if not (det and sto and det != sto):
        return df
    suffix = None
    for inst in sorted(sto):
        m = re.match(r"^(.+?)(_[^_0-9][^_]*)$", inst)
        if m and m.group(1) in det:
            suffix = m.group(2); break
    if not suffix:
        return df
    df = df.copy()
    df["instance"] = df["instance"].astype(str).apply(
        lambda x: x[:-len(suffix)] if x.endswith(suffix) else x)
    return df


def _categorise(df: pd.DataFrame) -> pd.DataFrame:
    """Assign ordered Categorical to instance column."""
    df = _strip_regime_suffix(df)
    df["instance"] = pd.Categorical(
        df["instance"], categories=_instance_order(df["instance"]), ordered=True)
    return df


def _load_pair(paths: list[str], tag_det="deterministic",
               tag_sto="stochastic_low") -> Optional[pd.DataFrame]:
    """Load one or two CSV paths, tag with regime, and return combined DataFrame."""
    if not paths:
        return None
    pieces = []
    for i, p in enumerate(paths):
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"[load] Could not read {p}: {e}")
            continue
        if "regime" not in df.columns:
            df["regime"] = tag_det if i == 0 else tag_sto
        # If results.csv has stochastic_tag, derive regime from it
        if "stochastic_tag" in df.columns and "regime" not in df.columns:
            df["regime"] = df["stochastic_tag"].apply(
                lambda t: "deterministic" if str(t) in ("none", "None", "nan", "0", "")
                else "stochastic_low")
        pieces.append(df)
    if not pieces:
        return None
    out = pd.concat(pieces, ignore_index=True)
    # derive regime from stochastic_tag if present and regime still missing
    if "stochastic_tag" in out.columns:
        out["regime"] = out["stochastic_tag"].apply(
            lambda t: "deterministic" if str(t) in ("none","None","nan","0","")
            else "stochastic_low")
    return _categorise(out)


def _save(fig, path: Path):
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved → {path.name}")


# ─────────────────────────────────────────────────────────────────────────────
#  AGV figures  (11–17)
# ─────────────────────────────────────────────────────────────────────────────

def fig11_agv_time_budget_stacked(agv: pd.DataFrame, out: Path):
    """
    Stacked bar: fraction of episode time each AGV state accounts for,
    split by regime. The key question: is time_waiting_route (red) large?
    """
    time_cols = ["time_idle","time_waiting_route","time_traveling",
                 "time_loading","time_unloading"]
    colors = ["#AAAAAA","#DC143C","#2E8B57","#F4A460","#4682B4"]
    labels = ["Idle","Waiting (blocked)","Traveling","Loading","Unloading"]

    g = agv.groupby(["regime","instance"], observed=True)[time_cols].mean().reset_index()
    totals = g[time_cols].sum(axis=1).replace(0, np.nan)
    for c in time_cols:
        g[c] = g[c] / totals

    regimes = [r for r in ["deterministic","stochastic_low"] if r in g["regime"].unique()]
    fig, axes = plt.subplots(1, len(regimes), figsize=(8*len(regimes), 6), sharey=True)
    if len(regimes) == 1:
        axes = [axes]

    for ax, regime in zip(axes, regimes):
        sub = g[g["regime"] == regime].set_index("instance")[time_cols]
        sub.plot(kind="bar", stacked=True, ax=ax, color=colors,
                 width=0.8, edgecolor="white")
        ax.set_title(f"AGV time budget — {regime}")
        ax.set_ylabel("Fraction of episode time")
        ax.set_xlabel("Instance")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(labels, loc="upper right", fontsize=8)
        ax.set_ylim(0, 1)
        ax.axhline(0.3, color="red", linestyle=":", alpha=0.4,
                   label="30% congestion warning")

    fig.suptitle("AGV time budget — where does fleet time actually go?", y=1.02)
    _save(fig, out)


def fig12_congestion_fraction(agv: pd.DataFrame, out: Path):
    """
    Congestion fraction (time_waiting_route / total_accounted) per instance.
    This is the single most diagnostic AGV metric.
    If >0.3 the floor is congestion-limited, not work-limited.
    """
    g = agv.groupby(["regime","instance"], observed=True)["congestion_fraction"].agg(
        ["mean","std"]).reset_index()

    fig, ax = plt.subplots(figsize=(13, 5))
    order = _instance_order(agv["instance"])
    sns.barplot(data=g, x="instance", y="mean", hue="regime", order=order,
                palette=REGIME_COLORS, ax=ax, errorbar=None)

    # overlay std as error bars manually
    for i, (_, row) in enumerate(g.iterrows()):
        regime_idx = 0 if row["regime"] == "deterministic" else 1
        x_pos = i  # approximate; fine for visual
        ax.errorbar(x_pos, row["mean"], yerr=row["std"], fmt="none",
                    color="black", capsize=3, linewidth=1)

    ax.axhline(0.30, color="red",    linestyle="--", alpha=0.6, label="30% warning")
    ax.axhline(0.15, color="orange", linestyle="--", alpha=0.4, label="15% caution")
    ax.set_title("AGV congestion fraction per instance\n"
                 "(time blocked waiting for zone clearance / total accounted time)")
    ax.set_ylabel("Congestion fraction")
    ax.set_xlabel("Instance")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(loc="upper left")
    _save(fig, out)


def fig13_trip_duration_vs_makespan(agv: pd.DataFrame, results: Optional[pd.DataFrame], out: Path):
    """
    Mean trip duration vs makespan inflation ratio.
    If longer trips drive inflation, they should correlate strongly.
    """
    ep = agv.groupby(["regime","instance","rule","seed"], observed=True).agg(
        mean_trip=("mean_trip_duration","mean"),
        total_trips=("total_trips","sum"),
        congestion=("congestion_fraction","mean"),
        makespan=("makespan","first")
    ).reset_index()

    if results is not None:
        det_ms = (results[results["regime"]=="deterministic"]
                  .groupby(["instance","rule","seed"], observed=True)["makespan"]
                  .mean().reset_index().rename(columns={"makespan":"makespan_det"}))
        ep = ep.merge(det_ms, on=["instance","rule","seed"], how="left")
        ep["makespan_ratio"] = ep["makespan"] / ep["makespan_det"].replace(0, np.nan)
    else:
        ep["makespan_ratio"] = np.nan

    sto = ep[ep["regime"]=="stochastic_low"].dropna(subset=["makespan_ratio"])
    if sto.empty:
        print("[fig13] No stochastic rows with makespan_ratio; skipping.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.scatterplot(data=sto, x="mean_trip", y="makespan_ratio", hue="instance",
                    size="total_trips", sizes=(30,200), ax=axes[0], alpha=0.7)
    axes[0].set_title("Mean trip duration vs makespan inflation")
    axes[0].set_xlabel("Mean trip duration (sim-s)")
    axes[0].set_ylabel("Makespan ratio (stoch / det)")
    axes[0].axhline(1, color="black", linestyle="--", alpha=0.3)
    if axes[0].get_legend():
        sns.move_legend(axes[0], "upper left", bbox_to_anchor=(1.02,1), fontsize=7, ncol=2)

    sns.scatterplot(data=sto, x="congestion", y="makespan_ratio", hue="instance",
                    ax=axes[1], alpha=0.7, legend=False)
    axes[1].set_title("Congestion fraction vs makespan inflation")
    axes[1].set_xlabel("Congestion fraction")
    axes[1].set_ylabel("")
    axes[1].axhline(1, color="black", linestyle="--", alpha=0.3)
    axes[1].axvline(0.30, color="red", linestyle="--", alpha=0.4)

    fig.suptitle("Does AGV congestion explain makespan inflation?  (gap = other causes)", y=1.02)
    _save(fig, out)


def fig14_agv_idle_vs_congestion(agv: pd.DataFrame, out: Path):
    """
    Scatter of time_idle vs time_waiting_route per AGV, coloured by instance.
    Bottom-right = overloaded fleet (low idle, high waiting).
    Top-left     = over-provisioned fleet (high idle, low waiting).
    Ideal point  = top-right (high idle = slack available, low wait = not congested).
    """
    g = agv.groupby(["regime","instance","agv_id"], observed=True).agg(
        idle=("time_idle","mean"),
        waiting=("time_waiting_route","mean"),
        traveling=("time_traveling","mean")
    ).reset_index()

    regimes = [r for r in ["deterministic","stochastic_low"] if r in g["regime"].unique()]
    fig, axes = plt.subplots(1, len(regimes), figsize=(7*len(regimes), 6))
    if len(regimes) == 1: axes = [axes]

    for ax, regime in zip(axes, regimes):
        sub = g[g["regime"]==regime]
        sns.scatterplot(data=sub, x="idle", y="waiting", hue="instance",
                        style="instance", ax=ax, alpha=0.75, s=60)
        ax.set_title(f"AGV idle vs blocked time — {regime}")
        ax.set_xlabel("Mean time idle (sim-s)")
        ax.set_ylabel("Mean time waiting for route clearance (sim-s)")
        if ax.get_legend():
            sns.move_legend(ax, "upper left", bbox_to_anchor=(1.02,1), fontsize=7)

    fig.suptitle("Over-provisioned (top-left) vs Congested (bottom-right)", y=1.02)
    _save(fig, out)


def fig15_fleet_size_vs_congestion(agv: pd.DataFrame, results: Optional[pd.DataFrame], out: Path):
    """
    If agvCount varies across instances (it does — set by machinesPerType × 1.5),
    plot congestion fraction vs agvCount to test whether more vehicles = more congestion.
    NOTE: this confounds instance complexity with fleet size; annotate clearly.
    """
    if results is None:
        print("[fig15] results.csv needed for agvCount; skipping.")
        return

    ep_agv = agv.groupby(["regime","instance","rule","seed"], observed=True).agg(
        congestion=("congestion_fraction","mean"),
        time_waiting=("time_waiting_route","mean")
    ).reset_index()

    ep_res = results[["instance","rule","seed","agvCount","machines"]].drop_duplicates()
    merged = ep_agv.merge(ep_res, on=["instance","rule","seed"], how="left")
    if merged["agvCount"].isna().all():
        print("[fig15] agvCount not resolved after merge; skipping.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(15,6))
    sns.scatterplot(data=merged, x="agvCount", y="congestion", hue="regime",
                    palette=REGIME_COLORS, ax=axes[0], alpha=0.6, s=60)
    axes[0].axhline(0.30, color="red", linestyle="--", alpha=0.5)
    axes[0].set_title("AGV congestion fraction vs fleet size\n"
                       "(⚠ fleet size ∝ problem size — not a controlled sweep)")
    axes[0].set_xlabel("agvCount"); axes[0].set_ylabel("Congestion fraction")

    # Normalise by machine count to get vehicles-per-machine ratio
    merged["agv_per_machine"] = merged["agvCount"] / merged["machines"].replace(0, np.nan)
    sns.scatterplot(data=merged, x="agv_per_machine", y="congestion", hue="regime",
                    palette=REGIME_COLORS, ax=axes[1], alpha=0.6, s=60, legend=False)
    axes[1].axhline(0.30, color="red", linestyle="--", alpha=0.5)
    axes[1].set_title("Congestion vs AGVs-per-machine ratio")
    axes[1].set_xlabel("AGVs per machine"); axes[1].set_ylabel("")
    _save(fig, out)


def fig16_reroute_analysis(agv: pd.DataFrame, out: Path):
    """
    Reroute count per AGV per episode (machine-failure redirections).
    Shows how much extra AGV work failures create beyond the direct downtime cost.
    """
    g = agv.groupby(["regime","instance"], observed=True)["reroute_count"].agg(
        ["mean","sum"]).reset_index()
    order = _instance_order(agv["instance"])

    fig, axes = plt.subplots(1,2, figsize=(15,5))
    sns.barplot(data=g, x="instance", y="mean", hue="regime", order=order,
                palette=REGIME_COLORS, ax=axes[0])
    axes[0].set_title("Mean reroutes per AGV per episode")
    axes[0].set_xlabel("Instance"); axes[0].set_ylabel("Mean reroutes")
    axes[0].tick_params(axis="x", rotation=45)

    sto = agv[agv["regime"]=="stochastic_low"]
    if not sto.empty:
        sns.violinplot(data=sto, x="instance", y="reroute_count", order=order,
                       ax=axes[1], color=REGIME_COLORS["stochastic_low"],
                       density_norm="width", cut=0, inner="quartile")
        axes[1].set_title("Distribution of reroutes per AGV (stochastic)")
        axes[1].set_xlabel("Instance"); axes[1].set_ylabel("Reroutes per AGV")
        axes[1].tick_params(axis="x", rotation=45)
    else:
        axes[1].axis("off")

    fig.suptitle("Machine-failure reroutes — additional AGV work caused by failures", y=1.02)
    _save(fig, out)


def fig17_path_length_comparison(agv: pd.DataFrame, out: Path):
    """
    Total path length per AGV per episode, det vs stoch.
    Longer paths in stochastic = longer detour routes after rerouting.
    """
    g = agv.groupby(["regime","instance","rule","seed","agv_id"], observed=True).agg(
        total_path_length=("total_path_length","mean"),
        total_trips=("total_trips","sum")
    ).reset_index()
    g["path_per_trip"] = g["total_path_length"] / g["total_trips"].replace(0, np.nan)

    order = _instance_order(agv["instance"])
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.boxplot(data=g, x="instance", y="total_path_length", hue="regime",
                order=order, palette=REGIME_COLORS, ax=axes[0],
                flierprops={"markersize": 3})
    axes[0].set_title("Total AGV path length per episode")
    axes[0].set_xlabel("Instance"); axes[0].set_ylabel("Path length (sim-units)")
    axes[0].tick_params(axis="x", rotation=45)

    g_trip = g.dropna(subset=["path_per_trip"])
    sns.boxplot(data=g_trip, x="instance", y="path_per_trip", hue="regime",
                order=order, palette=REGIME_COLORS, ax=axes[1],
                flierprops={"markersize": 3})
    axes[1].set_title("Path length per trip (normalises for trip count)")
    axes[1].set_xlabel("Instance"); axes[1].set_ylabel("Path length / trip")
    axes[1].tick_params(axis="x", rotation=45)

    fig.suptitle("AGV travel distance — reroutes inflate path length under failure", y=1.02)
    _save(fig, out)


# ─────────────────────────────────────────────────────────────────────────────
#  Segment / congestion figures  (18–22)
# ─────────────────────────────────────────────────────────────────────────────

def fig18_top_bottlenecks(seg: pd.DataFrame, out: Path, top_n: int = 15):
    """
    Horizontal bar chart of the top-N zones by mean block_rate, split by regime.
    Identifies structural bottlenecks — zones that are always congested.
    """
    g = seg.groupby(["regime","zone_name","aisle_type"], observed=True).agg(
        block_rate=("block_rate","mean"),
        mean_block_time=("mean_block_time","mean"),
        traversals=("traversal_count","mean")
    ).reset_index()

    fig, axes = plt.subplots(1,2, figsize=(18,8))
    for ax, regime in zip(axes, ["deterministic","stochastic_low"]):
        sub = g[g["regime"]==regime].nlargest(top_n, "block_rate")
        if sub.empty:
            ax.axis("off"); continue
        color_map = {"RowAisle":"#F4A460","SpineAisle":"#2E8B57","VerticalAisle":"#4682B4"}
        colors = sub["aisle_type"].map(color_map).fillna("grey")
        bars = ax.barh(range(len(sub)), sub["block_rate"], color=colors)
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels(sub["zone_name"], fontsize=8)
        ax.invert_yaxis()
        ax.axvline(0.30, color="red", linestyle="--", alpha=0.5, label="30% threshold")
        ax.set_title(f"Top-{top_n} bottleneck zones — {regime}")
        ax.set_xlabel("Block rate (blocked / (blocked+traversed))")
        # legend for aisle type
        from matplotlib.patches import Patch
        patches = [Patch(color=c, label=k) for k, c in color_map.items()]
        ax.legend(handles=patches + [plt.Line2D([],[],color="red",linestyle="--",
                  label="30% threshold")], fontsize=8)

    fig.suptitle("Structural bottleneck zones — sorted by contention rate", y=1.02)
    _save(fig, out)


def fig19_segment_heatmap(seg: pd.DataFrame, out: Path):
    """
    Heatmap of mean block_rate across (zone_name × instance).
    Chronic bottlenecks appear as horizontal bands — same zone hot across all instances.
    """
    g = seg.groupby(["regime","zone_name","instance"], observed=True)["block_rate"].mean().reset_index()

    regimes = [r for r in ["deterministic","stochastic_low"] if r in g["regime"].unique()]
    fig, axes = plt.subplots(1, len(regimes), figsize=(10*len(regimes), max(8, len(g["zone_name"].unique())//3)))
    if len(regimes) == 1: axes = [axes]

    for ax, regime in zip(axes, regimes):
        sub = g[g["regime"]==regime]
        pivot = sub.pivot(index="zone_name", columns="instance", values="block_rate")
        # Sort rows by mean block rate descending (hottest zones at top)
        pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
        sns.heatmap(pivot, ax=ax, cmap="YlOrRd", vmin=0, vmax=0.6,
                    cbar_kws={"label":"Block rate"},
                    linewidths=0.15, linecolor="white")
        ax.set_title(f"Zone block rate — {regime}")
        ax.set_xlabel("Instance"); ax.set_ylabel("Zone")
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.tick_params(axis="y", labelsize=7)

    fig.suptitle("Congestion heatmap — horizontal bands = structural floor bottlenecks", y=1.02)
    _save(fig, out)


def fig20_aisle_type_comparison(seg: pd.DataFrame, out: Path):
    """
    Violin of block_rate grouped by aisle_type and regime.
    RowAisles typically worse (single-width, direction-constrained).
    """
    fig, axes = plt.subplots(1, 2, figsize=(15,6))
    order = ["RowAisle","SpineAisle","VerticalAisle"]
    palette = {"RowAisle":"#F4A460","SpineAisle":"#2E8B57","VerticalAisle":"#4682B4"}

    sns.violinplot(data=seg, x="aisle_type", y="block_rate", hue="regime",
                   order=order, palette=REGIME_COLORS, ax=axes[0],
                   split=True, inner="quartile", density_norm="width", cut=0)
    axes[0].set_title("Block rate distribution by aisle type")
    axes[0].set_ylabel("Block rate"); axes[0].set_xlabel("Aisle type")
    axes[0].axhline(0.30, color="red", linestyle="--", alpha=0.5)

    sns.violinplot(data=seg, x="aisle_type", y="mean_block_time", hue="regime",
                   order=order, palette=REGIME_COLORS, ax=axes[1],
                   split=True, inner="quartile", density_norm="width", cut=0)
    axes[1].set_title("Mean block duration by aisle type")
    axes[1].set_ylabel("Mean block time (sim-s)"); axes[1].set_xlabel("Aisle type")

    fig.suptitle("Aisle type congestion profile — RowAisles expected to dominate", y=1.02)
    _save(fig, out)


def fig21_spine_vs_row_entry(seg: pd.DataFrame, out: Path):
    """
    Compare entry segments (Seg0 and last segment of each row aisle) vs mid-segments.
    Entry segments adjacent to spine intersections are predicted hotspots.
    """
    seg = seg.copy()
    # Extract aisle index and segment index from name
    seg["seg_index"] = seg["zone_name"].apply(
        lambda n: int(m.group(1)) if (m := re.search(r"Seg(\d+)$", str(n))) else -1)
    row_segs = seg[seg["aisle_type"]=="RowAisle"].copy()
    if row_segs.empty:
        print("[fig21] No RowAisle zones; skipping."); return

    max_seg_per_aisle = row_segs.groupby(["zone_name"], observed=True)["seg_index"].transform("max")
    row_segs["position"] = "mid"
    row_segs.loc[row_segs["seg_index"] == 0, "position"] = "entry (left)"
    row_segs.loc[row_segs["seg_index"] == max_seg_per_aisle, "position"] = "exit (right)"

    fig, axes = plt.subplots(1,2, figsize=(15,6))
    order = ["entry (left)","mid","exit (right)"]
    sns.boxplot(data=row_segs, x="position", y="block_rate", hue="regime",
                order=order, palette=REGIME_COLORS, ax=axes[0])
    axes[0].set_title("Row aisle block rate: entry vs mid vs exit segments")
    axes[0].set_ylabel("Block rate"); axes[0].set_xlabel("Segment position")
    axes[0].axhline(0.30, color="red", linestyle="--", alpha=0.4)

    sns.boxplot(data=row_segs, x="position", y="mean_block_time", hue="regime",
                order=order, palette=REGIME_COLORS, ax=axes[1])
    axes[1].set_title("Row aisle mean block duration: entry vs mid vs exit")
    axes[1].set_ylabel("Mean block time (sim-s)"); axes[1].set_xlabel("Segment position")

    fig.suptitle("Entry/exit segments of row aisles — predicted spine-intersection hotspots", y=1.02)
    _save(fig, out)


def fig22_congestion_vs_machine_utilization(seg: pd.DataFrame, mach: pd.DataFrame, out: Path):
    """
    Cross-domain: does zone-level congestion predict machine starvation?
    Per instance: mean row-aisle block_rate vs mean machine utilization_rate.
    Negative correlation confirms the AGV-bottleneck → machine-starvation chain.
    """
    seg_row = seg[seg["aisle_type"]=="RowAisle"]
    seg_g = seg_row.groupby(["regime","instance"], observed=True)["block_rate"].mean().reset_index()
    seg_g.columns = ["regime","instance","mean_row_block_rate"]

    mach_g = mach.groupby(["regime","instance"], observed=True)["utilization_rate"].mean().reset_index()

    merged = seg_g.merge(mach_g, on=["regime","instance"], how="inner")
    if merged.empty:
        print("[fig22] No overlapping data after merge; skipping."); return

    fig, ax = plt.subplots(figsize=(10,7))
    sns.scatterplot(data=merged, x="mean_row_block_rate", y="utilization_rate",
                    hue="regime", style="instance", palette=REGIME_COLORS,
                    ax=ax, s=100, alpha=0.8)

    # Fit and plot regression lines per regime
    for regime, color in REGIME_COLORS.items():
        sub = merged[merged["regime"]==regime].dropna()
        if len(sub) >= 3:
            z = np.polyfit(sub["mean_row_block_rate"], sub["utilization_rate"], 1)
            xs = np.linspace(sub["mean_row_block_rate"].min(), sub["mean_row_block_rate"].max(), 50)
            ax.plot(xs, np.polyval(z, xs), color=color, linestyle="--", alpha=0.7)

    ax.set_title("Row-aisle congestion vs machine utilization\n"
                 "(negative slope = AGV congestion starves machines)")
    ax.set_xlabel("Mean row-aisle block rate")
    ax.set_ylabel("Mean machine utilization rate")
    if ax.get_legend():
        sns.move_legend(ax, "upper right", bbox_to_anchor=(1.25,1), fontsize=8)
    _save(fig, out)


# ─────────────────────────────────────────────────────────────────────────────
#  Cross-domain combined figures  (23–25)
# ─────────────────────────────────────────────────────────────────────────────

def fig23_cascade_chain(mach: pd.DataFrame, agv: pd.DataFrame, seg: pd.DataFrame,
                        results: Optional[pd.DataFrame], out: Path):
    """
    Four-panel cascade chain summary per instance:
      1. Machine utilization
      2. AGV congestion fraction
      3. Row-aisle block rate
      4. Makespan inflation ratio
    Visual proof (or refutation) of: failures → congestion → starvation → inflation.
    """
    order = _instance_order(mach["instance"])

    mach_g = mach.groupby(["regime","instance"], observed=True)["utilization_rate"].mean().reset_index()
    agv_g  = agv.groupby(["regime","instance"],  observed=True)["congestion_fraction"].mean().reset_index()
    seg_g  = (seg[seg["aisle_type"]=="RowAisle"]
              .groupby(["regime","instance"], observed=True)["block_rate"].mean().reset_index())

    panels = [
        (mach_g, "utilization_rate", "Machine utilization", False),
        (agv_g,  "congestion_fraction", "AGV congestion fraction", False),
        (seg_g,  "block_rate", "Row-aisle block rate", False),
    ]

    if results is not None:
        det = (results[results["regime"]=="deterministic"]
               .groupby("instance", observed=True)["makespan"].mean()
               .reset_index().rename(columns={"makespan":"ms_det"}))
        sto = (results[results["regime"]=="stochastic_low"]
               .groupby("instance", observed=True)["makespan"].mean()
               .reset_index().rename(columns={"makespan":"ms_sto"}))
        inf = det.merge(sto, on="instance", how="inner")
        inf["inflation"] = inf["ms_sto"] / inf["ms_det"]
        inf["regime"] = "stochastic_low"
        panels.append((inf, "inflation", "Makespan inflation ratio", True))

    n_panels = len(panels)
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 4*n_panels), sharex=True)
    if n_panels == 1: axes = [axes]

    for ax, (df, col, title, single_regime) in zip(axes, panels):
        if single_regime:
            sub = df.set_index("instance")[col].reindex(order)
            ax.bar(range(len(order)), sub.values, color=REGIME_COLORS["stochastic_low"],
                   alpha=0.7, width=0.6)
            ax.axhline(1, color="black", linestyle="--", alpha=0.4)
        else:
            for regime, color in REGIME_COLORS.items():
                sub = df[df["regime"]==regime].set_index("instance")[col].reindex(order)
                ax.plot(range(len(order)), sub.values, "o-", color=color,
                        label=regime, linewidth=1.5, markersize=5, alpha=0.8)
            ax.legend(fontsize=8)
        ax.set_ylabel(title, fontsize=9)
        ax.set_title(title)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=45, ha="right", fontsize=8)

    fig.suptitle("Cascade chain: do failures → congestion → starvation → inflation?", y=1.01)
    fig.tight_layout()
    _save(fig, out)


def fig24_pdr_brittleness_vs_congestion(agv: pd.DataFrame, mach: pd.DataFrame, out: Path):
    """
    Does PDR choice interact with floor congestion?
    Scatter: per-(instance, rule) congestion vs utilization, coloured by rule.
    PDRs that consistently sit in the low-util / high-congestion quadrant
    are poor choices for AGV-bottlenecked floors.
    """
    agv_g = agv.groupby(["regime","instance","rule"], observed=True).agg(
        congestion=("congestion_fraction","mean")).reset_index()
    mach_g = mach.groupby(["regime","instance","rule"], observed=True).agg(
        util=("utilization_rate","mean")).reset_index()

    merged = agv_g.merge(mach_g, on=["regime","instance","rule"], how="inner")
    if merged.empty:
        print("[fig24] No overlapping data; skipping."); return

    regimes = [r for r in ["deterministic","stochastic_low"] if r in merged["regime"].unique()]
    fig, axes = plt.subplots(1, len(regimes), figsize=(8*len(regimes), 6))
    if len(regimes)==1: axes=[axes]

    for ax, regime in zip(axes, regimes):
        sub = merged[merged["regime"]==regime]
        sns.scatterplot(data=sub, x="congestion", y="util", hue="rule",
                        ax=ax, alpha=0.7, s=70)
        ax.set_title(f"PDR: congestion vs utilization — {regime}")
        ax.set_xlabel("Mean AGV congestion fraction")
        ax.set_ylabel("Mean machine utilization")
        ax.axvline(0.30, color="red",  linestyle="--", alpha=0.4)
        ax.axhline(0.10, color="grey", linestyle="--", alpha=0.4)
        if ax.get_legend():
            sns.move_legend(ax,"upper left",bbox_to_anchor=(1.02,1),fontsize=8)

    fig.suptitle("PDR choice and floor efficiency — do some PDRs worsen congestion?", y=1.02)
    _save(fig, out)


def fig25_summary_table(mach: pd.DataFrame, agv: pd.DataFrame,
                        seg: pd.DataFrame, results: Optional[pd.DataFrame],
                        out_csv: Path):
    """
    Combined per-(regime, instance) summary table merging key metrics from all domains.
    Saved as CSV for advisor discussions.
    """
    rows = []
    instances = _instance_order(mach["instance"])

    for regime in ["deterministic","stochastic_low"]:
        for inst in instances:
            row = {"regime": regime, "instance": str(inst)}

            # Machine
            m = mach[(mach["regime"]==regime) & (mach["instance"].astype(str)==str(inst))]
            if not m.empty:
                per_m = m.groupby("machine_id")["utilization_rate"].mean()
                row["mean_util"]       = round(per_m.mean(), 3)
                row["frac_below_5pct"] = round((per_m < 0.05).mean(), 3)
                row["mean_makespan"]   = round(m["makespan"].mean(), 1)

            # AGV
            a = agv[(agv["regime"]==regime) & (agv["instance"].astype(str)==str(inst))]
            if not a.empty:
                row["mean_congestion"]    = round(a["congestion_fraction"].mean(), 3)
                row["mean_trip_dur"]      = round(a["mean_trip_duration"].mean(), 1)
                row["mean_reroutes"]      = round(a["reroute_count"].mean(), 2)
                row["mean_time_waiting"]  = round(a["time_waiting_route"].mean(), 1)
                row["mean_time_idle"]     = round(a["time_idle"].mean(), 1)

            # Segments
            s = seg[(seg["regime"]==regime) & (seg["instance"].astype(str)==str(inst))]
            if not s.empty:
                row_aisles = s[s["aisle_type"]=="RowAisle"]
                row["mean_row_block_rate"]  = round(row_aisles["block_rate"].mean(), 3) if not row_aisles.empty else 0
                row["worst_zone"]           = s.loc[s["block_rate"].idxmax(), "zone_name"] if not s.empty else ""
                row["worst_zone_blockrate"] = round(s["block_rate"].max(), 3) if not s.empty else 0

            rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(out_csv, index=False)
    print(f"  saved → {out_csv.name}")

    # Also print to stdout
    with pd.option_context("display.max_rows", None, "display.width", 220, "display.float_format", "{:.3f}".format):
        print("\n=== Phase-2 Combined Summary ===")
        print(summary.to_string(index=False))

    return summary


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="DFJSP Phase-2 analysis suite")
    p.add_argument("--machine",  nargs="+", metavar="CSV",
                   help="machine_utilization CSV(s). Pass det then stoch, or single combined.")
    p.add_argument("--results",  nargs="+", metavar="CSV",
                   help="results CSV(s).")
    p.add_argument("--agv",      nargs="+", metavar="CSV",
                   help="agv_performance CSV(s).")
    p.add_argument("--segments", nargs="+", metavar="CSV",
                   help="segment_congestion CSV(s).")
    p.add_argument("--out", default="./figs", help="Output directory (default: ./figs)")
    args = p.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    mach    = _load_pair(args.machine  or [])
    results = _load_pair(args.results  or [])
    agv     = _load_pair(args.agv      or [])
    seg     = _load_pair(args.segments or [])

    present = {k: v is not None for k, v in
               [("machine",mach),("results",results),("agv",agv),("segments",seg)]}
    print("Loaded domains:", {k for k,v in present.items() if v})

    # ── AGV figures ───────────────────────────────────────────────────────────
    if present["agv"]:
        print("\n── AGV figures ──")
        fig11_agv_time_budget_stacked(agv,  out/"11_agv_time_budget.png")
        fig12_congestion_fraction(agv,      out/"12_congestion_fraction.png")
        fig13_trip_duration_vs_makespan(agv, results, out/"13_trip_vs_makespan.png")
        fig14_agv_idle_vs_congestion(agv,   out/"14_idle_vs_congestion.png")
        fig15_fleet_size_vs_congestion(agv, results, out/"15_fleet_vs_congestion.png")
        fig16_reroute_analysis(agv,         out/"16_reroutes.png")
        fig17_path_length_comparison(agv,   out/"17_path_length.png")

    # ── Segment figures ───────────────────────────────────────────────────────
    if present["segments"]:
        print("\n── Segment congestion figures ──")
        fig18_top_bottlenecks(seg,          out/"18_top_bottlenecks.png")
        fig19_segment_heatmap(seg,          out/"19_segment_heatmap.png")
        fig20_aisle_type_comparison(seg,    out/"20_aisle_type.png")
        fig21_spine_vs_row_entry(seg,       out/"21_spine_entry.png")
        if present["machine"]:
            fig22_congestion_vs_machine_utilization(seg, mach, out/"22_congestion_vs_util.png")

    # ── Cross-domain ──────────────────────────────────────────────────────────
    if present["machine"] and present["agv"]:
        print("\n── Cross-domain figures ──")
        fig23_cascade_chain(mach, agv, seg if present["segments"] else pd.DataFrame(),
                            results, out/"23_cascade_chain.png")
        fig24_pdr_brittleness_vs_congestion(agv, mach, out/"24_pdr_vs_congestion.png")

    if present["machine"] and present["agv"]:
        fig25_summary_table(mach, agv,
                            seg if present["segments"] else pd.DataFrame(),
                            results, out/"25_summary_table.csv")

    print(f"\nDone. Outputs written to {out}/")


if __name__ == "__main__":
    main()
