"""
DFJSP Phase-2 Analysis Suite (Single & Dual Regime Robust)
==========================================================
Covers all four CSVs from the updated logging schema.
Dynamically scales layouts if only deterministic or stochastic data is provided.
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
    df = _strip_regime_suffix(df)
    df["instance"] = pd.Categorical(
        df["instance"], categories=_instance_order(df["instance"]), ordered=True)
    return df


def _load_pair(paths: list[str], tag_det="deterministic",
               tag_sto="stochastic_low") -> Optional[pd.DataFrame]:
    if not paths:
        return None
    pieces = []
    for i, p in enumerate(paths):
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"[load] Could not read {p}: {e}")
            continue
        
        # If there's only 1 file passed, check if it already has internal tracking
        if "regime" not in df.columns:
            if "stochastic_tag" in df.columns:
                df["regime"] = df["stochastic_tag"].apply(
                    lambda t: "deterministic" if str(t) in ("none", "None", "nan", "0", "")
                    else "stochastic_low")
            else:
                # If fallback is needed, rely on positional assumption or name matching
                if len(paths) == 1:
                    df["regime"] = tag_det if "det" in str(p).lower() else tag_sto
                else:
                    df["regime"] = tag_det if i == 0 else tag_sto
        pieces.append(df)
        
    if not pieces:
        return None
    out = pd.concat(pieces, ignore_index=True)
    return _categorise(out)


def _save(fig, path: Path):
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved → {path.name}")


# ─────────────────────────────────────────────────────────────────────────────
#  AGV figures  (11–17)
# ─────────────────────────────────────────────────────────────────────────────

def fig11_agv_time_budget_stacked(agv: pd.DataFrame, out: Path):
    time_cols = ["time_idle","time_waiting_route","time_traveling",
                 "time_loading","time_unloading"]
    colors = ["#AAAAAA","#DC143C","#2E8B57","#F4A460","#4682B4"]
    labels = ["Idle","Waiting (blocked)","Traveling","Loading","Unloading"]

    g = agv.groupby(["regime","instance"], observed=True)[time_cols].mean().reset_index()
    totals = g[time_cols].sum(axis=1).replace(0, np.nan)
    for c in time_cols:
        g[c] = g[c] / totals

    regimes = [r for r in ["deterministic","stochastic_low"] if r in g["regime"].unique()]
    fig, axes = plt.subplots(1, len(regimes), figsize=(7 * len(regimes), 6), sharey=True)
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
        ax.axhline(0.3, color="red", linestyle=":", alpha=0.4)

    fig.suptitle("AGV time budget — where does fleet time actually go?", y=1.02)
    _save(fig, out)


def fig12_congestion_fraction(agv: pd.DataFrame, out: Path):
    g = agv.groupby(["regime","instance"], observed=True)["congestion_fraction"].agg(
        ["mean","std"]).reset_index()

    fig, ax = plt.subplots(figsize=(13, 5))
    order = _instance_order(agv["instance"])
    
    # Filter global palette map down to active regimes to prevent seaborn coloring shifts
    active_regimes = g["regime"].unique()
    current_palette = {r: REGIME_COLORS[r] for r in active_regimes}

    sns.barplot(data=g, x="instance", y="mean", hue="regime", order=order,
                palette=current_palette, ax=ax, errorbar=None)

    # Automatically calculate coordinates for error bars via matplotlib offsets
    for p in ax.patches:
        x = p.get_x() + p.get_width() / 2.0
        y = p.get_height()
        if y > 0:
            # Match back to the dataframe grouping to get matching std deviation
            # Use rounding to safely align float coordinates to categorical indices if needed
            inst_idx = int(round(p.get_x() + p.get_width() / 2.0 - 0.5 if len(active_regimes) > 1 else p.get_x()))
            if inst_idx < len(order):
                inst_name = order[inst_idx]
                std_val = g[g["instance"] == inst_name]["std"].mean()
                if not np.isnan(std_val):
                    ax.errorbar(x, y, yerr=std_val, fmt="none", color="black", capsize=3, linewidth=1)

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
    ep = agv.groupby(["regime","instance","rule","seed"], observed=True).agg(
        mean_trip=("mean_trip_duration","mean"),
        total_trips=("total_trips","sum"),
        congestion=("congestion_fraction","mean"),
        makespan=("makespan","first")
    ).reset_index()

    # Requires both regimes to track inflation ratio cleanly
    if results is not None and "deterministic" in ep["regime"].values and "stochastic_low" in ep["regime"].values:
        det_ms = (results[results["regime"]=="deterministic"]
                  .groupby(["instance","rule","seed"], observed=True)["makespan"]
                  .mean().reset_index().rename(columns={"makespan":"makespan_det"}))
        ep = ep.merge(det_ms, on=["instance","rule","seed"], how="left")
        ep["makespan_ratio"] = ep["makespan"] / ep["makespan_det"].replace(0, np.nan)
        sto = ep[ep["regime"]=="stochastic_low"].dropna(subset=["makespan_ratio"])
    else:
        print("[fig13] Single regime execution or missing results mapping; skipping relative cross-comparison.")
        return

    if sto.empty:
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

    fig.suptitle("Does AGV congestion explain makespan inflation?", y=1.02)
    _save(fig, out)


def fig14_agv_idle_vs_congestion(agv: pd.DataFrame, out: Path):
    g = agv.groupby(["regime","instance","agv_id"], observed=True).agg(
        idle=("time_idle","mean"),
        waiting=("time_waiting_route","mean"),
        traveling=("time_traveling","mean")
    ).reset_index()

    regimes = [r for r in ["deterministic","stochastic_low"] if r in g["regime"].unique()]
    fig, axes = plt.subplots(1, len(regimes), figsize=(7 * len(regimes), 6))
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
    if results is None:
        return

    ep_agv = agv.groupby(["regime","instance","rule","seed"], observed=True).agg(
        congestion=("congestion_fraction","mean"),
        time_waiting=("time_waiting_route","mean")
    ).reset_index()

    ep_res = results[["instance","rule","seed","agvCount","machines"]].drop_duplicates()
    merged = ep_agv.merge(ep_res, on=["instance","rule","seed"], how="left")
    if merged["agvCount"].isna().all():
        return

    active_regimes = merged["regime"].unique()
    current_palette = {r: REGIME_COLORS[r] for r in active_regimes}

    fig, axes = plt.subplots(1, 2, figsize=(15,6))
    sns.scatterplot(data=merged, x="agvCount", y="congestion", hue="regime",
                    palette=current_palette, ax=axes[0], alpha=0.6, s=60)
    axes[0].axhline(0.30, color="red", linestyle="--", alpha=0.5)
    axes[0].set_title("AGV congestion fraction vs fleet size")
    axes[0].set_xlabel("agvCount"); axes[0].set_ylabel("Congestion fraction")

    merged["agv_per_machine"] = merged["agvCount"] / merged["machines"].replace(0, np.nan)
    sns.scatterplot(data=merged, x="agv_per_machine", y="congestion", hue="regime",
                    palette=current_palette, ax=axes[1], alpha=0.6, s=60, legend=False)
    axes[1].axhline(0.30, color="red", linestyle="--", alpha=0.5)
    axes[1].set_title("Congestion vs AGVs-per-machine ratio")
    axes[1].set_xlabel("AGVs per machine"); axes[1].set_ylabel("")
    _save(fig, out)


def fig16_reroute_analysis(agv: pd.DataFrame, out: Path):
    g = agv.groupby(["regime","instance"], observed=True)["reroute_count"].agg(
        ["mean","sum"]).reset_index()
    order = _instance_order(agv["instance"])
    
    active_regimes = g["regime"].unique()
    current_palette = {r: REGIME_COLORS[r] for r in active_regimes}

    # If only deterministic processing runs, drop the secondary violin plot completely
    has_stochastic = "stochastic_low" in active_regimes
    fig, axes = plt.subplots(1, 2 if has_stochastic else 1, figsize=(15 if has_stochastic else 8, 5))
    if not has_stochastic:
        axes = [axes]

    sns.barplot(data=g, x="instance", y="mean", hue="regime", order=order,
                palette=current_palette, ax=axes[0])
    axes[0].set_title("Mean reroutes per AGV per episode")
    axes[0].set_xlabel("Instance"); axes[0].set_ylabel("Mean reroutes")
    axes[0].tick_params(axis="x", rotation=45)

    if has_stochastic:
        sto = agv[agv["regime"]=="stochastic_low"]
        sns.violinplot(data=sto, x="instance", y="reroute_count", order=order,
                       ax=axes[1], color=REGIME_COLORS["stochastic_low"],
                       density_norm="width", cut=0, inner="quartile")
        axes[1].set_title("Distribution of reroutes per AGV (stochastic)")
        axes[1].set_xlabel("Instance"); axes[1].set_ylabel("Reroutes per AGV")
        axes[1].tick_params(axis="x", rotation=45)

    fig.suptitle("Machine-failure reroutes — additional AGV workload profile", y=1.02)
    _save(fig, out)


def fig17_path_length_comparison(agv: pd.DataFrame, out: Path):
    g = agv.groupby(["regime","instance","rule","seed","agv_id"], observed=True).agg(
        total_path_length=("total_path_length","mean"),
        total_trips=("total_trips","sum")
    ).reset_index()
    g["path_per_trip"] = g["total_path_length"] / g["total_trips"].replace(0, np.nan)

    order = _instance_order(agv["instance"])
    active_regimes = g["regime"].unique()
    current_palette = {r: REGIME_COLORS[r] for r in active_regimes}

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.boxplot(data=g, x="instance", y="total_path_length", hue="regime",
                order=order, palette=current_palette, ax=axes[0], flierprops={"markersize": 3})
    axes[0].set_title("Total AGV path length per episode")
    axes[0].set_xlabel("Instance"); axes[0].set_ylabel("Path length (sim-units)")
    axes[0].tick_params(axis="x", rotation=45)

    g_trip = g.dropna(subset=["path_per_trip"])
    sns.boxplot(data=g_trip, x="instance", y="path_per_trip", hue="regime",
                order=order, palette=current_palette, ax=axes[1], flierprops={"markersize": 3})
    axes[1].set_title("Path length per trip (normalized)")
    axes[1].set_xlabel("Instance"); axes[1].set_ylabel("Path length / trip")
    axes[1].tick_params(axis="x", rotation=45)

    _save(fig, out)


# ─────────────────────────────────────────────────────────────────────────────
#  Segment / congestion figures  (18–22)
# ─────────────────────────────────────────────────────────────────────────────

def fig18_top_bottlenecks(seg: pd.DataFrame, out: Path, top_n: int = 15):
    g = seg.groupby(["regime","zone_name","aisle_type"], observed=True).agg(
        block_rate=("block_rate","mean")
    ).reset_index()

    active_regimes = g["regime"].unique()
    fig, axes = plt.subplots(1, len(active_regimes), figsize=(9 * len(active_regimes), 8), squeeze=False)
    axes = axes.flatten()

    for ax, regime in zip(axes, active_regimes):
        sub = g[g["regime"]==regime].nlargest(top_n, "block_rate")
        if sub.empty:
            continue
        color_map = {"RowAisle":"#F4A460","SpineAisle":"#2E8B57","VerticalAisle":"#4682B4"}
        colors = sub["aisle_type"].map(color_map).fillna("grey")
        ax.barh(range(len(sub)), sub["block_rate"], color=colors)
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels(sub["zone_name"], fontsize=8)
        ax.invert_yaxis()
        ax.axvline(0.30, color="red", linestyle="--", alpha=0.5)
        ax.set_title(f"Top-{top_n} bottleneck zones — {regime}")
        ax.set_xlabel("Block rate")
        
        from matplotlib.patches import Patch
        patches = [Patch(color=c, label=k) for k, c in color_map.items()]
        ax.legend(handles=patches + [plt.Line2D([],[],color="red",linestyle="--", label="30% threshold")], fontsize=8)

    fig.suptitle("Structural bottleneck zones — sorted by contention rate", y=1.02)
    _save(fig, out)


def fig19_segment_heatmap(seg: pd.DataFrame, out: Path):
    g = seg.groupby(["regime","zone_name","instance"], observed=True)["block_rate"].mean().reset_index()
    active_regimes = g["regime"].unique()
    
    fig, axes = plt.subplots(1, len(active_regimes), figsize=(10 * len(active_regimes), max(8, len(g["zone_name"].unique())//3)), squeeze=False)
    axes = axes.flatten()

    for ax, regime in zip(axes, active_regimes):
        sub = g[g["regime"]==regime]
        pivot = sub.pivot(index="zone_name", columns="instance", values="block_rate")
        pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
        sns.heatmap(pivot, ax=ax, cmap="YlOrRd", vmin=0, vmax=0.6, cbar_kws={"label":"Block rate"}, linewidths=0.15, linecolor="white")
        ax.set_title(f"Zone block rate — {regime}")
        ax.set_xlabel("Instance"); ax.set_ylabel("Zone")
        ax.tick_params(axis="x", rotation=45, labelsize=8)

    _save(fig, out)


def fig20_aisle_type_comparison(seg: pd.DataFrame, out: Path):
    active_regimes = seg["regime"].unique()
    current_palette = {r: REGIME_COLORS[r] for r in active_regimes}
    # split parameter on violinplot breaks if only 1 hue value exists
    should_split = len(active_regimes) > 1

    fig, axes = plt.subplots(1, 2, figsize=(15,6))
    order = ["RowAisle","SpineAisle","VerticalAisle"]

    sns.violinplot(data=seg, x="aisle_type", y="block_rate", hue="regime",
                   order=order, palette=current_palette, ax=axes[0],
                   split=should_split, inner="quartile", density_norm="width", cut=0)
    axes[0].set_title("Block rate distribution by aisle type")
    axes[0].axhline(0.30, color="red", linestyle="--", alpha=0.5)

    sns.violinplot(data=seg, x="aisle_type", y="mean_block_time", hue="regime",
                   order=order, palette=current_palette, ax=axes[1],
                   split=should_split, inner="quartile", density_norm="width", cut=0)
    axes[1].set_title("Mean block duration by aisle type")

    _save(fig, out)


def fig21_spine_vs_row_entry(seg: pd.DataFrame, out: Path):
    seg = seg.copy()
    seg["seg_index"] = seg["zone_name"].apply(lambda n: int(m.group(1)) if (m := re.search(r"Seg(\d+)$", str(n))) else -1)
    row_segs = seg[seg["aisle_type"]=="RowAisle"].copy()
    if row_segs.empty: return

    max_seg_per_aisle = row_segs.groupby(["zone_name"], observed=True)["seg_index"].transform("max")
    row_segs["position"] = "mid"
    row_segs.loc[row_segs["seg_index"] == 0, "position"] = "entry (left)"
    row_segs.loc[row_segs["seg_index"] == max_seg_per_aisle, "position"] = "exit (right)"

    active_regimes = row_segs["regime"].unique()
    current_palette = {r: REGIME_COLORS[r] for r in active_regimes}

    fig, axes = plt.subplots(1,2, figsize=(15,6))
    order = ["entry (left)","mid","exit (right)"]
    sns.boxplot(data=row_segs, x="position", y="block_rate", hue="regime", order=order, palette=current_palette, ax=axes[0])
    axes[0].set_title("Row aisle block rate: entry vs mid vs exit")
    axes[0].axhline(0.30, color="red", linestyle="--", alpha=0.4)

    sns.boxplot(data=row_segs, x="position", y="mean_block_time", hue="regime", order=order, palette=current_palette, ax=axes[1])
    _save(fig, out)


def fig22_congestion_vs_machine_utilization(seg: pd.DataFrame, mach: pd.DataFrame, out: Path):
    seg_row = seg[seg["aisle_type"]=="RowAisle"]
    seg_g = seg_row.groupby(["regime","instance"], observed=True)["block_rate"].mean().reset_index()
    seg_g.columns = ["regime","instance","mean_row_block_rate"]
    mach_g = mach.groupby(["regime","instance"], observed=True)["utilization_rate"].mean().reset_index()

    merged = seg_g.merge(mach_g, on=["regime","instance"], how="inner")
    if merged.empty: return

    active_regimes = merged["regime"].unique()
    current_palette = {r: REGIME_COLORS[r] for r in active_regimes}

    fig, ax = plt.subplots(figsize=(10,7))
    sns.scatterplot(data=merged, x="mean_row_block_rate", y="utilization_rate", hue="regime", style="instance", palette=current_palette, ax=ax, s=100, alpha=0.8)

    for regime in active_regimes:
        sub = merged[merged["regime"]==regime].dropna()
        if len(sub) >= 3:
            z = np.polyfit(sub["mean_row_block_rate"], sub["utilization_rate"], 1)
            xs = np.linspace(sub["mean_row_block_rate"].min(), sub["mean_row_block_rate"].max(), 50)
            ax.plot(xs, np.polyval(z, xs), color=REGIME_COLORS[regime], linestyle="--", alpha=0.7)

    ax.set_title("Row-aisle congestion vs machine utilization")
    _save(fig, out)


# ─────────────────────────────────────────────────────────────────────────────
#  Cross-domain combined figures  (23–25)
# ─────────────────────────────────────────────────────────────────────────────

def fig23_cascade_chain(mach: pd.DataFrame, agv: pd.DataFrame, seg: pd.DataFrame,
                        results: Optional[pd.DataFrame], out: Path):
    order = _instance_order(mach["instance"])
    active_regimes = mach["regime"].unique()

    mach_g = mach.groupby(["regime","instance"], observed=True)["utilization_rate"].mean().reset_index()
    agv_g  = agv.groupby(["regime","instance"],  observed=True)["congestion_fraction"].mean().reset_index()
    
    panels = [
        (mach_g, "utilization_rate", "Machine utilization", False),
        (agv_g,  "congestion_fraction", "AGV congestion fraction", False),
    ]

    if not seg.empty:
        seg_g  = (seg[seg["aisle_type"]=="RowAisle"]
                  .groupby(["regime","instance"], observed=True)["block_rate"].mean().reset_index())
        panels.append((seg_g, "block_rate", "Row-aisle block rate", False))

    # Only include makespan inflation panel if BOTH datasets are active
    if results is not None and "deterministic" in active_regimes and "stochastic_low" in active_regimes:
        det = (results[results["regime"]=="deterministic"].groupby("instance", observed=True)["makespan"].mean().reset_index().rename(columns={"makespan":"ms_det"}))
        sto = (results[results["regime"]=="stochastic_low"].groupby("instance", observed=True)["makespan"].mean().reset_index().rename(columns={"makespan":"ms_sto"}))
        inf = det.merge(sto, on="instance", how="inner")
        if not inf.empty:
            inf["inflation"] = inf["ms_sto"] / inf["ms_det"]
            inf["regime"] = "stochastic_low"
            panels.append((inf, "inflation", "Makespan inflation ratio", True))

    n_panels = len(panels)
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 4 * n_panels), sharex=True)
    if n_panels == 1: axes = [axes]

    for ax, (df, col, title, single_regime) in zip(axes, panels):
        if single_regime:
            sub = df.set_index("instance")[col].reindex(order)
            ax.bar(range(len(order)), sub.values, color=REGIME_COLORS["stochastic_low"], alpha=0.7, width=0.6)
            ax.axhline(1, color="black", linestyle="--", alpha=0.4)
        else:
            for regime in active_regimes:
                sub = df[df["regime"]==regime].set_index("instance")[col].reindex(order)
                ax.plot(range(len(order)), sub.values, "o-", color=REGIME_COLORS[regime], label=regime, linewidth=1.5, markersize=5, alpha=0.8)
            ax.legend(fontsize=8)
        ax.set_ylabel(title, fontsize=9)
        ax.set_title(title)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=45, ha="right", fontsize=8)

    fig.tight_layout()
    _save(fig, out)


def fig24_pdr_brittleness_vs_congestion(agv: pd.DataFrame, mach: pd.DataFrame, out: Path):
    agv_g = agv.groupby(["regime","instance","rule"], observed=True).agg(congestion=("congestion_fraction","mean")).reset_index()
    mach_g = mach.groupby(["regime","instance","rule"], observed=True).agg(util=("utilization_rate","mean")).reset_index()

    merged = agv_g.merge(mach_g, on=["regime","instance","rule"], how="inner")
    if merged.empty: return

    active_regimes = merged["regime"].unique()
    fig, axes = plt.subplots(1, len(active_regimes), figsize=(8 * len(active_regimes), 6))
    if len(active_regimes)==1: axes=[axes]

    for ax, regime in zip(axes, active_regimes):
        sub = merged[merged["regime"]==regime]
        sns.scatterplot(data=sub, x="congestion", y="util", hue="rule", ax=ax, alpha=0.7, s=70)
        ax.set_title(f"PDR: congestion vs utilization — {regime}")
        ax.set_xlabel("Mean AGV congestion fraction")
        ax.set_ylabel("Mean machine utilization")
        ax.axvline(0.30, color="red",  linestyle="--", alpha=0.4)
        if ax.get_legend():
            sns.move_legend(ax,"upper left",bbox_to_anchor=(1.02,1),fontsize=8)

    _save(fig, out)


def fig25_summary_table(mach: pd.DataFrame, agv: pd.DataFrame,
                        seg: pd.DataFrame, results: Optional[pd.DataFrame],
                        out_csv: Path):
    rows = []
    instances = _instance_order(mach["instance"])
    active_regimes = mach["regime"].unique()

    for regime in active_regimes:
        for inst in instances:
            row = {"regime": regime, "instance": str(inst)}

            m = mach[(mach["regime"]==regime) & (mach["instance"].astype(str)==str(inst))]
            if not m.empty:
                per_m = m.groupby("machine_id")["utilization_rate"].mean()
                row["mean_util"]       = round(per_m.mean(), 3)
                row["frac_below_5pct"] = round((per_m < 0.05).mean(), 3)
                row["mean_makespan"]   = round(m["makespan"].mean(), 1)

            a = agv[(agv["regime"]==regime) & (agv["instance"].astype(str)==str(inst))]
            if not a.empty:
                row["mean_congestion"]    = round(a["congestion_fraction"].mean(), 3)
                row["mean_trip_dur"]      = round(a["mean_trip_duration"].mean(), 1)
                row["mean_reroutes"]      = round(a["reroute_count"].mean(), 2)
                row["mean_time_waiting"]  = round(a["time_waiting_route"].mean(), 1)
                row["mean_time_idle"]     = round(a["time_idle"].mean(), 1)

            if not seg.empty:
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
    return summary


# ─────────────────────────────────────────────────────────────────────────────
#  Main Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="DFJSP Phase-2 analysis suite")
    p.add_argument("--machine",  nargs="+", metavar="CSV")
    p.add_argument("--results",  nargs="+", metavar="CSV")
    p.add_argument("--agv",      nargs="+", metavar="CSV")
    p.add_argument("--segments", nargs="+", metavar="CSV")
    p.add_argument("--out", default="./figs")
    args = p.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    mach    = _load_pair(args.machine  or [])
    results = _load_pair(args.results  or [])
    agv     = _load_pair(args.agv      or [])
    seg     = _load_pair(args.segments or [])

    present = {k: v is not None for k, v in [("machine",mach),("results",results),("agv",agv),("segments",seg)]}
    print("Loaded domains:", {k for k,v in present.items() if v})

    if present["agv"]:
        print("\n── AGV figures ──")
        fig11_agv_time_budget_stacked(agv,  out/"11_agv_time_budget.png")
        fig12_congestion_fraction(agv,      out/"12_congestion_fraction.png")
        fig13_trip_duration_vs_makespan(agv, results, out/"13_trip_vs_makespan.png")
        fig14_agv_idle_vs_congestion(agv,   out/"14_idle_vs_congestion.png")
        fig15_fleet_size_vs_congestion(agv, results, out/"15_fleet_vs_congestion.png")
        fig16_reroute_analysis(agv,         out/"16_reroutes.png")
        fig17_path_length_comparison(agv,   out/"17_path_length.png")

    if present["segments"]:
        print("\n── Segment congestion figures ──")
        fig18_top_bottlenecks(seg,          out/"18_top_bottlenecks.png")
        fig19_segment_heatmap(seg,          out/"19_segment_heatmap.png")
        fig20_aisle_type_comparison(seg,    out/"20_aisle_type.png")
        fig21_spine_vs_row_entry(seg,       out/"21_spine_entry.png")
        if present["machine"]:
            fig22_congestion_vs_machine_utilization(seg, mach, out/"22_congestion_vs_util.png")

    if present["machine"] and present["agv"]:
        print("\n── Cross-domain figures ──")
        fig23_cascade_chain(mach, agv, seg if present["segments"] else pd.DataFrame(), results, out/"23_cascade_chain.png")
        fig24_pdr_brittleness_vs_congestion(agv, mach, out/"24_pdr_vs_congestion.png")
        fig25_summary_table(mach, agv, seg if present["segments"] else pd.DataFrame(), results, out/"25_summary_table.csv")

    print(f"\nDone. Outputs written to {out}/")

if __name__ == "__main__":
    main()