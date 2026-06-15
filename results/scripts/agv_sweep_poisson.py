"""
random_agv_sweep_analysis.py
=====================
AGV fleet-size sweep analysis for the DFJSP-AGV simulation (Random Instances).

Answers two primary questions:
  1. At what fleet size does makespan start to degrade due to congestion?
  2. What is the recommended AGV count for each generated instance size?

Expected directory layout:
    <sweep-dir>/
      30j_10m_agv10/
        none/
          merged_results.csv
          ...
      30j_15m_agv12/
      30j_15m_agv15_msweep/
      ...

Usage:
    python random_agv_sweep_analysis.py --sweep-dir my_random_sweeps/ [--out figs_agv_sweep/]
    python random_agv_sweep_analysis.py --sweep-dir my_random_sweeps/ --instances 30j_10m 50j_15m

Output files:
    01_makespan_vs_agv.png          — makespan vs fleet size per instance (with congestion overlay)
    02_makespan_per_pdr.png         — per-PDR makespan curves across fleet sizes
    03_agv_time_budget.png          — stacked time-budget evolution as fleet grows
    04_congestion_vs_agv.png        — congestion fraction per instance with onset annotations
    05_machine_util_vs_agv.png      — machine utilization and starvation rate vs fleet size
    06_segment_heatmap.png          — zone-level block-rate heatmap across AGV counts
    07_reroutes_vs_agv.png          — reroute count evolution (early congestion signal)
    08_congestion_makespan_scatter  — congestion × makespan coloured by fleet size
    09_pdr_rank_stability.png       — how PDR rankings shift across fleet sizes
    10_summary_panel.png            — 4-panel overview
    recommended_agv_counts.csv      — per-instance recommended fleet size table
"""

from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)

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

CONGESTION_WARN  = 0.20   # first threshold — congestion is building
CONGESTION_ALARM = 0.35   # second threshold — floor is congestion-limited

# ── Instance label normalisation ───────────────────────────────────────────────

def _instance_order(vals) -> list[str]:
    """Sort instances logically by jobs then machines (e.g., 30j_5m, 30j_10m, 50j_15m)."""
    present = set(pd.Series(vals).dropna().unique())
    
    def parse_jm(s):
        m = re.match(r'^(\d+)j_(\d+)m', s)
        if m:
            return (int(m.group(1)), int(m.group(2)), s)
        return (999, 999, s)
        
    return sorted(list(present), key=parse_jm)


# ── Data loading ───────────────────────────────────────────────────────────────

CSV_NAMES = {
    "results":  "merged_results.csv",
    "machine":  "merged_machine_utilization.csv",
    "agv":      "merged_agv_performance.csv",
    "segments": "merged_segment_congestion.csv",
}


def load_sweep(sweep_dir: str | Path) -> dict[str, Optional[pd.DataFrame]]:
    """
    Discover matching instance/agv subdirectories under sweep_dir, load all four CSV types,
    tag each row with agv_count and instance group (from the folder name), and return a dict
    keyed by domain name.
    """
    sweep_dir = Path(sweep_dir)
    buckets: dict[str, list[pd.DataFrame]] = {k: [] for k in CSV_NAMES}

    # Find directories matching patterns like 30j_15m_agv10
    target_dirs = []
    for d in sweep_dir.iterdir():
        if not d.is_dir(): continue
        # Matches <jobs>j_<machines>m_...agv<count>
        m = re.search(r'^(\d+j_\d+m).*?agv(\d+)', d.name)
        if m:
            inst_name = m.group(1)
            agv_n = int(m.group(2))
            target_dirs.append((d, inst_name, agv_n))

    if not target_dirs:
        raise FileNotFoundError(f"No instance/AGV subdirectories found in {sweep_dir}. "
                                 "Expected folders like 30j_10m_agv10/, 50j_15m_agv15_msweep/, etc.")

    print(f"Discovered {len(target_dirs)} valid instance sweep folders.")

    for d, inst_name, agv_n in target_dirs:
        for key, fname in CSV_NAMES.items():
            fpath = d / 'none' / fname
            if not fpath.exists():
                print(f"  [warn] Not found: {fpath}")
                continue

            df = pd.read_csv(fpath)
            # Override CSV values with folder-derived truths
            df["agv_count"] = agv_n
            df["instance"]  = inst_name

            # Cross-check agvCount column (results CSV only)
            if key == "results" and "agvCount" in df.columns:
                mismatch = df[df["agvCount"].notna() & (df["agvCount"] != agv_n)]
                if not mismatch.empty:
                    found_vals = mismatch["agvCount"].unique()
                    print(f"  [warn] {d.name}: agvCount column has values "
                          f"{found_vals} but folder implies {agv_n}. "
                          "Using folder-derived agv_count throughout.")

            buckets[key].append(df)

    result: dict[str, Optional[pd.DataFrame]] = {}
    for key, frames in buckets.items():
        if frames:
            result[key] = pd.concat(frames, ignore_index=True)
            print(f"  Loaded '{key}': {len(result[key]):,} rows "
                  f"across {len(frames)} directory segments")
        else:
            result[key] = None
            print(f"  [warn] No data loaded for domain '{key}' — related figures will be skipped")

    return result


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved → {path.name}")


# ── Elbow / onset detection ────────────────────────────────────────────────────

def _find_elbow(agv_counts: np.ndarray, values: np.ndarray) -> int:
    """
    Kneedle-lite: find the elbow of a monotone curve using the index of
    maximum perpendicular distance from the straight line joining first
    and last points.  Returns the corresponding agv_count value.
    """
    if len(agv_counts) < 3:
        return int(agv_counts[np.argmin(values)])

    x = (agv_counts - agv_counts[0]).astype(float) / max(agv_counts[-1] - agv_counts[0], 1)
    y = (values    - values[0]).astype(float)       / max(np.ptp(values), 1e-9)

    # Perpendicular distance from each point to the line (0,0)→(1, y_end)
    dx, dy = 1.0, float(y[-1] - y[0])
    norm = np.hypot(dx, dy) + 1e-9
    dists = np.abs(dy * x - dx * y + 0) / norm   # simplified for origin line
    return int(agv_counts[np.argmax(dists)])


def _makespan_minimum(agv_counts: np.ndarray, makespan: np.ndarray) -> int:
    return int(agv_counts[np.argmin(makespan)])


def _congestion_onset(agv_counts: np.ndarray, congestion: np.ndarray,
                      threshold: float = CONGESTION_WARN) -> Optional[int]:
    """First AGV count where mean congestion_fraction >= threshold."""
    above = agv_counts[congestion >= threshold]
    return int(above[0]) if len(above) > 0 else None


# ── Figure helpers ─────────────────────────────────────────────────────────────

def _subplot_grid(n: int, max_cols: int = 3):
    """Return (fig, axes_2d, cols, rows) for a grid of n subplots."""
    cols = min(max_cols, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols,
                              figsize=(6.5 * cols, 4.5 * rows),
                              squeeze=False)
    return fig, axes, cols, rows


TAB10 = list(plt.cm.tab10.colors)


# ══════════════════════════════════════════════════════════════════════════════
#  Fig 01 — Makespan vs AGV count, one sub-panel per instance
# ══════════════════════════════════════════════════════════════════════════════

def fig01_makespan_vs_agv(
    results: pd.DataFrame,
    agv:     Optional[pd.DataFrame],
    out:     Path,
    instance_filter: Optional[list[str]] = None,
) -> None:
    instances = _instance_order(results["instance"])
    if instance_filter:
        instances = [i for i in instances if i in instance_filter]

    ms = (results.groupby(["instance", "agv_count"])["makespan"]
          .agg(mean="mean", std="std").reset_index())

    cong = None
    if agv is not None:
        cong = (agv.groupby(["instance", "agv_count"])["congestion_fraction"]
                .mean().reset_index()
                .rename(columns={"congestion_fraction": "congestion"}))

    n = len(instances)
    if n == 0:
        print("[fig01] No instances to plot; skipping."); return

    fig, axes, cols, rows = _subplot_grid(n)

    for idx, inst in enumerate(instances):
        ax = axes[idx // cols][idx % cols]
        sub = ms[ms["instance"] == inst].sort_values("agv_count")
        if sub.empty:
            ax.set_visible(False); continue

        x   = sub["agv_count"].values
        y   = sub["mean"].values
        err = sub["std"].fillna(0).values
        color = TAB10[idx % len(TAB10)]

        ax.plot(x, y, "o-", color=color, linewidth=2, markersize=6, zorder=3)
        ax.fill_between(x, y - err, y + err, alpha=0.15, color=color)

        # Minimum makespan marker
        min_i = int(np.argmin(y))
        ax.axvline(x[min_i], color="#2E8B57", linestyle="--",
                   linewidth=1.4, alpha=0.8, zorder=2)
        ax.annotate(f"min\n{x[min_i]}",
                    xy=(x[min_i], y[min_i]),
                    xytext=(8, 8), textcoords="offset points",
                    color="#2E8B57", fontsize=7, fontweight="bold")

        # Elbow marker
        elbow = _find_elbow(x, y)
        if elbow != x[min_i]:
            ax.axvline(elbow, color="darkorange", linestyle=":",
                       linewidth=1.2, alpha=0.7, zorder=2)
            ax.text(elbow + 0.3, y.max() * 0.97, f"elbow\n{elbow}",
                    color="darkorange", fontsize=6, va="top")

        # Congestion overlay
        if cong is not None:
            sc = cong[cong["instance"] == inst].sort_values("agv_count")
            if not sc.empty:
                ax2 = ax.twinx()
                ax2.plot(sc["agv_count"], sc["congestion"], "s--",
                         color="crimson", linewidth=1.2, markersize=4,
                         alpha=0.7, label="Congestion")
                ax2.axhline(CONGESTION_WARN,  color="crimson", linestyle=":",
                            linewidth=0.8, alpha=0.5)
                ax2.axhline(CONGESTION_ALARM, color="darkred", linestyle=":",
                            linewidth=0.8, alpha=0.4)
                ax2.set_ylabel("Congestion fraction", color="crimson", fontsize=7)
                ax2.tick_params(axis="y", colors="crimson", labelsize=7)
                ax2.set_ylim(0, max(CONGESTION_ALARM * 1.4,
                                    sc["congestion"].max() * 1.3))
                ax2.spines["right"].set_visible(True)
                ax2.spines["right"].set_color("crimson")
                ax2.spines["right"].set_alpha(0.4)

        ax.set_title(inst, fontsize=10, fontweight="bold")
        ax.set_xlabel("AGV fleet size")
        ax.set_ylabel("Makespan (sim-time units)")
        ax.set_xticks(x)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.suptitle(
        "Makespan vs AGV Fleet Size — per Instance\n"
        "Band = ±1 std (rules × seeds)  |  Green dashed = min makespan  "
        "|  Orange dotted = elbow  |  Red overlay = congestion fraction",
        y=1.01, fontsize=10,
    )
    fig.tight_layout()
    _save(fig, out)


# ══════════════════════════════════════════════════════════════════════════════
#  Fig 02 — Per-PDR makespan curves
# ══════════════════════════════════════════════════════════════════════════════

def fig02_makespan_per_pdr(
    results:           pd.DataFrame,
    out:               Path,
    instance_filter:   Optional[list[str]] = None,
) -> None:
    instances = _instance_order(results["instance"])
    if instance_filter:
        instances = [i for i in instances if i in instance_filter]

    rules   = sorted(results["rule"].unique())
    palette = dict(zip(rules, TAB10[:len(rules)]))

    g = (results.groupby(["instance", "rule", "agv_count"])["makespan"]
         .mean().reset_index())

    n = len(instances)
    if n == 0:
        print("[fig02] No instances to plot; skipping."); return

    fig, axes, cols, rows = _subplot_grid(n)

    for idx, inst in enumerate(instances):
        ax    = axes[idx // cols][idx % cols]
        sub   = g[g["instance"] == inst]
        if sub.empty:
            ax.set_visible(False); continue

        for rule in rules:
            r = sub[sub["rule"] == rule].sort_values("agv_count")
            if r.empty: continue
            ax.plot(r["agv_count"], r["makespan"], "o-",
                    color=palette[rule], linewidth=1.6, markersize=4,
                    label=rule, alpha=0.85)

        ax.set_title(inst, fontsize=10, fontweight="bold")
        ax.set_xlabel("AGV fleet size")
        ax.set_ylabel("Makespan")
        ax.set_xticks(sorted(sub["agv_count"].unique()))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

        if idx == 0:
            ax.legend(fontsize=6, ncol=2, loc="upper right",
                      framealpha=0.7, title="PDR", title_fontsize=6)

    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.suptitle(
        "Makespan vs AGV Count — Broken Out by PDR\n"
        "(seed-averaged; each line = one dispatching rule)",
        y=1.01, fontsize=10,
    )
    fig.tight_layout()
    _save(fig, out)


# ══════════════════════════════════════════════════════════════════════════════
#  Fig 03 — AGV time-budget evolution
# ══════════════════════════════════════════════════════════════════════════════

def fig03_agv_time_budget(agv: pd.DataFrame, out: Path) -> None:
    time_cols = [
        "time_idle", "time_waiting_route", "time_traveling",
        "time_loading", "time_unloading",
    ]
    colors = ["#AAAAAA", "#DC143C", "#2E8B57", "#F4A460", "#4682B4"]
    labels = ["Idle", "Waiting (blocked)", "Traveling", "Loading", "Unloading"]

    present = [c for c in time_cols if c in agv.columns]
    if not present:
        print("[fig03] No time-state columns found in AGV data; skipping.")
        return

    g     = agv.groupby("agv_count")[present].mean()
    tot   = g.sum(axis=1).replace(0, np.nan)
    g_frc = g.div(tot, axis=0)

    agv_counts = sorted(g_frc.index)
    bar_width  = min(2.5, (agv_counts[-1] - agv_counts[0]) / max(len(agv_counts), 2) * 0.7)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left: fractional stacked bar ─────────────────────────────────────────
    ax = axes[0]
    bottoms = np.zeros(len(agv_counts))
    for col, color, label in zip(present, colors, labels):
        if col not in g_frc.columns: continue
        vals = g_frc.loc[agv_counts, col].fillna(0).values
        ax.bar(agv_counts, vals, bottom=bottoms, color=color,
               label=label, edgecolor="white", linewidth=0.5,
               width=bar_width, zorder=2)
        bottoms += vals

    ax.axhline(CONGESTION_WARN,  color="crimson", linestyle="--",
               linewidth=1, alpha=0.6, zorder=3)
    ax.axhline(CONGESTION_ALARM, color="darkred",  linestyle=":",
               linewidth=0.8, alpha=0.5, zorder=3)
    ax.text(agv_counts[-1] + bar_width * 0.6, CONGESTION_WARN + 0.01,
            f"{int(CONGESTION_WARN*100)}% warn", fontsize=7, color="crimson")
    ax.text(agv_counts[-1] + bar_width * 0.6, CONGESTION_ALARM + 0.01,
            f"{int(CONGESTION_ALARM*100)}% alarm", fontsize=7, color="darkred")

    ax.set_xlabel("AGV fleet size")
    ax.set_ylabel("Fraction of mean AGV episode time")
    ax.set_title("AGV time-budget breakdown vs fleet size")
    ax.set_xticks(agv_counts)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.8)

    # ── Right: absolute waiting vs idle ──────────────────────────────────────
    ax2 = axes[1]
    if "time_waiting_route" in g.columns:
        ax2.plot(agv_counts, g.loc[agv_counts, "time_waiting_route"].fillna(0),
                 "o-", color="#DC143C", linewidth=2, markersize=6,
                 label="Waiting (blocked)", zorder=3)
    if "time_idle" in g.columns:
        ax2.plot(agv_counts, g.loc[agv_counts, "time_idle"].fillna(0),
                 "s--", color="#888888", linewidth=2, markersize=6,
                 label="Idle", zorder=3)
    if "time_traveling" in g.columns:
        ax2.plot(agv_counts, g.loc[agv_counts, "time_traveling"].fillna(0),
                 "^:", color="#2E8B57", linewidth=1.5, markersize=5,
                 label="Traveling", alpha=0.8, zorder=3)

    ax2.set_xlabel("AGV fleet size")
    ax2.set_ylabel("Mean AGV time (sim-time units)")
    ax2.set_title("Absolute waiting vs idle per AGV as fleet grows")
    ax2.set_xticks(agv_counts)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax2.legend(fontsize=8)

    fig.suptitle(
        "AGV Time-Budget Evolution with Fleet Size\n"
        "Congestion signature: idle ↓ and blocked-waiting ↑ simultaneously",
        y=1.03, fontsize=11,
    )
    fig.tight_layout()
    _save(fig, out)


# ══════════════════════════════════════════════════════════════════════════════
#  Fig 04 — Congestion fraction per instance vs AGV count
# ══════════════════════════════════════════════════════════════════════════════

def fig04_congestion_vs_agv(agv: pd.DataFrame, out: Path) -> list[dict]:
    g = (agv.groupby(["instance", "agv_count"])["congestion_fraction"]
         .agg(mean="mean", std="std").reset_index())
    instances  = _instance_order(g["instance"])

    fig, ax = plt.subplots(figsize=(12, 6))
    onset_records: list[dict] = []

    for idx, inst in enumerate(instances):
        sub = g[g["instance"] == inst].sort_values("agv_count")
        if sub.empty: continue

        x     = sub["agv_count"].values
        y     = sub["mean"].fillna(0).values
        color = TAB10[idx % len(TAB10)]

        ax.plot(x, y, "o-", color=color, linewidth=1.8, markersize=5,
                label=inst, zorder=3)
        ax.fill_between(x, y - sub["std"].fillna(0).values,
                             y + sub["std"].fillna(0).values,
                        alpha=0.08, color=color)

        onset = _congestion_onset(x, y, threshold=CONGESTION_WARN)
        if onset is not None:
            onset_records.append({"instance": inst, "congestion_onset_agv": onset})
            ax.axvline(onset, color=color, linestyle=":", linewidth=0.9, alpha=0.55)

    ax.axhline(CONGESTION_WARN,  color="crimson", linestyle="--",
               linewidth=1.3, alpha=0.7, label=f"{int(CONGESTION_WARN*100)}% warn threshold")
    ax.axhline(CONGESTION_ALARM, color="darkred", linestyle=":",
               linewidth=1.0, alpha=0.5, label=f"{int(CONGESTION_ALARM*100)}% alarm threshold")

    ax.set_xlabel("AGV fleet size")
    ax.set_ylabel("Mean congestion fraction per AGV")
    ax.set_title(
        f"Congestion Fraction vs AGV Fleet Size — per Instance\n"
        f"Dotted verticals = per-instance {int(CONGESTION_WARN*100)}% onset crossing"
    )
    ax.set_xticks(sorted(agv["agv_count"].unique()))
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left", framealpha=0.8)
    fig.tight_layout()
    _save(fig, out)
    return onset_records


# ══════════════════════════════════════════════════════════════════════════════
#  Fig 05 — Machine utilization & starvation vs AGV count
# ══════════════════════════════════════════════════════════════════════════════

def fig05_machine_util_vs_agv(machine: pd.DataFrame, out: Path) -> None:
    machine["utilization_rate"] = pd.to_numeric(machine["utilization_rate"], errors="coerce")

    ep = (machine.groupby(["instance", "rule", "seed", "agv_count"])
          .agg(
              mean_util    = ("utilization_rate", "mean"),
              frac_starved = ("utilization_rate", lambda x: float((x < 0.05).mean())),
          ).reset_index())

    g = (ep.groupby(["instance", "agv_count"])
         .agg(
             mean_util    = ("mean_util",    "mean"),
             std_util     = ("mean_util",    "std"),
             frac_starved = ("frac_starved", "mean"),
         ).reset_index())

    instances = _instance_order(g["instance"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, inst in enumerate(instances):
        sub   = g[g["instance"] == inst].sort_values("agv_count")
        if sub.empty: continue
        color = TAB10[idx % len(TAB10)]
        x     = sub["agv_count"].values

        axes[0].plot(x, sub["mean_util"].values, "o-",
                     color=color, linewidth=1.8, markersize=5, label=inst)
        axes[0].fill_between(x,
            sub["mean_util"] - sub["std_util"].fillna(0),
            sub["mean_util"] + sub["std_util"].fillna(0),
            alpha=0.07, color=color)

        axes[1].plot(x, sub["frac_starved"].values, "o-",
                     color=color, linewidth=1.8, markersize=5, label=inst)

    axes[0].axhline(0.60, color="steelblue", linestyle="--",
                    linewidth=0.9, alpha=0.5, label="60% utilization target")
    axes[0].set_ylabel("Mean machine utilization rate")
    axes[0].set_xlabel("AGV fleet size")
    axes[0].set_title("Machine utilization vs fleet size")
    axes[0].set_xticks(sorted(machine["agv_count"].unique()))
    axes[0].legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")

    axes[1].axhline(0.10, color="orange", linestyle="--",
                    linewidth=0.9, alpha=0.5, label="10% starvation ceiling")
    axes[1].set_ylabel("Fraction of machines with utilization < 5%")
    axes[1].set_xlabel("AGV fleet size")
    axes[1].set_title("Machine starvation rate vs fleet size")
    axes[1].set_xticks(sorted(machine["agv_count"].unique()))
    axes[1].legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")

    fig.suptitle(
        "Machine Utilization & Starvation vs AGV Fleet Size\n"
        "Adding AGVs should relieve starvation — plateau → transport saturation",
        y=1.03, fontsize=11,
    )
    fig.tight_layout()
    _save(fig, out)


# ══════════════════════════════════════════════════════════════════════════════
#  Fig 06 — Segment congestion heatmap evolution
# ══════════════════════════════════════════════════════════════════════════════

def fig06_segment_heatmap(seg: pd.DataFrame, out: Path) -> None:
    seg["block_rate"] = pd.to_numeric(seg["block_rate"], errors="coerce")
    aisle_types = sorted(seg["aisle_type"].dropna().unique())

    if not aisle_types:
        aisle_types = ["all"]
        seg = seg.copy()
        seg["aisle_type"] = "all"

    n = len(aisle_types)
    fig, axes = plt.subplots(1, n, figsize=(max(7, 8 * n), 8), squeeze=False)

    for ax, atype in zip(axes[0], aisle_types):
        sub = seg if atype == "all" else seg[seg["aisle_type"] == atype]
        pivot = (sub.groupby(["zone_name", "agv_count"])["block_rate"]
                 .mean()
                 .unstack("agv_count"))
        pivot = pivot.reindex(columns=sorted(pivot.columns))

        if pivot.empty:
            ax.set_visible(False); continue

        pivot = pivot.loc[pivot.max(axis=1).sort_values(ascending=False).index]

        display_rows = min(len(pivot), 30)
        pivot_disp   = pivot.iloc[:display_rows]

        annot = display_rows <= 20
        sns.heatmap(
            pivot_disp, ax=ax,
            cmap="YlOrRd",
            vmin=0, vmax=max(pivot_disp.values.max(), 0.01),
            cbar_kws={"label": "Mean block rate"},
            linewidths=0.2, linecolor="white",
            annot=annot, fmt=".2f" if annot else "",
        )
        title = f"{'All aisles' if atype == 'all' else atype} — block rate evolution"
        if len(pivot) > display_rows:
            title += f"\n(top {display_rows} of {len(pivot)} zones by peak block rate)"
        ax.set_title(title)
        ax.set_xlabel("AGV fleet size")
        ax.set_ylabel("Zone")
        ax.tick_params(axis="y", labelsize=7)
        ax.tick_params(axis="x", rotation=0)

    fig.suptitle(
        "Segment Congestion Heatmap Across Fleet Sizes\n"
        "Zones sorted by peak block rate — track where congestion begins",
        y=1.02, fontsize=11,
    )
    fig.tight_layout()
    _save(fig, out)


# ══════════════════════════════════════════════════════════════════════════════
#  Fig 07 — Reroute count vs AGV count
# ══════════════════════════════════════════════════════════════════════════════

def fig07_reroutes_vs_agv(agv: pd.DataFrame, out: Path) -> None:
    if "reroute_count" not in agv.columns:
        print("[fig07] reroute_count column absent; skipping."); return

    g = (agv.groupby(["instance", "agv_count"])["reroute_count"]
         .agg(mean="mean", std="std").reset_index())
    instances = _instance_order(g["instance"])

    fig, ax = plt.subplots(figsize=(11, 5))

    for idx, inst in enumerate(instances):
        sub   = g[g["instance"] == inst].sort_values("agv_count")
        if sub.empty: continue
        color = TAB10[idx % len(TAB10)]
        x, y  = sub["agv_count"].values, sub["mean"].fillna(0).values
        ax.plot(x, y, "o-", color=color, linewidth=1.8, markersize=5, label=inst)
        ax.fill_between(x, y - sub["std"].fillna(0).values,
                              y + sub["std"].fillna(0).values,
                         alpha=0.08, color=color)

    ax.set_xlabel("AGV fleet size")
    ax.set_ylabel("Mean reroute count per AGV per episode")
    ax.set_title(
        "Reroute Count vs AGV Fleet Size\n"
        "Rising reroutes = early path-conflict signal ahead of measurable congestion"
    )
    ax.set_xticks(sorted(agv["agv_count"].unique()))
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    _save(fig, out)


# ══════════════════════════════════════════════════════════════════════════════
#  Fig 08 — Congestion × makespan scatter, coloured by fleet size
# ══════════════════════════════════════════════════════════════════════════════

def fig08_congestion_makespan_scatter(
    results: pd.DataFrame,
    agv:     pd.DataFrame,
    out:     Path,
) -> None:
    ms   = (results.groupby(["instance", "agv_count"])["makespan"]
            .mean().reset_index()
            .rename(columns={"makespan": "mean_makespan"}))
    cong = (agv.groupby(["instance", "agv_count"])["congestion_fraction"]
            .mean().reset_index()
            .rename(columns={"congestion_fraction": "mean_congestion"}))
    merged = ms.merge(cong, on=["instance", "agv_count"])
    if merged.empty:
        print("[fig08] No matching (instance, agv_count) pairs; skipping."); return

    agv_counts = sorted(merged["agv_count"].unique())
    norm       = mpl.colors.Normalize(vmin=min(agv_counts), vmax=max(agv_counts))
    cmap       = plt.cm.viridis

    fig, ax = plt.subplots(figsize=(11, 7))

    for inst in _instance_order(merged["instance"]):
        sub = merged[merged["instance"] == inst].sort_values("agv_count")
        if sub.empty: continue
        ax.plot(sub["mean_congestion"], sub["mean_makespan"],
                "-", color="lightgray", linewidth=0.8, alpha=0.6, zorder=1)
        sc = ax.scatter(
            sub["mean_congestion"], sub["mean_makespan"],
            c=sub["agv_count"], cmap="viridis",
            vmin=min(agv_counts), vmax=max(agv_counts),
            s=60, alpha=0.85, edgecolors="white", linewidths=0.5, zorder=2,
        )
        last = sub.iloc[-1]
        ax.text(last["mean_congestion"] + 0.003, last["mean_makespan"],
                inst, fontsize=6.5, va="center", alpha=0.75)

    ax.axvline(CONGESTION_WARN,  color="crimson", linestyle="--",
               linewidth=1, alpha=0.5, label=f"{int(CONGESTION_WARN*100)}% congestion warn")
    ax.axvline(CONGESTION_ALARM, color="darkred",  linestyle=":",
               linewidth=0.9, alpha=0.4, label=f"{int(CONGESTION_ALARM*100)}% alarm")

    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("AGV fleet size")
    ax.set_xlabel("Mean AGV congestion fraction")
    ax.set_ylabel("Mean makespan (sim-time units)")
    ax.set_title(
        "Congestion vs Makespan Across Fleet Sizes\n"
        "Colour = fleet size  |  Lines connect sweep steps per instance"
    )
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    _save(fig, out)


# ══════════════════════════════════════════════════════════════════════════════
#  Fig 09 — PDR rank stability across fleet sizes
# ══════════════════════════════════════════════════════════════════════════════

def fig09_pdr_rank_stability(results: pd.DataFrame, out: Path) -> None:
    g = (results.groupby(["instance", "agv_count", "rule"])["makespan"]
         .mean().reset_index())
    g["rank"] = g.groupby(["instance", "agv_count"])["makespan"].rank(method="min")

    rank_mean = (g.groupby(["rule", "agv_count"])["rank"]
                 .mean().reset_index())

    rules   = sorted(results["rule"].unique())
    palette = dict(zip(rules, TAB10[:len(rules)]))

    fig, ax = plt.subplots(figsize=(11, 5))

    for rule in rules:
        sub = rank_mean[rank_mean["rule"] == rule].sort_values("agv_count")
        ax.plot(sub["agv_count"], sub["rank"], "o-",
                color=palette[rule], linewidth=1.8, markersize=6, label=rule)

    ax.invert_yaxis()  # rank 1 at top
    ax.set_xlabel("AGV fleet size")
    ax.set_ylabel("Mean rank across instances (1 = best makespan)")
    ax.set_title(
        "PDR Rank Stability Across AGV Fleet Sizes\n"
        "Rank 1 = best (top of chart)  |  Worsening rank at high fleet = congestion-sensitive"
    )
    ax.set_xticks(sorted(results["agv_count"].unique()))
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    _save(fig, out)


# ══════════════════════════════════════════════════════════════════════════════
#  Fig 10 — 4-panel summary overview
# ══════════════════════════════════════════════════════════════════════════════

def fig10_summary_panel(
    results: pd.DataFrame,
    agv:     Optional[pd.DataFrame],
    machine: Optional[pd.DataFrame],
    out:     Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    instances = _instance_order(results["instance"])
    agv_counts_x = sorted(results["agv_count"].unique())

    ms_g = (results.groupby(["instance", "agv_count"])["makespan"]
            .mean().reset_index())

    # ── TL: normalised makespan ───────────────────────────────────────────────
    ax = axes[0, 0]
    for idx, inst in enumerate(instances):
        sub      = ms_g[ms_g["instance"] == inst].sort_values("agv_count")
        if sub.empty: continue
        baseline = sub.iloc[0]["makespan"]
        if baseline == 0: continue
        ax.plot(sub["agv_count"], sub["makespan"] / baseline,
                "o-", color=TAB10[idx % len(TAB10)], linewidth=1.5,
                markersize=4, label=inst, alpha=0.85)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.4,
               label="Baseline (smallest fleet)")
    ax.set_xlabel("AGV fleet size")
    ax.set_ylabel("Normalised makespan (÷ smallest fleet)")
    ax.set_title("Relative makespan improvement vs fleet size")
    ax.set_xticks(agv_counts_x)
    ax.legend(fontsize=6, ncol=2, loc="upper right", framealpha=0.7)

    # ── TR: congestion ────────────────────────────────────────────────────────
    ax = axes[0, 1]
    if agv is not None:
        cong_g = (agv.groupby(["instance", "agv_count"])["congestion_fraction"]
                  .mean().reset_index())
        for idx, inst in enumerate(instances):
            sub = cong_g[cong_g["instance"] == inst].sort_values("agv_count")
            if sub.empty: continue
            ax.plot(sub["agv_count"], sub["congestion_fraction"],
                    "o-", color=TAB10[idx % len(TAB10)], linewidth=1.5,
                    markersize=4, label=inst, alpha=0.85)
        ax.axhline(CONGESTION_WARN,  color="crimson", linestyle="--",
                   linewidth=1, alpha=0.6, label=f"{int(CONGESTION_WARN*100)}% warn")
        ax.axhline(CONGESTION_ALARM, color="darkred",  linestyle=":",
                   linewidth=0.9, alpha=0.4, label=f"{int(CONGESTION_ALARM*100)}% alarm")
        ax.set_ylabel("Mean congestion fraction")
        ax.legend(fontsize=6, ncol=2, loc="upper left", framealpha=0.7)
    else:
        ax.text(0.5, 0.5, "AGV data not loaded", ha="center", va="center",
                transform=ax.transAxes, color="gray", fontsize=10)
    ax.set_xlabel("AGV fleet size")
    ax.set_title("Congestion fraction vs fleet size")
    ax.set_xticks(agv_counts_x)

    # ── BL: machine utilization ───────────────────────────────────────────────
    ax = axes[1, 0]
    if machine is not None:
        machine["utilization_rate"] = pd.to_numeric(machine["utilization_rate"], errors="coerce")
        util_g = (machine.groupby(["instance", "agv_count"])["utilization_rate"]
                  .mean().reset_index())
        for idx, inst in enumerate(instances):
            sub = util_g[util_g["instance"] == inst].sort_values("agv_count")
            if sub.empty: continue
            ax.plot(sub["agv_count"], sub["utilization_rate"],
                    "o-", color=TAB10[idx % len(TAB10)], linewidth=1.5,
                    markersize=4, label=inst, alpha=0.85)
        ax.axhline(0.60, color="steelblue", linestyle="--",
                   linewidth=0.9, alpha=0.5, label="60% target")
        ax.set_ylabel("Mean machine utilization rate")
        ax.legend(fontsize=6, ncol=2, loc="lower right", framealpha=0.7)
    else:
        ax.text(0.5, 0.5, "Machine data not loaded", ha="center", va="center",
                transform=ax.transAxes, color="gray", fontsize=10)
    ax.set_xlabel("AGV fleet size")
    ax.set_title("Machine utilization vs fleet size")
    ax.set_xticks(agv_counts_x)

    # ── BR: PDR spread (coefficient of variation) ─────────────────────────────
    ax = axes[1, 1]
    cv_g = (results.groupby(["instance", "agv_count", "rule"])["makespan"]
            .mean().reset_index()
            .groupby(["instance", "agv_count"])["makespan"]
            .agg(cv=lambda x: (x.std() / x.mean()) if x.mean() > 0 else 0)
            .reset_index())
    for idx, inst in enumerate(instances):
        sub = cv_g[cv_g["instance"] == inst].sort_values("agv_count")
        if sub.empty: continue
        ax.plot(sub["agv_count"], sub["cv"],
                "o-", color=TAB10[idx % len(TAB10)], linewidth=1.5,
                markersize=4, label=inst, alpha=0.85)
    ax.set_xlabel("AGV fleet size")
    ax.set_ylabel("CV of makespan across PDRs")
    ax.set_title("PDR spread — does fleet size compress rule sensitivity?")
    ax.set_xticks(agv_counts_x)
    ax.legend(fontsize=6, ncol=2, loc="upper right", framealpha=0.7)

    fig.suptitle("AGV Fleet-Size Sweep — Summary Overview", fontsize=13, y=1.01)
    fig.tight_layout()
    _save(fig, out)


# ══════════════════════════════════════════════════════════════════════════════
#  Recommendation table
# ══════════════════════════════════════════════════════════════════════════════

def make_recommendation_table(
    results: pd.DataFrame,
    agv:     Optional[pd.DataFrame],
    out_csv: Path,
) -> pd.DataFrame:
    ms_g = (results.groupby(["instance", "agv_count"])["makespan"]
            .mean().reset_index())

    cong_g: Optional[pd.DataFrame] = None
    if agv is not None:
        cong_g = (agv.groupby(["instance", "agv_count"])["congestion_fraction"]
                  .mean().reset_index())

    rows = []
    for inst in _instance_order(ms_g["instance"]):
        sub = ms_g[ms_g["instance"] == inst].sort_values("agv_count")
        if sub.empty: continue

        x, y  = sub["agv_count"].values, sub["makespan"].values
        opt   = _makespan_minimum(x, y)
        elbow = _find_elbow(x, y)

        pct_gain = 100.0 * (y.max() - y.min()) / y.max() if y.max() > 0 else 0.0

        if y[-1] < y[0]:
            regime = "monotone-decreasing (add more AGVs)"
        elif y[-1] > y[0] and np.argmin(y) not in (0, len(y) - 1):
            regime = "has-minimum"
        elif y[-1] > y[0]:
            regime = "monotone-increasing (congestion-dominated)"
        else:
            regime = "flat"

        onset = None
        if cong_g is not None:
            sc = cong_g[cong_g["instance"] == inst].sort_values("agv_count")
            if not sc.empty:
                onset = _congestion_onset(sc["agv_count"].values,
                                          sc["congestion_fraction"].values,
                                          threshold=CONGESTION_WARN)

        if onset is not None and opt > onset:
            recommended = onset
            note = f"capped at congestion onset (>{int(CONGESTION_WARN*100)}%)"
        elif regime == "monotone-decreasing (add more AGVs)":
            recommended = int(x[-1])
            note = "monotone: use largest tested fleet"
        else:
            recommended = opt
            note = "at makespan minimum"

        rows.append({
            "instance":              inst,
            "agv_at_min_makespan":   opt,
            "elbow_agv":             elbow,
            "makespan_improvement%": round(pct_gain, 1),
            "congestion_onset_agv":  int(onset) if onset is not None else "none",
            "recommended_agv":       recommended,
            "curve_shape":           regime,
            "note":                  note,
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"  saved → {out_csv.name}")

    print("\n" + "═" * 80)
    print("  RECOMMENDED AGV COUNT PER INSTANCE")
    print("═" * 80)
    with pd.option_context("display.max_rows", None, "display.width", 160,
                           "display.max_colwidth", 40):
        print(df.to_string(index=False))
    print("═" * 80)
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    global CONGESTION_WARN, CONGESTION_ALARM

    p = argparse.ArgumentParser(
        description="AGV fleet-size sweep analysis for DFJSP-AGV simulation results (Random Instances).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python random_agv_sweep_analysis.py --sweep-dir Results/random_sweeps/
  python random_agv_sweep_analysis.py --sweep-dir Results/random_sweeps/ --out figs/random_sweeps/
  python random_agv_sweep_analysis.py --sweep-dir Results/random_sweeps/ --instances 30j_10m 50j_15m
        """,
    )
    p.add_argument(
        "--sweep-dir", required=True, metavar="DIR",
        help="Root directory containing random instance subdirectories (e.g. 30j_10m_agv10/).",
    )
    p.add_argument(
        "--out", default="figs_agv_sweep", metavar="DIR",
        help="Output directory for all plots and CSVs (default: figs_agv_sweep/).",
    )
    p.add_argument(
        "--instances", nargs="+", metavar="INST",
        help="Restrict per-instance detail plots to these instances, e.g. 30j_10m 50j_15m. "
             "All instances are always used for aggregate figures.",
    )
    p.add_argument(
        "--congestion-warn", type=float, default=CONGESTION_WARN, metavar="FRAC",
        help=f"Congestion fraction warning threshold (default: {CONGESTION_WARN}).",
    )
    p.add_argument(
        "--congestion-alarm", type=float, default=CONGESTION_ALARM, metavar="FRAC",
        help=f"Congestion fraction alarm threshold (default: {CONGESTION_ALARM}).",
    )
    args = p.parse_args()

    CONGESTION_WARN  = args.congestion_warn
    CONGESTION_ALARM = args.congestion_alarm

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    instance_filter = args.instances if args.instances else None
    if instance_filter:
        print(f"Instance filter: {instance_filter}")

    # ── Load ──────────────────────────────────────────────────────────────────
    print("\n── Loading sweep data ──")
    data    = load_sweep(args.sweep_dir)
    results = data.get("results")
    machine = data.get("machine")
    agv     = data.get("agv")
    seg     = data.get("segments")

    if results is None:
        raise RuntimeError(
            "merged_results.csv could not be loaded from any AGV sweep folder. "
            "This file is required."
        )

    # ── Sanity summary ────────────────────────────────────────────────────────
    print("\n── Data inventory ──")
    print(f"  Instances  : {_instance_order(results['instance'])}")
    print(f"  AGV counts : {sorted(results['agv_count'].unique())}")
    print(f"  Rules      : {sorted(results['rule'].unique())}")
    print(f"  Seeds      : {sorted(results['seed'].unique())}")

    seeds_per = (results.groupby(["instance", "agv_count", "rule"])
                 .size()
                 .agg(["min", "max"]))
    print(f"  Seeds per (instance, agv_count, rule): {seeds_per['min']}–{seeds_per['max']}")

    expected_seeds = sorted(results["seed"].unique())
    combos         = results.groupby(["instance", "agv_count", "rule"]).size()
    short           = combos[combos < len(expected_seeds)]
    if not short.empty:
        print(f"\n  [warn] {len(short)} (instance, agv_count, rule) combos have "
              f"fewer than {len(expected_seeds)} seeds:")
        print("  " + short.reset_index().to_string(index=False))

    # ── Figures ───────────────────────────────────────────────────────────────
    print("\n── Generating figures ──")

    fig01_makespan_vs_agv(results, agv, out / "01_makespan_vs_agv.png",
                          instance_filter=instance_filter)

    fig02_makespan_per_pdr(results, out / "02_makespan_per_pdr.png",
                           instance_filter=instance_filter)

    if agv is not None:
        fig03_agv_time_budget(agv, out / "03_agv_time_budget.png")
        fig04_congestion_vs_agv(agv, out / "04_congestion_vs_agv.png")

    if machine is not None:
        fig05_machine_util_vs_agv(machine, out / "05_machine_util_vs_agv.png")

    if seg is not None:
        fig06_segment_heatmap(seg, out / "06_segment_heatmap.png")

    if agv is not None:
        fig07_reroutes_vs_agv(agv, out / "07_reroutes_vs_agv.png")
        fig08_congestion_makespan_scatter(results, agv,
                                          out / "08_congestion_makespan_scatter.png")

    fig09_pdr_rank_stability(results, out / "09_pdr_rank_stability.png")

    # ── Recommendation table ──────────────────────────────────────────────────
    print()
    make_recommendation_table(results, agv, out / "recommended_agv_counts.csv")

    # ── Summary panel ─────────────────────────────────────────────────────────
    fig10_summary_panel(results, agv, machine, out / "10_summary_panel.png")

    print(f"\nDone. All outputs written to: {out}/")


if __name__ == "__main__":
    main()