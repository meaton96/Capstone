"""
plot_sensitivity.py — Visualizations for the AGV count sensitivity analysis.

Usage:
    python plot_sensitivity.py <results.csv> [--out <output_dir>]
                               [--config 30j/15m]   # filter to one config
                               [--rule SPT_SMPT]     # filter to one rule

Produces:
    1. agv_vs_makespan_lines.png     — line chart: mean makespan vs AGV count per PDR
                                       (faceted by factory config)
    2. agv_improvement_heatmap.png   — heatmap: % improvement from min→max AGVs per
                                       PDR × config
    3. pdr_boxplot_per_agv.png       — box plots of makespan distribution (across seeds)
                                       per AGV count, for the largest config only
    4. agv_vs_makespan_by_config.png — line chart aggregated across PDRs,
                                       one line per factory config
"""

import argparse
import os
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from viz_utils import load_sensitivity, PDR_ORDER

warnings.filterwarnings("ignore", category=FutureWarning)

plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

PALETTE = plt.cm.tab10.colors


def _rule_color(rules):
    return {r: PALETTE[i % len(PALETTE)] for i, r in enumerate(rules)}


# ── Plot 1 — faceted line: makespan vs AGV count per PDR ────────────────────

def plot_agv_lines_faceted(df: pd.DataFrame, out_dir: str) -> None:
    configs = sorted(df["config"].unique(),
                     key=lambda c: (int(c.split("j/")[0]), int(c.split("/")[1].replace("m", ""))))
    rules = [r for r in PDR_ORDER if r in df["rule"].values]
    colors = _rule_color(rules)

    ncols = min(3, len(configs))
    nrows = int(np.ceil(len(configs) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(6 * ncols, 4 * nrows),
                             sharex=False, sharey=False)
    axes = np.array(axes).flatten()

    for ax_i, config in enumerate(configs):
        ax = axes[ax_i]
        sub = df[df["config"] == config]
        for rule in rules:
            rsub = sub[sub["rule"] == rule].groupby("agvCount")["makespan"].mean().reset_index()
            if rsub.empty:
                continue
            ax.plot(rsub["agvCount"], rsub["makespan"],
                    marker="o", label=rule, color=colors[rule],
                    linewidth=1.8, markersize=5, alpha=0.85)
        ax.set_title(config, fontsize=10, fontweight="bold")
        ax.set_xlabel("AGV Count")
        ax.set_ylabel("Mean Makespan")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    # Hide empty subplots
    for ax_i in range(len(configs), len(axes)):
        axes[ax_i].set_visible(False)

    # Single shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="PDR",
               loc="lower center", ncol=min(len(rules), 5),
               bbox_to_anchor=(0.5, -0.02), fontsize=8)

    fig.suptitle("Mean Makespan vs AGV Count by Factory Config and PDR", y=1.01, fontsize=13)
    fig.tight_layout()
    path = os.path.join(out_dir, "agv_vs_makespan_lines.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Plot 2 — heatmap: % improvement (min AGV → max AGV) ─────────────────────

def plot_improvement_heatmap(df: pd.DataFrame, out_dir: str) -> None:
    rules = [r for r in PDR_ORDER if r in df["rule"].values]
    configs = sorted(df["config"].unique(),
                     key=lambda c: (int(c.split("j/")[0]), int(c.split("/")[1].replace("m", ""))))

    rows = []
    for config in configs:
        for rule in rules:
            sub = df[(df["config"] == config) & (df["rule"] == rule)]
            if sub.empty:
                continue
            agg = sub.groupby("agvCount")["makespan"].mean()
            if len(agg) < 2:
                continue
            worst = agg.max()
            best  = agg.min()
            pct_improvement = (worst - best) / worst * 100
            rows.append({"config": config, "rule": rule, "pct_improvement": pct_improvement})

    if not rows:
        print("  [SKIP] Not enough data for improvement heatmap.")
        return

    pivot = pd.DataFrame(rows).pivot(index="rule", columns="config", values="pct_improvement")
    # Re-order rows
    ordered = [r for r in PDR_ORDER if r in pivot.index]
    pivot = pivot.loc[ordered, configs]

    fig, ax = plt.subplots(figsize=(max(8, len(configs) * 1.1), max(4, len(ordered) * 0.6)))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlGn",
                linewidths=0.4, ax=ax,
                cbar_kws={"label": "% Makespan Improvement (fewest → most AGVs)"})
    ax.set_title("% Makespan Improvement Across AGV Range\n(min AGVs → max AGVs per config)")
    ax.set_xlabel("Factory Configuration")
    ax.set_ylabel("PDR")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)

    fig.tight_layout()
    path = os.path.join(out_dir, "agv_improvement_heatmap.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Plot 3 — box plots: makespan spread per AGV count (largest config) ──────

def plot_boxplot_largest_config(df: pd.DataFrame, out_dir: str) -> None:
    # Pick the config with the most jobs as "stress test"
    largest = df.sort_values(["jobs", "machines"], ascending=False)["config"].iloc[0]
    sub = df[df["config"] == largest]
    agv_counts = sorted(sub["agvCount"].unique())

    data = [sub[sub["agvCount"] == a]["makespan"].values for a in agv_counts]

    fig, ax = plt.subplots(figsize=(max(8, len(agv_counts) * 0.9), 5))
    bp = ax.boxplot(data, labels=[str(a) for a in agv_counts],
                    patch_artist=True,
                    medianprops={"color": "black", "linewidth": 2})
    cmap = plt.cm.Blues(np.linspace(0.3, 0.85, len(agv_counts)))
    for patch, color in zip(bp["boxes"], cmap):
        patch.set_facecolor(color)

    ax.set_xlabel("AGV Count")
    ax.set_ylabel("Makespan (sim-time units)")
    ax.set_title(f"Makespan Distribution vs AGV Count\n(Config: {largest}, all PDRs & seeds)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    fig.tight_layout()
    path = os.path.join(out_dir, "pdr_boxplot_per_agv.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Plot 4 — aggregate: makespan vs AGV count, one line per config ───────────

def plot_agv_aggregate_by_config(df: pd.DataFrame, out_dir: str) -> None:
    configs = sorted(df["config"].unique(),
                     key=lambda c: (int(c.split("j/")[0]), int(c.split("/")[1].replace("m", ""))))

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(configs)))

    for config, color in zip(configs, colors):
        sub = df[df["config"] == config].groupby("agvCount")["makespan"].mean().reset_index()
        ax.plot(sub["agvCount"], sub["makespan"],
                marker="o", label=config, color=color,
                linewidth=1.8, markersize=5, alpha=0.85)

    ax.set_xlabel("AGV Count")
    ax.set_ylabel("Mean Makespan (all PDRs averaged)")
    ax.set_title("Mean Makespan vs AGV Count by Factory Configuration\n(Averaged across all PDRs and seeds)")
    ax.legend(title="Config", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    fig.tight_layout()
    path = os.path.join(out_dir, "agv_vs_makespan_by_config.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualise AGV sensitivity analysis results.")
    parser.add_argument("csv", help="Path to sensitivity results CSV")
    parser.add_argument("--out", default="plots_sensitivity",
                        help="Output directory (default: plots_sensitivity)")
    parser.add_argument("--config", default=None,
                        help="Filter to a single factory config, e.g. '30j/15m'")
    parser.add_argument("--rule", default=None,
                        help="Filter to a single PDR, e.g. 'SPT_SMPT'")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = load_sensitivity(args.csv)

    if args.config:
        df = df[df["config"] == args.config]
        print(f"  Filtered to config: {args.config}")
    if args.rule:
        df = df[df["rule"] == args.rule]
        print(f"  Filtered to rule: {args.rule}")

    print(f"\nConfigs : {sorted(df['config'].unique())}")
    print(f"Rules   : {sorted(df['rule'].unique())}")
    print(f"AGV counts: {sorted(df['agvCount'].unique())}\n")

    plot_agv_lines_faceted(df, args.out)
    plot_improvement_heatmap(df, args.out)
    plot_boxplot_largest_config(df, args.out)
    plot_agv_aggregate_by_config(df, args.out)

    print("\nDone.")


if __name__ == "__main__":
    main()
