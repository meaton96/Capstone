"""
plot_sensitivity.py — Visualizations for the AGV + machine-count sensitivity analysis.

Usage:
    python plot_sensitivity.py <results.csv> [--out <output_dir>]

The CSV contains two interleaved experiments:
  - AGV sweep   : 30j/15m and 50j/15m, multiple AGV counts (2-30)
  - Machine sweep: all other configs, single fixed AGV count per config

Produces:
    1. agv_sweep_lines.png         — mean makespan vs AGV count (AGV-sweep configs only)
                                     per PDR + mean overlay
    2. agv_improvement_heatmap.png — % improvement from fewest→most AGVs (AGV sweep only)
    3. agv_boxplot.png             — makespan distribution vs AGV count for each
                                     AGV-sweep config (all PDRs & seeds pooled)
    4. machine_sweep_bar.png       — makespan vs machine count (machine-sweep configs),
                                     grouped bar per PDR + mean overlay
    5. machine_sweep_deviation.png — % deviation from per-machine-count mean
"""

import argparse
import os
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

from results.scripts.old.viz_utils import load_sensitivity, PDR_ORDER

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
    return {r: PALETTE[i % len(PALETTE)] for i, r in enumerate(sorted(set(rules)))}


def _sort_configs(configs):
    return sorted(configs,
                  key=lambda c: (int(c.split("j/")[0]), int(c.split("/")[1].replace("m", ""))))


def _split_experiments(df: pd.DataFrame):
    """
    Separate AGV-sweep configs (multiple AGV counts) from machine-sweep configs
    (single AGV count per config).
    """
    agv_variety = df.groupby("config")["agvCount"].nunique()
    agv_sweep_configs = agv_variety[agv_variety > 1].index.tolist()
    machine_sweep_configs = agv_variety[agv_variety == 1].index.tolist()
    return (df[df["config"].isin(agv_sweep_configs)].copy(),
            df[df["config"].isin(machine_sweep_configs)].copy())


# ── Plot 1 — AGV sweep: mean makespan vs AGV count per PDR ──────────────────

def plot_agv_sweep_lines(agv_df: pd.DataFrame, out_dir: str) -> None:
    configs = _sort_configs(agv_df["config"].unique())
    rules = [r for r in PDR_ORDER if r in agv_df["rule"].values]
    colors = _rule_color(rules)

    ncols = min(2, len(configs))
    nrows = int(np.ceil(len(configs) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(7 * ncols, 5 * nrows),
                             sharey=False)
    axes = np.array(axes).flatten()

    for ax_i, config in enumerate(configs):
        ax = axes[ax_i]
        sub = agv_df[agv_df["config"] == config]

        # Faded individual PDR lines
        for rule in rules:
            rsub = sub[sub["rule"] == rule].groupby("agvCount")["makespan"].mean().reset_index()
            if rsub.empty:
                continue
            ax.plot(rsub["agvCount"], rsub["makespan"],
                    color=colors[rule], alpha=0.3, linewidth=1.2, zorder=2)

        # Mean across all PDRs + seeds
        mean_agg = sub.groupby("agvCount")["makespan"].agg(["mean", "std"]).reset_index()
        ax.fill_between(mean_agg["agvCount"],
                        mean_agg["mean"] - mean_agg["std"],
                        mean_agg["mean"] + mean_agg["std"],
                        alpha=0.2, color="steelblue", zorder=3)
        ax.plot(mean_agg["agvCount"], mean_agg["mean"],
                color="steelblue", linewidth=2.5, marker="o", markersize=6,
                label="Mean ± std", zorder=4)

        ax.set_title(config, fontsize=11, fontweight="bold")
        ax.set_xlabel("AGV Count")
        ax.set_ylabel("Mean Makespan")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    for ax_i in range(len(configs), len(axes)):
        axes[ax_i].set_visible(False)

    # Shared legend
    pdr_handle = Line2D([0], [0], color="gray", alpha=0.4, linewidth=1.2, label="Individual PDRs")
    mean_handle = Line2D([0], [0], color="steelblue", linewidth=2.5, marker="o", label="Mean across PDRs")
    fig.legend(handles=[mean_handle, pdr_handle],
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.04), fontsize=9)

    fig.suptitle("Mean Makespan vs AGV Count\n(AGV Sweep Configs)", y=1.01, fontsize=13)
    fig.tight_layout()
    path = os.path.join(out_dir, "agv_sweep_lines.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Plot 2 — AGV sweep: % improvement heatmap ────────────────────────────────

def plot_improvement_heatmap(agv_df: pd.DataFrame, out_dir: str) -> None:
    rules = [r for r in PDR_ORDER if r in agv_df["rule"].values]
    configs = _sort_configs(agv_df["config"].unique())

    rows = []
    for config in configs:
        for rule in rules:
            sub = agv_df[(agv_df["config"] == config) & (agv_df["rule"] == rule)]
            agg = sub.groupby("agvCount")["makespan"].mean()
            if len(agg) < 2:
                continue
            pct = (agg.max() - agg.min()) / agg.max() * 100
            rows.append({"config": config, "rule": rule, "pct_improvement": pct})

    if not rows:
        print("  [SKIP] No AGV-sweep configs with multiple AGV counts.")
        return

    pivot = pd.DataFrame(rows).pivot(index="rule", columns="config", values="pct_improvement")
    available_configs = [c for c in configs if c in pivot.columns]
    ordered = [r for r in PDR_ORDER if r in pivot.index]
    pivot = pivot.loc[ordered, available_configs]

    fig, ax = plt.subplots(figsize=(max(5, len(available_configs) * 2.5),
                                    max(4, len(ordered) * 0.6)))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlGn",
                linewidths=0.4, ax=ax,
                cbar_kws={"label": "% Improvement (fewest → most AGVs)"})
    ax.set_title("% Makespan Improvement Across AGV Range\n(Fewer AGVs → More AGVs)")
    ax.set_xlabel("Factory Configuration")
    ax.set_ylabel("PDR")
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)

    fig.tight_layout()
    path = os.path.join(out_dir, "agv_improvement_heatmap.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Plot 3 — AGV sweep: box plots per AGV count ───────────────────────────────

def plot_agv_boxplots(agv_df: pd.DataFrame, out_dir: str) -> None:
    configs = _sort_configs(agv_df["config"].unique())
    ncols = min(2, len(configs))
    nrows = int(np.ceil(len(configs) / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(7 * ncols, 5 * nrows),
                             sharey=False)
    axes = np.array(axes).flatten()

    for ax_i, config in enumerate(configs):
        ax = axes[ax_i]
        sub = agv_df[agv_df["config"] == config]
        agv_counts = sorted(sub["agvCount"].unique())
        data = [sub[sub["agvCount"] == a]["makespan"].values for a in agv_counts]

        bp = ax.boxplot(data, labels=[str(a) for a in agv_counts],
                        patch_artist=True,
                        medianprops={"color": "black", "linewidth": 2})
        cmap_vals = plt.cm.Blues(np.linspace(0.3, 0.85, len(agv_counts)))
        for patch, color in zip(bp["boxes"], cmap_vals):
            patch.set_facecolor(color)

        # Mean line
        means = [np.mean(d) for d in data]
        ax.plot(range(1, len(agv_counts) + 1), means,
                color="steelblue", marker="D", linewidth=1.5,
                markersize=5, linestyle="--", label="Mean", zorder=5)

        ax.set_title(config, fontsize=11, fontweight="bold")
        ax.set_xlabel("AGV Count")
        ax.set_ylabel("Makespan")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.legend(fontsize=8)

    for ax_i in range(len(configs), len(axes)):
        axes[ax_i].set_visible(False)

    fig.suptitle("Makespan Distribution vs AGV Count\n(All PDRs & Seeds Pooled)", y=1.01,
                 fontsize=13)
    fig.tight_layout()
    path = os.path.join(out_dir, "agv_boxplot.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Plot 4 — Machine sweep: grouped bar per PDR ──────────────────────────────

def plot_machine_sweep_bar(mach_df: pd.DataFrame, out_dir: str) -> None:
    if mach_df.empty:
        print("  [SKIP] No machine-sweep configs found.")
        return

    # Group by jobs separately (30j vs 50j)
    for job_count in sorted(mach_df["jobs"].unique()):
        sub_all = mach_df[mach_df["jobs"] == job_count]
        configs = _sort_configs(sub_all["config"].unique())
        rules = [r for r in PDR_ORDER if r in sub_all["rule"].values]
        colors = _rule_color(rules)

        # X-axis: machine count
        machine_counts = sorted(sub_all["machines"].unique())
        x = np.arange(len(machine_counts))
        width = 0.8 / max(len(rules), 1)

        fig, ax = plt.subplots(figsize=(max(9, len(machine_counts) * 1.5), 6))

        for i, rule in enumerate(rules):
            rsub = sub_all[sub_all["rule"] == rule].groupby("machines")["makespan"].mean()
            vals = [rsub.get(m, np.nan) for m in machine_counts]
            offset = (i - len(rules) / 2 + 0.5) * width
            ax.bar(x + offset, vals, width, label=rule, color=colors[rule], alpha=0.75)

        # Mean line
        mean_by_m = sub_all.groupby("machines")["makespan"].mean()
        means = [mean_by_m.get(m, np.nan) for m in machine_counts]
        ax.plot(x, means, color="black", linewidth=2, marker="D", markersize=6,
                zorder=5, label="Mean across PDRs", linestyle="--")

        ax.set_xticks(x)
        ax.set_xticklabels([str(m) for m in machine_counts])
        ax.set_xlabel("Total Machines")
        ax.set_ylabel("Makespan (sim-time units)")
        ax.set_title(f"Makespan vs Machine Count — {job_count} Jobs\n"
                     f"(Machine Sweep, fixed AGV count)")
        ax.legend(title="PDR", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

        # Zoom y-axis
        valid = [v for v in means if not np.isnan(v)]
        if valid:
            ax.set_ylim(bottom=min(valid) * 0.9)

        fig.tight_layout()
        path = os.path.join(out_dir, f"machine_sweep_bar_{job_count}j.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")


# ── Plot 5 — Machine sweep: % deviation from per-machine-count mean ──────────

def plot_machine_sweep_deviation(mach_df: pd.DataFrame, out_dir: str) -> None:
    if mach_df.empty:
        print("  [SKIP] No machine-sweep configs found.")
        return

    rules = [r for r in PDR_ORDER if r in mach_df["rule"].values]
    colors = _rule_color(rules)

    for job_count in sorted(mach_df["jobs"].unique()):
        sub_all = mach_df[mach_df["jobs"] == job_count]
        machine_counts = sorted(sub_all["machines"].unique())
        mean_by_m = sub_all.groupby("machines")["makespan"].mean()

        x = np.arange(len(machine_counts))
        width = 0.8 / max(len(rules), 1)

        fig, ax = plt.subplots(figsize=(max(9, len(machine_counts) * 1.5), 6))

        for i, rule in enumerate(rules):
            rsub = sub_all[sub_all["rule"] == rule].groupby("machines")["makespan"].mean()
            deviations = []
            for m in machine_counts:
                if m in rsub.index and m in mean_by_m.index:
                    deviations.append((rsub[m] - mean_by_m[m]) / mean_by_m[m] * 100)
                else:
                    deviations.append(np.nan)
            offset = (i - len(rules) / 2 + 0.5) * width
            ax.bar(x + offset, deviations, width, label=rule, color=colors[rule], alpha=0.85)

        ax.axhline(0, color="black", linewidth=1.2, linestyle="--", alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([str(m) for m in machine_counts])
        ax.set_xlabel("Total Machines")
        ax.set_ylabel("% Deviation from Per-Machine-Count Mean")
        ax.set_title(f"PDR Deviation from Mean — {job_count} Jobs (Machine Sweep)")
        ax.legend(title="PDR", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:+.2f}%"))

        fig.tight_layout()
        path = os.path.join(out_dir, f"machine_sweep_deviation_{job_count}j.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualise AGV + machine sensitivity results.")
    parser.add_argument("csv", help="Path to sensitivity results CSV")
    parser.add_argument("--out", default="plots_sensitivity",
                        help="Output directory (default: plots_sensitivity)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = load_sensitivity(args.csv)

    agv_df, mach_df = _split_experiments(df)

    print(f"\nAGV-sweep configs   : {_sort_configs(agv_df['config'].unique())}")
    print(f"Machine-sweep configs: {_sort_configs(mach_df['config'].unique())}")
    print(f"AGV counts in sweep  : {sorted(agv_df['agvCount'].unique())}\n")

    plot_agv_sweep_lines(agv_df, args.out)
    plot_improvement_heatmap(agv_df, args.out)
    plot_agv_boxplots(agv_df, args.out)
    plot_machine_sweep_bar(mach_df, args.out)
    plot_machine_sweep_deviation(mach_df, args.out)

    print("\nDone.")


if __name__ == "__main__":
    main()