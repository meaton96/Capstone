"""
plot_random_gen.py — Visualizations for the random-generated jobs baseline.

Usage:
    python plot_random_gen.py <results.csv> [--out <output_dir>]

Produces:
    1. makespan_scaling.png       — mean +/- std band vs total_ops, PDR lines faded behind
    2. pdr_deviation.png          — % deviation of each PDR from per-config mean
    3. makespan_by_config_bar.png — grouped bar with mean overlay + zoomed y-axis
    4. reward_vs_makespan.png     — scatter per individual row coloured by rule
    5. pdr_makespan_heatmap.png   — dual heatmap: absolute makespan + % of config mean
    6. pdr_rank_heatmap.png       — rank heatmap (1=best) per PDR x config
"""

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

from viz_utils import load_random_gen, PDR_ORDER

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


# ── Plot 1 — mean +/- std band + faded individual PDR lines ─────────────────

def plot_makespan_scaling(df, out_dir):
    rules = [r for r in PDR_ORDER if r in df["rule"].values]
    colors = _rule_color(rules)
    agg = df.groupby(["rule", "total_ops"])["makespan"].mean().reset_index()
    mean_by_ops = df.groupby("total_ops")["makespan"].agg(["mean", "std"]).reset_index()
    mean_by_ops = mean_by_ops.sort_values("total_ops")

    fig, ax = plt.subplots(figsize=(10, 6))
    for rule in rules:
        sub = agg[agg["rule"] == rule].sort_values("total_ops")
        ax.plot(sub["total_ops"], sub["makespan"],
                color=colors[rule], alpha=0.3, linewidth=1.2, zorder=2)
    ax.fill_between(mean_by_ops["total_ops"],
                    mean_by_ops["mean"] - mean_by_ops["std"],
                    mean_by_ops["mean"] + mean_by_ops["std"],
                    alpha=0.25, color="steelblue", label="+/-1 std across PDRs", zorder=3)
    ax.plot(mean_by_ops["total_ops"], mean_by_ops["mean"],
            color="steelblue", linewidth=2.5, marker="o", markersize=7,
            label="Mean across PDRs", zorder=4)
    pdr_handle = Line2D([0], [0], color="gray", alpha=0.4, linewidth=1.2,
                        label="Individual PDRs (nearly identical)")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [pdr_handle], loc="upper left", fontsize=9)
    ax.set_xlabel("Total Operations (problem complexity proxy)")
    ax.set_ylabel("Makespan (sim-time units)")
    ax.set_title("Makespan Scaling vs Problem Complexity\n"
                 "(Random-Generated Jobs, 3 AGVs)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    fig.tight_layout()
    path = os.path.join(out_dir, "makespan_scaling.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Plot 2 — % deviation from per-config mean ───────────────────────────────

def plot_pdr_deviation(df, out_dir):
    rules = [r for r in PDR_ORDER if r in df["rule"].values]
    colors = _rule_color(rules)
    configs = _sort_configs(df["config"].unique())
    config_mean = df.groupby("config")["makespan"].mean()

    fig, ax = plt.subplots(figsize=(max(10, len(configs) * 1.2), 6))
    x = np.arange(len(configs))
    width = 0.8 / max(len(rules), 1)
    for i, rule in enumerate(rules):
        sub = df[df["rule"] == rule].set_index("config")
        deviations = []
        for c in configs:
            if c in sub.index:
                deviations.append((sub.loc[c, "makespan"] - config_mean[c]) / config_mean[c] * 100)
            else:
                deviations.append(np.nan)
        offset = (i - len(rules) / 2 + 0.5) * width
        ax.bar(x + offset, deviations, width, label=rule, color=colors[rule], alpha=0.85)
    ax.axhline(0, color="black", linewidth=1.2, linestyle="--", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Factory Configuration")
    ax.set_ylabel("% Deviation from Per-Config Mean Makespan")
    ax.set_title("PDR Performance Relative to Per-Config Mean\n"
                 "(Positive = worse than average, Negative = better)")
    ax.legend(title="PDR", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:+.2f}%"))
    fig.tight_layout()
    path = os.path.join(out_dir, "pdr_deviation.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Plot 3 — grouped bar with mean overlay + zoomed y-axis ──────────────────

def plot_bar_by_config(df, out_dir):
    configs = _sort_configs(df["config"].unique())
    rules = [r for r in PDR_ORDER if r in df["rule"].values]
    colors = _rule_color(rules)
    x = np.arange(len(configs))
    width = 0.8 / max(len(rules), 1)

    fig, ax = plt.subplots(figsize=(max(12, len(configs) * 1.2), 6))
    for i, rule in enumerate(rules):
        sub = df[df["rule"] == rule].set_index("config")
        makespans = [sub.loc[c, "makespan"] if c in sub.index else np.nan for c in configs]
        offset = (i - len(rules) / 2 + 0.5) * width
        ax.bar(x + offset, makespans, width, label=rule, color=colors[rule], alpha=0.75)
    config_mean = df.groupby("config")["makespan"].mean()
    means = [config_mean.get(c, np.nan) for c in configs]
    ax.plot(x, means, color="black", linewidth=2, marker="D", markersize=6,
            zorder=5, label="Mean across PDRs", linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Factory Configuration (jobs/machines)")
    ax.set_ylabel("Makespan (sim-time units)")
    ax.set_title("Makespan by PDR and Factory Configuration\n(Random-Generated Jobs, 3 AGVs)")
    ax.legend(title="PDR", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    all_vals = df["makespan"].dropna().values
    if len(all_vals):
        ax.set_ylim(bottom=all_vals.min() * 0.90)
    fig.tight_layout()
    path = os.path.join(out_dir, "makespan_by_config_bar.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Plot 4 — scatter: reward vs makespan ────────────────────────────────────

def plot_reward_vs_makespan(df, out_dir):
    rules = [r for r in PDR_ORDER if r in df["rule"].values]
    colors = _rule_color(rules)
    fig, ax = plt.subplots(figsize=(9, 6))
    for rule in rules:
        sub = df[df["rule"] == rule]
        ax.scatter(sub["makespan"], sub["total_reward"],
                   label=rule, color=colors[rule], alpha=0.75, s=55, zorder=3)
    ax.set_xlabel("Makespan (sim-time units)")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title("Reward vs Makespan per PDR\n(each point = one run)")
    ax.legend(title="PDR", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    fig.tight_layout()
    path = os.path.join(out_dir, "reward_vs_makespan.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Plot 5 — dual heatmap: absolute makespan + % of config mean ─────────────

def plot_pdr_heatmap(df, out_dir):
    df = df.copy()
    df["rule_upper"] = df["rule"].str.upper()
    configs = _sort_configs(df["config"].unique())
    row_order = [r.upper() for r in PDR_ORDER if r.upper() in df["rule_upper"].values]

    pivot = (df.groupby(["rule_upper", "config"])["makespan"]
               .mean()
               .unstack("config")
               .reindex(index=row_order, columns=configs))
    pivot.index.name = "PDR"
    pivot_norm = pivot.div(pivot.mean(axis=0), axis=1) * 100

    fig, axes = plt.subplots(1, 2,
                             figsize=(max(14, len(configs) * 1.6), max(5, len(pivot) * 0.7)))

    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlOrRd",
                linewidths=0.4, ax=axes[0],
                cbar_kws={"label": "Mean Makespan"},
                annot_kws={"size": 7})
    axes[0].set_title("Mean Makespan (lower = better)", fontsize=10)
    axes[0].set_xlabel("Factory Configuration")
    axes[0].set_ylabel("PDR")
    axes[0].tick_params(axis="x", rotation=40, labelsize=8)
    axes[0].tick_params(axis="y", rotation=0, labelsize=8)

    sns.heatmap(pivot_norm, annot=True, fmt=".1f", cmap="RdYlGn_r",
                linewidths=0.4, ax=axes[1], center=100,
                cbar_kws={"label": "% of Per-Config Mean (100=avg)"},
                annot_kws={"size": 7})
    axes[1].set_title("Relative Performance\n(% of config mean, <100 = better than avg)",
                      fontsize=10)
    axes[1].set_xlabel("Factory Configuration")
    axes[1].set_ylabel("")
    axes[1].tick_params(axis="x", rotation=40, labelsize=8)
    axes[1].tick_params(axis="y", rotation=0, labelsize=8)

    fig.suptitle("PDR vs RANDOM Makespan Comparison — All Configurations",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    path = os.path.join(out_dir, "pdr_makespan_heatmap.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Plot 6 — rank heatmap: 1=best per config ────────────────────────────────

def plot_pdr_rank_heatmap(df, out_dir):
    df = df.copy()
    df["rule_upper"] = df["rule"].str.upper()
    configs = _sort_configs(df["config"].unique())
    row_order = [r.upper() for r in PDR_ORDER if r.upper() in df["rule_upper"].values]

    pivot = (df.groupby(["rule_upper", "config"])["makespan"]
               .mean()
               .unstack("config")
               .reindex(index=row_order, columns=configs))
    rank_df = pivot.rank(axis=0, method="min").astype(float)

    fig, ax = plt.subplots(figsize=(max(10, len(configs) * 1.4), max(4, len(rank_df) * 0.65)))
    sns.heatmap(rank_df, annot=True, fmt=".0f", cmap="RdYlGn_r",
                linewidths=0.4, ax=ax,
                cbar_kws={"label": "Rank (1 = best makespan)"},
                annot_kws={"size": 8})
    ax.set_title("PDR Rank per Factory Configuration\n"
                 "(1 = shortest makespan, includes RANDOM)", fontsize=11)
    ax.set_xlabel("Factory Configuration")
    ax.set_ylabel("PDR")
    ax.tick_params(axis="x", rotation=40, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)
    fig.tight_layout()
    path = os.path.join(out_dir, "pdr_rank_heatmap.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualise random-gen job baseline results.")
    parser.add_argument("csv", help="Path to random-gen results CSV")
    parser.add_argument("--out", default="plots_random_gen",
                        help="Output directory (default: plots_random_gen)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = load_random_gen(args.csv)

    print(f"\nRules found:   {sorted(df['rule'].unique())}")
    print(f"Configs found: {sorted(df['config'].unique())}\n")

    plot_makespan_scaling(df, args.out)
    plot_pdr_deviation(df, args.out)
    plot_bar_by_config(df, args.out)
    plot_reward_vs_makespan(df, args.out)
    plot_pdr_heatmap(df, args.out)
    plot_pdr_rank_heatmap(df, args.out)

    print("\nDone.")


if __name__ == "__main__":
    main()