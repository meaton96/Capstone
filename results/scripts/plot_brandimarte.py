"""
plot_brandimarte.py — Visualizations for Brandimarte MK01-MK15 benchmark results.

Usage:
    python plot_brandimarte.py <results.csv> [--out <output_dir>]

Produces:
    1. makespan_by_instance_bar.png  — grouped bar: PDR × MK instance
    2. pdr_rank_heatmap.png          — heatmap of PDR rank (1=best) per instance
    3. random_pdr_boxplot.png        — box plot of RANDOM rule variance per instance
    4. best_pdr_per_instance.png     — which PDR wins each instance

NOTE: The Brandimarte CSV mixes deterministic PDR runs (single seed) with RANDOM
      rule runs (10 seeds each). Instances are inferred from row order.
      If PDR runs and RANDOM runs used different AGV counts, interpret
      cross-rule makespan comparisons with caution (flagged in plot titles).
"""

import argparse
import os
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from viz_utils import load_brandimarte, PDR_ORDER, PDR_LABELS

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


# ── Plot 1 — grouped bar: makespan per PDR per MK instance ──────────────────

def plot_bar_by_instance(pdr_df: pd.DataFrame, rand_df: pd.DataFrame,
                         out_dir: str, agv_warning: bool) -> None:
    instances = sorted(pdr_df["instance"].unique())
    rules = [r for r in PDR_ORDER if r in pdr_df["rule"].values]
    colors = _rule_color(rules)

    # Add RANDOM mean per instance
    rand_means = rand_df.groupby("instance")["makespan"].mean()

    x = np.arange(len(instances))
    width = 0.8 / (len(rules) + 1)

    fig, ax = plt.subplots(figsize=(max(14, len(instances) * 1.1), 6))

    for i, rule in enumerate(rules):
        sub = pdr_df[pdr_df["rule"] == rule].set_index("instance")
        vals = [sub.loc[inst, "makespan"] if inst in sub.index else np.nan for inst in instances]
        offset = (i - (len(rules)) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=rule, color=colors[rule], alpha=0.85)

    # RANDOM mean as the last group
    rand_vals = [rand_means.get(inst, np.nan) for inst in instances]
    offset = (len(rules) - len(rules) / 2 + 0.5) * width
    ax.bar(x + offset, rand_vals, width, label="RANDOM (mean)", color="gray", alpha=0.6, hatch="//")

    ax.set_xticks(x)
    ax.set_xticklabels(instances, rotation=45, ha="right")
    ax.set_xlabel("Brandimarte Instance")
    ax.set_ylabel("Makespan (sim-time units)")
    title = "Makespan by PDR — Brandimarte Benchmark"
    if agv_warning:
        title += "\n⚠ PDR runs and RANDOM runs may have used different AGV counts"
    ax.set_title(title)
    ax.legend(title="PDR", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    fig.tight_layout()
    path = os.path.join(out_dir, "makespan_by_instance_bar.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Plot 2 — heatmap: PDR rank per instance (1 = best makespan) ─────────────

def plot_rank_heatmap(pdr_df: pd.DataFrame, rand_df: pd.DataFrame,
                      out_dir: str) -> None:
    instances = sorted(pdr_df["instance"].unique())
    rules = [r for r in PDR_ORDER if r in pdr_df["rule"].values]

    pivot = pdr_df.pivot_table(index="rule", columns="instance",
                                values="makespan", aggfunc="mean")

    # Include RANDOM mean
    rand_means = rand_df.groupby("instance")["makespan"].mean()
    pivot.loc["RANDOM"] = [rand_means.get(i, np.nan) for i in instances]

    # Rank: 1 = lowest makespan (best)
    rank_df = pivot[instances].rank(axis=0, method="min").astype(float)

    # Re-order rows to PDR_ORDER
    ordered_rules = [r for r in PDR_ORDER if r in rank_df.index] + \
                    [r for r in rank_df.index if r not in PDR_ORDER]
    rank_df = rank_df.loc[ordered_rules]

    fig, ax = plt.subplots(figsize=(max(10, len(instances) * 0.9), 5))
    sns.heatmap(rank_df, annot=True, fmt=".0f", cmap="RdYlGn_r",
                linewidths=0.4, ax=ax,
                cbar_kws={"label": "Rank (1 = best makespan)"})
    ax.set_title("PDR Rank per Brandimarte Instance\n(1 = shortest makespan)")
    ax.set_xlabel("Instance")
    ax.set_ylabel("PDR")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)

    fig.tight_layout()
    path = os.path.join(out_dir, "pdr_rank_heatmap.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Plot 3 — box plot: RANDOM rule variance per instance ─────────────────────

def plot_random_variance(rand_df: pd.DataFrame, out_dir: str) -> None:
    instances = sorted(rand_df["instance"].unique())
    data = [rand_df[rand_df["instance"] == i]["makespan"].values for i in instances]

    fig, ax = plt.subplots(figsize=(max(10, len(instances) * 0.9), 5))
    bp = ax.boxplot(data, labels=instances, patch_artist=True,
                    medianprops={"color": "black", "linewidth": 2})
    for patch in bp["boxes"]:
        patch.set_facecolor("steelblue")
        patch.set_alpha(0.6)

    ax.set_xlabel("Brandimarte Instance")
    ax.set_ylabel("Makespan (sim-time units)")
    ax.set_title("RANDOM PDR Makespan Variance per Instance\n(10 seeds each)")
    ax.tick_params(axis="x", rotation=45)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    fig.tight_layout()
    path = os.path.join(out_dir, "random_pdr_boxplot.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Plot 4 — which PDR wins each instance ────────────────────────────────────

def plot_best_pdr(pdr_df: pd.DataFrame, rand_df: pd.DataFrame, out_dir: str) -> None:
    instances = sorted(pdr_df["instance"].unique())
    rules = [r for r in PDR_ORDER if r in pdr_df["rule"].values]
    colors = _rule_color(rules + ["RANDOM"])

    best_rules, best_makespans = [], []
    for inst in instances:
        candidates = {
            rule: pdr_df[(pdr_df["instance"] == inst) & (pdr_df["rule"] == rule)]["makespan"].mean()
            for rule in rules
        }
        rand_mean = rand_df[rand_df["instance"] == inst]["makespan"].mean()
        if not np.isnan(rand_mean):
            candidates["RANDOM"] = rand_mean
        best_rule = min(candidates, key=candidates.get)
        best_rules.append(best_rule)
        best_makespans.append(candidates[best_rule])

    bar_colors = [colors.get(r, "gray") for r in best_rules]
    x = np.arange(len(instances))

    fig, ax = plt.subplots(figsize=(max(10, len(instances)), 5))
    bars = ax.bar(x, best_makespans, color=bar_colors, alpha=0.85, edgecolor="white")
    for bar, rule in zip(bars, best_rules):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(best_makespans) * 0.01,
                rule.replace("_", "\n"), ha="center", va="bottom",
                fontsize=6.5, rotation=0)

    ax.set_xticks(x)
    ax.set_xticklabels(instances, rotation=45, ha="right")
    ax.set_xlabel("Brandimarte Instance")
    ax.set_ylabel("Best Makespan Achieved")
    ax.set_title("Best-Performing PDR per Brandimarte Instance")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    # Legend for rule colours
    from matplotlib.patches import Patch
    legend_handles = [Patch(color=colors[r], alpha=0.85, label=r)
                      for r in (rules + ["RANDOM"]) if r in colors]
    ax.legend(handles=legend_handles, title="PDR", bbox_to_anchor=(1.01, 1),
              loc="upper left", fontsize=8)

    fig.tight_layout()
    path = os.path.join(out_dir, "best_pdr_per_instance.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualise Brandimarte benchmark results.")
    parser.add_argument("csv", help="Path to Brandimarte results CSV")
    parser.add_argument("--out", default="plots_brandimarte",
                        help="Output directory (default: plots_brandimarte)")
    parser.add_argument("--agv-warning", action="store_true",
                        help="Flag plots if PDR/RANDOM runs used different AGV counts")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    pdr_df, rand_df = load_brandimarte(args.csv)

    print(f"\nDeterministic PDR rows : {len(pdr_df)}")
    print(f"RANDOM rule rows        : {len(rand_df)}")
    print(f"Inferred instances      : {sorted(pdr_df['instance'].unique())}\n")

    plot_bar_by_instance(pdr_df, rand_df, args.out, agv_warning=args.agv_warning)
    plot_rank_heatmap(pdr_df, rand_df, args.out)
    plot_random_variance(rand_df, args.out)
    plot_best_pdr(pdr_df, rand_df, args.out)

    print("\nDone.")


if __name__ == "__main__":
    main()
