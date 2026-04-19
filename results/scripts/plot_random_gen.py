"""
plot_random_gen.py — Visualizations for the random-generated jobs baseline.

Usage:
    python plot_random_gen.py <results.csv> [--out <output_dir>]

Produces:
    1. makespan_by_config_bar.png  — grouped bar chart: PDR × config
    2. makespan_vs_complexity.png  — line chart: makespan vs total_ops per PDR
    3. reward_vs_makespan.png      — scatter: total_reward vs makespan coloured by rule
"""

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from viz_utils import load_random_gen, PDR_ORDER, PDR_LABELS

# ── Style ────────────────────────────────────────────────────────────────────

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
    unique = sorted(set(rules))
    return {r: PALETTE[i % len(PALETTE)] for i, r in enumerate(unique)}


# ── Plot 1 — grouped bar: makespan per PDR per config ───────────────────────

def plot_bar_by_config(df: pd.DataFrame, out_dir: str) -> None:
    configs = sorted(df["config"].unique(),
                     key=lambda c: (int(c.split("j/")[0]), int(c.split("/")[1].replace("m", ""))))
    rules   = [r for r in PDR_ORDER if r in df["rule"].values]
    colors  = _rule_color(rules)

    x = np.arange(len(configs))
    width = 0.8 / max(len(rules), 1)

    fig, ax = plt.subplots(figsize=(max(12, len(configs) * 1.2), 6))

    for i, rule in enumerate(rules):
        sub = df[df["rule"] == rule].set_index("config")
        makespans = [sub.loc[c, "makespan"] if c in sub.index else np.nan for c in configs]
        offset = (i - len(rules) / 2 + 0.5) * width
        ax.bar(x + offset, makespans, width, label=rule, color=colors[rule], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Factory Configuration (jobs/machines)")
    ax.set_ylabel("Makespan (sim-time units)")
    ax.set_title("Makespan by PDR and Factory Configuration\n(Random-Generated Jobs, 3 AGVs)")
    ax.legend(title="PDR", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    fig.tight_layout()
    path = os.path.join(out_dir, "makespan_by_config_bar.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Plot 2 — line: makespan vs problem complexity (total_ops) ───────────────

def plot_makespan_vs_complexity(df: pd.DataFrame, out_dir: str) -> None:
    rules  = [r for r in PDR_ORDER if r in df["rule"].values]
    colors = _rule_color(rules)

    fig, ax = plt.subplots(figsize=(10, 6))

    for rule in rules:
        sub = df[df["rule"] == rule].sort_values("total_ops")
        ax.plot(sub["total_ops"], sub["makespan"],
                marker="o", label=rule, color=colors[rule], alpha=0.85, linewidth=1.8)

    ax.set_xlabel("Total Operations (problem complexity proxy)")
    ax.set_ylabel("Makespan (sim-time units)")
    ax.set_title("Makespan Scaling with Problem Complexity\n(Random-Generated Jobs, 3 AGVs)")
    ax.legend(title="PDR", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    fig.tight_layout()
    path = os.path.join(out_dir, "makespan_vs_complexity.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Plot 3 — scatter: reward vs makespan ────────────────────────────────────

def plot_reward_vs_makespan(df: pd.DataFrame, out_dir: str) -> None:
    rules  = [r for r in PDR_ORDER if r in df["rule"].values]
    colors = _rule_color(rules)

    fig, ax = plt.subplots(figsize=(9, 6))

    for rule in rules:
        sub = df[df["rule"] == rule]
        ax.scatter(sub["makespan"], sub["total_reward"],
                   label=rule, color=colors[rule], alpha=0.75, s=50)

    ax.set_xlabel("Makespan (sim-time units)")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title("Reward vs Makespan per PDR\n(Random-Generated Jobs)")
    ax.legend(title="PDR", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    fig.tight_layout()
    path = os.path.join(out_dir, "reward_vs_makespan.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualise random-gen job baseline results.")
    parser.add_argument("csv", help="Path to random-gen results CSV")
    parser.add_argument("--out", default="plots_random_gen", help="Output directory (default: plots_random_gen)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = load_random_gen(args.csv)

    print(f"\nRules found:   {sorted(df['rule'].unique())}")
    print(f"Configs found: {sorted(df['config'].unique())}\n")

    plot_bar_by_config(df, args.out)
    plot_makespan_vs_complexity(df, args.out)
    plot_reward_vs_makespan(df, args.out)

    print("\nDone.")


if __name__ == "__main__":
    main()
