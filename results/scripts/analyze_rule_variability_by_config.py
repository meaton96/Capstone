"""
analyze_rule_variability_by_config.py — How much rules differ from each other
WITHIN each config (proc-time multiplier), and why that spread appears to
shrink at high load: MMUR rules don't get better, their completed-job sample
collapses (deadlock/non-completion), so the surviving jobs are a
survivorship-biased subset, not evidence of convergence.

Usage:
    python analyze_rule_variability_by_config.py <job_completions.csv> [--out <dir>]
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PDR_ORDER = [
    "SPT_SMPT", "SPT_SRWT", "SRT_SMPT", "SRT_SRWT",
    "LPT_SMPT", "LPT_MMUR", "LRT_MMUR", "FIFO_SRWT",
    "random",
]
PALETTE = plt.cm.tab10.colors
CENSORED_COMPLETION_THRESHOLD = 0.5  # below this, flag the bar as heavily censored


def _rule_order(present):
    return [r for r in PDR_ORDER if r in present] + [r for r in present if r not in PDR_ORDER]


def load(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df["rule"] = df["rule"].str.strip()
    df["instance"] = df["instance"].str.strip()
    return df


def per_config_rule_stats(df):
    """For every (instance, rule): completion rate over ALL dispatched jobs,
    and mean/CV of flow_time over completed, dynamic jobs only."""
    dyn = df[df["is_dynamic"] == 1].copy()
    total = dyn.groupby(["instance", "rule"]).size().rename("n_total")
    done = dyn[dyn["completed"] == 1].groupby(["instance", "rule"]).size().rename("n_completed")
    cov = pd.concat([total, done], axis=1).fillna(0)
    cov["completion_rate"] = cov["n_completed"] / cov["n_total"]

    completed = dyn[dyn["completed"] == 1]
    flow = completed.groupby(["instance", "rule"])["flow_time"].agg(mean="mean", std="std")
    flow["cv"] = flow["std"] / flow["mean"]

    out = cov.join(flow).reset_index()
    return out


def plot_variability_by_config(stats: pd.DataFrame, out_dir: str):
    instances = sorted(stats["instance"].unique())
    n = len(instances)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 5.5), squeeze=False)
    axes = axes[0]

    for ax, inst in zip(axes, instances):
        sub = stats[stats["instance"] == inst].dropna(subset=["mean"])
        rules = _rule_order(list(sub["rule"]))
        sub = sub.set_index("rule").reindex(rules).dropna(subset=["mean"]).reset_index()
        colors = [PALETTE[PDR_ORDER.index(r) % len(PALETTE)] if r in PDR_ORDER else "gray" for r in sub["rule"]]

        bars = ax.bar(sub["rule"], sub["mean"], color=colors, alpha=0.85)
        # hatch + red outline any bar built from a heavily censored (non-completing) sample
        for bar, rate in zip(bars, sub["completion_rate"]):
            if rate < CENSORED_COMPLETION_THRESHOLD:
                bar.set_hatch("////")
                bar.set_edgecolor("crimson")
                bar.set_linewidth(2)

        for bar, rate, n_c in zip(bars, sub["completion_rate"], sub["n_completed"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{rate*100:.0f}% done\n(n={int(n_c)})",
                    ha="center", va="bottom", fontsize=7,
                    color="crimson" if rate < CENSORED_COMPLETION_THRESHOLD else "dimgray")

        missing = [r for r in rules if r not in list(sub["rule"])]
        for r in missing:
            idx = rules.index(r)
            ax.text(idx, 0, f"{r}\n0% done\n(n=0)", ha="center", va="bottom",
                    fontsize=7, color="crimson", rotation=0)

        spread = (sub["mean"].max() - sub["mean"].min()) / sub["mean"].median() if len(sub) else float("nan")
        ax.set_title(f"{inst}\nspread (max-min)/median = {spread:.2f}")
        ax.set_ylabel("Mean time in system (sim-s)")
        ax.tick_params(axis="x", rotation=40)
        ax.set_xticks(range(len(rules)))
        ax.set_xticklabels(rules)

    fig.suptitle(
        "Between-Rule Variability per Config\n"
        "(hatched red bars = <50% of dispatched jobs completed — mean built on a survivorship-biased sample, not a real value)",
        y=1.04, fontsize=11)
    fig.tight_layout()
    path = os.path.join(out_dir, "rule_variability_by_config.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_completion_collapse(stats: pd.DataFrame, out_dir: str):
    """Completion rate per rule across configs — the actual driver of the
    'variability' trend: MMUR rules collapse to nearly 0% completion at high
    proc-time multipliers instead of converging with the other rules."""
    instances = sorted(stats["instance"].unique())
    rules = _rule_order(list(stats["rule"].unique()))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, r in enumerate(rules):
        color = PALETTE[PDR_ORDER.index(r) % len(PALETTE)] if r in PDR_ORDER else "gray"
        sub = stats[stats["rule"] == r].set_index("instance").reindex(instances)
        style = "-o" if "MMUR" in r else "--o"
        lw = 2.5 if "MMUR" in r else 1.2
        alpha = 1.0 if "MMUR" in r else 0.55
        ax.plot(instances, sub["completion_rate"] * 100, style, color=color, label=r, linewidth=lw, alpha=alpha)

    ax.axhline(50, color="crimson", linestyle=":", linewidth=1, label="50% completion")
    ax.set_ylabel("Completion rate (%)")
    ax.set_xlabel("Config")
    ax.set_title("Job Completion Rate by Rule Across Proc-Time Multipliers\n(MMUR rules collapse — they don't recover)")
    ax.legend(fontsize=8, ncol=2)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    path = os.path.join(out_dir, "completion_rate_by_rule.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    out_dir = args.out or os.path.join(os.path.dirname(os.path.abspath(args.csv)), "figs")
    os.makedirs(out_dir, exist_ok=True)

    df = load(args.csv)
    stats = per_config_rule_stats(df)
    stats.to_csv(os.path.join(out_dir, "rule_variability_stats.csv"), index=False)
    print(f"  Saved: {os.path.join(out_dir, 'rule_variability_stats.csv')}")

    plot_variability_by_config(stats, out_dir)
    plot_completion_collapse(stats, out_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
