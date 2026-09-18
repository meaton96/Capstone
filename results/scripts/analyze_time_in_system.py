"""
analyze_time_in_system.py — Time-in-system (flow time) and cross-rule variance
analysis for a job_completions.csv sweep, with warm-up/tail trimming.

Filtering applied, in order:
  1. completed == 1                (drop unfinished jobs from deadlocked/timed-out runs)
  2. is_dynamic == 1                (drop the static initial-batch "warm-up" jobs, arrival_time==0)
  3. drop first/last TRIM_FRAC of the remaining jobs per (instance, rule, seed),
     ordered by arrival_time         (drop boundary-affected head/tail of the arrival stream)

Usage:
    python analyze_time_in_system.py <job_completions.csv> [--out <dir>] [--trim 0.10]
"""

import argparse
import os
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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

PDR_ORDER = [
    "SPT_SMPT", "SPT_SRWT", "SRT_SMPT", "SRT_SRWT",
    "LPT_SMPT", "LPT_MMUR", "LRT_MMUR", "FIFO_SRWT",
    "random",
]
PALETTE = plt.cm.tab10.colors

# Configs known (deadlock-fix-verification memory) to still censor/deadlock heavily —
# excluded from the main comparison, reported separately.
CENSORED_INSTANCE_PREFIXES = ("med_agv10", "med_agv11")


def _rule_order(present: list) -> list:
    return [r for r in PDR_ORDER if r in present] + [r for r in present if r not in PDR_ORDER]


def _rule_color(rules: list) -> dict:
    return {r: PALETTE[i % len(PALETTE)] for i, r in enumerate(rules)}


def load_and_filter(csv_path: str, trim_frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (filtered_df, coverage_df). coverage_df reports completion rate per instance,
    used to flag censored configs."""
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df["rule"] = df["rule"].str.strip()
    df["instance"] = df["instance"].str.strip()

    total = df.groupby("instance").size().rename("n_jobs")
    completed = df[df["completed"] == 1].groupby("instance").size().rename("n_completed")
    coverage = pd.concat([total, completed], axis=1).fillna(0)
    coverage["completion_rate"] = coverage["n_completed"] / coverage["n_jobs"]

    df = df[df["completed"] == 1].copy()
    df = df[df["is_dynamic"] == 1].copy()

    df = df.sort_values("arrival_time")
    grp = df.groupby(["instance", "rule", "seed"])
    rank = grp.cumcount()
    size = grp["arrival_time"].transform("size")
    k = (size * trim_frac).round().astype(int)
    df = df[(rank >= k) & (rank < size - k)].reset_index(drop=True)

    return df, coverage


def plot_time_in_system_by_instance(df: pd.DataFrame, out_dir: str, trim_frac: float = 0.10) -> None:
    """Box plot of time-in-system (flow_time) per rule, one panel per instance."""
    instances = sorted(df["instance"].unique())
    n = len(instances)
    ncols = 2
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.5 * nrows), squeeze=False)

    for idx, inst in enumerate(instances):
        ax = axes[idx // ncols][idx % ncols]
        sub = df[df["instance"] == inst]
        rules = _rule_order(list(sub["rule"].unique()))
        colors = _rule_color(rules)
        data = [sub.loc[sub["rule"] == r, "flow_time"].values for r in rules]
        bp = ax.boxplot(data, tick_labels=rules, patch_artist=True, showfliers=False,
                         medianprops={"color": "black", "linewidth": 1.8})
        for patch, r in zip(bp["boxes"], rules):
            patch.set_facecolor(colors[r])
            patch.set_alpha(0.7)
        n_jobs = sub.groupby("rule").size().reindex(rules)
        ax.set_title(f"{inst}  (n≈{int(n_jobs.median())}/rule)")
        ax.set_ylabel("Time in system (sim-s)")
        ax.tick_params(axis="x", rotation=35)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    trim_note = f"warm-up + head/tail trimmed {trim_frac:.0%}" if trim_frac > 0 else "no head/tail trim, flat-duration run"
    fig.suptitle(f"Time in System by Rule, per Config\n({trim_note}, completed jobs only)", y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "time_in_system_by_instance.png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.join(out_dir, 'time_in_system_by_instance.png')}")


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-instance x rule: mean, std, CV of time-in-system, aggregated across seeds."""
    g = df.groupby(["instance", "rule"])["flow_time"]
    out = g.agg(mean="mean", std="std", n="count").reset_index()
    out["cv"] = out["std"] / out["mean"]
    return out.sort_values(["instance", "mean"])


def plot_cv_by_rule(summary: pd.DataFrame, out_dir: str) -> None:
    """Bar chart: coefficient of variation of time-in-system per rule, averaged across
    the validated (non-censored) instances — the direct answer to 'variance between rules'."""
    agg = summary.groupby("rule")["cv"].agg(mean_cv="mean", std_cv="std").reset_index()
    rules = _rule_order(list(agg["rule"]))
    agg = agg.set_index("rule").reindex(rules).reset_index()
    colors = _rule_color(rules)

    fig, ax = plt.subplots(figsize=(max(9, len(rules) * 1.3), 5.5))
    bars = ax.bar(agg["rule"], agg["mean_cv"], color=[colors[r] for r in agg["rule"]], alpha=0.85)
    ax.errorbar(agg["rule"], agg["mean_cv"], yerr=agg["std_cv"], fmt="none", color="black",
                capsize=3, linewidth=1)
    for bar, v in zip(bars, agg["mean_cv"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{v:.2f}",
                ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Coefficient of variation (std / mean) of time in system")
    ax.set_xlabel("PDR")
    ax.set_title("Time-in-System Variability by Rule\n(mean CV across validated configs — agv06/07/09; error bars = spread across configs)")
    ax.tick_params(axis="x", rotation=30)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "time_in_system_cv_by_rule.png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.join(out_dir, 'time_in_system_cv_by_rule.png')}")


def plot_trim_effect(raw_df: pd.DataFrame, trimmed_df: pd.DataFrame, out_dir: str,
                      instance: str, rule: str, seed: str) -> None:
    """Diagnostic: flow_time vs arrival order, before/after filtering, for one example run."""
    raw = raw_df[(raw_df["instance"] == instance) & (raw_df["rule"] == rule)
                 & (raw_df["seed"].astype(str) == str(seed)) & (raw_df["completed"] == 1)]
    raw = raw.sort_values("arrival_time")
    kept_ids = set(trimmed_df[(trimmed_df["instance"] == instance) & (trimmed_df["rule"] == rule)
                               & (trimmed_df["seed"].astype(str) == str(seed))]["job_id"])

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = np.where(raw["job_id"].isin(kept_ids), "steelblue", "lightgray")
    labels_done = set()
    for is_kept in (False, True):
        mask = raw["job_id"].isin(kept_ids) == is_kept
        label = "kept (analysis window)" if is_kept else "excluded (warm-up/tail/static)"
        ax.scatter(raw.loc[mask, "arrival_time"], raw.loc[mask, "flow_time"],
                   s=14, alpha=0.7, color="steelblue" if is_kept else "lightgray", label=label)

    ax.set_xlabel("Arrival time (sim-s)")
    ax.set_ylabel("Time in system (sim-s)")
    ax.set_title(f"Trim Effect Example — {instance} / {rule} / seed {seed}\n"
                 "(monotonic growth = overload buildup-then-drain, not steady state)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fname = f"trim_effect_example_{instance}_{rule}_{seed}.png"
    fig.savefig(os.path.join(out_dir, fname), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.join(out_dir, fname)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", help="Path to job_completions.csv")
    parser.add_argument("--out", default=None, help="Output dir (default: <csv_dir>/figs)")
    parser.add_argument("--trim", type=float, default=0.10,
                         help="Fraction to drop from each end, by arrival order, per (instance, rule, seed) (default 0.10)")
    args = parser.parse_args()

    out_dir = args.out or os.path.join(os.path.dirname(os.path.abspath(args.csv)), "figs")
    os.makedirs(out_dir, exist_ok=True)

    raw_df = pd.read_csv(args.csv)
    raw_df.columns = raw_df.columns.str.strip()
    raw_df["rule"] = raw_df["rule"].str.strip()
    raw_df["instance"] = raw_df["instance"].str.strip()

    df, coverage = load_and_filter(args.csv, args.trim)

    print("Completion rate by instance (censoring check):")
    print(coverage.sort_values("completion_rate").to_string())

    censored = [i for i in df["instance"].unique() if i.startswith(CENSORED_INSTANCE_PREFIXES)]
    validated = df[~df["instance"].isin(censored)].copy()
    censored_df = df[df["instance"].isin(censored)].copy()

    print(f"\nValidated instances (main analysis): {sorted(validated['instance'].unique())}")
    print(f"Censored instances (excluded, reported separately): {sorted(censored)}")
    print(f"Rows after filtering — validated: {len(validated)}, censored: {len(censored_df)}")

    plot_time_in_system_by_instance(validated, out_dir, trim_frac=args.trim)

    summary = summary_table(validated)
    summary_path = os.path.join(out_dir, "time_in_system_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"  Saved: {summary_path}")

    plot_cv_by_rule(summary, out_dir)

    if not censored_df.empty:
        cens_summary = summary_table(censored_df)
        cens_path = os.path.join(out_dir, "time_in_system_summary_censored.csv")
        cens_summary.to_csv(cens_path, index=False)
        print(f"  Saved (censored configs, for reference only): {cens_path}")

    fully_complete = coverage[coverage["completion_rate"] == 1.0].index
    candidates = validated[validated["instance"].isin(fully_complete)]
    example = (candidates if not candidates.empty else validated).iloc[0]
    plot_trim_effect(raw_df, df, out_dir, example["instance"], example["rule"], str(example["seed"]))

    print("\nDone.")


if __name__ == "__main__":
    main()
