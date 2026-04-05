#!/usr/bin/env python3
"""
ingest_results.py — Data ingestion & analysis for FJSSP simulation batch runs.

Reads the CSV files produced by ResultsLogger, aggregates across seeds/repeats,
and produces summary statistics + comparison tables per (config, rule) pair.

Usage:
    python ingest_results.py --results-dir ./Results
    python ingest_results.py --results-dir ./Results --output summary.csv --plot
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np


# ─────────────────────────────────────────────────────────────
#  CSV Ingestion
# ─────────────────────────────────────────────────────────────

def load_results(results_dir: str) -> pd.DataFrame:
    """Load all CSV files from the results directory into a single DataFrame."""
    csv_files = glob.glob(os.path.join(results_dir, "*.csv"))
    if not csv_files:
        print(f"[ERROR] No CSV files found in {results_dir}")
        sys.exit(1)

    frames = []
    for path in sorted(csv_files):
        try:
            df = pd.read_csv(path)
            df["source_file"] = os.path.basename(path)
            frames.append(df)
            print(f"  Loaded {path} ({len(df)} rows)")
        except Exception as e:
            print(f"  [WARN] Skipping {path}: {e}")

    if not frames:
        print("[ERROR] No valid CSV files loaded.")
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)
    print(f"\nTotal rows loaded: {len(combined)}")
    return combined


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to snake_case for consistency."""
    col_map = {}
    for col in df.columns:
        normalized = col.strip().lower().replace(" ", "_")
        col_map[col] = normalized
    df = df.rename(columns=col_map)
    return df


# ─────────────────────────────────────────────────────────────
#  Analysis
# ─────────────────────────────────────────────────────────────

def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate results by (config_name, rule).
    Produces mean, std, min, max for makespan and reward.
    """
    # Try to identify config grouping columns
    group_cols = []
    for candidate in ["config_name", "name", "config"]:
        if candidate in df.columns:
            group_cols.append(candidate)
            break

    # Fall back to (job_count, machine_count) if no name column
    if not group_cols:
        for col in ["job_count", "machine_count"]:
            if col in df.columns:
                group_cols.append(col)

    # Rule column
    rule_col = None
    for candidate in ["rule_name", "rule", "dispatching_rule"]:
        if candidate in df.columns:
            rule_col = candidate
            break

    if rule_col is None:
        print("[WARN] No rule column found. Aggregating across all rules.")
    else:
        group_cols.append(rule_col)

    if not group_cols:
        print("[WARN] No grouping columns found. Showing global stats.")
        group_cols = ["source_file"]

    # Identify metric columns
    metric_cols = []
    for candidate in ["makespan", "total_reward", "decision_count", "avg_timescale"]:
        if candidate in df.columns:
            metric_cols.append(candidate)

    if not metric_cols:
        print("[ERROR] No metric columns (makespan, total_reward, etc.) found.")
        print(f"  Available columns: {list(df.columns)}")
        return pd.DataFrame()

    agg_dict = {}
    for m in metric_cols:
        agg_dict[m] = ["count", "mean", "std", "min", "max"]

    summary = df.groupby(group_cols).agg(agg_dict)
    summary.columns = ["_".join(col).strip("_") for col in summary.columns]
    summary = summary.reset_index()

    return summary


def compute_rule_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot table: rows = config, columns = rule, values = mean makespan.
    Highlights the best rule per config.
    """
    # Find the right column names
    config_col = None
    for c in ["config_name", "name", "config"]:
        if c in df.columns:
            config_col = c
            break

    rule_col = None
    for c in ["rule_name", "rule", "dispatching_rule"]:
        if c in df.columns:
            rule_col = c
            break

    if config_col is None or rule_col is None or "makespan" not in df.columns:
        print("[WARN] Cannot build rule comparison — missing columns.")
        return pd.DataFrame()

    pivot = df.pivot_table(
        index=config_col,
        columns=rule_col,
        values="makespan",
        aggfunc="mean"
    )

    # Add a "best_rule" column
    pivot["best_rule"] = pivot.idxmin(axis=1)
    pivot["best_makespan"] = pivot.min(axis=1, numeric_only=True)

    return pivot


def compute_seed_variance(df: pd.DataFrame) -> pd.DataFrame:
    """Show how much variance comes from seed differences vs. rule differences."""
    config_col = None
    for c in ["config_name", "name", "config"]:
        if c in df.columns:
            config_col = c
            break

    rule_col = None
    for c in ["rule_name", "rule", "dispatching_rule"]:
        if c in df.columns:
            rule_col = c
            break

    if not all([config_col, rule_col, "makespan" in df.columns, "seed" in df.columns]):
        return pd.DataFrame()

    stats = df.groupby([config_col, rule_col]).agg(
        runs=("makespan", "count"),
        mean_makespan=("makespan", "mean"),
        std_makespan=("makespan", "std"),
        cv_makespan=("makespan", lambda x: x.std() / x.mean() if x.mean() > 0 else 0),
    ).reset_index()

    return stats


# ─────────────────────────────────────────────────────────────
#  Optional Plotting
# ─────────────────────────────────────────────────────────────

def plot_results(df: pd.DataFrame, output_dir: str):
    """Generate comparison plots if matplotlib is available."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless-safe
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not installed — skipping plots.")
        return

    config_col = None
    for c in ["config_name", "name", "config"]:
        if c in df.columns:
            config_col = c
            break

    rule_col = None
    for c in ["rule_name", "rule", "dispatching_rule"]:
        if c in df.columns:
            rule_col = c
            break

    if not all([config_col, rule_col, "makespan" in df.columns]):
        print("[WARN] Cannot plot — missing columns.")
        return

    os.makedirs(output_dir, exist_ok=True)
    configs = df[config_col].unique()

    # 1. Grouped bar chart: makespan by rule for each config
    fig, axes = plt.subplots(1, len(configs), figsize=(6 * len(configs), 5), squeeze=False)
    for i, cfg in enumerate(configs):
        subset = df[df[config_col] == cfg]
        means = subset.groupby(rule_col)["makespan"].mean().sort_values()
        stds = subset.groupby(rule_col)["makespan"].std().reindex(means.index).fillna(0)

        ax = axes[0, i]
        colors = ["#2ecc71" if r == means.index[0] else "#3498db" for r in means.index]
        means.plot.bar(ax=ax, yerr=stds, color=colors, capsize=3)
        ax.set_title(cfg, fontsize=11)
        ax.set_ylabel("Makespan")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45)

    plt.suptitle("Mean Makespan by Rule × Config", fontsize=13)
    plt.tight_layout()
    path = os.path.join(output_dir, "makespan_comparison.png")
    plt.savefig(path, dpi=150)
    print(f"  Plot saved: {path}")
    plt.close()

    # 2. Heatmap of normalized makespan
    pivot = df.pivot_table(index=config_col, columns=rule_col, values="makespan", aggfunc="mean")
    normalized = pivot.div(pivot.min(axis=1), axis=0)  # ratio to best

    fig, ax = plt.subplots(figsize=(10, max(4, len(configs))))
    im = ax.imshow(normalized.values, cmap="RdYlGn_r", aspect="auto", vmin=1.0)
    ax.set_xticks(range(len(normalized.columns)))
    ax.set_xticklabels(normalized.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(normalized.index)))
    ax.set_yticklabels(normalized.index)
    plt.colorbar(im, label="Ratio to Best")
    ax.set_title("Makespan Ratio to Best Rule per Config")

    # Annotate cells
    for y in range(normalized.shape[0]):
        for x in range(normalized.shape[1]):
            val = normalized.values[y, x]
            ax.text(x, y, f"{val:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if val > 1.3 else "black")

    plt.tight_layout()
    path = os.path.join(output_dir, "makespan_heatmap.png")
    plt.savefig(path, dpi=150)
    print(f"  Plot saved: {path}")
    plt.close()


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ingest and analyze FJSSP simulation results.")
    parser.add_argument("--results-dir", required=True, help="Path to Results/ folder with CSVs")
    parser.add_argument("--output", default=None, help="Path to save summary CSV")
    parser.add_argument("--plot", action="store_true", help="Generate comparison plots")
    parser.add_argument("--plot-dir", default=None, help="Directory for plots (default: results-dir/plots)")
    args = parser.parse_args()

    print(f"Loading results from: {args.results_dir}\n")
    df = load_results(args.results_dir)
    df = normalize_columns(df)

    print(f"\nColumns: {list(df.columns)}")
    print(f"Unique rules: {df.get('rule_name', df.get('rule', pd.Series())).unique()}")

    # Summary stats
    print("\n" + "=" * 60)
    print("  SUMMARY STATISTICS")
    print("=" * 60)
    summary = compute_summary(df)
    if not summary.empty:
        print(summary.to_string(index=False))

    # Rule comparison
    print("\n" + "=" * 60)
    print("  RULE COMPARISON (Mean Makespan)")
    print("=" * 60)
    comparison = compute_rule_comparison(df)
    if not comparison.empty:
        print(comparison.to_string())

    # Seed variance
    print("\n" + "=" * 60)
    print("  SEED VARIANCE ANALYSIS")
    print("=" * 60)
    variance = compute_seed_variance(df)
    if not variance.empty:
        print(variance.to_string(index=False))

    # Save summary
    if args.output:
        summary.to_csv(args.output, index=False)
        print(f"\nSummary saved to: {args.output}")

    # Plots
    if args.plot:
        plot_dir = args.plot_dir or os.path.join(args.results_dir, "plots")
        print(f"\nGenerating plots in: {plot_dir}")
        plot_results(df, plot_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()