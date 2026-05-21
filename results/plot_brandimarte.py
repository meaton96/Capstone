"""
plot_brandimarte.py — Visualizations for Brandimarte MK01-MK15 benchmark results.

Usage:
    python plot_brandimarte.py <results.csv> [--out <output_dir>]

Produces:
    1. makespan_by_instance_bar.png  — grouped bar: PDR x MK instance (seed-averaged)
    2. pdr_rank_heatmap.png          — heatmap of PDR rank (1=best) per instance
    3. random_pdr_boxplot.png        — box plot of RANDOM rule variance per instance
    4. best_pdr_per_instance.png     — which PDR wins each instance
"""

import argparse
import os
import re
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from viz_utils import PDR_ORDER, PDR_LABELS

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


# ── Brandimarte canonical specs: (jobs, machines, total_ops) → MK label ──────

BRANDIMARTE_SPECS = [
    (10,  6,  55),   # MK01
    (10,  6,  58),   # MK02
    (15,  8, 150),   # MK03
    (15,  8,  90),   # MK04
    (15,  4, 106),   # MK05
    (10, 10, 150),   # MK06
    (20,  5, 100),   # MK07
    (20, 10, 225),   # MK08
    (20, 10, 240),   # MK09
    (20, 15, 240),   # MK10
    (30,  5, 179),   # MK11
    (30, 10, 193),   # MK12
    (30, 10, 231),   # MK13
    (30, 15, 277),   # MK14
    (30, 15, 284),   # MK15
]
FINGERPRINT_TO_MK = {fp: f"MK{i+1:02d}" for i, fp in enumerate(BRANDIMARTE_SPECS)}
MK_ORDER = [f"MK{i+1:02d}" for i in range(len(BRANDIMARTE_SPECS))]


def load_brandimarte(csv_path: str) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    """Load Brandimarte CSV, clean/standardize MK instance labels, split PDR vs RANDOM.

    Tries to clean and zero-pad the explicit 'instance' column from the new schema.
    Falls back to matching (jobs, machines, total_ops) if the instance column is missing.
    Also auto-detects if agvCount varies across runs for individual instances.
    """
    df = pd.read_csv(csv_path)

    # 1. Clean or derive the instance identifier
    if "instance" in df.columns and df["instance"].notna().any():
        def clean_instance(val):
            if pd.isna(val):
                return None
            # Matches 'MK1', 'mk01', 'instances/MK_1.fjs', etc., extracting the digits
            match = re.search(r'MK\s*[_]*(\d+)', str(val), re.IGNORECASE)
            return f"MK{int(match.group(1)):02d}" if match else None

        df["instance"] = df["instance"].apply(clean_instance)
    else:
        # Fallback to original fingerprinting logic
        fps = list(zip(df["jobs"], df["machines"], df["total_ops"]))
        df["instance"] = [FINGERPRINT_TO_MK.get(fp) for fp in fps]

    # Drop rows that don't match valid canonical targets
    unknown = df[df["instance"].isna() | ~df["instance"].isin(MK_ORDER)]
    if len(unknown) > 0:
        bad = unknown[["instance", "jobs", "machines", "total_ops"]].drop_duplicates()
        print(f"⚠ Dropping {len(unknown)} rows that don't match any standard MK01-MK15 spec:")
        print(bad.to_string(index=False))
        df = df[df["instance"].isin(MK_ORDER)].copy()

    # 2. Automatically check for AGV count discrepancies across runs
    auto_agv_warning = False
    if "agvCount" in df.columns:
        # If any single instance contains multiple unique AGV counts, trigger the warning flag
        agv_variance = df.groupby("instance")["agvCount"].nunique()
        if (agv_variance > 1).any():
            auto_agv_warning = True
            print("⚠ Auto-detected varying AGV counts across runs within the same instance data.")

    rand_mask = df["rule"].str.lower() == "random"
    rand_df = df[rand_mask].copy()
    pdr_df = df[~rand_mask].copy()
    
    return pdr_df, rand_df, auto_agv_warning


def _sorted_instances(df: pd.DataFrame) -> list[str]:
    """Return instances present in df, ordered MK01 → MK15."""
    present = set(df["instance"].unique())
    return [m for m in MK_ORDER if m in present]


def _rule_color(rules):
    return {r: PALETTE[i % len(PALETTE)] for i, r in enumerate(rules)}


# ── Plot 1 — grouped bar: makespan per PDR per MK instance ──────────────────

def plot_bar_by_instance(pdr_df: pd.DataFrame, rand_df: pd.DataFrame,
                         out_dir: str, agv_warning: bool) -> None:
    instances = _sorted_instances(pdr_df)
    rules = [r for r in PDR_ORDER if r in pdr_df["rule"].values]
    colors = _rule_color(rules)

    # Seed-averaged makespan per (rule, instance)
    pdr_mean = (pdr_df.groupby(["rule", "instance"])["makespan"]
                      .mean()
                      .unstack("instance"))
    rand_means = rand_df.groupby("instance")["makespan"].mean()

    x = np.arange(len(instances))
    width = 0.8 / (len(rules) + 1)

    fig, ax = plt.subplots(figsize=(max(14, len(instances) * 1.1), 6))

    for i, rule in enumerate(rules):
        vals = [pdr_mean.loc[rule, inst] if inst in pdr_mean.columns else np.nan
                for inst in instances]
        offset = (i - len(rules) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=rule, color=colors[rule], alpha=0.85)

    # RANDOM mean as the last group
    rand_vals = [rand_means.get(inst, np.nan) for inst in instances]
    offset = (len(rules) - len(rules) / 2 + 0.5) * width
    ax.bar(x + offset, rand_vals, width, label="RANDOM (mean)",
           color="gray", alpha=0.6, hatch="//")

    ax.set_xticks(x)
    ax.set_xticklabels(instances, rotation=45, ha="right")
    ax.set_xlabel("Brandimarte Instance")
    ax.set_ylabel("Makespan (sim-time units, seed-averaged)")
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
    instances = _sorted_instances(pdr_df)

    # Seed-averaged makespan per (rule, instance)
    pivot = pdr_df.pivot_table(index="rule", columns="instance",
                               values="makespan", aggfunc="mean")

    rand_means = rand_df.groupby("instance")["makespan"].mean()
    pivot.loc["RANDOM"] = [rand_means.get(i, np.nan) for i in instances]

    pivot = pivot[instances]  # enforce MK01..MK15 column order

    # Rank: 1 = lowest makespan (best)
    rank_df = pivot.rank(axis=0, method="min").astype(float)

    ordered_rules = [r for r in PDR_ORDER if r in rank_df.index] + \
                    [r for r in rank_df.index if r not in PDR_ORDER]
    rank_df = rank_df.loc[ordered_rules]

    fig, ax = plt.subplots(figsize=(max(10, len(instances) * 0.9), 5))
    sns.heatmap(rank_df, annot=True, fmt=".0f", cmap="RdYlGn_r",
                linewidths=0.4, ax=ax,
                cbar_kws={"label": "Rank (1 = best makespan)"})
    ax.set_title("PDR Rank per Brandimarte Instance\n(1 = shortest makespan, seed-averaged)")
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
    # 1. Filter to include only instances that have data in rand_df
    present_instances = [i for i in MK_ORDER if i in rand_df["instance"].unique()]
    
    # 2. Build the data list and match the labels list to it
    data = [rand_df[rand_df["instance"] == i]["makespan"].values for i in present_instances]
    
    # 3. Only proceed if we actually have data to plot
    if not data:
        print("  Skipping random_pdr_boxplot: No RANDOM data found.")
        return

    n_per = [len(d) for d in data]

    fig, ax = plt.subplots(figsize=(max(10, len(present_instances) * 0.9), 5))
    
    # Use the filtered present_instances list for tick_labels
    bp = ax.boxplot(data, tick_labels=present_instances, patch_artist=True,
                    medianprops={"color": "black", "linewidth": 2})

    for patch in bp["boxes"]:
        patch.set_facecolor("steelblue")
        patch.set_alpha(0.6)

    ax.set_xlabel("Brandimarte Instance")
    ax.set_ylabel("Makespan (sim-time units)")
    n_note = f"{min(n_per)}" if min(n_per) == max(n_per) else f"{min(n_per)}–{max(n_per)}"
    ax.set_title(f"RANDOM PDR Makespan Variance per Instance\n({n_note} seeds per instance)")
    ax.tick_params(axis="x", rotation=45)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    fig.tight_layout()
    path = os.path.join(out_dir, "random_pdr_boxplot.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Plot 4 — which PDR wins each instance ────────────────────────────────────

def plot_best_pdr(pdr_df: pd.DataFrame, rand_df: pd.DataFrame, out_dir: str) -> None:
    instances = _sorted_instances(pdr_df)
    rules = [r for r in PDR_ORDER if r in pdr_df["rule"].values]
    colors = _rule_color(rules + ["RANDOM"])

    # Seed-averaged lookup
    pdr_mean = pdr_df.groupby(["rule", "instance"])["makespan"].mean()
    rand_mean = rand_df.groupby("instance")["makespan"].mean()

    best_rules, best_makespans = [], []
    for inst in instances:
        candidates = {rule: pdr_mean.get((rule, inst), np.nan) for rule in rules}
        candidates = {k: v for k, v in candidates.items() if not np.isnan(v)}
        if inst in rand_mean.index and not np.isnan(rand_mean.loc[inst]):
            candidates["RANDOM"] = rand_mean.loc[inst]
        if not candidates:
            best_rules.append("—")
            best_makespans.append(np.nan)
            continue
        best_rule = min(candidates, key=candidates.get)
        best_rules.append(best_rule)
        best_makespans.append(candidates[best_rule])

    bar_colors = [colors.get(r, "gray") for r in best_rules]
    x = np.arange(len(instances))

    fig, ax = plt.subplots(figsize=(max(10, len(instances)), 5))
    bars = ax.bar(x, best_makespans, color=bar_colors, alpha=0.85, edgecolor="white")
    ymax = np.nanmax(best_makespans) if best_makespans else 1.0
    for bar, rule in zip(bars, best_rules):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + ymax * 0.01,
                rule.replace("_", "\n"), ha="center", va="bottom",
                fontsize=6.5, rotation=0)

    ax.set_xticks(x)
    ax.set_xticklabels(instances, rotation=45, ha="right")
    ax.set_xlabel("Brandimarte Instance")
    ax.set_ylabel("Best Makespan Achieved (seed-averaged)")
    ax.set_title("Best-Performing PDR per Brandimarte Instance")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

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
    pdr_df, rand_df, auto_agv_warning = load_brandimarte(args.csv)

    # Combine manual CLI flag with automated column inspection
    effective_agv_warning = args.agv_warning or auto_agv_warning

    # Sanity summary — catches seed-count mismatches early
    print(f"\nDeterministic PDR rows  : {len(pdr_df)}")
    print(f"RANDOM rule rows        : {len(rand_df)}")
    print(f"Matched MK instances    : {_sorted_instances(pdr_df)}")

    seeds_per = (pdr_df.groupby(["rule", "instance"]).size()
                        .reset_index(name="n_seeds"))
    n_range = (seeds_per["n_seeds"].min(), seeds_per["n_seeds"].max())
    print(f"Seeds per (rule, inst)  : {n_range[0]} – {n_range[1]}\n")

    plot_bar_by_instance(pdr_df, rand_df, args.out, agv_warning=effective_agv_warning)
    plot_rank_heatmap(pdr_df, rand_df, args.out)
    plot_random_variance(rand_df, args.out)
    plot_best_pdr(pdr_df, rand_df, args.out)

    print("\nDone.")


if __name__ == "__main__":
    main()