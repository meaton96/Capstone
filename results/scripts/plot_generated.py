"""
plot_generated.py — Visualizations for randomly-generated FJSSP instance results.

Usage:
    python plot_generated.py <results.csv> [--out <output_dir>] [--regime <tag>]

Arguments:
    csv             Path to merged_results.csv from the generated job batch run
    --out           Output directory (default: plots_generated)
    --regime        Filter to a single stochastic_tag value, e.g. "none", "mf", "arr"
                    If omitted, deterministic runs (stochastic_tag == "none") are used
                    for rank/bar plots, and stochastic runs get a separate comparison.

Produces:
    1. makespan_by_instance_bar.png     — grouped bar: PDR × config (seed-averaged)
    2. pdr_rank_heatmap.png             — heatmap of PDR rank (1=best) per config
    3. random_pdr_boxplot.png           — box plot of RANDOM rule variance per config
    4. pdr_rank_distribution.png        — stacked bar: rank distribution per PDR
    5. makespan_vs_size.png             — line plot: PDR makespan scaling with problem size
    6. stochastic_inflation.png         — makespan inflation ratio vs deterministic baseline
                                          (only produced if stochastic rows are present)
    7. pdr_stochastic_sensitivity.png   — which PDRs degrade most under disruption
                                          (only produced if stochastic rows are present)
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

warnings.filterwarnings("ignore", category=FutureWarning)

plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# ── PDR ordering and display labels ──────────────────────────────────────────
# Import from viz_utils if available, otherwise use inline defaults.
try:
    from viz_utils import PDR_ORDER, PDR_LABELS  # type: ignore
except ImportError:
    PDR_ORDER = [
        "SPT_SMPT", "SPT_SRWT", "SRT_SMPT", "SRT_SRWT",
        "LPT_SMPT", "LPT_MMUR", "LRT_MMUR", "SDT_SRWT",
        "Random",
    ]
    PDR_LABELS = {r: r for r in PDR_ORDER}

PALETTE = plt.cm.tab10.colors

# Stochastic tag display names for plot labels
TAG_LABELS = {
    "none":     "Deterministic",
    "mf":       "Machine failures",
    "agv":      "AGV failures",
    "arr":      "Dynamic arrivals",
    "mf+arr":   "Failures + arrivals",
    "mf+agv":   "Failures + AGV",
    "mf+agv+arr": "Full disruption",
}


# ── Data loading ──────────────────────────────────────────────────────────────

def _instance_sort_key(name: str) -> tuple:
    """
    Sort config names by job count then machine count, extracted from the name
    or from the jobs/machines columns. Falls back to alphabetical.
    Pattern matches: small_5j_5m, medium_20j_15m, large_50j_15m, xlarge_100j_20m
    """
    m = re.search(r'(\d+)j[_]?(\d+)m', str(name))
    if m:
        return (int(m.group(1)), int(m.group(2)), str(name))
    return (0, 0, str(name))


def load_generated(csv_path: str, regime_filter: str | None = None
                   ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load generated results CSV. Returns (det_df, stoch_df, full_df).

    det_df   — rows where stochastic_tag == "none"
    stoch_df — all other rows
    full_df  — unfiltered

    If regime_filter is provided, det_df is replaced by rows matching that tag
    and stoch_df contains everything else.
    """
    df = pd.read_csv(csv_path)

    # Normalise column names
    df.columns = df.columns.str.strip()

    # Normalise stochastic_tag: treat NaN and empty as "none"
    if "stochastic_tag" in df.columns:
        df["stochastic_tag"] = (df["stochastic_tag"]
                                .fillna("none")
                                .astype(str)
                                .str.strip()
                                .replace({"": "none", "nan": "none", "None": "none"}))
    else:
        df["stochastic_tag"] = "none"

    # Normalise rule name capitalisation to match PDR_ORDER
    df["rule"] = df["rule"].str.strip()

    # Infer instance sort order from jobs/machines columns if available
    if "jobs" in df.columns and "machines" in df.columns:
        # Build a stable ordering: (jobs, machines, name)
        size_map = (df.groupby("instance")[["jobs", "machines"]]
                    .first()
                    .reset_index())
        size_map["_sort"] = list(zip(size_map["jobs"], size_map["machines"], size_map["instance"]))
        ordered = size_map.sort_values(["jobs", "machines", "instance"])["instance"].tolist()
    else:
        ordered = sorted(df["instance"].unique(), key=_instance_sort_key)

    df["instance"] = pd.Categorical(df["instance"], categories=ordered, ordered=True)

    # Split
    if regime_filter:
        det_df   = df[df["stochastic_tag"] == regime_filter].copy()
        stoch_df = df[df["stochastic_tag"] != regime_filter].copy()
    else:
        det_df   = df[df["stochastic_tag"] == "none"].copy()
        stoch_df = df[df["stochastic_tag"] != "none"].copy()

    return det_df, stoch_df, df


def _sorted_instances(df: pd.DataFrame) -> list[str]:
    """Return instances in their categorical order."""
    if hasattr(df["instance"], "cat"):
        return [i for i in df["instance"].cat.categories if i in df["instance"].values]
    return sorted(df["instance"].unique(), key=_instance_sort_key)


def _rule_color(rules: list[str]) -> dict:
    return {r: PALETTE[i % len(PALETTE)] for i, r in enumerate(rules)}


def _save(fig: plt.Figure, out_dir: str, filename: str) -> None:
    path = os.path.join(out_dir, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Plot 1 — grouped bar: makespan per PDR per config ────────────────────────

def plot_bar_by_instance(det_df: pd.DataFrame, out_dir: str) -> None:
    instances = _sorted_instances(det_df)
    rand_mask = det_df["rule"].str.lower() == "random"
    pdr_df    = det_df[~rand_mask]
    rand_df   = det_df[rand_mask]

    rules  = [r for r in PDR_ORDER if r in pdr_df["rule"].values]
    colors = _rule_color(rules)

    pdr_mean  = (pdr_df.groupby(["rule", "instance"], observed=True)["makespan"]
                        .mean().unstack("instance"))
    rand_means = rand_df.groupby("instance", observed=True)["makespan"].mean()

    x     = np.arange(len(instances))
    width = 0.8 / (len(rules) + 1)

    fig, ax = plt.subplots(figsize=(max(12, len(instances) * 2.2), 6))

    for i, rule in enumerate(rules):
        vals   = [pdr_mean.loc[rule, inst] if (rule in pdr_mean.index and inst in pdr_mean.columns)
                  else np.nan for inst in instances]
        offset = (i - len(rules) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=rule, color=colors[rule], alpha=0.85)

    rand_vals = [rand_means.get(inst, np.nan) for inst in instances]
    offset    = (len(rules) - len(rules) / 2 + 0.5) * width
    ax.bar(x + offset, rand_vals, width, label="Random (mean)",
           color="gray", alpha=0.6, hatch="//")

    ax.set_xticks(x)
    ax.set_xticklabels(instances, rotation=30, ha="right")
    ax.set_xlabel("Config")
    ax.set_ylabel("Makespan (sim-time units, seed-averaged)")
    ax.set_title("Makespan by PDR — Generated Instances (Deterministic)")
    ax.legend(title="PDR", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    _save(fig, out_dir, "makespan_by_instance_bar.png")


# ── Plot 2 — heatmap: PDR rank per config ────────────────────────────────────

def plot_rank_heatmap(det_df: pd.DataFrame, out_dir: str) -> None:
    instances = _sorted_instances(det_df)
    rand_mask = det_df["rule"].str.lower() == "random"

    pivot = (det_df[~rand_mask]
             .pivot_table(index="rule", columns="instance",
                          values="makespan", aggfunc="mean", observed=True))

    rand_means = det_df[rand_mask].groupby("instance", observed=True)["makespan"].mean()
    pivot.loc["Random"] = [rand_means.get(i, np.nan) for i in instances]
    pivot = pivot.reindex(columns=instances)

    rank_df = pivot.rank(axis=0, method="min").astype(float)
    ordered_rules = [r for r in PDR_ORDER if r in rank_df.index] + \
                    [r for r in rank_df.index if r not in PDR_ORDER]
    rank_df = rank_df.loc[ordered_rules]

    fig, ax = plt.subplots(figsize=(max(8, len(instances) * 1.6), 5))
    sns.heatmap(rank_df, annot=True, fmt=".0f", cmap="RdYlGn_r",
                linewidths=0.4, ax=ax,
                cbar_kws={"label": "Rank (1 = best makespan)"})
    ax.set_title("PDR Rank per Generated Instance\n(1 = shortest makespan, seed-averaged)")
    ax.set_xlabel("Config")
    ax.set_ylabel("PDR")
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)

    _save(fig, out_dir, "pdr_rank_heatmap.png")


# ── Plot 3 — box plot: RANDOM variance per config ────────────────────────────

def plot_random_variance(det_df: pd.DataFrame, out_dir: str) -> None:
    rand_df   = det_df[det_df["rule"].str.lower() == "random"]
    instances = _sorted_instances(rand_df)

    data = [rand_df[rand_df["instance"] == i]["makespan"].values for i in instances]
    data = [(d, inst) for d, inst in zip(data, instances) if len(d) > 0]

    if not data:
        print("  Skipping random_pdr_boxplot: No RANDOM data found.")
        return

    vals, labels = zip(*data)
    n_per = [len(d) for d in vals]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.6), 5))
    bp = ax.boxplot(vals, tick_labels=labels, patch_artist=True,
                    medianprops={"color": "black", "linewidth": 2})
    for patch in bp["boxes"]:
        patch.set_facecolor("steelblue")
        patch.set_alpha(0.6)

    ax.set_xlabel("Config")
    ax.set_ylabel("Makespan (sim-time units)")
    n_note = (f"{min(n_per)}" if min(n_per) == max(n_per)
              else f"{min(n_per)}–{max(n_per)}")
    ax.set_title(f"Random PDR Makespan Variance per Generated Instance\n"
                 f"({n_note} seeds per config)")
    ax.tick_params(axis="x", rotation=30)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    _save(fig, out_dir, "random_pdr_boxplot.png")


# ── Plot 4 — PDR rank distribution (stacked bar) ─────────────────────────────

def plot_pdr_rank_distribution(det_df: pd.DataFrame, out_dir: str) -> None:
    instances  = _sorted_instances(det_df)
    n_instances = len(instances)
    rand_mask  = det_df["rule"].str.lower() == "random"

    pivot = (det_df[~rand_mask]
             .pivot_table(index="rule", columns="instance",
                          values="makespan", aggfunc="mean", observed=True))
    rand_means = det_df[rand_mask].groupby("instance", observed=True)["makespan"].mean()
    pivot.loc["Random"] = [rand_means.get(i, np.nan) for i in instances]
    pivot = pivot.reindex(columns=instances)

    rank_df = pivot.rank(axis=0, method="min").astype(int)
    ordered_rules = [r for r in PDR_ORDER if r in rank_df.index] + \
                    [r for r in rank_df.index if r not in PDR_ORDER]
    rank_df = rank_df.loc[ordered_rules]

    n_rules      = len(ordered_rules)
    rank_counts  = np.zeros((n_rules, n_rules), dtype=int)
    for r_idx, rule in enumerate(ordered_rules):
        for rank_val in range(1, n_rules + 1):
            rank_counts[r_idx, rank_val - 1] = (rank_df.loc[rule] == rank_val).sum()

    rank_colors = plt.cm.RdYlGn(np.linspace(1.0, 0.0, n_rules))

    fig, ax = plt.subplots(figsize=(11, 5))
    bottoms = np.zeros(n_rules)
    for rank_idx in range(n_rules):
        vals = rank_counts[:, rank_idx]
        bars = ax.bar(ordered_rules, vals, bottom=bottoms,
                      color=rank_colors[rank_idx],
                      label=f"Rank {rank_idx + 1}",
                      edgecolor="white", linewidth=0.4)
        if rank_idx == 0:
            for bar, val in zip(bars, vals):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bottoms[list(ordered_rules).index(
                                ordered_rules[list(bars).index(bar)]
                            )] + val / 2,
                            str(val),
                            ha="center", va="center",
                            fontsize=9, fontweight="bold", color="white")
        bottoms += vals

    ax.set_ylim(0, n_instances * 1.05)
    ax.set_yticks(range(0, n_instances + 1, max(1, n_instances // 5)))
    ax.set_ylabel("Number of configs")
    ax.set_xlabel("PDR")
    ax.set_title(
        f"PDR Rank Distribution Across Generated Instances (n={n_instances})\n"
        "White numbers = rank-1 (win) count"
    )
    ax.tick_params(axis="x", rotation=30)

    handles, labels = ax.get_legend_handles_labels()
    keep = [0, n_rules // 2, n_rules - 1]
    ax.legend([handles[i] for i in keep], [labels[i] for i in keep],
              title="Rank", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)

    expected = n_instances / n_rules
    ax.axhline(expected, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.text(n_rules - 0.5, expected + 0.1,
            f"chance baseline ({expected:.1f})",
            ha="right", va="bottom", fontsize=7, color="gray")

    _save(fig, out_dir, "pdr_rank_distribution.png")


# ── Plot 5 — makespan scaling with problem size ───────────────────────────────

def plot_makespan_vs_size(det_df: pd.DataFrame, out_dir: str) -> None:
    """
    Line plot: x = total_ops (proxy for problem size), y = seed-averaged makespan.
    One line per PDR. Shows whether PDR relative performance changes as problems scale.
    Requires jobs/machines/total_ops columns (present in results.csv).
    """
    if "total_ops" not in det_df.columns:
        print("  Skipping makespan_vs_size: total_ops column not found.")
        return

    # Use mean total_ops per instance as the x-axis value
    size_map = (det_df.groupby("instance", observed=True)["total_ops"]
                .mean().reset_index()
                .rename(columns={"total_ops": "mean_total_ops"}))

    g = (det_df.groupby(["rule", "instance"], observed=True)["makespan"]
         .mean().reset_index())
    g = g.merge(size_map, on="instance")

    rules  = [r for r in PDR_ORDER if r in g["rule"].values]
    rand_rules = [r for r in g["rule"].unique() if r.lower() == "random"]
    all_rules  = rules + [r for r in rand_rules if r not in rules]
    colors = _rule_color(all_rules)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Left: absolute makespan
    ax = axes[0]
    for rule in all_rules:
        sub = g[g["rule"] == rule].sort_values("mean_total_ops")
        ls  = "--" if rule.lower() == "random" else "-"
        ax.plot(sub["mean_total_ops"], sub["makespan"], "o" + ls,
                color=colors[rule], linewidth=1.8, markersize=5,
                label=rule, alpha=0.85)

    ax.set_xlabel("Total operations (proxy for problem size)")
    ax.set_ylabel("Mean makespan (sim-time units)")
    ax.set_title("Makespan Scaling with Problem Size")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")

    # Right: makespan normalised to SPT_SMPT baseline (or first rule)
    ax = axes[1]
    baseline_rule = "SPT_SMPT" if "SPT_SMPT" in all_rules else all_rules[0]
    baseline = (g[g["rule"] == baseline_rule]
                .set_index("instance")["makespan"]
                .rename("baseline"))
    g2 = g.merge(baseline, on="instance")
    g2["relative_makespan"] = g2["makespan"] / g2["baseline"]

    for rule in all_rules:
        sub = g2[g2["rule"] == rule].sort_values("mean_total_ops")
        ls  = "--" if rule.lower() == "random" else "-"
        ax.plot(sub["mean_total_ops"], sub["relative_makespan"], "o" + ls,
                color=colors[rule], linewidth=1.8, markersize=5,
                label=rule, alpha=0.85)

    ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8, alpha=0.4)
    ax.set_xlabel("Total operations (proxy for problem size)")
    ax.set_ylabel(f"Makespan relative to {baseline_rule}")
    ax.set_title("Relative PDR Performance vs Problem Size")
    ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")

    fig.suptitle("Makespan Scaling — Generated Instances (Deterministic)", y=1.02)
    fig.tight_layout()
    _save(fig, out_dir, "makespan_vs_size.png")


# ── Plot 6 — stochastic makespan inflation ────────────────────────────────────

def plot_stochastic_inflation(det_df: pd.DataFrame, stoch_df: pd.DataFrame,
                               out_dir: str) -> None:
    """
    Per-(instance, rule) inflation ratio = stochastic makespan / deterministic makespan.
    One subplot per stochastic tag. Reveals which configs are most disruption-sensitive.
    """
    if stoch_df.empty:
        print("  Skipping stochastic_inflation: no stochastic rows found.")
        return

    det_mean = (det_df.groupby(["instance", "rule"], observed=True)["makespan"]
                .mean().reset_index()
                .rename(columns={"makespan": "det_makespan"}))

    tags = sorted(stoch_df["stochastic_tag"].unique())
    n    = len(tags)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6), squeeze=False)

    instances = _sorted_instances(det_df)
    rules     = [r for r in PDR_ORDER if r in det_df["rule"].values] + \
                [r for r in det_df["rule"].unique() if r not in PDR_ORDER]
    colors    = _rule_color(rules)

    for ax, tag in zip(axes[0], tags):
        sub = stoch_df[stoch_df["stochastic_tag"] == tag]
        sto_mean = (sub.groupby(["instance", "rule"], observed=True)["makespan"]
                    .mean().reset_index()
                    .rename(columns={"makespan": "sto_makespan"}))
        merged = sto_mean.merge(det_mean, on=["instance", "rule"])
        merged["inflation"] = merged["sto_makespan"] / merged["det_makespan"]

        x = np.arange(len(instances))
        width = 0.8 / max(len(rules), 1)

        for i, rule in enumerate(rules):
            vals = [merged.loc[(merged["instance"] == inst) & (merged["rule"] == rule),
                               "inflation"].values
                    for inst in instances]
            vals = [v[0] if len(v) > 0 else np.nan for v in vals]
            offset = (i - len(rules) / 2 + 0.5) * width
            ax.bar(x + offset, vals, width, label=rule,
                   color=colors[rule], alpha=0.8)

        ax.axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(instances, rotation=30, ha="right")
        ax.set_xlabel("Config")
        ax.set_ylabel("Makespan inflation (stoch / det)")
        ax.set_title(TAG_LABELS.get(tag, tag))
        if ax == axes[0][0]:
            ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")

    fig.suptitle("Stochastic Makespan Inflation vs Deterministic Baseline", y=1.02)
    fig.tight_layout()
    _save(fig, out_dir, "stochastic_inflation.png")


# ── Plot 7 — PDR sensitivity to disruption ────────────────────────────────────

def plot_pdr_stochastic_sensitivity(det_df: pd.DataFrame, stoch_df: pd.DataFrame,
                                     out_dir: str) -> None:
    """
    For each PDR: mean inflation across all instances per stochastic tag.
    Reveals which rules are robust vs brittle under disruption.
    """
    if stoch_df.empty:
        print("  Skipping pdr_stochastic_sensitivity: no stochastic rows found.")
        return

    det_mean = (det_df.groupby(["instance", "rule"], observed=True)["makespan"]
                .mean().reset_index()
                .rename(columns={"makespan": "det_makespan"}))

    records = []
    for tag in stoch_df["stochastic_tag"].unique():
        sub = stoch_df[stoch_df["stochastic_tag"] == tag]
        sto_mean = (sub.groupby(["instance", "rule"], observed=True)["makespan"]
                    .mean().reset_index()
                    .rename(columns={"makespan": "sto_makespan"}))
        merged = sto_mean.merge(det_mean, on=["instance", "rule"])
        merged["inflation"] = merged["sto_makespan"] / merged["det_makespan"]
        agg = (merged.groupby("rule")["inflation"]
               .agg(mean_inflation="mean", std_inflation="std")
               .reset_index())
        agg["tag"] = tag
        records.append(agg)

    if not records:
        return

    all_inf   = pd.concat(records, ignore_index=True)
    tags      = sorted(all_inf["tag"].unique())
    rules     = [r for r in PDR_ORDER if r in all_inf["rule"].values] + \
                [r for r in all_inf["rule"].unique() if r not in PDR_ORDER]
    tag_colors = dict(zip(tags, plt.cm.Set2.colors[:len(tags)]))

    fig, ax = plt.subplots(figsize=(11, 6))
    x     = np.arange(len(rules))
    width = 0.8 / max(len(tags), 1)

    for i, tag in enumerate(tags):
        sub    = all_inf[all_inf["tag"] == tag].set_index("rule")
        means  = [sub.loc[r, "mean_inflation"] if r in sub.index else np.nan for r in rules]
        stds   = [sub.loc[r, "std_inflation"]  if r in sub.index else 0       for r in rules]
        offset = (i - len(tags) / 2 + 0.5) * width
        ax.bar(x + offset, means, width,
               label=TAG_LABELS.get(tag, tag),
               color=tag_colors[tag], alpha=0.85, edgecolor="white")
        ax.errorbar(x + offset, means, yerr=stds,
                    fmt="none", color="black", capsize=3, linewidth=1)

    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.4,
               label="No inflation (= det baseline)")
    ax.set_xticks(x)
    ax.set_xticklabels(rules, rotation=30, ha="right")
    ax.set_xlabel("PDR")
    ax.set_ylabel("Mean makespan inflation (stoch / det)")
    ax.set_title("PDR Sensitivity to Stochastic Disruption\n"
                 "Higher bar = more degradation under disruption")
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")

    _save(fig, out_dir, "pdr_stochastic_sensitivity.png")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualise randomly-generated FJSSP instance results."
    )
    parser.add_argument("csv",
                        help="Path to merged_results.csv from the generated batch run")
    parser.add_argument("--out", default="plots_generated",
                        help="Output directory (default: plots_generated)")
    parser.add_argument("--regime", default=None,
                        help="Treat this stochastic_tag as the 'deterministic' baseline "
                             "(default: 'none'). Use if you ran a non-standard baseline tag.")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    det_df, stoch_df, full_df = load_generated(args.csv, regime_filter=args.regime)

    # ── Sanity summary ────────────────────────────────────────────────────────
    print(f"\nTotal rows          : {len(full_df)}")
    print(f"Deterministic rows  : {len(det_df)}")
    print(f"Stochastic rows     : {len(stoch_df)}")
    print(f"Instances (det)     : {_sorted_instances(det_df)}")
    print(f"Rules               : {sorted(full_df['rule'].unique())}")
    if not stoch_df.empty:
        print(f"Stochastic tags     : {sorted(stoch_df['stochastic_tag'].unique())}")

    if not det_df.empty:
        seeds_per = (det_df.groupby(["rule", "instance"], observed=True)
                    .size().reset_index(name="n_seeds"))
        n_range = (seeds_per["n_seeds"].min(), seeds_per["n_seeds"].max())
        print(f"Seeds per (rule, instance): {n_range[0]} – {n_range[1]}")

    print()

    # ── Deterministic plots ───────────────────────────────────────────────────
    if det_df.empty:
        print("  No deterministic rows found — skipping plots 1–5.")
    else:
        plot_bar_by_instance(det_df, args.out)
        plot_rank_heatmap(det_df, args.out)
        plot_random_variance(det_df, args.out)
        plot_pdr_rank_distribution(det_df, args.out)
        plot_makespan_vs_size(det_df, args.out)

    # ── Stochastic comparison plots ───────────────────────────────────────────
    if not stoch_df.empty and not det_df.empty:
        plot_stochastic_inflation(det_df, stoch_df, args.out)
        plot_pdr_stochastic_sensitivity(det_df, stoch_df, args.out)
    elif not stoch_df.empty:
        print("  Stochastic rows found but no deterministic baseline — "
              "skipping inflation plots.")

    print("\nDone.")


if __name__ == "__main__":
    main()
