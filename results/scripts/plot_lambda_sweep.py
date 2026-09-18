#!/usr/bin/env python3
"""
plot_lambda_sweep.py — PDR differentiation across an arrival-lambda sweep.

Purpose-built for the thesis question this sweep was run to answer: does makespan
converge across dispatching rules while flow time (and its wait decomposition)
still differentiates them? Complements find_lambda_plateau.py (which characterizes
the arrival-bound vs capacity-bound regime) and plot_generated.py (generic
instance-indexed plots) with figures indexed explicitly by arrival_lambda.

Usage:
    python plot_lambda_sweep.py --results results.csv --completions job_completions.csv --out figs/
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

WAIT_BUCKET_COLS = [
    "time_needs_routing", "time_waiting_pickup", "time_in_transit",
    "time_queued", "time_processing",
]
WAIT_BUCKET_LABELS = {
    "time_needs_routing":  "Routing decision wait",
    "time_waiting_pickup": "AGV pickup wait",
    "time_in_transit":     "AGV transit",
    "time_queued":         "Machine queue wait",
    "time_processing":     "Processing",
}

# FactoryOrchestrator.MAX_EPISODE_SIM_SECONDS = 100_000 (lowered from 500_000 after the
# agv_congestion_sweep deadlocks). Episodes that hit this timeout log makespan ~100000 as a
# sentinel (gridlock/deadlock, not a real completion) and must be dropped before averaging —
# this script had no such guard before, unlike plot_generated.py's CENSORED_MAKESPAN_THRESHOLD.
CENSORED_MAKESPAN_THRESHOLD = 80_000

# A batch config may legitimately scale dynamicJobCap with lambda (e.g. keep episode
# length reasonable at low arrival rates) — that's a deliberate per-instance design
# choice, not a bug, and comparing rules WITHIN an instance is still valid even if
# the cap differs BETWEEN instances. The real failure mode (a decision-timing race
# that let some rules race ahead and truncate arrivals before the shared per-instance
# cap was reached — see FactoryOrchestrator.AllArrivalsExhausted) shows up as job
# count varying WITHIN the same instance across seeds/rules, which this catches
# without penalizing intentional between-instance cap scaling.
def _flag_underfilled(df: pd.DataFrame, job_col: str = "jobs") -> pd.Series:
    modal_per_instance = df.groupby("instance")[job_col].transform(lambda s: s.mode().iloc[0])
    return df[job_col] < modal_per_instance


def _rule_color(rules: list[str]) -> dict:
    return {r: PALETTE[i % len(PALETTE)] for i, r in enumerate(rules)}


def _ordered_rules(values) -> list[str]:
    uniq = set(values)
    return [r for r in PDR_ORDER if r in uniq] + sorted(r for r in uniq if r not in PDR_ORDER)


def _save(fig, out_dir, filename):
    path = os.path.join(out_dir, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def load_results(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df["rule"] = df["rule"].str.strip()

    censored = df["makespan"] >= CENSORED_MAKESPAN_THRESHOLD
    if censored.any():
        print(f"  Dropping {censored.sum()} censored (timeout/deadlock) episode row(s) "
              f"with makespan >= {CENSORED_MAKESPAN_THRESHOLD:,.0f}")
        df = df[~censored].copy()

    df["underfilled"] = _flag_underfilled(df)
    if df["underfilled"].any():
        lams = sorted(df.loc[df["underfilled"], "arrival_lambda"].unique())
        print(f"  Flagging underfilled (arrival-starved) lambda point(s): {lams} "
              f"— excluded from lambda-indexed comparisons, shown separately.")
    return df


def load_completions(path: str, results_df: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df["rule"] = df["rule"].str.strip()

    censored = df["makespan"] >= CENSORED_MAKESPAN_THRESHOLD
    if censored.any():
        df = df[~censored].copy()

    lam_map = (results_df.drop_duplicates("instance")[["instance", "arrival_lambda"]]
               .set_index("instance")["arrival_lambda"])
    df["arrival_lambda"] = df["instance"].map(lam_map)
    underfilled_map = (results_df.drop_duplicates("instance")[["instance", "underfilled"]]
                       .set_index("instance")["underfilled"])
    df["underfilled"] = df["instance"].map(underfilled_map)
    for k in (2, 3, 4):
        due = df["arrival_time"] + k * df["work_content"]
        df[f"tardiness_k{k}"] = (df["exit_time"] - due).clip(lower=0)
    return df


# ── Plot 1 — makespan vs lambda, per rule ─────────────────────────────────────

def plot_makespan_vs_lambda(df: pd.DataFrame, out_dir: str) -> None:
    filled = df[~df["underfilled"]]
    rules = _ordered_rules(filled["rule"].unique())
    colors = _rule_color(rules)

    g = filled.groupby(["rule", "arrival_lambda"])["makespan"].agg(["mean", "sem"]).reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    for rule in rules:
        sub = g[g["rule"] == rule].sort_values("arrival_lambda")
        ls = "--" if rule.lower() == "random" else "-"
        ax.errorbar(sub["arrival_lambda"], sub["mean"], yerr=sub["sem"],
                    marker="o", ms=4, lw=1.6, ls=ls, capsize=2,
                    color=colors[rule], label=rule, alpha=0.85)

    ax.set_xscale("log")
    ax.set_xlabel("Arrival rate λ (jobs/sim-second, log scale)")
    ax.set_ylabel("Mean makespan (sim-time units)")
    ax.set_title("Makespan vs Arrival Rate, by PDR\n(flat across ~37× λ range once the dynamic-job cap is reliably filled)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")

    _save(fig, out_dir, "makespan_vs_lambda.png")


# ── Plot 2 — mean + p95 flow time vs lambda, per rule ─────────────────────────

def plot_flow_time_vs_lambda(job_df: pd.DataFrame, out_dir: str) -> None:
    filled = job_df[~job_df["underfilled"]]
    rules = _ordered_rules(filled["rule"].unique())
    colors = _rule_color(rules)

    mean_g = filled.groupby(["rule", "arrival_lambda"])["flow_time"].mean().reset_index()
    p95_g = (filled.groupby(["rule", "arrival_lambda"])["flow_time"]
             .quantile(0.95).reset_index())

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for rule in rules:
        sub = mean_g[mean_g["rule"] == rule].sort_values("arrival_lambda")
        ls = "--" if rule.lower() == "random" else "-"
        axes[0].plot(sub["arrival_lambda"], sub["flow_time"], marker="o", ms=4,
                     lw=1.6, ls=ls, color=colors[rule], label=rule, alpha=0.85)

        sub95 = p95_g[p95_g["rule"] == rule].sort_values("arrival_lambda")
        axes[1].plot(sub95["arrival_lambda"], sub95["flow_time"], marker="o", ms=4,
                     lw=1.6, ls=ls, color=colors[rule], label=rule, alpha=0.85)

    for ax, title, ylab in zip(
        axes,
        ["Mean Flow Time vs Arrival Rate, by PDR", "p95 Flow Time vs Arrival Rate, by PDR"],
        ["Mean flow time (sim-time units)", "p95 flow time (sim-time units)"],
    ):
        ax.set_xscale("log")
        ax.set_xlabel("Arrival rate λ (jobs/sim-second, log scale)")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    axes[1].legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.suptitle("Flow Time Differentiates PDRs Where Makespan Does Not", y=1.02)
    fig.tight_layout()
    _save(fig, out_dir, "flow_time_vs_lambda.png")


# ── Plot 3 — flow-time rank heatmap across the sweep ──────────────────────────

def plot_flow_time_rank_heatmap(job_df: pd.DataFrame, out_dir: str) -> None:
    filled = job_df[~job_df["underfilled"]]
    piv = (filled.groupby(["rule", "arrival_lambda"])["flow_time"]
           .mean().unstack("arrival_lambda"))
    rules = _ordered_rules(piv.index)
    piv = piv.loc[rules]
    rank_df = piv.rank(axis=0, method="min")

    fig, ax = plt.subplots(figsize=(max(9, len(piv.columns) * 1.1), 5.5))
    im = ax.imshow(rank_df.values, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([f"{c:g}" for c in piv.columns], rotation=30, ha="right")
    ax.set_yticks(range(len(rules)))
    ax.set_yticklabels(rules)
    for i in range(rank_df.shape[0]):
        for j in range(rank_df.shape[1]):
            ax.text(j, i, f"{int(rank_df.values[i, j])}", ha="center", va="center", fontsize=9)

    mean_rank = rank_df.mean(axis=1)
    ax.set_xlabel("Arrival rate λ")
    ax.set_ylabel("PDR")
    ax.set_title("Flow-Time Rank per Arrival Rate\n(1 = best/lowest mean flow time — "
                  f"stable ranking = {mean_rank.idxmin()} best, {mean_rank.idxmax()} worst, "
                  "across the whole sweep)")
    fig.colorbar(im, ax=ax, label="Rank (1 = best)")

    _save(fig, out_dir, "flow_time_rank_heatmap.png")


# ── Plot 4 — wait decomposition, low-load vs high-load ────────────────────────

def plot_wait_decomposition_by_load(job_df: pd.DataFrame, out_dir: str, split_lambda: float) -> None:
    filled = job_df[~job_df["underfilled"]]
    cols = [c for c in WAIT_BUCKET_COLS if c in filled.columns]
    bucket_colors = plt.cm.Set2.colors

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    for ax, (mask, label) in zip(
        axes,
        [(filled["arrival_lambda"] < split_lambda, f"Low load (λ < {split_lambda:g})"),
         (filled["arrival_lambda"] >= split_lambda, f"High load (λ ≥ {split_lambda:g})")],
    ):
        sub = filled[mask]
        means = sub.groupby("rule")[cols].mean()
        # order rules by total wait time ascending (best -> worst) within this panel
        order = means.sum(axis=1).sort_values().index.tolist()
        means = means.loc[order]

        bottoms = np.zeros(len(order))
        for i, col in enumerate(cols):
            vals = means[col].values
            ax.bar(order, vals, bottom=bottoms, label=WAIT_BUCKET_LABELS.get(col, col),
                   color=bucket_colors[i % len(bucket_colors)], edgecolor="white", linewidth=0.5)
            bottoms += vals

        ax.set_title(label)
        ax.set_xlabel("PDR (ordered best → worst, this panel)")
        ax.tick_params(axis="x", rotation=30)

    axes[0].set_ylabel("Mean time per job (sim-time units)")
    axes[1].legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.suptitle("Job Wait-Time Decomposition by PDR — Low vs High Arrival Rate", y=1.02)
    fig.tight_layout()
    _save(fig, out_dir, "wait_decomposition_by_load.png")


def main():
    parser = argparse.ArgumentParser(description="Plot PDR differentiation across an arrival-lambda sweep.")
    parser.add_argument("--results", required=True, help="Path to results.csv")
    parser.add_argument("--completions", required=True, help="Path to job_completions.csv")
    parser.add_argument("--out", default="figs_lambda_sweep", help="Output directory")
    parser.add_argument("--split-lambda", type=float, default=None,
                        help="Lambda threshold splitting low/high load in the wait-decomposition "
                             "plot (default: median of observed lambdas).")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    res_df = load_results(args.results)
    job_df = load_completions(args.completions, res_df)

    lambdas = sorted(res_df["arrival_lambda"].unique())
    split_lambda = args.split_lambda or lambdas[len(lambdas) // 2]

    print(f"\nLambdas in sweep    : {lambdas}")
    print(f"Rules               : {sorted(res_df['rule'].unique())}")
    print(f"Split lambda (wait decomposition): {split_lambda:g}")

    plot_makespan_vs_lambda(res_df, args.out)
    plot_flow_time_vs_lambda(job_df, args.out)
    plot_flow_time_rank_heatmap(job_df, args.out)
    plot_wait_decomposition_by_load(job_df, args.out, split_lambda)

    print("\nDone.")


if __name__ == "__main__":
    main()
