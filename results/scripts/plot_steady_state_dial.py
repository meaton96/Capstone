"""
plot_steady_state_dial.py — Visualize the fixed-duration lambda dial-in sweep
(episodeDurationSeconds + dynamicJobCap=0). Answers: does WIP plateau (slack)
or keep climbing (saturated) at each candidate arrival rate?

Usage:
    python plot_steady_state_dial.py <results_dir> [--out <dir>]

Expects <results_dir> to contain throughput.csv, machine_utilization.csv,
agv_performance.csv, results.csv from a single batch run.
"""

import argparse
import os
import re
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
    "LPT_SMPT", "LPT_MMUR", "LRT_MMUR", "SDT_SRWT",
    "random",
]
PALETTE = plt.cm.tab10.colors


def _rule_order(present) -> list:
    present = list(present)
    return [r for r in PDR_ORDER if r in present] + [r for r in present if r not in PDR_ORDER]


def _rule_color(rules) -> dict:
    return {r: PALETTE[i % len(PALETTE)] for i, r in enumerate(rules)}


def _lambda_from_instance(name: str) -> float:
    """agv07_dial_l004 -> 0.004, agv07_dial_l010 -> 0.010"""
    m = re.search(r'l(\d+)$', name)
    return int(m.group(1)) / 1000.0 if m else float("nan")


def _sorted_instances(names) -> list:
    return sorted(set(names), key=_lambda_from_instance)


def plot_wip_by_lambda(throughput: pd.DataFrame, out_dir: str) -> None:
    instances = _sorted_instances(throughput["instance"])
    n = len(instances)
    ncols = 2
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4.3 * nrows), squeeze=False)

    rules = _rule_order(throughput["rule"].unique())
    colors = _rule_color(rules)

    for idx, inst in enumerate(instances):
        ax = axes[idx // ncols][idx % ncols]
        sub = throughput[throughput["instance"] == inst]
        lam = _lambda_from_instance(inst)
        for rule in rules:
            rs = sub[sub["rule"] == rule].sort_values("window_start")
            if rs.empty:
                continue
            ax.plot(rs["window_start"], rs["work_in_progress"], color=colors[rule],
                    linewidth=1.3, alpha=0.85, label=rule)
        ax.set_title(f"λ = {lam:.3f} jobs/s  ({inst})")
        ax.set_xlabel("Sim time (s)")
        ax.set_ylabel("Work in progress (jobs)")
        if idx == 0:
            ax.legend(fontsize=7, ncol=2, loc="upper left")

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.suptitle("WIP Over Time by Candidate Arrival Rate\n"
                  "(plateau = steady state with slack; unbounded climb = still overloaded)", y=1.02)
    fig.tight_layout()
    path = os.path.join(out_dir, "wip_by_lambda.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_saturation_by_lambda(machine_util: pd.DataFrame, agv_perf: pd.DataFrame,
                               results: pd.DataFrame, out_dir: str) -> None:
    instances = _sorted_instances(results["instance"])
    lambdas = [_lambda_from_instance(i) for i in instances]

    mu = machine_util.groupby("instance")["utilization_rate"].mean().reindex(instances)

    makespan_by_inst = results.groupby("instance")["makespan"].mean().reindex(instances)
    agv_perf = agv_perf.merge(results[["instance", "rule", "seed", "makespan"]].drop_duplicates(),
                               on=["instance", "rule", "seed"], how="left", suffixes=("", "_r"))
    agv_perf["busy_fraction"] = 1.0 - (agv_perf["time_idle"] / agv_perf["makespan"])
    agv_perf["waiting_route_fraction"] = agv_perf["time_waiting_route"] / agv_perf["makespan"]
    agv_busy = agv_perf.groupby("instance")["busy_fraction"].mean().reindex(instances)
    agv_wait = agv_perf.groupby("instance")["waiting_route_fraction"].mean().reindex(instances)

    x = np.arange(len(instances))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(x - width, mu.values, width, label="Machine utilization", color=PALETTE[0], alpha=0.85)
    ax.bar(x, agv_busy.values, width, label="AGV busy fraction (1 - idle)", color=PALETTE[1], alpha=0.85)
    ax.bar(x + width, agv_wait.values, width, label="AGV waiting-for-route fraction", color=PALETTE[3], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f"λ={l:.3f}" for l in lambdas])
    ax.set_ylabel("Fraction of episode duration")
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle=":", alpha=0.5)
    ax.set_title("System Saturation by Candidate Arrival Rate\n(mean across all 9 rules, single seed)")
    ax.legend(fontsize=8, loc="upper left")

    for i, v in enumerate(mu.values):
        ax.text(i - width, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    for i, v in enumerate(agv_busy.values):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    for i, v in enumerate(agv_wait.values):
        ax.text(i + width, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)

    fig.tight_layout()
    path = os.path.join(out_dir, "saturation_by_lambda.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def print_censoring_summary(results: pd.DataFrame) -> None:
    instances = _sorted_instances(results["instance"])
    print("\nCensoring (jobs still in-flight at episode cutoff) by lambda:")
    for inst in instances:
        sub = results[results["instance"] == inst]
        lam = _lambda_from_instance(inst)
        mean_censored = sub["jobs_censored"].mean()
        mean_jobs = sub["jobs"].mean()
        print(f"  λ={lam:.3f}  mean jobs_censored={mean_censored:.1f} / {mean_jobs:.0f} total "
              f"({100*mean_censored/mean_jobs:.0f}%)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", help="Directory containing throughput.csv, machine_utilization.csv, agv_performance.csv, results.csv")
    parser.add_argument("--out", default=None, help="Output dir (default: <results_dir>/figs)")
    args = parser.parse_args()

    out_dir = args.out or os.path.join(args.results_dir, "figs")
    os.makedirs(out_dir, exist_ok=True)

    def load(name):
        df = pd.read_csv(os.path.join(args.results_dir, name))
        df.columns = df.columns.str.strip()
        if "rule" in df.columns:
            df["rule"] = df["rule"].str.strip()
        df["instance"] = df["instance"].str.strip()
        return df

    throughput = load("throughput.csv")
    machine_util = load("machine_utilization.csv")
    agv_perf = load("agv_performance.csv")
    results = load("results.csv")

    plot_wip_by_lambda(throughput, out_dir)
    plot_saturation_by_lambda(machine_util, agv_perf, results, out_dir)
    print_censoring_summary(results)

    print("\nDone.")


if __name__ == "__main__":
    main()
