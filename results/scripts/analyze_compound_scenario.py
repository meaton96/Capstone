#!/usr/bin/env python3
"""
analyze_compound_scenario.py

Plots WIP and time-in-system (flow-time) over sim-time for the
`compound_scenario` hand-crafted instance (linux_server/BatchConfigs/
Scenarios/generate_scenarios.py), ALL rules overlaid on shared axes (not
small multiples -- one line per rule per figure) so cross-rule divergence is
visible at a glance, with phase-boundary bands pulled straight from the
scenario JSON's "_phases" metadata (not hardcoded here, so it can't drift
out of sync with the generator).

v1 of this script used small-multiples (one panel per rule) and raw
per-step WIP; both made it hard to actually see rules differ, especially
once compound_scenario v1's phases turned out too short to reach anything
past an arrival-spike shape. compound_scenario v2 is long enough that a
rolling window is now the right unit, and overlaying (not faceting) is what
actually shows one rule diverging from another.

Usage:
    python3 analyze_compound_scenario.py \
        --results-dir /path/to/results/scenarios_merged \
        --scenario-json /path/to/linux_server/BatchConfigs/Scenarios/compound_scenario.json \
        --out-dir /path/to/output/figs
"""
import argparse
import csv
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

RULES_ORDER = ["FIFO_SRWT", "SPT_SRWT", "SPT_SMPT", "LPT_SMPT", "LRT_MMUR",
               "LPT_MMUR", "SRT_SRWT", "SRT_SMPT", "random"]
COLORS = dict(zip(RULES_ORDER, cm.tab10(np.linspace(0, 1, len(RULES_ORDER)))))


def load_phases(scenario_json_path):
    with open(scenario_json_path) as f:
        data = json.load(f)
    return data.get("_phases", [])


def load_job_completions(results_dir, instance_name):
    path = os.path.join(results_dir, "job_completions.csv")
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["instance"] != instance_name:
                continue
            completed = r["completed"] in ("1", "True")
            rows.append({
                "rule": r["rule"],
                "seed": r["seed"],
                "arrival": float(r["arrival_time"]),
                "exit": float(r["exit_time"]) if completed else None,
                "flow": float(r["flow_time"]) if completed else None,
            })
    return rows


def first_seed_only(rows):
    """Scenarios are deterministic (except `random`) -- collapse repeats to
    one seed so a repeated-run isn't triple-weighted in the series."""
    if not rows:
        return rows
    seeds = sorted(set(r["seed"] for r in rows))
    return [r for r in rows if r["seed"] == seeds[0]]


def rolling_wip(rows, t_max, window=200.0, step=25.0):
    """WIP averaged over a sliding window (not instantaneous) -- smooths the
    per-job step function into something comparable across rules with very
    different job counts in flight at once."""
    centers = np.arange(window / 2, t_max + step, step)
    wip = np.zeros_like(centers)
    for j in rows:
        start = j["arrival"]
        end = j["exit"] if j["exit"] is not None else t_max
        # fraction of each window during which this job was in-system
        lo = np.maximum(centers - window / 2, start)
        hi = np.minimum(centers + window / 2, end)
        overlap = np.clip(hi - lo, 0, None)
        wip += overlap / window
    return centers, wip


def rolling_flowtime(rows, t_max, window=300.0, step=50.0):
    """Mean flow-time of jobs whose exit falls within each sliding window,
    keyed on exit time (not arrival) -- this is "how bad is it to finish a
    job right now", the direct time-in-system reading. NaN where no job
    exited in that window (gap, not zero)."""
    centers = np.arange(window / 2, t_max + step, step)
    result = np.full_like(centers, np.nan)
    exits = [(j["exit"], j["flow"]) for j in rows if j["exit"] is not None]
    for i, c in enumerate(centers):
        lo, hi = c - window / 2, c + window / 2
        vals = [f for e, f in exits if lo <= e < hi]
        if vals:
            result[i] = sum(vals) / len(vals)
    return centers, result


def _shade_phases(ax, phases, y_top):
    for i, p in enumerate(phases):
        ax.axvspan(p["start"], p["end"], color="gray", alpha=0.10 if i % 2 == 0 else 0.18)
        ax.annotate(p["name"], (p["start"], y_top), fontsize=6.5, rotation=90,
                    va="top", ha="left", color="#444444")


def plot_overlay(rows_by_rule, phases, t_max, series_fn, ylabel, title, out_path, **series_kwargs):
    fig, ax = plt.subplots(figsize=(16, 6))
    y_max = 0
    for rule in RULES_ORDER:
        rows = first_seed_only(rows_by_rule.get(rule, []))
        if not rows:
            continue
        xs, ys = series_fn(rows, t_max, **series_kwargs)
        ax.plot(xs, ys, label=rule, color=COLORS[rule], linewidth=1.4, alpha=0.9)
        finite = ys[np.isfinite(ys)]
        if finite.size:
            y_max = max(y_max, finite.max())
    _shade_phases(ax, phases, y_max * 1.02 if y_max > 0 else 1.0)
    ax.set_xlabel("sim-time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--scenario-json", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--instance", default="compound_scenario")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    phases = load_phases(args.scenario_json)
    rows = load_job_completions(args.results_dir, args.instance)

    rows_by_rule = defaultdict(list)
    for r in rows:
        rows_by_rule[r["rule"]].append(r)

    t_max = max((p["end"] for p in phases), default=2500.0) + 300.0

    plot_overlay(
        rows_by_rule, phases, t_max, rolling_wip,
        ylabel="work-in-progress (200s rolling average)",
        title="compound_scenario: WIP over time, all rules overlaid\n"
              "(shaded bands = scripted phases; a rule not settling back down between "
              "bands is carrying backlog forward)",
        out_path=os.path.join(args.out_dir, "compound_wip_overlay.png"),
        window=200.0, step=25.0,
    )
    plot_overlay(
        rows_by_rule, phases, t_max, rolling_flowtime,
        ylabel="mean time-in-system of jobs exiting in window (300s rolling)",
        title="compound_scenario: time-in-system over time, all rules overlaid",
        out_path=os.path.join(args.out_dir, "compound_flowtime_overlay.png"),
        window=300.0, step=50.0,
    )


if __name__ == "__main__":
    main()
