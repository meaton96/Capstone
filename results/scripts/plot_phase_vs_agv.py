#!/usr/bin/env python3
"""E1 figure: mean flow time of each phase kind of a compound scenario vs AGV count, failures off vs on.

One small panel per phase kind (own y-scale: the kinds differ by 10x), two lines per panel, band = +-1 standard error
over layouts x rules x seeds. Reads results/E1_runs.csv (analyze_load_sweep.py). 3-AGV points are off-chart by default
(AGV-limited, flow 1000-2600 s would flatten every panel); --min-agv 3 keeps them.

Usage: python3 results/scripts/plot_phase_vs_agv.py [--scenario compound_scenario_v2] [--protocol releasePrevious]
"""
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PANELS = [("flow_quiet", "quiet (no Weld op)"), ("flow_bottleneck", "bottleneck (pinned, 1 Weld)"),
          ("flow_standoff", "standoff (2-machine pool)"), ("flow_speed_trap", "speed trap (2-machine pool)"),
          ("flow_burst", "burst (pinned)"), ("flow_starvation", "starvation (pinned)")]
OFF, ON = "#2a78d6", "#eb6834"      # validated categorical slots 1-2 (dataviz palette), CVD dE >= 24


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/E1_runs.csv")
    ap.add_argument("--scenario", default="compound_scenario_v2")
    ap.add_argument("--protocol", default="releasePrevious")
    ap.add_argument("--min-agv", type=int, default=5)
    ap.add_argument("--out", default="docs/figures/e1_phase_flow_vs_agv.png")
    a = ap.parse_args()
    d = pd.read_csv(a.csv)
    d = d[(d.protocol == a.protocol) & (d.unfinished == 0) & (d.agv >= a.min_agv)]
    d = d[d.scenario.isin([a.scenario, a.scenario + "_fail"])]
    d["fail"] = d.scenario.str.endswith("_fail")
    panels = [p for p in PANELS if d[p[0]].notna().any()]

    fig, axes = plt.subplots(2, 3, figsize=(11, 6.2), sharex=True)
    for ax, (col, title) in zip(axes.flat, panels):
        for fail, colour, label in ((False, OFF, "failures off"), (True, ON, "failures on")):
            g = d[d.fail == fail].groupby("agv")[col]
            m, se = g.mean(), g.sem()
            ax.fill_between(m.index, m - se, m + se, color=colour, alpha=0.18, linewidth=0)
            ax.plot(m.index, m.values, color=colour, lw=2, marker="o", ms=6, mfc=colour, mec="#fcfcfb", mew=1.5,
                    label=label)
        ax.set_title(title, fontsize=10, loc="left", color="#0b0b0b")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", color="#e4e3df", lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#a8a7a0")
        ax.tick_params(colors="#52514e", labelsize=9)
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("AGVs", color="#52514e")
    for ax in axes[:, 0]:
        ax.set_ylabel("mean flow time (s)", color="#52514e")
    axes.flat[0].set_xticks(sorted(d.agv.unique()))
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=2, frameon=False, fontsize=10)
    fig.suptitle(f"E1: flow time per phase vs fleet size - {a.scenario}, {a.protocol}", x=0.01, ha="left",
                 fontsize=12, color="#0b0b0b")
    n = d.groupby("fail").size()
    fig.text(0.01, 0.005, f"Mean over layouts D/C/G/J x 2 rules x seeds; band +-1 SE. n = {n.get(False, 0)} runs (fail off), "
             f"{n.get(True, 0)} (fail on). 3-AGV runs omitted (AGV-limited). Pinned phases include repair time when failures are on.",
             fontsize=8, color="#52514e", ha="left")
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    fig.savefig(a.out, dpi=160, facecolor="#fcfcfb")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
