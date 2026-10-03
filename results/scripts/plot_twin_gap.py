#!/usr/bin/env python3
"""Thesis figure for twin-gap (G1): mean flow time against fleet size, Unity vs the event-based twins.

  python results/scripts/plot_twin_gap.py [--cells results/outdated/des_twin/G1_cells.csv] [--out docs/Thesis/figures]

Four panels (transport-bound compound on D and J, _mfsweep_control on D, machine-bound rnd_load_s0 on D), one line
per model, log scale. Reads the output of des_twin_gap.py.
"""
import argparse
import os

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PANELS = [("compound_scenario", "D", "compound, layout D (transport-bound)"),
          ("compound_scenario", "J", "compound, layout J (transport-bound)"),
          ("_mfsweep_control", "D", "137-job steady scenario, layout D (transport-bound)"),
          ("rnd_load_s0", "D", "rnd_load seed 0, layout D (machine-bound, linear axis)")]
LINES = [("unity_flow", "Unity (physical floor)", dict(marker="o", lw=2, color="#1f4e79")),
         ("kinematic_flow", "DES-1k (free-flow motion)", dict(marker="s", ls="--", color="#c55a11")),
         ("geometric_flow", "DES-1g (distance / speed)", dict(marker="^", ls=":", color="#548235")),
         ("instant_flow", "DES-0 (instant transfers)", dict(ls="-.", color="#7f7f7f"))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=os.path.join(ROOT, "results", "des_twin", "G1_cells.csv"))
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "Thesis", "figures"))
    a = ap.parse_args()
    c = pd.read_csv(a.cells)
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.2), sharex=True)
    for ax, (scen, lay, title) in zip(axes.flat, PANELS):
        d = c[(c.scenario == scen) & (c.layout == lay)].sort_values("agvs")
        for col, label, style in LINES:
            ax.plot(d.agvs, d[col], label=label, **style)
        if not scen.startswith("rnd_load"):      # rnd_load spans 3,200-3,900 s: a log axis hides it
            ax.set_yscale("log")
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.3, which="both")
        ax.set_xticks([2, 3, 5, 7, 10, 15, 20, 25])
    for ax in axes[1]:
        ax.set_xlabel("AGVs")
    for ax in axes[:, 0]:
        ax.set_ylabel("mean flow time (s)")
    axes[0, 0].legend(fontsize=7.5, loc="upper right")
    fig.tight_layout()
    os.makedirs(a.out, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(a.out, f"twin_gap_fleet.{ext}"), dpi=150)
    print("wrote", os.path.join(a.out, "twin_gap_fleet.pdf"))


if __name__ == "__main__":
    main()
