"""
compare_parking.py — Compare AGV parking-method results (single vs multiple).

The batch was run twice on the same fixed-seed instances, once per parking
method. Instance names carry a `_single` / `_multiple` suffix, and the run also
logs an explicit `parking_method` column. This script pairs the two arms on a
common (base instance, rule, seed) key and reports the makespan delta, so any
difference is attributable to the parking layout rather than to a different
problem instance.

Pairing is the whole point: we never compare aggregate single vs aggregate
multiple, we compare matched cells. A row only enters the comparison if BOTH
parking methods ran that exact (base, rule, seed).

Usage:
    python compare_parking.py <results.csv> [--out <dir>] [--metric makespan]

Produces (in --out, default `plots_parking`):
    parking_delta_by_instance.png   grouped bar: % makespan change per instance,
                                    averaged over rules+seeds, with per-rule spread
    parking_delta_heatmap.png       instance x rule heatmap of % change (signed)
    parking_paired_scatter.png      single vs multiple makespan, point per matched
                                    (base,rule,seed); off-diagonal = a real change
    parking_summary.csv             the matched table + deltas, for the writeup
    console summary                 mean/median delta, win/loss counts, unpaired rows
"""

import argparse
import os
import warnings

import matplotlib.pyplot as plt
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

SINGLE = "single"
MULTIPLE = "multiple"


# ── Loading & pairing ─────────────────────────────────────────────────────────

def _infer_parking(row) -> str:
    """Trust the explicit column; fall back to the instance-name suffix."""
    pm = str(row.get("parking_method", "")).strip().lower()
    if pm in (SINGLE, MULTIPLE):
        return pm
    name = str(row.get("instance", "")).lower()
    if name.endswith("_multiple"):
        return MULTIPLE
    if name.endswith("_single"):
        return SINGLE
    return ""  # unknown — dropped from comparison


def _base_instance(name: str) -> str:
    """Strip the parking suffix so single/multiple share a join key."""
    s = str(name)
    for suf in ("_multiple", "_single"):
        if s.lower().endswith(suf):
            return s[: -len(suf)]
    return s


def load_paired(csv_path: str, metric: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (paired, raw).

    raw    — every row with a resolved parking_method and base instance.
    paired — wide table with one row per (base, rule, seed) that ran under BOTH
             methods, columns single/multiple/<metric> + delta + pct_delta.
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df["rule"] = df["rule"].astype(str).str.strip()

    df["parking_method"] = df.apply(_infer_parking, axis=1)
    df = df[df["parking_method"].isin([SINGLE, MULTIPLE])].copy()
    df["base"] = df["instance"].map(_base_instance)

    if metric not in df.columns:
        raise SystemExit(f"metric '{metric}' not a column. Have: {list(df.columns)}")

    # Average any accidental duplicate (base,rule,seed,method) rows so the pivot
    # is well-defined; normally this is a no-op.
    key = ["base", "rule", "seed", "parking_method"]
    agg = df.groupby(key, observed=True)[metric].mean().reset_index()

    wide = agg.pivot_table(index=["base", "rule", "seed"],
                           columns="parking_method", values=metric).reset_index()

    # Keep only fully-paired cells.
    both = wide.dropna(subset=[SINGLE, MULTIPLE]).copy()
    both["delta"] = both[MULTIPLE] - both[SINGLE]
    both["pct_delta"] = both["delta"] / both[SINGLE] * 100.0

    # Report what didn't pair, so silent half-runs don't masquerade as agreement.
    unpaired = wide[wide[SINGLE].isna() | wide[MULTIPLE].isna()]
    if len(unpaired):
        print(f"[warn] {len(unpaired)} (base,rule,seed) cells ran under only one "
              f"method and are excluded from the comparison.")
        for _, r in unpaired.head(10).iterrows():
            have = SINGLE if pd.notna(r[SINGLE]) else MULTIPLE
            print(f"        {r['base']} / {r['rule']} / seed {r['seed']}: only {have}")

    return both, df


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(paired: pd.DataFrame, metric: str) -> None:
    n = len(paired)
    mean_d = paired["pct_delta"].mean()
    med_d = paired["pct_delta"].median()
    # "win" = multiple is better = lower makespan = negative delta.
    wins = (paired["delta"] < -1e-9).sum()
    ties = (paired["delta"].abs() <= 1e-9).sum()
    losses = (paired["delta"] > 1e-9).sum()

    print("\n── Parking comparison (multiple vs single) ──")
    print(f"Metric                : {metric}")
    print(f"Matched cells         : {n}  (base x rule x seed)")
    print(f"Mean   % change       : {mean_d:+.3f}%   (negative = multiple better)")
    print(f"Median % change       : {med_d:+.3f}%")
    print(f"Multiple better/tie/worse : {wins} / {ties} / {losses}")

    # Per-instance roll-up — the level most likely to show a real effect.
    inst = (paired.groupby("base")
            .agg(mean_pct=("pct_delta", "mean"),
                 median_pct=("pct_delta", "median"),
                 n=("pct_delta", "size"))
            .reset_index()
            .sort_values("mean_pct"))
    print("\nBy instance (mean % change, sorted best→worst for multiple):")
    for _, r in inst.iterrows():
        print(f"  {r['base']:<22} {r['mean_pct']:+7.2f}%   (n={int(r['n'])})")

    # Magnitude reality check — is anything even outside seed noise?
    big = paired[paired["pct_delta"].abs() >= 2.0]
    print(f"\nCells with |change| >= 2%: {len(big)} / {n}")
    if mean_d > -0.5 and mean_d < 0.5:
        print("Interpretation: makespan is effectively unchanged by parking method "
              "in this batch. Under fixed dispatch at low occupancy this is the "
              "expected null — parking layout only bites when AGVs contend for the "
              "lot, which needs higher sustained load to observe.")


# ── Plot 1 — % delta per instance (bars) with rule spread ─────────────────────

def plot_delta_by_instance(paired: pd.DataFrame, out_dir: str, metric: str) -> None:
    order = (paired.groupby("base")["pct_delta"].mean().sort_values().index.tolist())
    means = paired.groupby("base")["pct_delta"].mean().reindex(order)
    # spread across rules+seeds within each instance
    stds = paired.groupby("base")["pct_delta"].std().reindex(order)

    fig, ax = plt.subplots(figsize=(max(8, len(order) * 1.6), 5))
    x = np.arange(len(order))
    colors = ["#2a9d8f" if v < 0 else "#e76f51" for v in means]  # green better, red worse
    ax.bar(x, means, 0.6, yerr=stds, color=colors, alpha=0.85,
           error_kw=dict(capsize=4, linewidth=1, ecolor="black"))
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=25, ha="right")
    ax.set_ylabel(f"% change in {metric}\n(multiple − single, negative = better)")
    ax.set_title("Parking method effect by instance\n"
                 "bars = mean over rules+seeds, whiskers = std across rules+seeds")
    _save(fig, out_dir, "parking_delta_by_instance.png")


# ── Plot 2 — instance x rule heatmap of signed % delta ────────────────────────

def plot_delta_heatmap(paired: pd.DataFrame, out_dir: str, metric: str) -> None:
    grid = (paired.groupby(["base", "rule"])["pct_delta"].mean().unstack("rule"))
    # order instances by size if the name encodes it, else as-is
    grid = grid.reindex(sorted(grid.index, key=str))

    vmax = np.nanmax(np.abs(grid.values)) if grid.size else 1.0
    vmax = max(vmax, 0.5)

    fig, ax = plt.subplots(figsize=(max(8, grid.shape[1] * 1.0),
                                    max(4, grid.shape[0] * 0.7)))
    sns.heatmap(grid, annot=True, fmt="+.1f", cmap="RdYlGn_r", center=0,
                vmin=-vmax, vmax=vmax, linewidths=0.5, linecolor="white",
                cbar_kws={"label": f"% change in {metric} (multiple − single)"},
                ax=ax)
    ax.set_title("Parking effect per instance × rule\n"
                 "red = multiple worse, green = multiple better")
    ax.set_xlabel("Rule")
    ax.set_ylabel("Instance")
    _save(fig, out_dir, "parking_delta_heatmap.png")


# ── Plot 3 — paired scatter, single vs multiple ───────────────────────────────

def plot_paired_scatter(paired: pd.DataFrame, out_dir: str, metric: str) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    bases = sorted(paired["base"].unique(), key=str)
    palette = plt.cm.tab10.colors
    for i, b in enumerate(bases):
        sub = paired[paired["base"] == b]
        ax.scatter(sub[SINGLE], sub[MULTIPLE], s=28, alpha=0.8,
                   color=palette[i % len(palette)], label=b, edgecolor="white",
                   linewidth=0.4)

    lo = min(paired[SINGLE].min(), paired[MULTIPLE].min())
    hi = max(paired[SINGLE].max(), paired[MULTIPLE].max())
    pad = (hi - lo) * 0.05
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", linewidth=1,
            alpha=0.6, label="parity (no change)")
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel(f"{metric} — single parking")
    ax.set_ylabel(f"{metric} — multiple parking")
    ax.set_title("Paired makespan: each point is one (instance, rule, seed)\n"
                 "below the line = multiple faster")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_aspect("equal", adjustable="box")
    _save(fig, out_dir, "parking_paired_scatter.png")


# ── IO ────────────────────────────────────────────────────────────────────────

def _save(fig, out_dir, filename):
    path = os.path.join(out_dir, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    ap = argparse.ArgumentParser(description="Compare single vs multiple parking results.")
    ap.add_argument("csv", help="results.csv with single+multiple runs")
    ap.add_argument("--out", default="plots_parking")
    ap.add_argument("--metric", default="makespan",
                    help="numeric column to compare (default: makespan)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    paired, raw = load_paired(args.csv, args.metric)

    if paired.empty:
        raise SystemExit("No paired (base,rule,seed) cells found — did both "
                         "parking methods run on the same instances/seeds?")

    print_summary(paired, args.metric)
    plot_delta_by_instance(paired, args.out, args.metric)
    plot_delta_heatmap(paired, args.out, args.metric)
    plot_paired_scatter(paired, args.out, args.metric)

    out_csv = os.path.join(args.out, "parking_summary.csv")
    paired.sort_values(["base", "rule", "seed"]).to_csv(out_csv, index=False)
    print(f"  Saved: {out_csv}")
    print("\nDone.")


if __name__ == "__main__":
    main()
