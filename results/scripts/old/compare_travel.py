"""
compare_travel.py — Paired single-vs-multiple parking comparison on AGV transport
metrics, with significance testing.

Why this is separate from compare_parking.py:
  agv_performance.csv has ONE ROW PER AGV, so each (instance, rule, seed) run
  contributes several correlated rows. Treating those as independent samples
  would inflate significance (pseudo-replication). This script first aggregates
  AGV rows to a PER-RUN value, then pairs single vs multiple on
  (base_instance, rule, seed) — the correct experimental unit.

Aggregation per metric:
  time_traveling, time_waiting_route, total_path_length  -> SUM over AGVs
      (fleet total work; the quantity the layout is supposed to reduce)
  makespan                                               -> MEAN (identical
      across AGV rows of a run, but mean is safe)

Significance (across the matched pairs):
  - Wilcoxon signed-rank  (paired, non-parametric; primary — no normality assumption)
  - Paired t-test         (secondary, reported for completeness)
  - Cohen's d_z           (paired effect size)
  - 95% CI on mean % change (bootstrap)

Pairs are matched cells, so the test asks: across the matched runs, is the
multiple-parking value reliably different from single?

Usage:
    python compare_travel.py <agv_performance.csv> [--metric time_traveling]
                             [--out <dir>] [--by-instance]
"""

import argparse
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

plt.rcParams.update({
    "figure.dpi": 150, "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3,
})

SINGLE, MULTIPLE = "single", "multiple"
SUM_METRICS = {"time_traveling", "time_waiting_route", "time_loading",
               "time_unloading", "time_idle", "total_path_length",
               "total_trips", "reroute_count"}


def _parking(row):
    pm = str(row.get("parking_method", "")).strip().lower()
    if pm in (SINGLE, MULTIPLE):
        return pm
    name = str(row.get("instance", "")).lower()
    if name.endswith("_multiple"):
        return MULTIPLE
    if name.endswith("_single"):
        return SINGLE
    return ""


def _base(name):
    s = str(name)
    for suf in ("_multiple", "_single"):
        if s.lower().endswith(suf):
            return s[:-len(suf)]
    return s


def load_paired(csv_path, metric):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    if metric not in df.columns:
        raise SystemExit(f"metric '{metric}' not found. have: {list(df.columns)}")
    df["rule"] = df["rule"].astype(str).str.strip()
    df["parking_method"] = df.apply(_parking, axis=1)
    df = df[df["parking_method"].isin([SINGLE, MULTIPLE])].copy()
    df["base"] = df["instance"].map(_base)

    aggfn = "sum" if metric in SUM_METRICS else "mean"
    run = (df.groupby(["base", "rule", "seed", "parking_method"], observed=True)[metric]
           .agg(aggfn).reset_index())

    wide = run.pivot_table(index=["base", "rule", "seed"],
                           columns="parking_method", values=metric).reset_index()
    both = wide.dropna(subset=[SINGLE, MULTIPLE]).copy()
    both["delta"] = both[MULTIPLE] - both[SINGLE]
    both["pct_delta"] = both["delta"] / both[SINGLE].replace(0, np.nan) * 100
    return both, aggfn


def bootstrap_ci(x, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def significance(paired, metric, aggfn):
    s = paired[SINGLE].to_numpy()
    m = paired[MULTIPLE].to_numpy()
    d = m - s
    n = len(d)

    # Wilcoxon (primary). Guard the degenerate all-zero-diff case.
    if np.allclose(d, 0):
        w_p = 1.0; w_stat = 0.0
    else:
        w_stat, w_p = stats.wilcoxon(m, s)
    t_stat, t_p = stats.ttest_rel(m, s)
    dz = d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else 0.0
    pct = paired["pct_delta"].dropna()
    ci = bootstrap_ci(pct.to_numpy()) if len(pct) else (np.nan, np.nan)

    print("\n" + "=" * 68)
    print(f"PAIRED SIGNIFICANCE — {metric}  (per-run {aggfn} over AGVs)")
    print("=" * 68)
    print(f"Matched pairs (base x rule x seed) : {n}")
    print(f"Mean single                        : {s.mean():.1f}")
    print(f"Mean multiple                      : {m.mean():.1f}")
    print(f"Mean change                        : {d.mean():+.1f}  "
          f"({pct.mean():+.1f}%)")
    print(f"95% CI on mean % change (bootstrap): [{ci[0]:+.1f}%, {ci[1]:+.1f}%]")
    print(f"Multiple lower / tie / higher      : "
          f"{(d < 0).sum()} / {(d == 0).sum()} / {(d > 0).sum()}")
    print("-" * 68)
    print(f"Wilcoxon signed-rank   W={w_stat:.1f}   p = {w_p:.2e}   <- primary")
    print(f"Paired t-test          t={t_stat:.2f}   p = {t_p:.2e}")
    print(f"Cohen's d_z (paired)   {dz:+.2f}   "
          f"({_d_label(abs(dz))} effect)")
    print("-" * 68)
    sig = w_p < 0.05
    direction = "REDUCES" if d.mean() < 0 else "INCREASES"
    if sig:
        print(f"CONCLUSION: multiple parking {direction} {metric} significantly "
              f"(p={w_p:.1e}).")
        if metric == "time_traveling":
            print("  This is the transport-cost reduction makespan could not show.")
            print("  A layout change that moves fleet travel by this much while")
            print("  leaving makespan flat is the core spatial-vs-DES evidence.")
    else:
        print(f"CONCLUSION: no significant difference in {metric} (p={w_p:.2f}).")
    print("=" * 68)
    return dict(n=n, mean_single=s.mean(), mean_multiple=m.mean(),
                mean_pct=pct.mean(), ci_lo=ci[0], ci_hi=ci[1],
                wilcoxon_p=w_p, t_p=t_p, cohen_dz=dz)


def _d_label(d):
    return ("negligible" if d < 0.2 else "small" if d < 0.5
            else "medium" if d < 0.8 else "large")


def per_instance(paired, metric):
    print(f"\nPer-instance breakdown ({metric}):")
    print(f"  {'instance':<22} {'single':>10} {'multiple':>10} {'%chg':>7} "
          f"{'wilcox_p':>9}")
    rows = []
    for base, g in paired.groupby("base"):
        s, m = g[SINGLE].to_numpy(), g[MULTIPLE].to_numpy()
        if np.allclose(m - s, 0):
            p = 1.0
        else:
            try:
                _, p = stats.wilcoxon(m, s)
            except ValueError:
                p = np.nan
        pct = ((m - s) / s * 100).mean()
        flag = "*" if (p < 0.05) else " "
        print(f"  {base:<22} {s.mean():10.1f} {m.mean():10.1f} {pct:+6.1f}% "
              f"{p:9.2e}{flag}")
        rows.append(dict(instance=base, mean_single=s.mean(),
                         mean_multiple=m.mean(), pct_change=pct, wilcoxon_p=p, n=len(g)))
    print("  (* = p<0.05; note small_5j_5m has only 1 machine/type and is noisiest)")
    return pd.DataFrame(rows)


def plot_paired(paired, metric, out_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left: paired scatter, single vs multiple.
    bases = sorted(paired["base"].unique(), key=str)
    pal = plt.cm.tab10.colors
    for i, b in enumerate(bases):
        sub = paired[paired["base"] == b]
        ax1.scatter(sub[SINGLE], sub[MULTIPLE], s=30, alpha=0.8,
                    color=pal[i % len(pal)], label=b, edgecolor="white", linewidth=0.4)
    lo = min(paired[SINGLE].min(), paired[MULTIPLE].min())
    hi = max(paired[SINGLE].max(), paired[MULTIPLE].max())
    pad = (hi - lo) * 0.05
    ax1.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", alpha=0.6,
             label="parity")
    ax1.set_xlabel(f"{metric} — single (per-run fleet total)")
    ax1.set_ylabel(f"{metric} — multiple")
    ax1.set_title("Paired per run — below line = multiple lower")
    ax1.legend(fontsize=6.5, loc="upper left")
    ax1.set_aspect("equal", adjustable="box")

    # Right: % change distribution per instance.
    order = sorted(paired["base"].unique(), key=str)
    data = [paired[paired["base"] == b]["pct_delta"].dropna().to_numpy() for b in order]
    bp = ax2.boxplot(data, vert=True, patch_artist=True, showmeans=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#4c72b0"); patch.set_alpha(0.6)
    ax2.axhline(0, color="black", linewidth=1)
    ax2.set_xticks(range(1, len(order) + 1))
    ax2.set_xticklabels(order, rotation=25, ha="right", fontsize=8)
    ax2.set_ylabel(f"% change in {metric} (multiple − single)")
    ax2.set_title("Per-instance % change (negative = multiple better)")

    fig.suptitle(f"Parking effect on {metric}", y=1.02, fontsize=13)
    fig.tight_layout()
    p = os.path.join(out_dir, f"travel_compare_{metric}.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print(f"\n  Saved: {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--metric", default="time_traveling")
    ap.add_argument("--out", default="plots_travel")
    ap.add_argument("--by-instance", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    paired, aggfn = load_paired(args.csv, args.metric)
    if paired.empty:
        raise SystemExit("no matched pairs — did both methods run on the same cells?")

    res = significance(paired, args.metric, aggfn)
    if args.by_instance:
        pi = per_instance(paired, args.metric)
        pi.to_csv(os.path.join(args.out, f"per_instance_{args.metric}.csv"), index=False)
    plot_paired(paired, args.metric, args.out)
    paired.sort_values(["base", "rule", "seed"]).to_csv(
        os.path.join(args.out, f"paired_{args.metric}.csv"), index=False)
    print(f"  Saved: {os.path.join(args.out, f'paired_{args.metric}.csv')}")
    print("\nDone.")


if __name__ == "__main__":
    main()
