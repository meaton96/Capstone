#!/usr/bin/env python3
"""
Find the makespan plateau in an arrival-lambda sweep.

Reads a results.csv from a lambda sweep (fixed hardware, only arrivalLambda
varying), aggregates makespan per arrival_lambda, and identifies where makespan
stops falling -- the capacity-bound floor M*. The smallest lambda on that
plateau is the operating point to reuse for the AGV sweep and the later
op-duration (mu/sigma) tests.

Reasoning the script encodes:
  - Arrival-bound regime: makespan ~ C/lambda, so (makespan * lambda) ~ C (the
    realized job cap) and makespan FALLS as lambda rises.
  - Capacity-bound regime: makespan ~ M* (flat), so (makespan * lambda) RISES.
  The plateau begins where mean makespan first comes within `--tol` of the floor.

Usage:
    python find_lambda_plateau.py
    python find_lambda_plateau.py --results results.csv --out lambda_plateau.png --tol 0.10
"""
import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LAM = "arrival_lambda"

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 150, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
})


def load(path):
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        sys.exit(f"[error] file not found: {path}")
    n0 = len(df)
    df = df.drop_duplicates().reset_index(drop=True)   # kill the 2x double-logging
    if n0 != len(df):
        print(f"[info] dropped {n0 - len(df)} exact-duplicate rows")
    for c in (LAM, "makespan"):
        if c not in df.columns:
            sys.exit(f"[error] results.csv is missing required column '{c}'")
    df[LAM] = pd.to_numeric(df[LAM], errors="coerce")
    df["makespan"] = pd.to_numeric(df["makespan"], errors="coerce")
    df = df.dropna(subset=[LAM, "makespan"])
    df["lam"] = df[LAM].round(6)                         # guard float-dup keys
    return df


def hardware_check(df):
    """Warn if the 'fixed hardware' assumption is violated (would confound)."""
    print("[hardware check]")
    for col in ("agvCount", "machines"):
        if col in df.columns:
            u = sorted(df[col].dropna().unique())
            flag = "" if len(u) == 1 else "  <-- VARIES, sweep is confounded!"
            print(f"  {col:10s}: {u}{flag}")
    if "jobs" in df.columns:                              # expected to vary (stochastic cap)
        j = df["jobs"]
        print(f"  jobs      : {int(j.min())}..{int(j.max())} (varies = expected with dynamic cap)")
    print(f"  lambdas   : {sorted(df['lam'].unique())}")
    if "rule" in df.columns:
        print(f"  rules     : {sorted(df['rule'].unique())}")


def summarize(df):
    s = (df.groupby("lam")["makespan"].agg(["mean", "std", "count"])
           .reset_index().sort_values("lam").reset_index(drop=True))
    s["sem"] = s["std"].fillna(0) / np.sqrt(s["count"].clip(lower=1))
    floor = s["mean"].min()
    s["pct_above_floor"] = s["mean"] / floor - 1.0
    s["rel_drop_from_prev"] = (-s["mean"].pct_change()).fillna(np.nan)   # +ve = makespan fell
    s["mk_x_lam"] = s["mean"] * s["lam"]                                 # ~const when arrival-bound
    return s, floor


def find_plateau(s, tol):
    cand = s[s["pct_above_floor"] <= tol]
    return float(cand["lam"].iloc[0]) if not cand.empty else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results.csv")
    ap.add_argument("--out", default="lambda_plateau.png")
    ap.add_argument("--tol", type=float, default=0.10,
                    help="plateau = within this fraction of the floor (default 0.10 = 10%%)")
    args = ap.parse_args()

    df = load(args.results)
    hardware_check(df)
    s, floor = summarize(df)
    plateau = find_plateau(s, args.tol)

    # estimate the arrival-bound constant C from the lowest 3 lambdas
    C = float(s["mk_x_lam"].iloc[:min(3, len(s))].median())

    # ---- printed table -------------------------------------------------------
    print("\n[makespan by arrival_lambda]")
    print(f"  {'lambda':>9} {'mean_mk':>10} {'sem':>7} {'n':>3} "
          f"{'%>floor':>8} {'drop_prev':>9} {'mk*lam':>8}  regime")
    for _, r in s.iterrows():
        regime = "capacity-bound" if r["pct_above_floor"] <= args.tol else (
                 "transition" if r["pct_above_floor"] <= 0.5 else "arrival-bound")
        drop = "" if np.isnan(r["rel_drop_from_prev"]) else f"{r['rel_drop_from_prev']:+.1%}"
        print(f"  {r['lam']:>9.4f} {r['mean']:>10.1f} {r['sem']:>7.1f} {int(r['count']):>3} "
              f"{r['pct_above_floor']:>8.1%} {drop:>9} {r['mk_x_lam']:>8.1f}  {regime}")

    print("\n[result]")
    print(f"  estimated capacity floor M*  : {floor:.1f}")
    print(f"  arrival-bound constant C     : {C:.1f}  (makespan*lambda in starved regime)")
    if plateau is not None:
        print(f"  plateau starts at lambda     : {plateau:.4f}  "
              f"(first lambda within {args.tol:.0%} of the floor)")
        print(f"  --> use lambda >= {plateau:.4f} as the loaded operating point for the AGV /")
        print(f"      op-duration experiments; below it the system is still arrival-starved.")
    else:
        print(f"  no lambda reached within {args.tol:.0%} of the floor -- extend the sweep "
              f"to higher lambda (system never fully saturates in this range).")

    # ---- figure --------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    # left: makespan vs lambda
    if "rule" in df.columns and df["rule"].nunique() > 1:
        for rule, gr in df.groupby("rule"):
            ss = gr.groupby("lam")["makespan"].mean().reset_index().sort_values("lam")
            ax[0].plot(ss["lam"], ss["makespan"], marker="o", ms=3, lw=1,
                       alpha=0.45, label=str(rule))
    ax[0].errorbar(s["lam"], s["mean"], yerr=s["sem"], marker="o", color="k",
                   lw=2, capsize=3, label="all rules (mean)")
    grid = np.linspace(s["lam"].min(), s["lam"].max(), 200)
    ax[0].plot(grid, C / grid, "r:", lw=1.2, label="arrival-bound  C / lambda")
    ax[0].axhline(floor, color="#6a8d3f", ls="--", lw=1, label=f"floor M* = {floor:.0f}")
    if plateau is not None:
        ax[0].axvspan(plateau, s["lam"].max(), color="#6a8d3f", alpha=0.08)
        ax[0].axvline(plateau, color="#6a8d3f", lw=1)
        ax[0].annotate(f"plateau\nλ ≥ {plateau:.4f}", xy=(plateau, floor),
                       xytext=(plateau, floor * 2.2),
                       ha="center", fontsize=9, color="#3a5a1e",
                       arrowprops=dict(arrowstyle="->", color="#3a5a1e"))
    ax[0].set(xscale="log", yscale="log", xlabel="arrival_lambda (log)",
              ylabel="makespan (log)", title="Makespan vs arrival rate")
    ax[0].legend(fontsize=8)

    # right: makespan*lambda -- flat = arrival-bound, rising = capacity-bound
    ax[1].plot(s["lam"], s["mk_x_lam"], marker="s", color="#3b6ea5", lw=1.5)
    ax[1].axhline(C, color="r", ls=":", lw=1.2, label=f"C ≈ {C:.0f} (arrival-bound)")
    if plateau is not None:
        ax[1].axvspan(plateau, s["lam"].max(), color="#6a8d3f", alpha=0.08)
    ax[1].set(xscale="log", xlabel="arrival_lambda (log)",
              ylabel="makespan × lambda",
              title="Load diagnostic (rises once capacity-bound)")
    ax[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    print(f"\n  figure written to: {args.out}")


if __name__ == "__main__":
    main()