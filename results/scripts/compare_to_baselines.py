"""Score evaluate.py episodes (e.g. a trained checkpoint) against the saved 12 head-rule baselines.

The baselines are results/eval_pdr12_heads/episodes_all.csv (docs/experiments/EVAL_BASELINES_12HEAD_0927.md).
The candidate run must use the same instances: --scenario-generator randomized --episode-duration-seconds 5400,
seeds within 0-19, a two-branch player. For another condition (evaluate.py --agvs / --layout), pass that
condition's own 12-pair run as --baselines.

    python results/scripts/compare_to_baselines.py results/eval_rnd02/episodes.csv [more episodes.csv ...]
    python results/scripts/compare_to_baselines.py results/eval_rnd02_agv9/episodes.csv \
        --baselines results/eval_pdr12_heads_agv9/episodes.csv --csv results/eval_rnd02_agv9/compare.csv

Per policy it prints the mean gap to the per-seed best baseline rule (negative = beats every fixed rule on that
seed) and to the reference rule, on total flow and mean flow, plus jobs exited and seed wins.

The reference is the best fixed pair on average for these baselines (lowest mean gap to the per-seed best on
total flow; SPT-ECT on the D / 7-AGV set), unless --reference names one. Against it, per policy, on total flow
(the RQ3 metric, docs/WIP/RUNNING_EXPERIMENTS.md "GEN"):
  median   median per-seed gap, policy / reference - 1, with a bootstrap 95% CI over seeds;
  p_better one-sided Wilcoxon signed-rank p that the gaps sit below 0 (p_worse: above 0);
  verdict  "better" if p_better < alpha, "worse" if p_worse < alpha, otherwise "n.s.".
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[2]
BASELINES = REPO / "results" / "eval_pdr12_heads" / "episodes_all.csv"
FLOOR_COLUMNS = ("agv_count", "layout")


def best_on_average(base):
    """The baseline pair with the lowest mean gap to the per-seed best on total flow."""
    best = base.groupby("seed")["total_flow_time"].transform("min")
    return (base["total_flow_time"] / best - 1).groupby(base["policy"]).mean().idxmin()


def check_same_floor(base, cand):
    """Refuse candidate episodes run on a different fleet or layout than the baselines (when both record it)."""
    for col in FLOOR_COLUMNS:
        if col in base.columns and col in cand.columns:
            b, c = set(base[col].dropna().unique()), set(cand[col].dropna().unique())
            if b and c and b != c:
                raise SystemExit(f"{col}: candidate episodes have {sorted(c)}, baselines have {sorted(b)}. "
                                 f"Pass the matching condition's baselines with --baselines.")
        elif col in cand.columns and cand[col].notna().any():
            print(f"Note: the baselines do not record {col}; candidate episodes have "
                  f"{sorted(cand[col].dropna().unique())}. Check they are the same condition.")


def bootstrap_median_ci(x, n_boot, rng, level=0.95):
    """Percentile bootstrap CI of the median, resampling seeds."""
    if len(x) < 2:
        return np.nan, np.nan
    medians = np.median(rng.choice(x, size=(n_boot, len(x)), replace=True), axis=1)
    tail = (1 - level) / 2
    return np.quantile(medians, tail), np.quantile(medians, 1 - tail)


def paired_test(gaps, alpha, n_boot, rng):
    """Median gap, its bootstrap CI, one-sided Wilcoxon p in each direction, and the verdict."""
    x = np.asarray(gaps, dtype=float)
    lo, hi = bootstrap_median_ci(x, n_boot, rng)
    if np.count_nonzero(x) == 0:             # identical to the reference on every seed: nothing to test
        p_better = p_worse = 1.0
    else:
        p_better = wilcoxon(x, alternative="less").pvalue
        p_worse = wilcoxon(x, alternative="greater").pvalue
    verdict = "better" if p_better < alpha else "worse" if p_worse < alpha else "n.s."
    return {"n": len(x), "median": np.median(x), "ci_lo": lo, "ci_hi": hi,
            "p_better": p_better, "p_worse": p_worse, "verdict": verdict}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("episodes", nargs="+", help="evaluate.py episodes.csv file(s) to score")
    ap.add_argument("--baselines", default=str(BASELINES),
                    help="12-pair episodes.csv on the same instances and floor (default: D, 7 AGVs)")
    ap.add_argument("--reference", default="auto",
                    help="reference pair, or 'auto' = best pair on average in --baselines (default)")
    ap.add_argument("--alpha", type=float, default=0.05, help="significance level of the one-sided tests")
    ap.add_argument("--bootstrap", type=int, default=10000, help="bootstrap resamples for the median CI")
    ap.add_argument("--rng-seed", type=int, default=0, help="bootstrap RNG seed (keeps the CI reproducible)")
    ap.add_argument("--csv", default=None, help="also write the per-policy table here (unformatted numbers)")
    a = ap.parse_args()

    base = pd.read_csv(a.baselines)
    base = base[base["kind"] == "pdr"]
    cand = pd.concat([pd.read_csv(p) for p in a.episodes], ignore_index=True)
    cand = cand[cand["kind"] != "pdr"]          # PDR rows in the candidate run are ignored: the baselines cover them
    if cand.empty:
        raise SystemExit("No non-PDR episodes found in the given files.")
    check_same_floor(base, cand)

    missing = sorted(set(cand["seed"]) - set(base["seed"]))
    if missing:
        raise SystemExit(f"Seeds {missing} have no baselines (baselines cover seeds {sorted(base['seed'].unique())}).")

    reference = best_on_average(base) if a.reference == "auto" else a.reference
    if reference not in set(base["policy"]):
        raise SystemExit(f"Reference {reference!r} is not in the baselines ({sorted(base['policy'].unique())}).")
    ref_seeds = set(base.loc[base["policy"] == reference, "seed"])
    if set(cand["seed"]) - ref_seeds:
        raise SystemExit(f"Reference {reference} is missing seeds {sorted(set(cand['seed']) - ref_seeds)}.")

    best = base.groupby("seed").agg(best_total=("total_flow_time", "min"), best_mean=("mean_flow_time", "min"))
    ref = base[base["policy"] == reference].set_index("seed")[["total_flow_time", "mean_flow_time"]]
    ref.columns = ["ref_total", "ref_mean"]
    c = cand.join(best, on="seed").join(ref, on="seed")
    c["gap_best_total"] = c["total_flow_time"] / c["best_total"] - 1
    c["gap_best_mean"] = c["mean_flow_time"] / c["best_mean"] - 1
    c["gap_ref_total"] = c["total_flow_time"] / c["ref_total"] - 1
    c["gap_ref_mean"] = c["mean_flow_time"] / c["ref_mean"] - 1
    c["beats_all_total"] = c["gap_best_total"] < 0

    out = c.groupby("policy").agg(
        seeds=("seed", "nunique"),
        total_vs_best=("gap_best_total", "mean"), total_vs_ref=("gap_ref_total", "mean"),
        mean_vs_best=("gap_best_mean", "mean"), mean_vs_ref=("gap_ref_mean", "mean"),
        jobs_exited=("jobs_exited", "mean"), seeds_beating_all=("beats_all_total", "sum"),
        deadlocks=("deadlock", "sum"),
    )
    if (c.groupby("policy")["seed"].count() != out["seeds"]).any():
        print("Note: some policy has more than one episode per seed; the test below treats each as a pair.")

    rng = np.random.default_rng(a.rng_seed)
    tests = pd.DataFrame({p: paired_test(g["gap_ref_total"], a.alpha, a.bootstrap, rng)
                          for p, g in c.groupby("policy")}).T
    out = out.join(tests.drop(columns="n"))
    if a.csv:
        out.assign(reference=reference).to_csv(a.csv)

    shown = out.copy()
    for col in ["total_vs_best", "total_vs_ref", "mean_vs_best", "mean_vs_ref", "median", "ci_lo", "ci_hi"]:
        shown[col] = (100 * shown[col].astype(float)).map("{:+.1f}%".format)
    for col in ["p_better", "p_worse"]:
        shown[col] = shown[col].astype(float).map("{:.3g}".format)
    shown["jobs_exited"] = shown["jobs_exited"].map("{:.1f}".format)
    pd.set_option("display.width", 250)
    print(f"Reference: {reference} ({'best on average' if a.reference == 'auto' else 'given'}). "
          f"*_vs_ref, median and CI are gaps to it on total flow; vs_best < 0 beats every fixed rule on that seed.\n")
    print(shown.to_string())

    base_best = base.groupby("seed")["total_flow_time"].transform("min")
    ref_gap = (base["total_flow_time"] / base_best - 1)[base["policy"] == reference].mean()
    base_jobs = base.groupby("policy")["jobs_exited"].mean()
    print(f"\nBaselines ({base['seed'].nunique()} seeds): {reference} total flow {100 * ref_gap:+.1f}% vs per-seed best; "
          f"jobs exited {base_jobs.min():.1f}-{base_jobs.max():.1f} across rules. "
          f"Verdict: one-sided Wilcoxon on the per-seed gaps to {reference}, alpha {a.alpha}.")


if __name__ == "__main__":
    main()
