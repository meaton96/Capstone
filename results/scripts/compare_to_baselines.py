"""Score evaluate.py episodes (e.g. a trained checkpoint) against the 12 head-rule baselines, on episode return.

The default baselines are the clean-clock rq4-agvfail/reg runs (at most 24 episodes per player, so no float32-clock
drift), with their FIFO rows replaced by rq4-agvfail-qfifo/reg (strict queue FIFO since 2026-10-02). Later
--baselines files override earlier ones on (policy, seed). The candidate run must use the same instances:
--scenario-generator randomized --episode-duration-seconds 5400, seeds within 0-19, a two-branch player, and the
same --reward-spec as the baselines (flow_time), since the score is `return`. For another condition (evaluate.py
--agvs / --layout), pass that condition's own 12-pair run as --baselines.

    python results/scripts/compare_to_baselines.py results/eval_rnd02/episodes.csv [more episodes.csv ...]
    python results/scripts/compare_to_baselines.py results/eval_rnd02_agv9/episodes.csv \
        --baselines results/eval_pdr12_heads_agv9/episodes.csv --csv results/eval_rnd02_agv9/compare.csv

Scores are on `return` (time in system of every job, finished or not; CLAUDE.md "Measuring headroom and targets").
Returns are negative, so gap = return / reference_return - 1 is positive when the policy is worse and negative when
it is better. Total flow of exited jobs is censored (finishing fewer jobs lowers it): it is printed only next to
jobs exited, never ranked on. Deadlocked or timed-out episodes stop the clock early, so they are left out of the
gaps (baselines and candidates), counted, and a policy with any of them cannot get a "better" verdict.

Per policy it prints the mean return gap to the per-seed best baseline pair (negative = beats every fixed pair on
that seed) and to the reference pair. The reference is the best pair on average (lowest mean return gap to the
per-seed best), unless --reference names one. Against it, per policy:
  median   median per-seed gap, with a bootstrap 95% CI over seeds;
  p_better one-sided Wilcoxon signed-rank p that the gaps sit below 0 (p_worse: above 0);
  verdict  "better" if p_better < alpha (and no deadlocked/timed-out episodes), "worse" if p_worse < alpha,
           otherwise "n.s.".
"""
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[2]
BASELINES = [str(REPO / "results" / "rq4-agvfail" / "reg" / "s*" / "episodes.csv"),
             str(REPO / "results" / "rq4-agvfail-qfifo" / "reg" / "s*" / "episodes.csv")]
FLOOR_COLUMNS = ("agv_count", "layout", "tiles", "job_scope", "agv_assignment", "release_rule", "release_weights",
                 "scenario_travel_price")
KEY = ["policy", "seed"]


def clean(df):
    """Episodes that ran to their cap or their last job: not deadlocked, not timed out."""
    return ~(df["deadlock"].astype(bool) | df["timed_out"].astype(bool))


def load_baselines(patterns):
    """PDR episodes from each pattern in order; a later file replaces earlier rows with the same (policy, seed)."""
    base = None
    for pattern in patterns:
        paths = sorted(glob.glob(pattern))
        if not paths:
            raise SystemExit(f"--baselines {pattern}: no files")
        frame = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
        frame = frame[frame["kind"] == "pdr"]
        if base is not None:
            keep = ~base.set_index(KEY).index.isin(frame.set_index(KEY).index)
            frame = pd.concat([base[keep], frame], ignore_index=True)
        base = frame
    if base.duplicated(KEY).any():
        raise SystemExit("The baselines have more than one episode per (policy, seed).")
    return base


def return_gap(ret, ref):
    """Relative gap on return; returns are negative, so > 0 is worse than the reference and < 0 better."""
    return ret / ref - 1


def best_on_average(base):
    """The baseline pair with the lowest mean return gap to the per-seed best (max return)."""
    best = base.groupby("seed")["return"].transform("max")
    return return_gap(base["return"], best).groupby(base["policy"]).mean().idxmin()


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
    ap.add_argument("--baselines", nargs="+", default=BASELINES,
                    help="12-pair episodes.csv files or globs on the same instances and floor; later ones override "
                         "earlier (policy, seed) rows (default: rq4-agvfail/reg + rq4-agvfail-qfifo/reg, D, 7 AGVs)")
    ap.add_argument("--reference", default="auto",
                    help="reference pair, or 'auto' = best pair on average in --baselines (default)")
    ap.add_argument("--alpha", type=float, default=0.05, help="significance level of the one-sided tests")
    ap.add_argument("--bootstrap", type=int, default=10000, help="bootstrap resamples for the median CI")
    ap.add_argument("--rng-seed", type=int, default=0, help="bootstrap RNG seed (keeps the CI reproducible)")
    ap.add_argument("--csv", default=None, help="also write the per-policy table here (unformatted numbers)")
    a = ap.parse_args()

    base_all = load_baselines(a.baselines)
    base = base_all[clean(base_all)]
    if len(base) < len(base_all):
        print(f"Note: {len(base_all) - len(base)} deadlocked/timed-out baseline episodes left out of the per-seed best.")
    cand = pd.concat([pd.read_csv(p) for p in a.episodes], ignore_index=True)
    cand = cand[cand["kind"] != "pdr"]          # PDR rows in the candidate run are ignored: the baselines cover them
    if cand.empty:
        raise SystemExit("No non-PDR episodes found in the given files.")
    for name, df in (("baseline", base), ("candidate", cand)):
        if (df["return"] == 0).all():
            raise SystemExit(f"Every {name} return is 0: that run had no --reward-spec, so it cannot be scored.")
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

    best = base.groupby("seed")["return"].max().rename("best_return")
    ref = base[base["policy"] == reference].set_index("seed")["return"].rename("ref_return")
    c = cand.join(best, on="seed").join(ref, on="seed")
    c["ok"] = clean(c)
    c["gap_best"] = return_gap(c["return"], c["best_return"]).where(c["ok"])
    c["gap_ref"] = return_gap(c["return"], c["ref_return"]).where(c["ok"])
    c["beats_all"] = c["gap_best"] < 0
    c["dead_or_timeout"] = ~c["ok"]

    out = c.groupby("policy").agg(
        seeds=("seed", "nunique"),
        return_vs_best=("gap_best", "mean"), return_vs_ref=("gap_ref", "mean"),
        seeds_beating_all=("beats_all", "sum"), dead_or_timeout=("dead_or_timeout", "sum"),
        jobs_exited=("jobs_exited", "mean"), total_flow_censored=("total_flow_time", "mean"),
    )
    if (c.groupby("policy")["seed"].count() != out["seeds"]).any():
        print("Note: some policy has more than one episode per seed; the test below treats each as a pair.")

    rng = np.random.default_rng(a.rng_seed)
    tests = pd.DataFrame({p: paired_test(g.loc[g["ok"], "gap_ref"], a.alpha, a.bootstrap, rng)
                          for p, g in c.groupby("policy")}).T
    out = out.join(tests.drop(columns="n"))
    refused = (out["dead_or_timeout"] > 0) & (out["verdict"] == "better")
    out.loc[refused, "verdict"] = "n.s. (deadlocks)"
    if a.csv:
        out.assign(reference=reference).to_csv(a.csv)

    shown = out.copy()
    for col in ["return_vs_best", "return_vs_ref", "median", "ci_lo", "ci_hi"]:
        shown[col] = (100 * shown[col].astype(float)).map("{:+.1f}%".format)
    for col in ["p_better", "p_worse"]:
        shown[col] = shown[col].astype(float).map("{:.3g}".format)
    shown["jobs_exited"] = shown["jobs_exited"].map("{:.1f}".format)
    shown["total_flow_censored"] = shown["total_flow_censored"].map("{:,.0f}".format)
    pd.set_option("display.width", 250)
    print(f"Reference: {reference} ({'best on average' if a.reference == 'auto' else 'given'}). "
          "Gaps are on return (> 0 worse, < 0 better); return_vs_best < 0 beats every fixed pair on that seed. "
          "Deadlocked/timed-out episodes are left out of the gaps.\n")
    print(shown.to_string())

    base_best = base.groupby("seed")["return"].transform("max")
    ref_gap = return_gap(base["return"], base_best)[base["policy"] == reference].mean()
    base_jobs = base.groupby("policy")["jobs_exited"].mean()
    print(f"\nBaselines ({base['seed'].nunique()} seeds): {reference} return {100 * ref_gap:+.1f}% vs per-seed best; "
          f"jobs exited {base_jobs.min():.1f}-{base_jobs.max():.1f} across pairs. "
          f"Verdict: one-sided Wilcoxon on the per-seed return gaps to {reference}, alpha {a.alpha}. "
          "Total flow is of exited jobs only (censored): read it with jobs exited, not as a score.")


if __name__ == "__main__":
    main()
