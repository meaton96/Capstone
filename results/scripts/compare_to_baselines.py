"""Score evaluate.py episodes (e.g. a trained checkpoint) against the saved 12 head-rule baselines.

The baselines are results/eval_pdr12_heads/episodes_all.csv (docs/experiments/EVAL_BASELINES_12HEAD_0927.md).
The candidate run must use the same instances: --scenario-generator randomized --episode-duration-seconds 5400,
seeds within 0-19, a two-branch player.

    python results/scripts/compare_to_baselines.py results/eval_rnd02/episodes.csv [more episodes.csv ...]

Per policy it prints the mean gap to the per-seed best baseline rule (negative = beats every fixed rule on that
seed) and to SPT-ECT (the best fixed rule on average), on total flow and mean flow, plus jobs exited and seed wins.
"""
import argparse
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
BASELINES = REPO / "results" / "eval_pdr12_heads" / "episodes_all.csv"
REFERENCE_RULE = "SPT-ECT"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("episodes", nargs="+", help="evaluate.py episodes.csv file(s) to score")
    ap.add_argument("--baselines", default=str(BASELINES))
    a = ap.parse_args()

    base = pd.read_csv(a.baselines)
    cand = pd.concat([pd.read_csv(p) for p in a.episodes], ignore_index=True)
    cand = cand[cand["kind"] != "pdr"]          # PDR rows in the candidate run are ignored: the baselines cover them
    if cand.empty:
        raise SystemExit("No non-PDR episodes found in the given files.")

    missing = sorted(set(cand["seed"]) - set(base["seed"]))
    if missing:
        raise SystemExit(f"Seeds {missing} have no baselines (baselines cover seeds {sorted(base['seed'].unique())}).")

    best = base.groupby("seed").agg(best_total=("total_flow_time", "min"), best_mean=("mean_flow_time", "min"))
    ref = base[base["policy"] == REFERENCE_RULE].set_index("seed")[["total_flow_time", "mean_flow_time"]]
    ref.columns = ["ref_total", "ref_mean"]
    c = cand.join(best, on="seed").join(ref, on="seed")
    c["gap_best_total"] = c["total_flow_time"] / c["best_total"] - 1
    c["gap_best_mean"] = c["mean_flow_time"] / c["best_mean"] - 1
    c["gap_ref_total"] = c["total_flow_time"] / c["ref_total"] - 1
    c["gap_ref_mean"] = c["mean_flow_time"] / c["ref_mean"] - 1
    c["beats_all_total"] = c["gap_best_total"] < 0

    out = c.groupby("policy").agg(
        seeds=("seed", "nunique"),
        total_vs_best=("gap_best_total", "mean"), total_vs_spt_ect=("gap_ref_total", "mean"),
        mean_vs_best=("gap_best_mean", "mean"), mean_vs_spt_ect=("gap_ref_mean", "mean"),
        jobs_exited=("jobs_exited", "mean"), seeds_beating_all=("beats_all_total", "sum"),
        deadlocks=("deadlock", "sum"),
    )
    base_jobs = base.groupby("policy")["jobs_exited"].mean()
    for col in ["total_vs_best", "total_vs_spt_ect", "mean_vs_best", "mean_vs_spt_ect"]:
        out[col] = (100 * out[col]).map("{:+.1f}%".format)
    out["jobs_exited"] = out["jobs_exited"].map("{:.1f}".format)
    pd.set_option("display.width", 200)
    print(out.to_string())
    print(f"\nReference (baselines, seeds 0-19): SPT-ECT total flow +8.1% vs per-seed best; jobs exited "
          f"{base_jobs.min():.1f}-{base_jobs.max():.1f} across rules. vs_best < 0 beats every fixed rule on that seed.")


if __name__ == "__main__":
    main()
