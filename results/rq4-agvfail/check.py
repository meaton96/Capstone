"""
@file check.py
@brief Merge rq4-agvfail episodes, check reg against base-pdr12, and print a first dose-response table.

@details Run from the repo root after run.sh (it calls this at the end):
    .venv/bin/python results/rq4-agvfail/check.py
Writes results/rq4-agvfail/episodes_all.csv (every variant, with a `variant` column). The regression check pairs
reg with results/eval_pdr12_heads/episodes_all.csv on (policy, seed) and compares makespan, total flow, decisions
and jobs exited; any mismatch has to be explained before the old baseline is reused. The dose table is paired per
seed against reg on the same build, so it stays valid even if reg differs from the old baseline.
"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
BASELINE = HERE.parent / "eval_pdr12_heads" / "episodes_all.csv"
VARIANTS = ["reg", "l8400", "l3000", "l1500"]
KEY = ["policy", "seed"]


def load() -> pd.DataFrame:
    frames = []
    for variant in VARIANTS:
        for path in sorted((HERE / variant).glob("s*/episodes.csv")):
            frames.append(pd.read_csv(path).assign(variant=variant))
    if not frames:
        raise SystemExit("no episodes.csv found")
    episodes = pd.concat(frames, ignore_index=True)
    episodes.to_csv(HERE / "episodes_all.csv", index=False)
    return episodes


def regression(reg: pd.DataFrame) -> None:
    base = pd.read_csv(BASELINE)
    merged = reg.merge(base, on=KEY, suffixes=("", "_base"), how="outer", indicator=True)
    print(f"== reg vs base-pdr12: {len(reg)} reg episodes, {len(base)} baseline, "
          f"{(merged._merge == 'both').sum()} paired")
    both = merged[merged._merge == "both"]
    mismatched = pd.Series(False, index=both.index)
    for column in ["makespan", "total_flow_time", "decisions", "jobs_exited"]:
        rel = (both[column] - both[f"{column}_base"]).abs() / both[f"{column}_base"].abs().clip(lower=1e-9)
        bad = rel > 1e-6
        mismatched |= bad
        print(f"   {column:16s} mismatches {int(bad.sum()):3d}   max rel diff {rel.max():.2e}")
    if mismatched.any():
        print("   MISMATCHED episodes (policy, seed):")
        print(both.loc[mismatched, KEY + ["total_flow_time", "total_flow_time_base"]].to_string(index=False))
    else:
        print("   REPRODUCED: every paired episode matches")


def dose(episodes: pd.DataFrame) -> None:
    reg = episodes[episodes.variant == "reg"].set_index(KEY)
    rows = []
    for variant in VARIANTS[1:]:
        cur = episodes[episodes.variant == variant].set_index(KEY)
        if cur.empty:
            continue
        paired = cur.join(reg, rsuffix="_reg", how="inner")
        paired["d_total_flow"] = paired.total_flow_time / paired.total_flow_time_reg - 1
        paired["d_mean_flow"] = paired.mean_flow_time / paired.mean_flow_time_reg - 1
        paired["d_jobs"] = paired.jobs_exited - paired.jobs_exited_reg
        for policy, group in paired.groupby(level="policy"):
            rows.append({"variant": variant, "policy": policy, "n": len(group),
                         "agv_failures": group.agv_failures.mean(),
                         "repair_s": group.agv_repair_time.mean(),
                         "blocked_s": group.agv_blocked_by_failure_time.mean(),
                         "d_total_flow_median": group.d_total_flow.median(),
                         "d_total_flow_mean": group.d_total_flow.mean(),
                         "d_mean_flow_median": group.d_mean_flow.median(),
                         "d_jobs_mean": group.d_jobs.mean(),
                         "deadlocks": int(group.deadlock.astype(str).eq("True").sum())})
    if rows:
        print("\n== dose-response, paired per seed vs reg (same build)")
        print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    all_episodes = load()
    print(all_episodes.groupby("variant").size().to_string())
    regression(all_episodes[all_episodes.variant == "reg"])
    dose(all_episodes)
