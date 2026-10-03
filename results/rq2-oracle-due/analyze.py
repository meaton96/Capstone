"""
@file analyze.py
@brief rq2-oracle-due: Unity switching-oracle headroom on the tardiness objective, and the twin cross-check.

@details Per seed, from result.json (switch_oracle.py): fixed-pair returns and the oracle return of the tardiness reward
(return = -tardiness integral over the window / 1000). Gains are reductions of tardiness (positive = better):
  total_%   oracle vs the best-on-average fixed pair (the go/no-go figure: >= 3% median, >= 60% of seeds above 2%)
  switch_%  oracle vs the seed's own best fixed pair
plus the same vs SRT-TECT (the best flow rule) and the time in system of the oracle's schedule (stages.csv
window_time_in_system). Failure-free seeds (0, 2, 4 of the default generator) are compared with the twin screen
(results/rq2-twin-due/tasks/warm_s<seed>_tard_H15_c2.json, same instances, DES-1k).

@par Usage
@code{.sh}
.venv/bin/python results/rq2-oracle-due/analyze.py [results/rq2-oracle-due-agv4]   # default: this folder
@endcode
The twin cross-check runs only for this folder (the regime variants are not the twin screen's instances).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
TWIN = REPO / "results/rq2-twin-due/tasks"
pd.set_option("display.width", 220)


DIR = HERE


def load():
    rows, stages = [], []
    for p in sorted(DIR.glob("s*/result.json")):
        r = json.loads(p.read_text())
        fixed = {k: -v for k, v in r["fixed_returns"].items()}
        rows.append({"seed": r["seed"], "oracle": -r["oracle_return"], "schedule": "|".join(r["schedule"]),
                     **{f"fix:{k}": v for k, v in fixed.items()}})
        st = pd.read_csv(p.parent / "stages.csv")
        st["seed"] = r["seed"]
        stages.append(st)
    return pd.DataFrame(rows).set_index("seed"), (pd.concat(stages) if stages else pd.DataFrame())


def main():
    import sys
    global DIR
    if len(sys.argv) > 1:
        DIR = Path(sys.argv[1]).resolve()
    print(f"{DIR.name}: {len(list(DIR.glob('s*/result.json')))} seeds finished")
    df, stages = load()
    if df.empty:
        print("no finished seeds yet")
        return
    fixed = df[[c for c in df.columns if c.startswith("fix:")]]
    fixed.columns = [c[4:] for c in fixed.columns]
    zero = fixed.min(axis=1) <= 1e-9
    if zero.any():
        print(f"seeds with no tardiness under some pair (left out of the % figures): {list(fixed.index[zero])}")
    f, d = fixed[~zero], df[~zero]
    best = f.mean().idxmin()
    total = (1 - d["oracle"] / f[best]) * 100
    switch = (1 - d["oracle"] / f.min(axis=1)) * 100
    vs_srt = (1 - d["oracle"] / f["SRT-TECT"]) * 100 if "SRT-TECT" in f else None
    out = pd.DataFrame({"best_pair_value": f[best], "per_seed_best": f.min(axis=1), "per_seed_best_pair": f.idxmin(axis=1),
                        "oracle": d["oracle"], "total_%": total, "switch_%": switch, "vs_SRT-TECT_%": vs_srt,
                        "schedule": d["schedule"]})
    print(f"== Per seed (tardiness, 1000 job-s; best-on-average pair {best}) ==")
    print(out.round(3).to_string(), "\n")
    print(f"total_% vs {best}: mean {total.mean():.2f}, median {total.median():.2f}, seeds > 2%: "
          f"{int((total > 2).sum())}/{len(total)} ({(total > 2).mean() * 100:.0f}%)")
    print(f"switch_% vs per-seed best: mean {switch.mean():.2f}, median {switch.median():.2f}")
    if vs_srt is not None:
        print(f"oracle vs SRT-TECT: mean {vs_srt.mean():.2f}, median {vs_srt.median():.2f}")
    gaps = (f.div(f.min(axis=1), axis=0) - 1) * 100
    print("\n== Fixed pairs: mean gap to the per-seed best (%), wins ==")
    print(pd.DataFrame({"gap_%": gaps.mean(), "wins": f.idxmin(axis=1).value_counts()}).fillna(0)
          .sort_values("gap_%").round(2).to_string(), "\n")
    job_rules = d["schedule"].str.split("|").explode().str.split("-").str[0].value_counts(normalize=True)
    print("oracle job rules:", ", ".join(f"{k} {v:.0%}" for k, v in job_rules.items()))

    if not stages.empty and "window_time_in_system" in stages:
        st1 = stages[stages.stage == 1].pivot(index="seed", columns="tail", values="window_time_in_system")
        last = stages[stages.stage == stages.groupby("seed").stage.transform("max")]
        oracle_tis = last.loc[last.groupby("seed")["return"].idxmax(), ["seed", "window_time_in_system"]].set_index("seed")
        tis_best = st1.min(axis=1)
        print(f"\noracle schedule's time in system vs the seed's best fixed pair on time in system: "
              f"{((oracle_tis['window_time_in_system'] / tis_best - 1) * 100).mean():+.2f}% (mean)")

    rows = []
    failure_free = {0, 2, 4} if DIR == HERE else set()   # seeds without machine failures; twin = this regime only
    for seed in df.index:
        if seed not in failure_free:
            continue
        p = TWIN / f"warm_s{seed}_tard_H15_c2.json"
        if not p.exists():
            continue
        t = json.loads(p.read_text())
        twin = pd.Series({k: v["tard"] for k, v in t["fixed"].items()})
        unity = fixed.loc[seed]          # already 1000 job-s (-return)
        common = [k for k in unity.index if k in twin.index]
        rows.append({"seed": seed, "twin_best": twin[common].idxmin(), "unity_best": unity[common].idxmin(),
                     "rank_corr": twin[common].rank().corr(unity[common].rank()),
                     "mean_abs_diff_%": ((twin[common] / unity[common].replace(0, np.nan) - 1).abs() * 100).mean(),
                     "twin_oracle_gain_%": (1 - t["oracle"] / t["best_fixed"]) * 100 if t["best_fixed"] > 0 else np.nan,
                     "unity_oracle_gain_%": (1 - df.loc[seed, "oracle"] / fixed.loc[seed].min()) * 100
                     if fixed.loc[seed].min() > 0 else np.nan})
    if rows:
        print("\n== Twin cross-check (failure-free seeds; gain vs the seed's best fixed pair) ==")
        print(pd.DataFrame(rows).round(2).to_string(index=False))
    out.to_csv(DIR / "per_seed.csv", float_format="%.4f")


if __name__ == "__main__":
    main()
