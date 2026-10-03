"""
@file analyze.py
@brief base-due-c2: fixed-pair baseline on the tardiness objective (21 head pairs x held-out seeds 0-19, TWK due
       dates c = 2, random warm-up, 5,400 s window): per pair mean tardiness and time in system over the window, share
       of exited jobs that were late, gap to the per-seed best, wins. Checks seeds 0-7 against rq2-oracle-due's stage-1
       fixed runs (same player build and instances: equal returns expected).

@par Usage
@code{.sh}
.venv/bin/python results/base-due-c2/analyze.py
@endcode
"""
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
pd.set_option("display.width", 200)


def main():
    files = sorted(HERE.glob("s*/episodes.csv"))
    if not files:
        print("no finished seeds yet")
        return
    e = pd.concat([pd.read_csv(f) for f in files])
    print(f"{e.seed.nunique()} seeds, {len(e)} episodes, deadlocks {int(e.deadlock.sum())}, timeouts {int(e.timed_out.sum())}")
    e["tard"] = e["window_tardiness"] / 1000.0
    e["tis"] = e["window_time_in_system"] / 1000.0
    e["pct_late"] = 100.0 * e["jobs_exited_late"] / e["jobs_exited"].where(e["jobs_exited"] > 0)
    tard = e.pivot(index="seed", columns="policy", values="tard")
    tis = e.pivot(index="seed", columns="policy", values="tis")
    ok = tard.min(axis=1) > 1e-9
    gap = (tard[ok].div(tard[ok].min(axis=1), axis=0) - 1) * 100
    tis_gap = (tis.div(tis.min(axis=1), axis=0) - 1) * 100
    table = pd.DataFrame({
        "tardiness": tard.mean(), "gap_%": gap.mean(), "gap_median_%": gap.median(),
        "wins": tard[ok].idxmin(axis=1).value_counts(), "time_in_system": tis.mean(), "tis_gap_%": tis_gap.mean(),
        "pct_late": e.groupby("policy")["pct_late"].mean(),
    }).fillna({"wins": 0}).sort_values("gap_%")
    print("\n== Fixed pairs (tardiness and time in system: 1000 job-s over the window; gaps to the per-seed best) ==")
    print(table.round(2).to_string())
    print(f"\nbest on average: {table.index[0]}; seeds without tardiness under some pair: {int((~ok).sum())}")

    rows = []
    for seed in tard.index:
        p = REPO / f"results/rq2-oracle-due/s{seed}/result.json"
        if not p.exists():
            continue
        fixed = json.loads(p.read_text())["fixed_returns"]
        for pair, ret in fixed.items():
            mine = e[(e.seed == seed) & (e.policy == pair)]["return"]
            if len(mine):
                rows.append(abs(float(mine.iloc[0]) - ret))
    if rows:
        print(f"check vs rq2-oracle-due stage 1: {len(rows)} pairs compared, max |return difference| {max(rows):.3g}")
    table.to_csv(HERE / "pairs.csv", float_format="%.4f")


if __name__ == "__main__":
    main()
