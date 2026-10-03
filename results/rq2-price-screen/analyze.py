"""
@file analyze.py
@brief rq2-price-screen: does TECT's travel price change the best fixed machine rule on the 15-machine floor?

@details Reads <root>/<regime>/<variant>/s<seed>/episodes.csv. Each row is one rule: ECT, or TECT at the
travel price λ in its travel_price column (job rule SRT throughout). Time in system = -return of the flow_time
reward. Per regime and rule it reports: mean time in system, the mean gap to the per-seed best rule, seeds won,
jobs exited, and the per-seed change against plain TECT. It also checks SRT-ECT / SRT-TECT λ 0 against the
stage-1 fixed returns of rq2-oracle-screen-qfifo (or rq2-oracle-screen) wherever that seed exists; they must be
equal. Writes summary.csv and episodes_all.csv next to the results.

@par Usage
@code{.sh}
.venv/bin/python results/rq2-price-screen/analyze.py [results/rq2-price-screen]
@endcode
"""

import json
import sys
from pathlib import Path

import pandas as pd


def load(root: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(root.glob("*/*/s*/episodes.csv")):
        df = pd.read_csv(path)
        df["regime"] = path.parts[-4]
        frames.append(df)
    if not frames:
        raise SystemExit(f"no episodes under {root}")
    df = pd.concat(frames, ignore_index=True)
    df["tis"] = -df["return"]
    df["rule"] = [("ECT" if p == "SRT-ECT" else f"TECT λ{lam:g}") for p, lam in zip(df["policy"], df["travel_price"])]
    return df


def oracle_check(df: pd.DataFrame, repo: Path) -> None:
    checked, bad = 0, []
    for _, row in df[df["rule"].isin(["ECT", "TECT λ0"])].iterrows():
        for name in ("rq2-oracle-screen-qfifo", "rq2-oracle-screen"):
            res = repo / "results" / name / row["regime"] / f"s{row['seed']}" / "result.json"
            if res.exists():
                ref = json.loads(res.read_text())["fixed_returns"][row["policy"]]
                checked += 1
                if abs(ref - row["return"]) > 1e-6:
                    bad.append((row["regime"], row["seed"], row["policy"], row["return"], ref, name))
                break
    print(f"Oracle stage-1 check: {checked} episodes compared, {len(bad)} differ")
    for b in bad:
        print("  DIFFERS:", b)


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
    repo = Path(__file__).resolve().parents[2]
    df = load(root)
    df.to_csv(root / "episodes_all.csv", index=False)
    oracle_check(df, repo)

    df["best"] = df.groupby(["regime", "seed"])["tis"].transform("min")
    df["gap_pct"] = 100.0 * (df["tis"] / df["best"] - 1.0)
    df["won"] = df["tis"] == df["best"]
    plain = df[df["rule"] == "TECT λ0"].set_index(["regime", "seed"])["tis"]
    df["vs_tect_pct"] = [100.0 * (t / plain.get((r, s), float("nan")) - 1.0)
                         for r, s, t in zip(df["regime"], df["seed"], df["tis"])]

    order = {"base-w": 0, "agv4": 1, "agv3": 2, "amax1": 3}
    summary = (df.groupby(["regime", "rule"])
                 .agg(seeds=("seed", "nunique"), time_in_system=("tis", "mean"), vs_plain_tect_pct=("vs_tect_pct", "mean"),
                      gap_to_best_pct=("gap_pct", "mean"), seeds_won=("won", "sum"), jobs_exited=("jobs_exited", "mean"))
                 .reset_index())
    summary["_o"] = summary["regime"].map(order).fillna(9)
    summary = summary.sort_values(["_o", "regime", "rule"]).drop(columns="_o")
    summary.to_csv(root / "summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(summary.round(2).to_string(index=False))

    print("\nBest rule on average per regime (smallest mean gap to the per-seed best):")
    for reg, g in summary.groupby("regime", sort=False):
        b = g.sort_values("gap_to_best_pct").iloc[0]
        print(f"  {reg}: {b['rule']} (gap {b['gap_to_best_pct']:.2f}%, won {int(b['seeds_won'])}/{int(b['seeds'])})")

    print("\nPer seed, time in system relative to plain TECT (%):")
    table = df.pivot_table(index=["regime", "seed"], columns="rule", values="vs_tect_pct")
    print(table.round(2).to_string())


if __name__ == "__main__":
    main()
