"""Summarize rq2-linked-price (run.sh): which fixed machine rule is best on a linked 7-tile floor, per fleet and release.

Reads <condition>/s<seed>/episodes.csv, writes episodes_all.csv and summary.csv next to this file. Score = time in
system of every job over the window (-episode return of the flow_time reward), lower is better; gaps are per seed
against the best variant of the same (fleet, release) on that seed. Works on partial results.

    .venv/bin/python results/rq2-linked-price/analyze.py
"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
VARIANTS = ["l0", "l1", "l4", "loc"]
LABEL = {"l0": "TECT (λ 0)", "l1": "TECT λ 1", "l4": "TECT λ 4", "loc": "stay in tile"}


def load():
    frames = []
    for f in sorted(HERE.glob("*/s*/episodes.csv")):
        d = pd.read_csv(f)
        fleet, rel, var = f.parent.parent.name.split("-")
        d["fleet"], d["release"], d["variant"] = fleet, rel, var
        frames.append(d)
    if not frames:
        raise SystemExit("no episodes yet")
    df = pd.concat(frames, ignore_index=True)
    df["tis"] = -df["return"]                                   # time in system (reward units: s / 1000)
    df["cross_pct"] = 100 * df["cross_tile_moves"] / df["routed_moves"]
    df["tiles_per_cross"] = df["tiles_crossed"] / df["cross_tile_moves"].where(df["cross_tile_moves"] > 0)
    df["agv_busy_pct"] = 100 * (1 - df["agv_idle_fraction"])
    df["best"] = df.groupby(["fleet", "release", "seed"])["tis"].transform("min")
    df["gap_pct"] = 100 * (df["tis"] / df["best"] - 1)
    return df


def main():
    df = load()
    df.to_csv(HERE / "episodes_all.csv", index=False)
    pd.set_option("display.width", 200)
    g = df.groupby(["fleet", "release", "variant"])
    summary = pd.DataFrame({
        "seeds": g["seed"].nunique(),
        "time_in_system": g["tis"].mean(),
        "gap_to_best_pct": g["gap_pct"].mean(),
        "seeds_won": g["gap_pct"].agg(lambda x: int((x < 1e-9).sum())),
        "jobs_exited": g["jobs_exited"].mean(),
        "cross_tile_pct": g["cross_pct"].mean(),
        "tiles_per_cross": g["tiles_per_cross"].mean(),
        "agv_busy_pct": g["agv_busy_pct"].mean(),
        "deadlocks": g["deadlock"].sum(),
    }).round(2)
    summary.to_csv(HERE / "summary.csv")
    print(summary.to_string())

    print("\nBest variant on average per (fleet, release); does it change?")
    best = summary.reset_index().loc[lambda s: s.groupby(["fleet", "release"])["gap_to_best_pct"].idxmin()]
    for _, r in best.iterrows():
        print(f"  {r['fleet']} {r['release']}: {LABEL.get(r['variant'], r['variant'])} "
              f"(gap {r['gap_to_best_pct']:.2f}%, won {r['seeds_won']}/{r['seeds']})")

    print("\nPer seed, time in system relative to TECT λ 0 (%):")
    piv = df.pivot_table(index=["fleet", "release", "seed"], columns="variant", values="tis")
    rel = piv.div(piv["l0"], axis=0).sub(1).mul(100).round(1) if "l0" in piv else piv
    print(rel.reindex(columns=[v for v in VARIANTS if v in rel.columns]).to_string())

    rc = df.dropna(subset=["release_counts"]).groupby("release")["release_counts"].first()
    print("\nJobs released per tile (first episode of each release rule):")
    for rel_name, counts in rc.items():
        print(f"  {rel_name}: {counts}")


if __name__ == "__main__":
    main()
