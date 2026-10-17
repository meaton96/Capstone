"""
@file analyze.py
@brief rq4-uncert-v5-policy (2026-10-17): the final policy (cmix pick s3), the PPO families and the 6 v5 pairs under the
       twin uncertainty layer, B2 test seeds 0-39, 24 cells + none (B2: ../rq3-gen-v5 base, ../eval-v5-cmix/test).
       Per cell: every pick / family mean vs MDD-PTECT (the noise-free rule NF of ../rq4-uncert-screen) and vs the cell's
       best v5 pair (in-sample), the degradation from none, and the reproduction check against the screen (MDD-PTECT
       here vs MDD-PTECT@3 in ../rq4-uncert-screen/out/rules: only where it matches do the screen's planner / oracle /
       level-tuned-rule numbers carry over). -> analysis.out, summary.csv
"""
import importlib.util
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
R = HERE.parent
_s = importlib.util.spec_from_file_location("size", R / "rq3-size-v5" / "analyze.py")
size = importlib.util.module_from_spec(_s)
_s.loader.exec_module(size)                     # name() / table() for the four groups
rq3 = size.rq3
sys.path.insert(0, str(HERE))
import task as T                                 # noqa: E402  (CELLS, cell_dir)
SEEDS = list(range(40))
SCREEN = R / "rq4-uncert-screen" / "out" / "rules"
POLS = ["cmix s3", "cmix mean of 5", "loose s2", "loose mean of 5", "PPO s4", "PPO mean of 5", "hindsight best v5 pair"]


def cell_table(cell):
    if cell == "none":
        return size.floor_table("m15")
    return size.table([HERE / T.cell_dir(cell) / "base" / g / "episodes.csv" for g in ("pdr", "ppo", "cmix", "loose")])


def screen_match(cell, t):
    ok = n = 0
    for s in t.index:
        f = SCREEN / ("none" if cell == "none" else T.cell_dir(cell)) / f"s{s}.json"
        if f.exists():
            n += 1
            ok += abs(json.loads(f.read_text())["tardiness"]["MDD-PTECT@3"] - t.loc[s, rq3.REF]) < 1e-3
    return ok, n


def main():
    rng = np.random.default_rng(0)
    print("== rq4-uncert-v5-policy: twin uncertainty cells, B2 test seeds 0-39; final policy = cmix pick s3; < 0 better")
    rows, base = [], None
    for cell in ["none"] + T.CELLS:
        t = cell_table(cell)
        pairs = [p for p in rq3.V5 if p in t.columns]
        if len(pairs) < 6 or rq3.REF not in t.columns:
            print(f"\n-- {cell}: pairs not complete yet")
            continue
        for fam in ("cmix", "loose", "PPO"):
            mem = [f"{fam} s{k}" for k in range(5)]
            if all(m in t.columns for m in mem):
                t[f"{fam} mean of 5"] = t[mem].mean(axis=1, skipna=False)
        best = t[pairs].sum().idxmin()
        t["hindsight best v5 pair"] = t[pairs].min(axis=1)
        ok, n = screen_match(cell, t)
        if base is None:
            base = t
        deg = 100 * (t[rq3.REF].sum() / base[rq3.REF].reindex(t.index).sum() - 1)
        print(f"\n-- {cell}: MDD-PTECT mean {t[rq3.REF].mean():.1f} k job-s ({deg:+.1f}% vs none); best pair {best}; "
              f"MDD-PTECT = screen's on {ok} / {n} seeds{'' if ok == n else '  (screen numbers do NOT carry over)'}")
        for pol in POLS:
            if pol not in t.columns:
                continue
            for vs, b in (("MDD-PTECT", rq3.REF), ("best pair", best)):
                c = rq3.compare(t[pol].to_dict(), t[b].to_dict(), rng)
                if c:
                    rows.append({"cell": cell, "policy": pol, "vs": vs, "best_pair": best, "screen_match": f"{ok}/{n}", **c})
                    if vs == "MDD-PTECT":
                        print(f"   {pol:24s} vs MDD-PTECT {c['pooled_%']:+6.2f} [{c['ci_lo']:+6.2f}, {c['ci_hi']:+6.2f}] "
                              f"{c['better']:2d}/{c['worse']:<2d} p={c['wilcoxon_p']:.2g}")
    pd.DataFrame(rows).to_csv(HERE / "summary.csv", index=False)


if __name__ == "__main__":
    buf = io.StringIO()
    with redirect_stdout(buf):
        main()
    (HERE / "analysis.out").write_text(buf.getvalue())
    sys.stdout.write(buf.getvalue())
