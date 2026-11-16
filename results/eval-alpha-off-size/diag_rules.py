"""
@file diag_rules.py
@brief eval-alpha-off-size, exploratory (10-27, after the pre-declared verdict): why the law + offset arms miss the
       reachable fixed-alpha room on large floors. Per floor: the v5 pairs vs MDD-PTECT; per arm policy (training seed):
       % vs MDD-PTECT, share of slots with the PECT machine rule, job-rule shares, mean played alpha (3 x offset, cap 4)
       overall and in PTECT slots; then the Spearman correlation over the 10 policies of the result with the PECT
       share, the ATC share and the alpha in PTECT slots.
@par Usage (from the repo root)
@code{.sh}
.venv/bin/python results/eval-alpha-off-size/diag_rules.py > results/eval-alpha-off-size/diag_rules.out
@endcode
"""
import collections
import glob
import importlib.util

import numpy as np
from scipy.stats import spearmanr

_spec = importlib.util.spec_from_file_location("aos", "results/eval-alpha-off-size/analyze.py")
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)
SEEDS = set(range(40))
OFF = {"x1": 1, "x0.5": .5, "x0.25": .25, "x1.5": 1.5, "x2": 2}


def dirs(pattern):
    return glob.glob(f"results/{pattern}") if "*" in pattern else [f"results/{pattern}"]


def main():
    for floor in ("m50", "m100", "m200"):
        pdr, _, _ = A.read(dirs(A.SOURCES[floor]["pdr"]), SEEDS)
        base = pdr["MDD-PTECT"]

        def pct(d):
            s = [k for k in base if k in d]
            return 100 * (sum(d[k] for k in s) / sum(base[k] for k in s) - 1)
        print(f"== {floor}: v5 pairs vs MDD-PTECT: " + ", ".join(
            f"{p} {pct(v):+.1f}" for p, v in sorted(pdr.items(), key=lambda x: pct(x[1])) if p != "MDD-PTECT"))
        al, _, _ = A.read(glob.glob(f"results/eval-alpha-off-size/out/{floor}/alpha_s*"), SEEDS)
        print("   fixed alpha: " + ", ".join(f"{p} {pct(v):+.1f}" for p, v in al.items()))
        print("   policy                       % vs PTECT | PECT share | ATC / MDD / SRT | mean alpha | alpha in PTECT slots")
        rows = []
        for grp in ("off-b2", "off-het"):
            pol, dec, _ = A.read(glob.glob(f"results/eval-alpha-off-size/out/{floor}/{grp}_s*"), SEEDS)
            by = collections.defaultdict(list)
            for r in dec:
                by[r["policy"]].append(r["rule"])
            for p in sorted(by):
                rules = by[p]
                pect = np.array([r.split("@")[0].split("-")[1] == "PECT" for r in rules])
                job = collections.Counter(r.split("-")[0] for r in rules)
                alpha = np.array([min(4, 3 * OFF[r.split("@")[1]]) if "@" in r else 3.0 for r in rules])
                row = (pct(pol[p]), pect.mean(), job["ATC"] / len(rules), alpha[~pect].mean())
                rows.append(row)
                print(f"   {p.split('/')[0][5:]:<28} {row[0]:+6.1f} | {row[1]:.2f} | {job['ATC'] / len(rules):.2f} / "
                      f"{job['MDD'] / len(rules):.2f} / {job['SRT'] / len(rules):.2f} | {alpha.mean():.2f} | {row[3]:.2f}")
        d = np.array(rows)
        for j, n in ((1, "PECT share"), (2, "ATC share"), (3, "alpha in PTECT slots")):
            r = spearmanr(d[:, 0], d[:, j])
            print(f"   Spearman rho(result, {n}) over {len(d)} policies: {r.statistic:+.2f} (p {r.pvalue:.3f})")
        lo = d[:, 1] < 0.05
        print(f"   PECT share < 5%: {lo.sum()} policies, mean {d[lo, 0].mean():+.1f}%; others {d[~lo, 0].mean():+.1f}%")


if __name__ == "__main__":
    main()
