"""
@file analyze.py
@brief agv-study-fail-resume (2026-10-28): the failures-on cells of the AGV study (b2tf seeds 0-2, layout D,
       MDD-PTECT, 5-20 AGVs, release / hold) under preempt-resume, next to agv-study-rerun's restart runs of the same
       cells and its failure-free b2t runs (same jobs, seeds 0-1). Runs are read with agv-study-rerun/analyze.py's
       loader. -> analysis.out
"""
import importlib.util
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("agv_study", HERE.parent / "agv-study-rerun" / "analyze.py")
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)
NEW = {"agv-study-fail-resume-rel": ("resume", "release"), "agv-study-fail-resume-hold": ("resume", "hold"),
       "agv-v5-fail-rel": ("restart", "release"), "agv-v5-fail-hold": ("restart", "hold")}
A.SWEEPS = list(NEW) + ["agv-v5-rel"]


def main():
    pd.set_option("display.width", 200)
    d = A.runs()
    fa = d[d.sweep.isin(NEW)].copy()
    fa["model"] = fa.sweep.map(lambda s: NEW[s][0])
    fa["proto"] = fa.sweep.map(lambda s: NEW[s][1])
    n = fa.groupby(["model", "proto"]).size()
    print("== cells:", ", ".join(f"{m} {p} {k}" for (m, p), k in n.items()))
    print("\n== total tardiness (1000 job-s), mean over seeds 0-2, by fleet")
    print(fa.pivot_table(index="agvs", columns=["proto", "model"], values="tard_k", aggfunc="mean").round(1).to_string())
    print("\n== machine failures per run / deadlocks / collisions / stalls (means)")
    print(fa.pivot_table(index="agvs", columns=["proto", "model"], values=["failures", "deadlock", "collisions", "stalls"],
                         aggfunc="mean").round(2).to_string())
    print("\n== incomplete runs (deadlock or censored jobs)")
    print(fa.groupby(["proto", "model"]).incomplete.sum().to_string())
    nf = d[(d.sweep == "agv-v5-rel") & (d.regime == "b2t") & (d.layout == "D") & (d.rule == "MDD_PTECT")]
    rel = fa[fa.proto == "release"]
    if len(nf) and len(rel):
        b = nf.groupby(["inst", "agvs"]).tard_k.mean().rename("no failures")
        parts = [b]
        for m in ("restart", "resume"):
            a = rel[rel.model == m].groupby(["inst", "agvs"]).tard_k.mean()
            parts.append(a.rename(m))
        both = pd.concat(parts, axis=1).dropna(subset=["no failures"])
        for m in ("restart", "resume"):
            both[f"cost_{m}_%"] = 100 * (both[m] / both["no failures"] - 1)
        print("\n== cost of machine failures under release (same jobs, seeds 0-1), restart vs resume")
        print(both.round(1).to_string())


if __name__ == "__main__":
    main()
