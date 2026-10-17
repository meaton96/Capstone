"""
@file analyze.py
@brief rq3-gen-v5 (2026-10-14): RQ3 on the v5 rules. Per condition, the v5 policies (PPO 5 seeds, BC of the v5 oracle
       3 seeds) vs that condition's best fixed v5 pair on the B2 test seeds 0-39 (twin, window tardiness). Same
       statistics as ../rq3-gen-twin/analyze.py: pooled change with a seed-bootstrap 95% CI, mean / median per-seed
       change, seeds better / worse, paired two-sided Wilcoxon p.

@details
  - Best fixed pair = the v5 pair with the lowest pooled window tardiness over the same 40 seeds (in-sample, which
    flatters the baseline slightly). Its identity, the runner-up, MDD-PTECT (the B2 reference, chosen on fitting
    seeds) and MDD-TECT (the H15 best on B2) are printed with their gaps, and the condition's best H15 pair from
    ../rq3-gen-twin/best_pairs.csv for context.
  - Rows: each PPO seed, "PPO mean of 5 seeds" (per-instance mean tardiness of the 5 policies), the final policy
    (the seed ../eval-v5-final picked on the fitting seeds, selected.txt), each BC seed and their mean, and the
    per-instance hindsight best of the 6 v5 pairs.
  - Run-to-run check: MDD-PTECT in the ppo / bc groups must equal the pdr group per seed (mdd_run_mismatch = 0).

@par Usage
@code{.sh}
.venv/bin/python results/rq3-gen-v5/analyze.py      # -> analysis.out, summary.csv, best_pairs.csv
@endcode
"""
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
R = HERE.parent
SEEDS = list(range(40))
HASH_STR = {"config_hash": str, "instance_hash": str}
CONDS = ["base", "load-0.8", "load-1.2", "due-tight", "due-loose", "agv5-bay9", "agv7-bay9", "agv9-bay9",
         "lay-c", "lay-g", "lay-j"]
V5 = ["SRT-PECT", "SRT-PTECT", "MDD-PECT", "MDD-PTECT", "ATC-PECT", "ATC-PTECT"]
REF = "MDD-PTECT"
SELECTED = R / "eval-v5-final" / "selected.txt"


def short(p):
    p = str(p).replace("ckpt:", "").replace("/checkpoint", "").replace("results/", "")
    if "_s" in p and ("train-v5-nokl" in p or "train-v5-final" in p):
        return "PPO s" + p.rsplit("_s", 1)[1]
    if "bc_full_s" in p:
        return "BC s" + p.split("bc_full_s")[1].replace(".pt", "")
    return p


def load(cond):
    """{pair: {seed: tard}}, {policy: {seed: tard}}, run-to-run mismatches (window tardiness in 1000 job-s)."""
    pairs, pols, mism = {}, {}, 0
    for g in ("pdr", "ppo", "bc"):
        f = HERE / cond / g / "episodes.csv"
        if not (f.exists() and (f.parent / "DONE").exists()):
            continue
        e = pd.read_csv(f, dtype=HASH_STR)
        e = e[e.seed.isin(SEEDS)]
        for pol, d in e.groupby("policy"):
            v = {int(s): t / 1000.0 for s, t in zip(d.seed, d.window_tardiness)}
            name = short(pol)
            if g == "pdr":
                pairs[name] = v
            elif name == REF:
                pols.setdefault("_ref_" + g, v)
            else:
                pols[name] = v
    for k in [k for k in pols if k.startswith("_ref_")]:
        ref = pairs.get(REF, {})
        mism += sum(abs(pols[k][s] - ref[s]) > 1e-6 for s in pols[k] if s in ref)
        del pols[k]
    for fam, n in (("PPO", 5), ("BC", 3)):
        mem = [k for k in pols if k.startswith(fam + " s")]
        if len(mem) == n and all(all(s in pols[k] for s in SEEDS) for k in mem):
            pols[f"{fam} mean of {n} seeds"] = {s: float(np.mean([pols[k][s] for k in mem])) for s in SEEDS}
    return pairs, pols, mism


def compare(x_by_seed, b_by_seed, rng):
    s = [k for k in SEEDS if k in x_by_seed and k in b_by_seed]
    if len(s) < 5:
        return None
    x, b = np.array([x_by_seed[k] for k in s]), np.array([b_by_seed[k] for k in s])
    idx = rng.integers(0, len(s), (5000, len(s)))
    boot = 100 * (x[idx].sum(1) / b[idx].sum(1) - 1)
    per, diff = 100 * (x / b - 1), x - b
    return {"n": len(s), "pooled_%": 100 * (x.sum() / b.sum() - 1), "ci_lo": np.percentile(boot, 2.5),
            "ci_hi": np.percentile(boot, 97.5), "mean_%": per.mean(), "median_%": np.median(per),
            "better": int((diff < 0).sum()), "worse": int((diff > 0).sum()),
            "wilcoxon_p": float(wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0}


def fmt(v):
    return f"{v:.3f}" if abs(v) < 0.01 else f"{v:.2f}"


def main():
    pd.set_option("display.width", 240)
    rng = np.random.default_rng(0)
    sel = SELECTED.read_text().split()[0] if SELECTED.exists() else None
    final = f"PPO {sel}" if sel else None
    h15 = {}
    f = R / "rq3-gen-twin" / "best_pairs.csv"
    if f.exists():
        b = pd.read_csv(f)
        h15 = dict(zip(b.condition, b.best_pair))
    rows, ident = [], []
    for cond in CONDS:
        pairs, pols, mism = load(cond)
        full = {p: v for p, v in pairs.items() if p in V5 and all(s in v for s in SEEDS)}
        if len(full) < len(V5):
            if pols or pairs:
                print(f"[{cond}] pdr group missing or incomplete; skipped")
            continue
        totals = sorted((sum(v.values()), p) for p, v in full.items())
        best, base = totals[0][1], full[totals[0][1]]
        tot0 = totals[0][0]
        ident.append({"condition": cond, "best_v5_pair": best, "best_total": tot0, "runner_up": totals[1][1],
                      "runner_up_%": 100 * (totals[1][0] / tot0 - 1),
                      "MDD-PTECT_rank": [p for _, p in totals].index(REF) + 1,
                      "MDD-PTECT_%": 100 * (sum(full[REF].values()) / tot0 - 1),
                      "MDD-TECT_%": 100 * (sum(pairs["MDD-TECT"].values()) / tot0 - 1) if "MDD-TECT" in pairs else np.nan,
                      "best_H15_pair (rq3-gen-twin)": h15.get(cond, ""), "mdd_run_mismatch": mism})
        cand = dict(pols)
        if final and final in pols:
            cand[f"final policy ({sel})"] = pols[final]
        cand["hindsight best v5 pair"] = {s: min(v[s] for v in full.values()) for s in SEEDS}
        if best != REF:
            cand["MDD-PTECT (B2 reference)"] = full[REF]
        for name, v in cand.items():
            r = compare(v, base, rng)
            if r:
                rows.append({"condition": cond, "method": name, "vs": best, **r})
    if not ident:
        print("no finished conditions yet")
        return
    t, i = pd.DataFrame(rows), pd.DataFrame(ident)
    print("== Best fixed v5 pair per condition (lowest pooled window tardiness over seeds 0-39, in-sample); "
          "gaps are pooled % to it; mdd_run_mismatch = seeds where MDD-PTECT in ppo / bc differs from pdr (must be 0)")
    if final:
        print(f"   final policy: {final} (chosen on fitting seeds 2000-2159 by eval-v5-final)")
    print(i.to_string(index=False, float_format=fmt))
    print("\n== Per condition: pooled change in window tardiness vs that condition's best v5 pair (< 0 better); "
          "CI = seed bootstrap 95%; p = paired two-sided Wilcoxon")
    for cond, g in t.groupby("condition", sort=False):
        print(f"\n-- {cond} (vs {g.vs.iloc[0]})")
        print(g.drop(columns=["condition", "vs"]).sort_values("pooled_%").to_string(index=False, float_format=fmt))
    print("\n== Overview: pooled % vs the condition's best v5 pair [95% CI] (seeds better / 40); * = p < 0.05")
    piv = {}
    for _, r in t.iterrows():
        star = "*" if r.wilcoxon_p < 0.05 else " "
        piv.setdefault(r.condition, {})[r.method] = f"{r['pooled_%']:+.1f} [{r.ci_lo:+.1f},{r.ci_hi:+.1f}] ({r.better}){star}"
    cols = ([f"final policy ({sel})"] if final else []) + ["PPO mean of 5 seeds", "BC mean of 3 seeds",
                                                           "hindsight best v5 pair"]
    ov = pd.DataFrame.from_dict(piv, orient="index").reindex(columns=cols)
    ov.insert(0, "best v5 pair", i.set_index("condition").best_v5_pair)
    print(ov.fillna("-").to_string())
    t.to_csv(HERE / "summary.csv", index=False)
    i.to_csv(HERE / "best_pairs.csv", index=False)


if __name__ == "__main__":
    buf = io.StringIO()
    with redirect_stdout(buf):
        main()
    (HERE / "analysis.out").write_text(buf.getvalue())
    sys.stdout.write(buf.getvalue())
