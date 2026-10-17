"""
@file analyze.py
@brief rq4-agvfail-v5 (2026-10-14): the RQ4 rerun on the v5 rules, Unity, B2 test seeds 0-39, AGV breakdowns.
       Reads both halves when present:
       - rq4-agvfail-v5-rules: the 6 v5 pairs + MDD-TECT at l3000 / l1500 (groups pa, pb) and the 4 pairs
         eval-transfer-unity-v5 did not run at level none (group pn);
       - level none otherwise from ../eval-transfer-unity-v5 (same code, player and seeds);
       - rq4-agvfail-v5-policy (if run): PPO s0-s4 and BC s0-s2 at l3000 / l1500, PPO s3 / s4 at none.
       Per level: every policy's pooled window tardiness vs MDD-PTECT at the same level (seed-bootstrap 95% CI, seeds
       better / worse, paired Wilcoxon), the pair ranking and its Kendall tau against no breakdowns, the per-instance
       hindsight best of the 6 v5 pairs, and each policy's degradation from no breakdowns. Run-to-run checks: the
       repeated MDD-PTECT runs must agree.
@par Usage
@code{.sh}
.venv/bin/python results/rq4-agvfail-v5-rules/analyze.py     # -> results/rq4-agvfail-v5-rules/analysis.out
@endcode
"""
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, wilcoxon

HERE = Path(__file__).resolve().parent
R = HERE.parent
POLICY = R / "rq4-agvfail-v5-policy"
NONE = R / "eval-transfer-unity-v5" / "runs"
SELECTED = R / "eval-v5-final" / "selected.txt"
HASH_STR = {"config_hash": str, "instance_hash": str}   # a hex hash like 933e1442... crashes pandas as a float
V5 = ["SRT-PECT", "SRT-PTECT", "MDD-PECT", "MDD-PTECT", "ATC-PECT", "ATC-PTECT"]
PAIRS = V5 + ["MDD-TECT"]
REF = "MDD-PTECT"
LEVELS = ["none", "l3000", "l1500"]


def short(p):
    p = str(p).replace("ckpt:", "").replace("/checkpoint", "").replace("results/", "")
    if p.startswith("train-v5-nokl_s") or p.startswith("train-v5-final_s"):
        return "PPO s" + p.rsplit("_s", 1)[1]
    if "bc_full_s" in p:
        return "BC s" + p.split("bc_full_s")[1].replace(".pt", "")
    return p


def load_dir(root, level, group, source):
    rows = []
    for f in sorted((root / group).glob("s*/episodes.csv")):
        if not (f.parent / "DONE").exists():
            continue
        e = pd.read_csv(f, dtype=HASH_STR)
        e["policy"] = e.policy.map(short)
        e["level"], e["group"], e["source"] = level, group, source
        rows.append(e)
    return rows


def load():
    parts = []
    for g in ("ppo", "bc"):
        parts += load_dir(NONE, "none", g, "eval-transfer-unity-v5")
    parts += load_dir(HERE / "runs" / "none", "none", "pn", "rules")
    for lv in ("l3000", "l1500"):
        for g in ("pa", "pb"):
            parts += load_dir(HERE / "runs" / lv, lv, g, "rules")
    for lv in LEVELS:
        for g in ("ppoa", "ppob", "bc"):
            parts += load_dir(POLICY / "runs" / lv, lv, g, "policy")
    d = pd.concat(parts, ignore_index=True)
    d["tard"] = d.window_tardiness / 1000.0
    return d


def stats(y, b, rng):
    ok = y.notna() & b.notna()
    yy, bb = y[ok].to_numpy(), b[ok].to_numpy()
    if len(yy) < 5:
        return None
    idx = rng.integers(0, len(yy), (5000, len(yy)))
    boot = 100 * (yy[idx].sum(1) / bb[idx].sum(1) - 1)
    d = yy - bb
    return {"n": len(yy), "pooled_%": 100 * (yy.sum() / bb.sum() - 1), "ci_lo": np.percentile(boot, 2.5),
            "ci_hi": np.percentile(boot, 97.5), "better": int((d < 0).sum()), "worse": int((d > 0).sum()),
            "wilcoxon_p": float(wilcoxon(d).pvalue) if np.any(d) else 1.0, "mean_tard": yy.mean()}


def run_to_run(d):
    """The repeated MDD-PTECT runs (pa vs pb vs ppoa at a breakdown level; pn vs eval-transfer-unity-v5 at none)."""
    m = d[d.policy == REF]
    out = []
    for lv, x in m.groupby("level"):
        t = x.pivot_table(index="seed", columns="group", values="tard")
        cols = list(t.columns)
        for c in cols[1:]:
            ok = t[cols[0]].notna() & t[c].notna()
            diff = (t[c][ok] - t[cols[0]][ok]).abs()
            out.append({"level": lv, "a": cols[0], "b": c, "n": int(ok.sum()), "max_abs_diff_1000js": diff.max(),
                        "identical": int((diff < 1e-6).sum())})
    return pd.DataFrame(out)


def main():
    rng = np.random.default_rng(0)
    d = load()
    print("== rq4-agvfail-v5: Unity, B2 test seeds 0-39, window tardiness (1000 job-s), AGV breakdowns "
          "(Weibull lambda 3000 / 1500 s); vs MDD-PTECT at the same level, < 0 better")
    print("   episodes per level x source:")
    print(d.groupby(["level", "source"]).size().unstack(fill_value=0).to_string())
    rr = run_to_run(d)
    print("\n== run-to-run checks (repeated MDD-PTECT; the player is deterministic per seed, so expect identical)")
    print(rr.to_string(index=False) if len(rr) else "   none yet")
    # one value per (level, policy, seed): first occurrence (pa before pb / ppoa; eval-transfer-unity-v5 before pn)
    order = {"eval-transfer-unity-v5": 0, "rules": 1, "policy": 2}
    d = d.assign(o=d.source.map(order)).sort_values(["o", "group"])
    d = d.drop_duplicates(["level", "policy", "seed"], keep="first")
    t = d.pivot_table(index=["level", "seed"], columns="policy", values="tard")
    ppo = [c for c in t.columns if c.startswith("PPO s")]
    bc = [c for c in t.columns if c.startswith("BC s")]
    sel = SELECTED.read_text().split()[0] if SELECTED.exists() else None
    rows, ranks, nrank = [], {}, {}
    for lv in LEVELS:
        if lv not in t.index.get_level_values(0):
            continue
        x = t.loc[lv]
        if REF not in x:
            continue
        b = x[REF]
        have = [p for p in V5 if p in x and x[p].notna().sum() >= 35]
        if len(have) == len(V5):
            x = x.assign(**{"hindsight best v5 pair": x[V5].min(axis=1, skipna=False)})
        if len(ppo) == 5:
            x = x.assign(**{"PPO mean of 5 seeds": x[ppo].mean(axis=1, skipna=False)})
        if len(bc) == 3:
            x = x.assign(**{"BC mean of 3 seeds": x[bc].mean(axis=1, skipna=False)})
        pr = [p for p in PAIRS if p in x and x[p].notna().sum() >= 35]
        common = x[pr].dropna()                      # rank on the seeds every pair has, so totals are comparable
        ranks[lv] = {p: common[p].sum() for p in pr}
        nrank[lv] = len(common)
        for pol in [c for c in x.columns]:
            s = stats(x[pol], b, rng)
            if s is None or pol == REF:
                continue
            label = pol + (" (selected final)" if sel and pol == f"PPO {sel}" else "")
            rows.append({"level": lv, "policy": label, **s})
    r = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print("\n== every policy vs MDD-PTECT at the same level (pooled %, seed bootstrap 95% CI, paired Wilcoxon)")
    if sel:
        print(f"   final policy (eval-v5-final, chosen on fitting seeds): PPO {sel}")
    for lv in LEVELS:
        x = r[r.level == lv]
        if len(x):
            print(f"\n-- {lv}")
            print(x.drop(columns="level").to_string(index=False, float_format=lambda v: f"{v:.3g}" if abs(v) < 0.01 else f"{v:.2f}"))
    print("\n== pair ranking per level (pooled tardiness, best first) and Kendall tau vs no breakdowns")
    for lv, tot in ranks.items():
        order_ = sorted(tot, key=tot.get)
        line = "  ".join(f"{p} {tot[p] / tot[order_[0]] * 100 - 100:+.1f}%" for p in order_)
        tau = ""
        if lv != "none" and "none" in ranks:
            common = [p for p in order_ if p in ranks["none"]]
            if len(common) >= 3:
                k = kendalltau([ranks["none"][p] for p in common], [tot[p] for p in common])
                tau = f"   tau vs none {k.statistic:+.2f} (p {k.pvalue:.2g}, {len(common)} pairs)"
        print(f"  {lv:6s} ({nrank[lv]} seeds) best {order_[0]}: {line}{tau}")
    print("\n== degradation from no breakdowns (pooled tardiness on the same seeds, % increase)")
    deg = []
    if "none" in t.index.get_level_values(0):
        for pol in t.columns:
            for lv in ("l3000", "l1500"):
                if lv in t.index.get_level_values(0):
                    a, c = t.loc["none"][pol], t.loc[lv][pol]
                    ok = a.notna() & c.notna()
                    if ok.sum() >= 35:
                        deg.append({"policy": pol, "level": lv, "increase_%": 100 * (c[ok].sum() / a[ok].sum() - 1)})
    if deg:
        print(pd.DataFrame(deg).pivot(index="policy", columns="level", values="increase_%").round(2).to_string())


if __name__ == "__main__":
    buf = io.StringIO()
    with redirect_stdout(buf):
        main()
    (HERE / "analysis.out").write_text(buf.getvalue())
    sys.stdout.write(buf.getvalue())
