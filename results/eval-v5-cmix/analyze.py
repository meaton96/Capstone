"""
@file analyze.py
@brief eval-v5-cmix (2026-10-14): do PPO policies trained at loose or mixed due dates beat the best fixed rule where
       due dates are loose, and does one mixed-c policy adapt to tight / base / loose? Twin, B2 settings, conditions
       due-tight (c 1.25-1.75), base (1.75-2.5), due-loose (2.5-3.5); test seeds 0-39 and fitting seeds 2000-2159.

@details Families (5 PPO seeds each, same stack and code): base (trained at c 1.75-2.5: train-v5-nokl s0-s2 +
train-v5-final s3-s4), loose (c 2.5-3.5: train-v5-loose), cmix (c 1.25-3.5: train-v5-cmix). Each family's pick is
made on the fitting seeds only:
  base   s4, eval-v5-final's pick (fitting seeds at B2's own c)
  loose  lowest pooled tardiness on the fitting seeds at due-loose (its training range)
  cmix   lowest mean over the 3 conditions of (pooled tardiness / MDD-PTECT's) on the fitting seeds
Rows per condition and split: every seed, the family's per-instance mean of 5, its pick, BC of the v5 oracle (test,
from rq3-gen-v5), the hindsight best v5 pair, and at due-loose on the test seeds the switching oracle
(../rq3-oracle-loose). Scores: pooled window tardiness vs MDD-PTECT and vs the split's best v5 pair (in-sample), with
seed-bootstrap 95% CIs, seeds better / worse, Wilcoxon p, and the mean job-seconds saved per instance.
Test x {pdr, ppo, bc} comes from ../rq3-gen-v5 (identical code, settings and instances). -> analysis.out, summary.csv
"""
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
R = HERE.parent
H = {"config_hash": str, "instance_hash": str}
CONDS = ["due-tight", "base", "due-loose"]
FITS = ["fit2000", "fit2040", "fit2080", "fit2120"]
V5 = ["SRT-PECT", "SRT-PTECT", "MDD-PECT", "MDD-PTECT", "ATC-PECT", "ATC-PTECT"]
REF = "MDD-PTECT"
FAM = {"base": [f"train-v5-nokl_s{k}" for k in range(3)] + [f"train-v5-final_s{k}" for k in (3, 4)],
       "loose": [f"train-v5-loose_s{k}" for k in range(5)], "cmix": [f"train-v5-cmix_s{k}" for k in range(5)]}


def name(p):
    p = str(p).replace("ckpt:", "").replace("/checkpoint", "").replace("results/", "")
    for fam, runs in FAM.items():
        if p in runs:
            return f"{fam} s{p.rsplit('_s', 1)[1]}"
    if "bc_full_s" in p:
        return "BC s" + p.split("bc_full_s")[1].replace(".pt", "")
    return p


def read(f):
    if not (f.exists() and (f.parent / "DONE").exists()):
        return None
    e = pd.read_csv(f, dtype=H)
    e["policy"] = e.policy.map(name)
    return e[["policy", "seed", "window_tardiness"]]


def table(cond, split):
    parts = []
    if split == "test":
        for g in ("pdr", "ppo", "bc"):
            parts.append(read(R / "rq3-gen-v5" / cond / g / "episodes.csv"))
        for g in ("loose", "cmix"):
            parts.append(read(HERE / "test" / cond / g / "episodes.csv"))
    else:
        for s in FITS:
            for g in ("pdr", "ppo", "loose", "cmix"):
                parts.append(read(HERE / s / cond / g / "episodes.csv"))
    parts = [p for p in parts if p is not None]
    if not parts:
        return pd.DataFrame()
    e = pd.concat(parts, ignore_index=True).drop_duplicates(["policy", "seed"])
    return e.pivot_table(index="seed", columns="policy", values="window_tardiness") / 1000.0


def stats(y, b, rng):
    ok = y.notna() & b.notna()
    yy, bb = y[ok].to_numpy(), b[ok].to_numpy()
    if len(yy) < 10:
        return None
    idx = rng.integers(0, len(yy), (5000, len(yy)))
    bs = 100 * (yy[idx].sum(1) / bb[idx].sum(1) - 1)
    d = yy - bb
    return {"n": len(yy), "pooled_%": 100 * (yy.sum() / bb.sum() - 1), "ci_lo": np.percentile(bs, 2.5),
            "ci_hi": np.percentile(bs, 97.5), "better": int((d < 0).sum()), "worse": int((d > 0).sum()),
            "p": float(wilcoxon(d).pvalue) if np.any(d) else 1.0, "saved_k_per_inst": float(-d.mean())}


def picks(fit):
    out = {"base": "s4"}
    lo = fit.get("due-loose")
    if lo is not None and len(lo) and all(f"loose s{k}" in lo for k in range(5)):
        out["loose"] = min(range(5), key=lambda k: lo[f"loose s{k}"].sum())
        out["loose"] = f"s{out['loose']}"
    if all(c in fit and len(fit[c]) for c in CONDS) and all(f"cmix s{k}" in fit[c] for c in CONDS for k in range(5)):
        score = {k: np.mean([fit[c][f"cmix s{k}"].sum() / fit[c][REF].sum() for c in CONDS]) for k in range(5)}
        out["cmix"] = f"s{min(score, key=score.get)}"
    return out


def main():
    rng = np.random.default_rng(0)
    pd.set_option("display.width", 240)
    fit = {c: table(c, "fit") for c in CONDS}
    pk = picks(fit)
    print("== eval-v5-cmix: twin, B2; PPO families trained at c 1.75-2.5 (base), 2.5-3.5 (loose), 1.25-3.5 (cmix)")
    print(f"   picks on the fitting seeds: {pk}")
    oracle = {}
    for f in (R / "rq3-oracle-loose" / "out").glob("s*.json"):
        d = json.loads(f.read_text())
        oracle[int(d["seed"])] = d["oracle"]
    rows = []
    for split in ("test", "fit"):
        for cond in CONDS:
            t = fit[cond] if split == "fit" else table(cond, "test")
            if t.empty or REF not in t:
                print(f"[{split} {cond}] no data yet")
                continue
            have = [p for p in V5 if p in t and t[p].notna().all()]
            best = min(have, key=lambda p: t[p].sum()) if len(have) == 6 else REF
            cand = {}
            for fam in FAM:
                cols = [f"{fam} s{k}" for k in range(5) if f"{fam} s{k}" in t]
                for c in cols:
                    cand[c] = t[c]
                if len(cols) == 5:
                    cand[f"{fam} mean of 5"] = t[cols].mean(axis=1, skipna=False)
                if fam in pk and f"{fam} {pk[fam]}" in t:
                    cand[f"{fam} pick ({pk[fam]})"] = t[f"{fam} {pk[fam]}"]
            for c in [c for c in t.columns if c.startswith("BC s")]:
                cand[c] = t[c]
            if len(have) == 6:
                cand["hindsight best v5 pair"] = t[V5].min(axis=1)
            cand[f"best v5 pair ({best})"] = t[best]
            if split == "test" and cond == "due-loose" and len(oracle) >= 30:
                cand["switching oracle"] = pd.Series(oracle).reindex(t.index)
            for nm, y in cand.items():
                for vs, b in ((REF, t[REF]), ("best pair", t[best])):
                    if vs == "best pair" and best == REF:
                        continue
                    s = stats(y, b, rng)
                    if s:
                        rows.append({"split": split, "cond": cond, "method": nm, "vs": vs, **s})
            print(f"[{split} {cond}] n={len(t)} best v5 pair {best}; MDD-PTECT mean {t[REF].mean():.1f} k job-s")
    if not rows:
        return
    r = pd.DataFrame(rows)
    r.to_csv(HERE / "summary.csv", index=False)
    fmt = lambda v: f"{v:.3g}" if abs(v) < 0.01 else f"{v:.2f}"   # noqa: E731
    order = ["mean of 5", "pick"]
    for split in ("test", "fit"):
        for cond in CONDS:
            x = r[(r.split == split) & (r.cond == cond) & (r.vs == REF)]
            if not len(x):
                continue
            print(f"\n-- {split} {cond}: vs MDD-PTECT (pooled %, CI, better / worse, Wilcoxon, k job-s saved per instance)")
            print(x.drop(columns=["split", "cond", "vs"]).sort_values("pooled_%").to_string(index=False, float_format=fmt))
    print("\n== overview: family mean of 5 and pick vs MDD-PTECT, pooled % [CI]; * = CI below 0 and p < 0.05")
    ov = {}
    for _, q in r[(r.vs == REF) & r.method.str.contains("mean of 5|pick|oracle|BC s0|hindsight")].iterrows():
        star = "*" if (q.ci_hi < 0 and q.p < 0.05) else " "
        ov.setdefault((q.split, q.cond), {})[q.method] = f"{q['pooled_%']:+.1f} [{q.ci_lo:+.1f},{q.ci_hi:+.1f}]{star}"
    print(pd.DataFrame.from_dict(ov, orient="index").fillna("-").to_string())


if __name__ == "__main__":
    buf = io.StringIO()
    with redirect_stdout(buf):
        main()
    (HERE / "analysis.out").write_text(buf.getvalue())
    sys.stdout.write(buf.getvalue())
