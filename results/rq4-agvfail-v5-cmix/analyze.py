"""
@file analyze.py
@brief rq4-agvfail-v5-cmix (2026-10-17): the RQ4 AGV-breakdown half for the final policy (PPO trained on mixed due
       dates, train-v5-cmix pick s3), its 5-seed family and the loose pick s2, in Unity, B2 seeds 0-39, at no breakdowns
       and AGV Weibull lambda 3000 / 1500 s. Fixed pairs, the B2-trained PPO family and BC come from
       ../rq4-agvfail-v5-rules / -policy and ../eval-transfer-unity-v5 (loaded by ../rq4-agvfail-v5-rules/analyze.py).
       Per level: pooled window tardiness vs MDD-PTECT at the same level (seed-bootstrap 95% CI, Wilcoxon), the
       degradation from no breakdowns, and run-to-run checks. -> analysis.out, summary.csv
"""
import importlib.util
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
R = HERE.parent
_spec = importlib.util.spec_from_file_location("rq4", R / "rq4-agvfail-v5-rules" / "analyze.py")
rq4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rq4)
CMIX_UNITY = R / "eval-v5-cmix-unity" / "runs" / "base" / "pc"


def name(p):
    p = str(p).replace("ckpt:", "").replace("/checkpoint", "").replace("results/", "")
    if p.startswith("train-v5-cmix_s"):
        return "cmix s" + p.rsplit("_s", 1)[1]
    if p.startswith("train-v5-loose_s"):
        return "loose s" + p.rsplit("_s", 1)[1]
    return rq4.short(p)


def load_cmix():
    parts = []
    for lv in rq4.LEVELS:
        for g in ("cmixa", "cmixb"):
            for e in rq4.load_dir(HERE / "runs" / lv, lv, g, "cmix"):
                parts.append(e)
    for f in sorted(CMIX_UNITY.glob("s*/episodes.csv")):          # eval-v5-cmix-unity base = level none
        if (f.parent / "DONE").exists():
            e = pd.read_csv(f, dtype=rq4.HASH_STR)
            e["level"], e["group"], e["source"] = "none", "pc", "eval-v5-cmix-unity"
            parts.append(e)
    if not parts:
        return pd.DataFrame()
    d = pd.concat(parts, ignore_index=True)
    d["policy"] = d.policy.map(name)
    d["tard"] = d.window_tardiness / 1000.0
    return d


def main():
    rng = np.random.default_rng(0)
    base = rq4.load()
    cm = load_cmix()
    print("== rq4-agvfail-v5-cmix: Unity, B2 seeds 0-39, AGV breakdowns none / lambda 3000 / 1500 s; final policy = cmix s3")
    if cm.empty:
        print("   no cmix runs yet")
        return
    print(cm.groupby(["level", "source"]).size().unstack(fill_value=0).to_string())
    # run-to-run: cmixa's MDD-PTECT vs the rules half; cmixb's s3 / loose s2 at none vs eval-v5-cmix-unity
    print("\n-- run-to-run checks (identical seeds / compared)")
    ref = base[base.policy == rq4.REF].drop_duplicates(["level", "seed"]).set_index(["level", "seed"]).tard
    m = cm[(cm.policy == rq4.REF) & (cm.group == "cmixa")].set_index(["level", "seed"]).tard
    j = pd.concat([m.rename("a"), ref.rename("b")], axis=1).dropna()
    print(f"   cmixa MDD-PTECT vs rules half: {int(((j.a - j.b).abs() < 1e-6).sum())} / {len(j)}")
    for pol in ("cmix s3", "loose s2"):
        x = cm[(cm.policy == pol) & (cm.level == "none")]
        t = x.pivot_table(index="seed", columns="source", values="tard")
        if {"cmix", "eval-v5-cmix-unity"} <= set(t.columns):
            ok = t.dropna()
            print(f"   {pol} at none, cmixb vs eval-v5-cmix-unity: "
                  f"{int(((ok['cmix'] - ok['eval-v5-cmix-unity']).abs() < 1e-6).sum())} / {len(ok)}")
    d = pd.concat([base, cm[cm.policy != rq4.REF]], ignore_index=True)
    d = d.drop_duplicates(["level", "policy", "seed"], keep="first")
    t = d.pivot_table(index=["level", "seed"], columns="policy", values="tard")
    rows = []
    for lv in rq4.LEVELS:
        if lv not in t.index.get_level_values(0):
            continue
        x = t.loc[lv]
        cmix = [f"cmix s{k}" for k in range(5)]
        ppo = [f"PPO s{k}" for k in range(5)]
        if all(c in x for c in cmix):
            x = x.assign(**{"cmix mean of 5": x[cmix].mean(axis=1, skipna=False)})
        if all(c in x for c in ppo):
            x = x.assign(**{"B2-trained mean of 5": x[ppo].mean(axis=1, skipna=False)})
        if all(p in x for p in rq4.V5):
            x = x.assign(**{"hindsight best v5 pair": x[rq4.V5].min(axis=1, skipna=False)})
        for pol in ["cmix s3", "cmix mean of 5", "loose s2", "PPO s4", "B2-trained mean of 5", "MDD-PECT",
                    "hindsight best v5 pair"]:
            if pol in x:
                s = rq4.stats(x[pol], x[rq4.REF], rng)
                if s:
                    rows.append({"level": lv, "policy": pol, **s})
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "summary.csv", index=False)
    print("\n-- vs MDD-PTECT at the same level: pooled % [95% CI] better/worse p")
    for _, r in df.iterrows():
        print(f"   {r.level:6s} {r.policy:24s} {r['pooled_%']:+6.2f} [{r.ci_lo:+6.2f}, {r.ci_hi:+6.2f}] "
              f"{r.better:2d}/{r.worse:<2d} p={r.wilcoxon_p:.2g} n={r.n}")
    print("\n-- degradation from no breakdowns (pooled, same seeds)")
    for pol in ["cmix s3", "loose s2", "PPO s4", rq4.REF]:
        if pol not in t.columns or "none" not in t.index.get_level_values(0):
            continue
        n0 = t.loc["none"][pol]
        out = []
        for lv in ("l3000", "l1500"):
            if lv in t.index.get_level_values(0):
                y = t.loc[lv][pol]
                ok = y.notna() & n0.notna()
                if ok.sum() >= 5:
                    out.append(f"{lv} {100 * (y[ok].sum() / n0[ok].sum() - 1):+.1f}% (n {int(ok.sum())})")
        print(f"   {pol:10s} " + ", ".join(out))


if __name__ == "__main__":
    buf = io.StringIO()
    with redirect_stdout(buf):
        main()
    (HERE / "analysis.out").write_text(buf.getvalue())
    sys.stdout.write(buf.getvalue())
