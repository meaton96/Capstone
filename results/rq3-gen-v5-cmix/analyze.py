"""
@file analyze.py
@brief rq3-gen-v5-cmix (2026-10-17): the RQ3 conditions sweep (11 twin conditions, B2 test seeds 0-39) for the final
       policy (PPO trained on mixed due dates, train-v5-cmix s3) and its family, next to the loose-trained family, the
       B2-trained family and BC (../rq3-gen-v5), all against each condition's best v5 pair (in-sample on these seeds)
       and against MDD-PTECT. base / due-tight / due-loose come from ../eval-v5-cmix/test, the other 8 from here.
       Pooled window tardiness, seed-bootstrap 95% CIs, Wilcoxon. -> analysis.out, summary.csv
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
_spec = importlib.util.spec_from_file_location("rq3an", R / "rq3-gen-v5" / "analyze.py")
rq3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rq3)
H = {"config_hash": str, "instance_hash": str}
SEEDS = list(range(40))
PICKS = {"cmix": "s3", "loose": "s2", "PPO": "s4"}       # picks on the fitting seeds (eval-v5-cmix, eval-v5-final)
EVAL_CMIX = {"base", "due-tight", "due-loose"}


def family_runs(cond, group):
    """{'<group> sK': {seed: tard}} and the MDD-PTECT check run, from here or eval-v5-cmix/test."""
    root = (R / "eval-v5-cmix" / "test") if cond in EVAL_CMIX else HERE
    f = root / cond / group / "episodes.csv"
    if not (f.exists() and (f.parent / "DONE").exists()):
        return {}, {}
    e = pd.read_csv(f, dtype=H)
    e = e[e.seed.isin(SEEDS)]
    out, ref = {}, {}
    for pol, d in e.groupby("policy"):
        v = {int(s): t / 1000.0 for s, t in zip(d.seed, d.window_tardiness)}
        p = str(pol)
        if p == rq3.REF:
            ref = v
        else:
            out[f"{group} s" + p.replace("/checkpoint", "").rsplit("_s", 1)[1].split("/")[0].replace(".pt", "")] = v
    return out, ref


def main():
    rng = np.random.default_rng(0)
    pd.set_option("display.width", 240)
    rows, checks = [], []
    for cond in rq3.CONDS:
        pairs, pols, mism = rq3.load(cond)
        if not pairs:
            print(f"{cond}: rq3-gen-v5 pdr group missing")
            continue
        best = min((p for p in rq3.V5 if p in pairs), key=lambda p: sum(pairs[p][s] for s in SEEDS))
        cand = {}
        for group in ("cmix", "loose"):
            fam, ref = family_runs(cond, group)
            if ref:
                checks.append((cond, group, sum(abs(ref[s] - pairs[rq3.REF][s]) > 1e-6 for s in ref if s in pairs[rq3.REF])))
            if len(fam) == 5 and all(all(s in fam[k] for s in SEEDS) for k in fam):
                cand[f"{group} pick ({PICKS[group]})"] = fam[f"{group} {PICKS[group]}"]
                cand[f"{group} mean of 5"] = {s: float(np.mean([fam[k][s] for k in fam])) for s in SEEDS}
        if "PPO s4" in pols:
            cand["B2-trained pick (s4)"] = pols["PPO s4"]
        if "PPO mean of 5 seeds" in pols:
            cand["B2-trained mean of 5"] = pols["PPO mean of 5 seeds"]
        if "BC mean of 3 seeds" in pols:
            cand["BC mean of 3"] = pols["BC mean of 3 seeds"]
        cand["hindsight best v5 pair"] = {s: min(pairs[p][s] for p in rq3.V5 if p in pairs) for s in SEEDS}
        for name, v in cand.items():
            for vs, b in (("best pair", pairs[best]), ("MDD-PTECT", pairs[rq3.REF])):
                c = rq3.compare(v, b, rng)
                if c:
                    rows.append({"cond": cond, "best_pair": best, "method": name, "vs": vs, **c})
        checks.append((cond, "rq3-gen-v5 groups", mism))
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "summary.csv", index=False)
    print("== rq3-gen-v5-cmix: 11 twin conditions, B2 test seeds 0-39; final policy = cmix pick s3 (trained c 1.25-3.5)")
    bad = ", ".join(f"{c}/{g} {m}" for c, g, m in checks if m)
    print(f"   run-to-run checks (MDD-PTECT vs rq3-gen-v5's pdr group, {len(checks)} groups): "
          + (bad or "all identical"))
    if not all(m == 0 for *_, m in checks):
        print("   WARNING: run-to-run mismatches above")
    for vs in ("best pair", "MDD-PTECT"):
        d = df[df.vs == vs]
        if d.empty:
            continue
        print(f"\n-- vs {'each condition' + chr(39) + 's best v5 pair' if vs == 'best pair' else 'MDD-PTECT'}: "
              "pooled % [95% CI] (seeds better of 40), * = Wilcoxon p < 0.05")
        d = d.assign(cell=[f"{r['pooled_%']:+.1f} [{r.ci_lo:+.1f}, {r.ci_hi:+.1f}] ({r.better}){'*' if r.wilcoxon_p < 0.05 else ''}"
                           for _, r in d.iterrows()])
        order = [m for m in ["cmix pick (s3)", "cmix mean of 5", "loose pick (s2)", "loose mean of 5",
                             "B2-trained pick (s4)", "B2-trained mean of 5", "BC mean of 3", "hindsight best v5 pair"]
                 if m in set(d.method)]
        t = d.pivot(index="cond", columns="method", values="cell").reindex(index=[c for c in rq3.CONDS if c in set(d.cond)],
                                                                          columns=order)
        if vs == "best pair":
            t.insert(0, "best pair", d.groupby("cond").best_pair.first())
        print(t.to_string())


if __name__ == "__main__":
    buf = io.StringIO()
    with redirect_stdout(buf):
        main()
    (HERE / "analysis.out").write_text(buf.getvalue())
    sys.stdout.write(buf.getvalue())
