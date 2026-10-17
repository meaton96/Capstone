"""
@file analyze.py
@brief rq3-gen-v5-unity (2026-10-14): the RQ3 Unity subset next to the twin sweep (../rq3-gen-v5) on the same B2
       seeds 0-19. Per condition (load-1.2, due-tight, lay-g; plus base from ../eval-transfer-unity-v5 and the
       no-breakdown rows of ../rq4-agvfail-v5-*): the 6 v5 pairs' ranking in Unity and in the twin (Kendall tau, same
       best pair or not), and the final PPO policy vs the condition's best v5 pair and vs MDD-PTECT in each simulator
       (pooled window tardiness, seed-bootstrap 95% CI, seeds better / worse, Wilcoxon).
@par Usage
@code{.sh}
.venv/bin/python results/rq3-gen-v5-unity/analyze.py     # -> analysis.out
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
HASH_STR = {"config_hash": str, "instance_hash": str}
V5 = ["SRT-PECT", "SRT-PTECT", "MDD-PECT", "MDD-PTECT", "ATC-PECT", "ATC-PTECT"]
REF = "MDD-PTECT"
SEEDS = list(range(20))


def short(p):
    p = str(p).replace("ckpt:", "").replace("/checkpoint", "").replace("results/", "")
    if "_s" in p and ("train-v5-nokl" in p or "train-v5-final" in p):
        return "PPO s" + p.rsplit("_s", 1)[1]
    return p


def table(files):
    fr = []
    for f in files:
        if (f.parent / "DONE").exists():
            fr.append(pd.read_csv(f, dtype=HASH_STR))
    if not fr:
        return pd.DataFrame()
    e = pd.concat(fr, ignore_index=True)
    e["policy"] = e.policy.map(short)
    e = e[e.seed.isin(SEEDS)].drop_duplicates(["policy", "seed"])
    return e.pivot_table(index="seed", columns="policy", values="window_tardiness") / 1000.0


def cmp(y, b, rng):
    ok = y.notna() & b.notna()
    yy, bb = y[ok].to_numpy(), b[ok].to_numpy()
    if len(yy) < 5:
        return f"n={len(yy)}"
    idx = rng.integers(0, len(yy), (10000, len(yy)))
    bs = 100 * (yy[idx].sum(1) / bb[idx].sum(1) - 1)
    d = yy - bb
    p = float(wilcoxon(d).pvalue) if np.any(d) else 1.0
    return (f"{100 * (yy.sum() / bb.sum() - 1):+6.2f} [{np.percentile(bs, 2.5):+6.2f}, {np.percentile(bs, 97.5):+6.2f}] "
            f"{int((d < 0).sum()):2d}/{int((d > 0).sum()):<2d} p={p:.2g} n={len(yy)}")


def main():
    rng = np.random.default_rng(0)
    sel = (R / "eval-v5-final" / "selected.txt").read_text().split()[0]
    final = f"PPO {sel}"
    print(f"== rq3-gen-v5-unity: Unity vs twin on B2 seeds 0-19, v5 pairs + the final policy ({final}); pooled window "
          "tardiness, < 0 better")
    conds = {"base": ([*(R / "eval-transfer-unity-v5" / "runs").glob("*/s*/episodes.csv"),
                       *(R / "rq4-agvfail-v5-rules" / "runs" / "none").glob("*/s*/episodes.csv"),
                       *(R / "rq4-agvfail-v5-policy" / "runs" / "none").glob("*/s*/episodes.csv")],
                      [R / "rq3-gen-v5" / "base" / g / "episodes.csv" for g in ("pdr", "ppo")])}
    for c in ("load-1.2", "due-tight", "lay-g"):
        conds[c] = ([*(HERE / "runs" / c).glob("*/s*/episodes.csv")],
                    [R / "rq3-gen-v5" / c / g / "episodes.csv" for g in ("pdr", "ppo")])
    for c, (uf, tf) in conds.items():
        u, t = table(uf), table(tf)
        print(f"\n-- {c}: Unity {len(u)} seeds, twin {len(t)} seeds")
        have = [p for p in V5 if p in u and p in t and u[p].notna().sum() >= 15]
        if len(have) < len(V5) or final not in u:
            print(f"   incomplete (pairs {have}, final in Unity: {final in u})")
            continue
        s = [k for k in SEEDS if k in u.index and k in t.index and u.loc[k, have + [final]].notna().all()]
        u, t = u.loc[s], t.loc[s]
        tu, tt = u[V5].sum(), t[V5].sum()
        bu, bt = tu.idxmin(), tt.idxmin()
        k = kendalltau(tu[V5], tt[V5])
        print(f"   best v5 pair: Unity {bu}, twin {bt}; pair ranking Kendall tau {k.statistic:+.2f} (p {k.pvalue:.2g}); "
              f"MDD-PTECT Unity vs twin {cmp(u[REF], t[REF], rng)}")
        print("   pairs vs MDD-PTECT (Unity | twin): " + "  ".join(
            f"{p} {100 * (tu[p] / tu[REF] - 1):+.1f}|{100 * (tt[p] / tt[REF] - 1):+.1f}" for p in V5 if p != REF))
        print(f"   {final} vs best pair:  Unity {cmp(u[final], u[bu], rng)} | twin {cmp(t[final], t[bt], rng)}")
        print(f"   {final} vs MDD-PTECT:  Unity {cmp(u[final], u[REF], rng)} | twin {cmp(t[final], t[REF], rng)}")
        hu, ht = u[V5].min(axis=1), t[V5].min(axis=1)
        print(f"   hindsight best v5 vs best pair: Unity {cmp(hu, u[bu], rng)} | twin {cmp(ht, t[bt], rng)}")


if __name__ == "__main__":
    buf = io.StringIO()
    with redirect_stdout(buf):
        main()
    (HERE / "analysis.out").write_text(buf.getvalue())
    sys.stdout.write(buf.getvalue())
