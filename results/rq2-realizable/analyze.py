"""
@file analyze.py
@brief rq2-realizable: realizable vs hindsight headroom. Per setting (B2, S6): each policy family's best member is
       chosen on the training seeds (2000-2159) by mean tardiness and scored on the test seeds (0-39), next to the
       hindsight references (each test instance's own best fixed pair; the switching oracle) and the cloned oracle
       policies (dev-oracle-bc for B2, rq2-oracle-steady for S6) where available.
@par Usage
@code{.sh}
.venv/bin/python results/rq2-realizable/analyze.py
@endcode
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
RES = HERE.parent
TEST = list(range(40))


def load(setting):
    rows = {}
    for f in sorted((HERE / setting).glob("s*.json")):
        r = json.loads(f.read_text())
        rows[r["seed"]] = r["tard"]
    return pd.DataFrame(rows).T.sort_index()


def references(setting, seeds):
    """Hindsight oracle and cloned-policy tardiness per test seed, where they exist."""
    out = {}
    if setting == "B2":
        o = {}
        for s in seeds:
            f = RES / "rq2-twin-fleet" / "tasks" / f"B2_s{s}.json"
            if f.exists():
                o[s] = json.loads(f.read_text())["oracle"]
        out["oracle (hindsight)"] = pd.Series(o)
        ep = RES / "dev-oracle-bc" / "rollout" / "episodes.csv"
        if ep.exists():
            e = pd.read_csv(ep)
            e = e[e.policy.str.contains("bc_full")]
            for pol, g in e.groupby("policy"):
                out[f"cloned oracle {pol.split('/')[-1]}"] = g.set_index("seed").window_tardiness / 1000
    else:
        o = {}
        for s in seeds:
            f = RES / "rq2-oracle-steady" / "data" / f"s{s}.npz"
            if f.exists():
                o[s] = float(np.load(f)["oracle"])
        if o:
            out["oracle (hindsight)"] = pd.Series(o)
        roll = {}
        for f in sorted((RES / "rq2-oracle-steady" / "rollout").glob("s*.json")):
            r = json.loads(f.read_text())
            for ck, v in r.items():
                roll.setdefault(Path(ck).stem, {})[int(f.stem[1:])] = v["tard"]
        for k, v in roll.items():
            out[f"cloned oracle {k}"] = pd.Series(v)
    return out


def main():
    pd.set_option("display.width", 220)
    for setting in ("B2", "S6"):
        d = load(setting)
        if d.empty:
            print(f"== {setting}: no results yet\n")
            continue
        train = d[d.index >= 2000]
        test = d[d.index.isin(TEST)]
        print(f"== {setting}: {len(train)} training seeds, {len(test)} test seeds, {d.shape[1]} policies")
        if len(train) == 0 or len(test) == 0:
            continue
        fam = {"fixed": [c for c in d if c.startswith("fixed:")], "regime map": [c for c in d if c.startswith("map:")],
               "threshold": [c for c in d if c.startswith("thr:")]}
        fam["any of the 375"] = list(d.columns)
        best_fixed = train[fam["fixed"]].mean().idxmin()
        base = test[best_fixed]
        mdd = test["fixed:MDD-TECT"]
        rows = []

        def row(name, chosen, p):
            p = p.reindex(test.index).dropna()
            b, m = base.loc[p.index], mdd.loc[p.index]
            g = 100 * (p - b) / b
            rows.append({"policy": name, "chosen": chosen, "seeds": len(p),
                         "vs_best_fixed_%": 100 * (p.mean() / b.mean() - 1), "vs_MDD-TECT_%": 100 * (p.mean() / m.mean() - 1),
                         "median_vs_best_fixed_%": g.median(), "better_than_best_fixed": int((g < -1e-3).sum()),
                         "worse": int((g > 1e-3).sum()),
                         "wilcoxon_p": wilcoxon(p, b).pvalue if (p - b).abs().max() > 1e-9 else 1.0})

        row("MDD-TECT", "fixed:MDD-TECT", mdd)
        row("best fixed pair (chosen on training seeds)", best_fixed, base)
        for f, cols in fam.items():
            if f == "fixed":
                continue
            ch = train[cols].mean().idxmin()
            row(f"best {f} (chosen on training seeds)", ch, test[ch])
        for f in ("regime map", "threshold"):
            ch = test[fam[f]].mean().idxmin()
            row(f"best {f} chosen on the TEST seeds (optimistic)", ch, test[ch])
        row("each instance's own best fixed pair (hindsight)", "per instance", test[fam["fixed"]].min(axis=1))
        for name, s in references(setting, test.index).items():
            row(name, "", s)
        t = pd.DataFrame(rows).set_index("policy")
        print(t.round(3).to_string())
        if "oracle (hindsight)" in t.index and t.loc["oracle (hindsight)", "vs_best_fixed_%"] < 0:
            o = t.loc["oracle (hindsight)", "vs_best_fixed_%"]
            print("  share of the oracle's gain over the best fixed pair: " + "; ".join(
                f"{n}: {t.loc[n, 'vs_best_fixed_%'] / o:.0%}" for n in t.index if n.startswith(("best", "cloned", "each"))))
        # Which regime maps / thresholds win on training seeds: top 5 per family.
        for f in ("regime map", "threshold"):
            top = train[fam[f]].mean().sort_values().head(5)
            print(f"  top {f}s on training seeds (mean tardiness; best fixed {train[best_fixed].mean():.1f}): "
                  + "; ".join(f"{k.split(':', 1)[1]} {v:.1f}" for k, v in top.items()))
        print()


if __name__ == "__main__":
    main()
