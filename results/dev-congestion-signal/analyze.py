"""
@file analyze.py
@brief dev-congestion-signal: do the congestion features predict "ATC vs MDD-TECT from here on" beyond obs v3?

Per slot (rows/s*.csv from run.py): target d = tardiness(ATC-x held from slot k) - tardiness(MDD-TECT held from slot
k), in % (< 0: ATC better). Feature sets:
  - obs:  obs v3 global scalars + event flags (what the policy's MLP sees directly; constant columns dropped)
  - cong: the candidate congestion features (c_*)
  - both: obs + cong
Models (classification of d < 0 and regression of d) are scored out of fold with folds by seed (no seed in both train
and test), repeated over 10 random seed partitions; the gain of "both" over "obs" is paired per partition.

@par Usage
@code{.sh}
OMP_NUM_THREADS=4 .venv/bin/python results/dev-congestion-signal/analyze.py > results/dev-congestion-signal/analysis.out
@endcode
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
REPEATS, FOLDS = 10, 5

OBS_NAMES = {0: "time", 1: "wip", 2: "share needs routing", 3: "share waiting pickup", 4: "share in transit",
             5: "share queued", 6: "share processing", 7: "machines busy", 8: "machines down", 9: "AGVs busy",
             10: "mean machine load", 11: "job rows overflow", 12: "machine count", 13: "AGVs on duty / machine",
             14: "deferred jobs", 15: "options", 16: "share late", 17: "share behind"}


def oof(model_fn, X, y, groups, rng, kind):
    seeds = np.unique(groups)
    perm = rng.permutation(seeds)
    fold = {s: i % FOLDS for i, s in enumerate(perm)}
    f = np.array([fold[g] for g in groups])
    pred = np.zeros(len(y))
    for k in range(FOLDS):
        tr, te = f != k, f == k
        m = model_fn()
        m.fit(X[tr], y[tr])
        pred[te] = m.predict_proba(X[te])[:, 1] if kind == "cls" else m.predict(X[te])
    return pred


def models(kind):
    if kind == "cls":
        return {"logistic": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                                  LogisticRegression(C=0.5, max_iter=3000)),
                "boosting": lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                                                   min_samples_leaf=20, random_state=0)}
    return {"ridge": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=3.0)),
            "boosting": lambda: HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200,
                                                              min_samples_leaf=20, random_state=0)}


def compare(d, obs, cong, target, label):
    y_reg = d[target].values
    y_cls = (y_reg < 0).astype(int)
    groups = d.seed.values
    sets = {"obs": obs, "cong": cong, "both": obs + cong}
    print(f"\n== {label}: target {target} ({len(d)} slots, {d.seed.nunique()} seeds; ATC better in "
          f"{y_cls.mean():.0%}; median d {np.median(y_reg):+.1f}%)")
    for kind, y, metric in (("cls", y_cls, "AUC"), ("reg", y_reg, "Spearman(pred, d)")):
        for mname, fn in models(kind).items():
            scores = {s: [] for s in sets}
            for r in range(REPEATS):
                for s, cols in sets.items():
                    p = oof(fn, d[cols].values.astype(float), y, groups, np.random.default_rng(r), kind)
                    scores[s].append(roc_auc_score(y, p) if kind == "cls" else spearmanr(p, y)[0])
            sc = {s: np.array(v) for s, v in scores.items()}
            gain = sc["both"] - sc["obs"]
            print(f"  {metric:18s} {mname:9s} obs {sc['obs'].mean():.3f}  cong {sc['cong'].mean():.3f}  "
                  f"both {sc['both'].mean():.3f}   both - obs {gain.mean():+.3f} "
                  f"(range over {REPEATS} partitions {gain.min():+.3f} .. {gain.max():+.3f})")


def main():
    d = pd.concat([pd.read_csv(p) for p in sorted((HERE / "rows").glob("s*.csv"))], ignore_index=True)
    dev_files = sorted((HERE / "dev").glob("s*.csv"))
    if dev_files:
        d = d.merge(pd.concat([pd.read_csv(p) for p in dev_files]), on=["seed", "slot"], how="left")
    print(f"{d.seed.nunique()} seeds, {len(d)} slots; oracle replay max |diff| {d.oracle_replay_diff.max():.2e}")
    obs = [c for c in d.columns if (c[0] in "sf" and c[1:].isdigit()) and d[c].nunique() > 1]
    cong = [c for c in d.columns if c.startswith("c_") and d[c].nunique() > 1]
    print("obs columns used:", ", ".join(f"{c} ({OBS_NAMES.get(int(c[1:]), 'flag') if c[0] == 's' else 'flag ' + c[1:]})"
                                         for c in obs))

    print("\n== how much of the congestion features obs v3 already carries: best |Spearman| with any obs column")
    for c in cong:
        r = {o: abs(spearmanr(d[c], d[o], nan_policy="omit")[0]) for o in obs}
        o = max(r, key=r.get)
        # the job count waiting for an AGV is recoverable as WIP x share: check that product too
        print(f"  {c:28s} best {o} ({OBS_NAMES.get(int(o[1:]), 'flag') if o[0] == 's' else 'flag'}) {r[o]:.2f}")

    print("\n== univariate Spearman with d (ATC-TECT tail minus MDD-TECT tail), pooled slots")
    for c in cong + obs:
        rho = spearmanr(d[c], d["d_ATC-TECT"], nan_policy="omit")[0]
        rho2 = spearmanr(d[c], d["d_ATC-ECT"], nan_policy="omit")[0]
        if c in cong or abs(rho) > 0.15:
            print(f"  {c:28s} ATC-TECT {rho:+.2f}   ATC-ECT {rho2:+.2f}")

    for target in ("d_ATC-TECT", "d_ATC-ECT"):
        compare(d, obs, cong, target, "all slots")
    compare(d[d.slot < 16], obs, cong, "d_ATC-TECT", "first 4 h (slots 0-15; later tails are short)")
    for L in (1, 4):
        if f"dev{L}" in d:
            dd = d.dropna(subset=[f"dev{L}"])
            dd = dd[dd[f"dev{L}"].abs() > 1e-9]          # slots where the swap changes nothing carry no label
            print(f"\n(dev{L}: {len(dd)} of {len(d.dropna(subset=[f'dev{L}']))} slots where the swap changes tardiness)")
            for c in cong:
                print(f"  {c:28s} Spearman with dev{L} {spearmanr(dd[c], dd[f'dev{L}'], nan_policy='omit')[0]:+.2f}")
            compare(dd, obs, cong, f"dev{L}", f"local swap of {L} slot(s) in the oracle schedule")
    d.to_csv(HERE / "slots.csv", index=False, float_format="%.5g")


if __name__ == "__main__":
    main()
