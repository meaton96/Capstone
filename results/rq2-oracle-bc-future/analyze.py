"""
@file analyze.py
@brief rq2-oracle-bc-future (2026-10-15): does BC of the v5 switching oracle use a future job table?
       -> analysis.out, summary.csv
       --variant 4k (data scaling, 10-16): the arms retrained on 4,000 seeds (bc4k/, out/rollout4k/) -> analysis4k.out,
       summary4k.csv, plus the learning curve over 250-4,000 seeds and the 1,000- vs 4,000-seed models side by side.

1. Per slot (bc/scores_*.json): oracle-pair match, job-rule match and hold-to-end regret of the clone's pick on the
   held-out evaluation slots, per arm (mean and sd over BC seeds), and the learning curve (250 / 500 / 1000 seeds).
2. Rollouts (out/rollout), B2 window tardiness, pooled over seeds (fitting 2000-2159 + test 0-39, and per set):
   - each arm's family (per-instance mean over its BC seeds) and each BC seed vs MDD-PTECT; seed-bootstrap 95% CI,
     Wilcoxon, seeds better / worse; share of the switching oracle's gain (rq2-rebase-v5 oracle);
   - future arms vs the state arm (paired, family) and vs their own exact run (paired), per forecast level;
   - on the test seeds, the planner on the same published schedule (rq2-perturbed-v5, 900 s and 300 s re-plans);
   - training-seed level: one pooled % per trained model, Welch t and Mann-Whitney U between arms.
   Reference: rq2-oracle-bc-v5 (state BC trained on fitting seeds 2000-2159, 3 seeds; test seeds only).
"""
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
R = HERE.parent
REF = "MDD-PTECT"
ARMS = ("state", "future64", "future128")
FUTURE_ARMS = ("future64", "future128")
LEVELS = ["exact", "jit1", "jit10", "jit30", "churn10", "churn25", "redrawn"]
NAMES = {"exact": "exact schedule", "jit1": "arrivals ± 1 min", "jit10": "arrivals ± 10 min",
         "jit30": "arrivals ± 30 min", "churn10": "10% churn + 10 min", "churn25": "25% churn + 10 min",
         "redrawn": "redrawn (no information)"}
TEST = list(range(40))
FIT = list(range(2000, 2160))
RNG = np.random.default_rng(0)
VARIANT = {"bc": "bc", "rollout": "rollout", "suffix": ""}      # set by --variant


def pct(x, ref):
    return 100.0 * (x.sum() - ref.sum()) / ref.sum()


def cmp(x, y, ref, n_boot=2000):
    """Pooled % of x vs y relative to sum(ref), seed bootstrap CI, Wilcoxon on per-seed x - y, better / worse."""
    x, y, ref = np.asarray(x, float), np.asarray(y, float), np.asarray(ref, float)
    d = x - y
    est = 100.0 * d.sum() / ref.sum()
    idx = RNG.integers(0, len(d), (n_boot, len(d)))
    boot = 100.0 * d[idx].sum(1) / ref[idx].sum(1)
    nz = d[np.abs(d) > 1e-9]
    p = stats.wilcoxon(nz).pvalue if len(nz) > 0 else 1.0
    return est, np.percentile(boot, 2.5), np.percentile(boot, 97.5), int((d < -1e-9).sum()), int((d > 1e-9).sum()), p


def fmt(c):
    est, lo, hi, b, w, p = c
    return f"{est:+6.2f} [{lo:+6.2f}, {hi:+6.2f}] {b:3d}/{w:<3d} p={p:.2g}"


def per_slot(folder="bc"):
    rows = []
    for f in sorted((HERE / folder).glob("scores_*.json")):
        d = json.loads(f.read_text())
        row = {"arm": d["arm"], "bc_seed": d["bc_seed"], "n_train": d["n_train_seeds"], "epochs": d["epochs"]}
        for split in ("train", "val", "eval_test", "eval_fit", "eval_all", "eval_all_redrawn"):
            for m in ("pair_match_%", "job_rule_match_%", "mean_regret_%", "mddptect_regret_%", "share_mddptect_%"):
                row[f"{split}:{m}"] = (d.get(split) or {}).get(m, np.nan)
        row["redrawn_pick_changed_%"] = d.get("redrawn_pick_changed_%", np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def rollouts():
    """{(level, checkpoint stem): pd.Series seed -> tardiness}."""
    out = {}
    for f in sorted((HERE / "out" / VARIANT["rollout"]).glob("s*.json")):
        seed = int(f.stem[1:])
        d = json.loads(f.read_text())
        for lv, cks in d.items():
            if lv.startswith("_"):
                continue
            for ck, v in cks.items():
                out.setdefault((lv, ck[:-3]), {})[seed] = v["tard"]
    return {k: pd.Series(v) for k, v in out.items()}


def references(seeds):
    mdd, oracle = {}, {}
    for s in seeds:
        mdd[s] = json.loads((R / "rq2-rebase-v5" / "out" / "policies" / f"s{s}.json").read_text())["tard"][f"fixed:{REF}"]
        oracle[s] = float(np.load(R / "rq2-rebase-v5" / "out" / "oracle" / f"s{s}.npz")["oracle"])
    return pd.Series(mdd), pd.Series(oracle)


def planner(level, slot=""):
    out = {}
    for s in TEST:
        f = R / "rq2-perturbed-v5" / "B2" / f"{level}{slot}" / f"s{s}.json"
        if f.exists():
            out[s] = json.loads(f.read_text())["real"]
    return pd.Series(out, dtype=float)


def bc_v5_reference():
    out = {}
    for s in TEST:
        f = R / "rq2-oracle-bc-v5" / "out" / "rollout" / f"s{s}.json"
        if f.exists():
            d = json.loads(f.read_text())
            out[s] = np.mean([d[f"bc_full_s{k}.pt"]["tard"] for k in range(3)])
    return pd.Series(out, dtype=float)


def main():
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print("== rq2-oracle-bc-future: behaviour cloning of the v5 switching oracle (B2), state vs state + future job "
          "table (64 / 128 rows, 2 h)\n")
    ps = per_slot(VARIANT["bc"])
    if len(ps):
        print("-- 1. per slot, held-out evaluation slots (test 0-39 + fitting 2000-2159 = 4,800 slots; deterministic "
              "argmax pair), mean (sd) over BC seeds")
        cols = ["epochs", "train:pair_match_%", "val:pair_match_%", "eval_all:pair_match_%",
                "eval_all:job_rule_match_%", "eval_all:mean_regret_%", "eval_test:pair_match_%",
                "eval_fit:pair_match_%", "eval_all:share_mddptect_%"]
        g = ps.groupby(["n_train", "arm"])[cols]
        tab = g.mean().round(2).astype(str) + " (" + g.std().round(2).astype(str) + ")"
        tab["models"] = g.size()
        print(tab.to_string())
        print(f"   always {REF}: mean hold-to-end regret {ps['eval_all:mddptect_regret_%'].iloc[0]:.2f}% per slot; "
              f"chance pair match 16.7%")
        print("\n   the same slots with the redrawn (unrelated) future in the table (future arms):")
        cols = ["eval_all:pair_match_%", "eval_all_redrawn:pair_match_%", "eval_all:mean_regret_%",
                "eval_all_redrawn:mean_regret_%", "redrawn_pick_changed_%"]
        f = ps[ps.arm != "state"].groupby(["n_train", "arm"])[cols]
        print((f.mean().round(2).astype(str) + " (" + f.std().round(2).astype(str) + ")").to_string())
        main_ps = ps[ps.n_train == ps.n_train.max()]
        for arm in FUTURE_ARMS:
            a = main_ps[main_ps.arm == arm]["eval_all:pair_match_%"]
            b = main_ps[main_ps.arm == "state"]["eval_all:pair_match_%"]
            if len(a) > 1 and len(b) > 1:
                print(f"   pair match {arm} vs state across BC seeds: {a.mean():.2f} vs {b.mean():.2f}, Welch p "
                      f"{stats.ttest_ind(a, b, equal_var=False).pvalue:.3g}, Mann-Whitney p "
                      f"{stats.mannwhitneyu(a, b).pvalue:.3g}")
            a = main_ps[main_ps.arm == arm]["eval_all:mean_regret_%"]
            b = main_ps[main_ps.arm == "state"]["eval_all:mean_regret_%"]
            if len(a) > 1 and len(b) > 1:
                print(f"   per-slot regret {arm} vs state across BC seeds: {a.mean():.2f} vs {b.mean():.2f}, Welch p "
                      f"{stats.ttest_ind(a, b, equal_var=False).pvalue:.3g}, Mann-Whitney p "
                      f"{stats.mannwhitneyu(a, b).pvalue:.3g}")
        print()
    f160 = per_slot("bc_fit160") if not VARIANT["suffix"] else []
    if len(f160):
        print("-- 1b. early read: trained on rq2-oracle-bc-v5's data (fitting seeds 2000-2159, the eval 'fit' slots are "
              "in-sample), per slot on test seeds 0-39, mean (sd) over BC seeds")
        cols = ["epochs", "train:pair_match_%", "eval_test:pair_match_%", "eval_test:job_rule_match_%",
                "eval_test:mean_regret_%", "redrawn_pick_changed_%"]
        g = f160.groupby("arm")[cols]
        print((g.mean().round(2).astype(str) + " (" + g.std().round(2).astype(str) + ")").to_string())
        print(f"   always {REF} on the test slots: regret {f160['eval_test:mddptect_regret_%'].iloc[0]:.2f}%\n")

    ro = rollouts()
    if not ro:
        print("no rollouts yet")
        return
    seeds = sorted(set().union(*[set(v.index) for v in ro.values()]))
    mdd, oracle = references(seeds)
    cks = {arm: sorted({ck for (lv, ck) in ro if ck.startswith(f"bc_{arm}_s")}) for arm in ARMS}

    def fam(arm, lv, idx):
        lv = "exact" if arm == "state" else lv
        cols = [ro[(lv, ck)].reindex(idx) for ck in cks[arm] if (lv, ck) in ro]
        if not cols:
            return None
        df = pd.concat(cols, axis=1)
        return df.mean(axis=1, skipna=False)

    summary = []
    for split, idx in (("all 200", [s for s in seeds if s in TEST + FIT]), ("test 0-39", [s for s in seeds if s in TEST]),
                       ("fit 2000-2159", [s for s in seeds if s in FIT])):
        idx = pd.Index(idx)
        m, o = mdd.reindex(idx), oracle.reindex(idx)
        print(f"-- 2. rollouts, {split} (n = {len(idx)}): pooled % vs {REF} [95% CI] better/worse Wilcoxon p; "
              f"switching oracle {pct(o, m):+.2f}%")
        st = fam("state", "exact", idx)
        if st is not None and st.notna().all():
            c = cmp(st, m, m)
            print(f"   {'state (BC on state only)':<44} family {fmt(c)}  oracle share {100 * c[0] / pct(o, m):5.1f}%  "
                  f"[{len(cks['state'])} seeds]")
            summary.append({"split": split, "arm": "state", "level": "exact", "vs_ref": c[0], "lo": c[1], "hi": c[2],
                            "p": c[5], "oracle_share": 100 * c[0] / pct(o, m)})
        for arm in FUTURE_ARMS:
            ex = fam(arm, "exact", idx)
            for lv in LEVELS:
                f = fam(arm, lv, idx)
                if f is None or f.isna().any():
                    continue
                c = cmp(f, m, m)
                line = f"   {arm + ' ' + NAMES[lv]:<44} family {fmt(c)}  oracle share {100 * c[0] / pct(o, m):5.1f}%"
                rec = {"split": split, "arm": arm, "level": lv, "vs_ref": c[0], "lo": c[1], "hi": c[2], "p": c[5],
                       "oracle_share": 100 * c[0] / pct(o, m)}
                if st is not None and st.notna().all():
                    cs = cmp(f, st, m)
                    line += f" | vs state {fmt(cs)}"
                    rec.update({"vs_state": cs[0], "vs_state_lo": cs[1], "vs_state_hi": cs[2], "vs_state_p": cs[5]})
                if lv != "exact" and ex is not None and ex.notna().all():
                    ce = cmp(f, ex, m)
                    line += f" | vs own exact {fmt(ce)}"
                    rec.update({"vs_exact": ce[0], "vs_exact_p": ce[5]})
                if split == "test 0-39":
                    pl = planner(lv) if lv != "exact" else oracle.reindex(idx)
                    if len(pl) == len(idx):
                        line += f" | planner 900 s {pct(pl.reindex(idx), m):+.2f}"
                        rec["planner_900"] = pct(pl.reindex(idx), m)
                    pl3 = planner(lv, "-slot300")
                    if len(pl3) == len(idx):
                        line += f", 300 s {pct(pl3.reindex(idx), m):+.2f}"
                        rec["planner_300"] = pct(pl3.reindex(idx), m)
                print(line)
                summary.append(rec)
        if split == "test 0-39":
            ref = bc_v5_reference().reindex(idx)
            if ref.notna().all():
                print(f"   {'reference: rq2-oracle-bc-v5 (state, 160 seeds)':<44} family {fmt(cmp(ref, m, m))}")
        print()

    print("-- 3. training-seed level (one pooled % vs MDD-PTECT per trained model, all 200 seeds)")
    idx = pd.Index([s for s in seeds if s in TEST + FIT])
    m = mdd.reindex(idx)
    per_model = {}
    for arm in ARMS:
        for lv in (["exact"] if arm == "state" else LEVELS):
            vals = []
            for ck in cks[arm]:
                if (lv, ck) in ro and ro[(lv, ck)].reindex(idx).notna().all():
                    vals.append(pct(ro[(lv, ck)].reindex(idx), m))
            if vals:
                per_model[(arm, lv)] = vals
                print(f"   {arm:<10} {lv:<8} " + " / ".join(f"{v:+.2f}" for v in vals)
                      + f"  mean {np.mean(vals):+.2f} sd {np.std(vals, ddof=1) if len(vals) > 1 else 0:.2f}")
    base = per_model.get(("state", "exact"))
    for arm in FUTURE_ARMS:
        for lv in ("exact", "redrawn"):
            v = per_model.get((arm, lv))
            if base and v and len(base) > 1 and len(v) > 1:
                print(f"   {arm} {lv} vs state: Welch p {stats.ttest_ind(v, base, equal_var=False).pvalue:.3g}, "
                      f"Mann-Whitney p {stats.mannwhitneyu(v, base).pvalue:.3g}")
        ex, rd = per_model.get((arm, "exact")), per_model.get((arm, "redrawn"))
        if ex and rd and len(ex) == len(rd) and len(ex) > 1:
            d = np.array(ex) - np.array(rd)
            print(f"   {arm} exact - redrawn per model: " + " / ".join(f"{x:+.2f}" for x in d)
                  + f"  (paired t p {stats.ttest_rel(ex, rd).pvalue:.3g})")

    print("\n-- pairs played (share of slots, exact; all seeds)")
    for arm in ARMS:
        played = {}
        for f in sorted((HERE / "out" / VARIANT["rollout"]).glob("s*.json")):
            d = json.loads(f.read_text()).get("exact", {})
            for ck, v in d.items():
                if ck.startswith(f"bc_{arm}_s"):
                    for p in v["played"]:
                        played[p] = played.get(p, 0) + 1
        tot = sum(played.values()) or 1
        print(f"   {arm:<10} " + ", ".join(f"{p} {100 * n / tot:.0f}%" for p, n in sorted(played.items(), key=lambda x: -x[1])))
    pd.DataFrame(summary).to_csv(HERE / f"summary{VARIANT['suffix']}.csv", index=False)
    if VARIANT["suffix"]:
        scaling(mdd.reindex(idx), idx)


def scaling(m, idx):
    """Data scaling (10-16): per-slot learning curve over bc/ (250-1,000 seeds) and bc4k/ (2,000-4,000), and the
    rollouts of the 1,000- vs 4,000-seed models (one pooled % vs MDD-PTECT per model, 200 seeds)."""
    print("\n-- 4. data scaling: per-slot learning curve, all 4,800 evaluation slots, mean (sd) over BC seeds")
    ps = pd.concat([per_slot("bc"), per_slot("bc4k")])
    cols = ["eval_all:pair_match_%", "eval_all:mean_regret_%", "eval_all_redrawn:mean_regret_%",
            "redrawn_pick_changed_%"]
    g = ps.groupby(["arm", "n_train"])[cols]
    tab = g.mean().round(2).astype(str) + " (" + g.std().round(2).astype(str) + ")"
    tab["models"] = g.size()
    print(tab.to_string())
    print("\n-- 5. rollouts, 1,000 vs 4,000 training seeds (one pooled % vs MDD-PTECT per model, all 200 seeds)")
    runs = {}
    for name, sub in (("1k", "rollout"), ("4k", "rollout4k")):
        VARIANT["rollout"] = sub
        runs[name] = rollouts()
    for arm in ARMS:
        for lv in (["exact"] if arm == "state" else ["exact", "redrawn"]):
            vals = {}
            for name, ro in runs.items():
                vals[name] = [pct(ro[(lv, ck)].reindex(idx), m) for (l, ck) in sorted(ro)
                              if l == lv and ck.startswith(f"bc_{arm}_s") and ro[(lv, ck)].reindex(idx).notna().all()]
            a, b = vals["1k"], vals["4k"]
            if len(a) > 1 and len(b) > 1:
                print(f"   {arm:<10} {lv:<8} 1k mean {np.mean(a):+.2f} (sd {np.std(a, ddof=1):.2f})  4k mean "
                      f"{np.mean(b):+.2f} (sd {np.std(b, ddof=1):.2f})  Welch p "
                      f"{stats.ttest_ind(b, a, equal_var=False).pvalue:.3g}, MW p {stats.mannwhitneyu(b, a).pvalue:.3g}")
    VARIANT["rollout"] = "rollout4k"


if __name__ == "__main__":
    if "--variant" in sys.argv and sys.argv[sys.argv.index("--variant") + 1] == "4k":
        VARIANT.update(bc="bc4k", rollout="rollout4k", suffix="4k")
    buf = io.StringIO()
    with redirect_stdout(buf):
        main()
    text = buf.getvalue()
    (HERE / f"analysis{VARIANT['suffix']}.out").write_text(text)
    sys.stdout.write(text)
