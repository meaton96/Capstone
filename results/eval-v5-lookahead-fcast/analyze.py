"""
@file analyze.py
@brief eval-v5-lookahead-fcast (2026-10-15): how much of the look-ahead PPO's gain survives a perturbed published
       schedule, next to the twin planner at the same forecast error. Per level: the 5-seed look-ahead family vs
       MDD-PTECT, vs the state-only family (eval-v5-final) and vs its own exact-schedule run (eval-v5-lookahead),
       pooled over all 200 held-out seeds and per seed set; on the test seeds also the planner (rq2-perturbed-v5,
       re-planning every 900 s, and 300 s where it ran). Pooled window tardiness, seed-bootstrap 95% CIs, Wilcoxon.
       -> analysis.out, summary.csv
"""
import importlib.util
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
R = HERE.parent
H = {"config_hash": str, "instance_hash": str}
REF = "MDD-PTECT"
LEVELS = ["exact", "jit1", "jit10", "jit30", "churn10", "churn25", "redrawn"]
NAMES = {"exact": "exact schedule", "jit1": "arrivals ± 1 min", "jit10": "arrivals ± 10 min",
         "jit30": "arrivals ± 30 min", "churn10": "10% churn + 10 min", "churn25": "25% churn + 10 min",
         "redrawn": "redrawn (no information)"}
PICK = "s3"              # the look-ahead seed with the lowest tardiness on the fitting seeds (eval-v5-lookahead)

_spec = importlib.util.spec_from_file_location("la_an", R / "eval-v5-lookahead" / "analyze.py")
la_an = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(la_an)
cmp = la_an.cmp


def level_table(level, split):
    """seed x look-ahead seed (k = 0..4) tardiness (k job-s) at a perturbed level; None for missing tasks."""
    suffix = "__test" if split == "test" else ""
    cols = {}
    for k in range(5):
        d = HERE / level / f"train-v5-lookahead_s{k}__checkpoint{suffix}"
        if (d / "DONE").exists():
            e = pd.read_csv(d / "episodes.csv", dtype=H)
            cols[f"s{k}"] = e.set_index("seed").window_tardiness / 1000.0
    return pd.DataFrame(cols)


def planner(level, seeds, slot=""):
    out = {}
    for s in seeds:
        f = R / "rq2-perturbed-v5" / "B2" / f"{level}{slot}" / f"s{s}.json"
        if f.exists():
            out[s] = json.loads(f.read_text())["real"]
    return pd.Series(out, dtype=float)


def main():
    rng = np.random.default_rng(0)
    pd.set_option("display.width", 240)
    base = {sp: la_an.table(sp) for sp in ("fit", "test")}       # MDD-PTECT, state-only, exact look-ahead, oracle
    frames = {}
    for sp, t in base.items():
        f = pd.DataFrame(index=t.index)
        f[REF] = t[REF]
        f["state-only"] = t[[f"state-only s{k}" for k in range(5)]].mean(axis=1, skipna=False)
        for k in range(5):
            f[f"exact s{k}"] = t[f"look-ahead s{k}"]
        f["exact"] = f[[f"exact s{k}" for k in range(5)]].mean(axis=1, skipna=False)
        f["split"] = sp
        frames[sp] = f
    rows, seen = [], []
    for level in LEVELS:
        per = {}
        for sp, f in frames.items():
            g = f.copy()
            if level == "exact":
                g["family"] = g["exact"]
                g["pick"] = g[f"exact {PICK}"]
                g["n_ckpt"] = 5
            else:
                lt = level_table(level, sp)
                if lt.empty:
                    continue
                g["family"] = lt.reindex(g.index).mean(axis=1, skipna=False) if lt.shape[1] == 5 else np.nan
                g["pick"] = lt[PICK].reindex(g.index) if PICK in lt else np.nan
                g["n_ckpt"] = lt.shape[1]
            if sp == "test":
                g["planner 900 s"] = planner(level, g.index).reindex(g.index)
                g["planner 300 s"] = planner(level, g.index, "-slot300").reindex(g.index)
            per[sp] = g
        if not per:
            continue
        seen.append(level)
        both = pd.concat(per.values())
        row = {"level": level, "schedule": NAMES[level],
               "ckpts": int(min(g["n_ckpt"].iloc[0] for g in per.values())),
               "seed_sets": "+".join(per)}
        ref_gain = both[REF].sum() - both["exact"].sum()
        for col, against, key in (("family", REF, "fam_vs_ref"), ("family", "state-only", "fam_vs_stateonly"),
                                  ("family", "exact", "fam_vs_exact"), ("pick", REF, "pick_vs_ref")):
            s = cmp(both[col], both[against], rng)
            if s:
                row[key] = s["pooled_%"]
                row[key + "_ci"] = f"[{s['ci_lo']:+.2f}, {s['ci_hi']:+.2f}]"
                row[key + "_bw"] = f"{s['better']}/{s['worse']}"
                row[key + "_p"] = s["p"]
                if key == "fam_vs_ref":
                    row["n"] = s["n"]
                    row["saved_k_per_inst"] = (both[REF] - both[col]).mean()
                    row["share_of_exact_gain_%"] = 100 * (both[REF].sum() - both[col].sum()) / ref_gain
        if "test" in per:
            g = per["test"]
            for col, key in (("family", "test_fam"), ("planner 900 s", "test_planner900"),
                             ("planner 300 s", "test_planner300")):
                s = cmp(g[col], g[REF], rng)
                if s:
                    row[key] = s["pooled_%"]
                    row[key + "_ci"] = f"[{s['ci_lo']:+.2f}, {s['ci_hi']:+.2f}]"
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "summary.csv", index=False)

    def fmt(r, k):
        return f"{r[k]:+.2f} {r[k + '_ci']}" if k in r and pd.notna(r.get(k)) else "-"

    print("== eval-v5-lookahead-fcast: look-ahead PPO (train-v5-lookahead, 5 seeds) with features from a perturbed "
          "published schedule; B2 base c")
    print(f"   levels with outputs: {seen}; family = per-instance mean of the 5 seeds; pick = {PICK}\n")
    print("-- pooled over the held-out seeds (fitting 2000-2159 + test 0-39): pooled % [95% CI], better / worse, p")
    for _, r in df.iterrows():
        print(f"{r['schedule']:26s} n={int(r.get('n', 0)):3d} ckpts={r['ckpts']}  "
              f"vs MDD-PTECT {fmt(r, 'fam_vs_ref')} {r.get('fam_vs_ref_bw', '')} p={r.get('fam_vs_ref_p', np.nan):.2g}  "
              f"| vs state-only {fmt(r, 'fam_vs_stateonly')} {r.get('fam_vs_stateonly_bw', '')} "
              f"p={r.get('fam_vs_stateonly_p', np.nan):.2g}  "
              f"| vs own exact {fmt(r, 'fam_vs_exact')} {r.get('fam_vs_exact_bw', '')} "
              f"p={r.get('fam_vs_exact_p', np.nan):.2g}  "
              f"| kept {r.get('share_of_exact_gain_%', np.nan):.0f}% of the exact gain, "
              f"{r.get('saved_k_per_inst', np.nan):.2f} k job-s / inst  | pick {fmt(r, 'pick_vs_ref')}")
    print("\n-- test seeds 0-39, vs MDD-PTECT: look-ahead family vs the planner on the same published schedule")
    for _, r in df.iterrows():
        print(f"{r['schedule']:26s} look-ahead {fmt(r, 'test_fam'):24s} planner 900 s {fmt(r, 'test_planner900'):24s} "
              f"planner 300 s {fmt(r, 'test_planner300')}")


if __name__ == "__main__":
    buf = io.StringIO()
    with redirect_stdout(buf):
        main()
    (HERE / "analysis.out").write_text(buf.getvalue())
    sys.stdout.write(buf.getvalue())
