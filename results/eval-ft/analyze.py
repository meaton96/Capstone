"""
@file analyze.py
@brief eval-ft (2026-10-25, plan §2b, pre-declared in README.md before results): can a short fine-tune adapt the final
       policy to a very different floor? Window tardiness (1000 job-s), test seeds 0-39, % vs MDD-PTECT pooled over
       instances, 95% instance-bootstrap CI, test seeds better / worse, Wilcoxon. A family = the mean over its 5
       training seeds per instance; each training seed's own pooled % is shown with a one-sample t-test of the 5.
Arms: m100 = train-ft-m100 (fine-tuned on 100 machines), het = train-ft-het (fine-tuned on B2 with speed spread 1).
Budget b = fine-tune updates (0 = the final policy train-v5-cmix itself; 2 / 5 / 10 / 24).
Per arm:
  target cell (m100 or s1): every budget next to the references; paired b vs the final policy; the sample-efficiency
    read-out: m100 = share of the from-scratch result, (E_final - E_b) / (E_final - E_T100); het = share of the room
    to the per-instance best static v5 pair, (E_final - E_b) / (E_final - E_static) (no from-scratch v5 arm at s 1);
  B2 (s0): every budget vs the final policy (paired) and the retention E_b / E_final.
References: m100 = ../rq3-size-v5/m100 (pdr: v5 pairs + MDD-TECT; cmix = the final policy) and ../rq3-size-grid/m100
size100 (T100); s1 = ../eval-alpha-spread (out/1/final; grid/out/m15_s1: the v5 pairs = <pair>@3); B2 =
../eval-v5-cmix/test/base/cmix and ../rq3-gen-v5/base/pdr. Run-to-run check: every group's MDD-PTECT = the reference's.
@par Usage
@code{.sh}
.venv/bin/python results/eval-ft/analyze.py het     # -> analysis_het.out, summary_het.csv
.venv/bin/python results/eval-ft/analyze.py m100    # -> analysis_m100.out, summary_m100.csv
@endcode
"""
import io
import json
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp, wilcoxon

HERE = Path(__file__).resolve().parent
R = HERE.parent
OUT = HERE / "out"
SEEDS = list(range(40))
REF = "MDD-PTECT"
H = {"config_hash": str, "instance_hash": str}
BUDGETS = (2, 5, 10, 24)
V5 = ["SRT-PECT", "SRT-PTECT", "MDD-PECT", "MDD-PTECT", "ATC-PECT", "ATC-PTECT"]
B = 5000
## @brief Quoted yardsticks (not recomputed here): m100 = ../rq3-size-oracle (v5 switching ceiling) and
##        ../rq3-size-portfolio (short fleet -> MDD-TECT); s1 = ../eval-alpha-spread (outside the v5 action set).
QUOTED = {"m100": [("v5 switching ceiling (rq3-size-oracle)", -3.3),
                   ("short fleet -> MDD-TECT (rq3-size-portfolio)", -12.2)],
          "s1": [("spread law, MDD-PTECT at alpha(sigma) (outside v5)", -15.34),
                 ("best fixed alpha, MDD-PTECT@0.5 (outside v5)", -17.22),
                 ("train-alpha-het, v6 alpha head (other action set)", -15.92)]}


def name(p):
    """Episode policy name -> 'final s<k>', '<arm> b<b> s<k>', 'T100 s<k>' or the rule name."""
    p = str(p).replace("ckpt:", "").replace("results/", "")
    m = re.match(r"train-ft-(m100|het)_s(\d)/checkpoint(?:_step(\d+))?$", p)
    if m:
        step = int(m.group(3)) if m.group(3) else 98304 + 24 * 2048
        return f"{m.group(1)} b{(step - 98304) // 2048} s{m.group(2)}"
    for pre, lab in (("train-v5-cmix_s", "final s"), ("train-size-m100_s", "T100 s")):
        if p.startswith(pre):
            return lab + p.split("/")[0].rsplit("_s", 1)[1]
    return p


def done(files):
    """Finished files: DONE next to them, or ../eval-alpha-spread's out/<cell>/DONE_<group>_<seeds>."""
    return [f for f in files if f.exists() and ((f.parent / "DONE").exists()
                                                or any(f.parent.parent.glob(f"DONE_{f.parent.name}_*")))]


def table(files):
    """seed x policy window tardiness (1000 job-s) from finished episodes.csv files; MDD-PTECT kept per file for the
    run-to-run check (refs: {file: {seed: value}})."""
    frames, refs = [], {}
    for f in done(files):
        e = pd.read_csv(f, dtype=H)
        e = e[e.seed.isin(SEEDS)].copy()
        e["policy"] = e.policy.map(name)
        r = e[e.policy == REF]
        refs[str(f)] = dict(zip(r.seed.astype(int), r.window_tardiness / 1000.0))
        frames.append(e)
    if not frames:
        return pd.DataFrame(), refs
    e = pd.concat(frames, ignore_index=True).drop_duplicates(["policy", "seed"])
    return e.pivot_table(index="seed", columns="policy", values="window_tardiness") / 1000.0, refs


def ref_check(refs, ref):
    ok = n = 0
    for v in refs.values():
        for s, x in v.items():
            if s in ref.index:
                n += 1
                ok += abs(x - ref[s]) < 1e-6
    return ok, n


def family(t, prefix):
    mem = [f"{prefix} s{k}" for k in range(5)]
    if not all(m in t.columns for m in mem) or t[mem].notna().all(axis=1).sum() < len(SEEDS):
        return None
    return t[mem].mean(axis=1)


def compare(x, b, rng):
    """Pooled % of x vs b over instances, bootstrap CI, better / worse, Wilcoxon (x, b: seed-indexed Series)."""
    s = [k for k in SEEDS if k in x.index and k in b.index and not (np.isnan(x[k]) or np.isnan(b[k]))]
    xs, bs = x[s].to_numpy(), b[s].to_numpy()
    idx = rng.integers(0, len(s), (B, len(s)))
    boot = 100 * (xs[idx].sum(1) / bs[idx].sum(1) - 1)
    d = xs - bs
    return {"n": len(s), "pct": 100 * (xs.sum() / bs.sum() - 1), "lo": np.percentile(boot, 2.5),
            "hi": np.percentile(boot, 97.5), "better": int((d < 0).sum()), "worse": int((d > 0).sum()),
            "p": float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0}


def line(label, c, extra=""):
    return (f"   {label:<44} {c['pct']:+6.2f} [{c['lo']:+6.2f}, {c['hi']:+6.2f}] {c['better']:>2}/{c['worse']:<2} "
            f"p={c['p']:.2g}{extra}")


def seed_spread(t, prefix, ref):
    per = [100 * (t[f"{prefix} s{k}"].sum() / ref.sum() - 1) for k in range(5)]
    p = ttest_1samp(per, 0.0).pvalue
    return per, f"   | seeds {' '.join(f'{v:+.1f}' for v in per)} (sd {np.std(per, ddof=1):.2f}, t-test p={p:.2g})"


def share(t_start, t_b, t_goal, ref, rng):
    """(E_start - E_b) / (E_start - E_goal), E = pooled % vs ref; point and instance-bootstrap 95% CI."""
    a = [x.to_numpy() for x in (t_start, t_b, t_goal, ref)]
    e = lambda v, r: v.sum(-1) / r.sum(-1) - 1
    pt = (e(a[0], a[3]) - e(a[1], a[3])) / (e(a[0], a[3]) - e(a[2], a[3]))
    idx = rng.integers(0, len(a[3]), (B, len(a[3])))
    num = e(a[0][idx], a[3][idx]) - e(a[1][idx], a[3][idx])
    den = e(a[0][idx], a[3][idx]) - e(a[2][idx], a[3][idx])
    boot = num / np.where(np.abs(den) < 1e-12, np.nan, den)
    return pt, np.nanpercentile(boot, 2.5), np.nanpercentile(boot, 97.5)


def grid_s1():
    """../eval-alpha-spread's fixed-rule grid at s 1: the v5 pairs (= <pair>@3) per seed, 1000 job-s."""
    rows = {}
    for f in (R / "eval-alpha-spread" / "grid" / "out" / "m15_s1").glob("s*.json"):
        d = json.loads(f.read_text())
        rows[int(f.stem[1:])] = {p: d[f"{p}@3"]["total"] for p in V5}
    return pd.DataFrame(rows).T.sort_index()


def target_cell(arm, rng, rows):
    cell = "m100" if arm == "m100" else "s1"
    if arm == "m100":
        d = R / "rq3-size-v5" / "m100"
        refs_files = [d / "pdr" / "episodes.csv"] + sorted(d.glob("pdr_s*-*/episodes.csv"))
        base, _ = table(refs_files + [d / "cmix" / "episodes.csv"] + sorted(d.glob("cmix_s*-*/episodes.csv"))
                        + sorted((R / "rq3-size-grid" / "m100").glob("size100_s*-*/episodes.csv")))
        ft, frefs = table(sorted((OUT / "m100").glob("m100-b*_s*-*/episodes.csv")))
        pairs = V5 + ["MDD-TECT"]
    else:
        base, _ = table([R / "eval-alpha-spread" / "out" / "1" / "final" / "episodes.csv"])
        g = grid_s1()
        for p in V5:
            if p != REF:
                base[p] = g[p]
        ft, frefs = table(sorted((OUT / "s1").glob("het-b*/episodes.csv")))
        pairs = V5
        gr = g[REF].reindex(base.index)
        print(f"   check: the grid's MDD-PTECT@3 = evaluate.py's MDD-PTECT, max |rel| "
              f"{float(((gr - base[REF]) / base[REF]).abs().max()):.1e}")
    t = base.join(ft.drop(columns=[REF], errors="ignore"), how="left") if not ft.empty else base
    ref = t[REF]
    ok, n = ref_check(frefs, ref)
    print(f"-- {cell}: MDD-PTECT mean {ref.mean():.1f} k job-s; run-to-run MDD-PTECT = reference on {ok} / {n} "
          f"seed-runs")
    for p in pairs:
        if p in t.columns and p != REF:
            c = compare(t[p], ref, rng)
            print(line(p, c))
            rows.append({"arm": arm, "cell": cell, "row": p, **c})
    static = t[V5].min(axis=1)
    c = compare(static, ref, rng)
    print(line("per-instance best static v5 pair (hindsight)", c))
    rows.append({"arm": arm, "cell": cell, "row": "best static v5 pair", **c})
    for lab, v in QUOTED[cell]:
        print(f"   {lab:<44} {v:+6.2f} (quoted)")
    fams = {0: family(t, "final")}
    for b in BUDGETS:
        fams[b] = family(t, f"{arm} b{b}")
    goal = family(t, "T100") if arm == "m100" else static
    print("   budget (fine-tune updates; 0 = final policy), family mean of 5 vs MDD-PTECT:")
    for b, f in fams.items():
        if f is None:
            print(f"   b{b:<3} pending")
            continue
        c = compare(f, ref, rng)
        prefix = "final" if b == 0 else f"{arm} b{b}"
        per, txt = seed_spread(t, prefix, ref)
        print(line(f"b{b}", c, txt))
        rows.append({"arm": arm, "cell": cell, "row": f"b{b}", **c, "seed_pct": " ".join(f"{v:+.2f}" for v in per)})
    if arm == "m100" and goal is not None:
        c = compare(goal, ref, rng)
        _, txt = seed_spread(t, "T100", ref)
        print(line("T100 (from scratch, 48 updates)", c, txt))
        rows.append({"arm": arm, "cell": cell, "row": "T100", **c})
    goal_lab = "T100" if arm == "m100" else "best static v5 pair"
    print(f"   paired: each budget vs the final policy and vs {goal_lab}; share of the way from the final policy to "
          f"{goal_lab}:")
    for b in BUDGETS:
        if fams[b] is None or fams[0] is None or goal is None:
            continue
        c0 = compare(fams[b], fams[0], rng)
        cg = compare(fams[b], goal, rng)
        sh = share(fams[0], fams[b], goal, ref, rng)
        print(line(f"b{b} vs final", c0))
        print(line(f"b{b} vs {goal_lab}", cg, f"   | share {100 * sh[0]:+.0f}% [{100 * sh[1]:+.0f}, {100 * sh[2]:+.0f}]"))
        rows.append({"arm": arm, "cell": cell, "row": f"b{b} vs final", **c0})
        rows.append({"arm": arm, "cell": cell, "row": f"b{b} vs {goal_lab}", **cg, "share": sh[0],
                     "share_lo": sh[1], "share_hi": sh[2]})
    return fams, ref, goal


def b2_cell(arm, rng, rows):
    base, _ = table([R / "eval-v5-cmix" / "test" / "base" / "cmix" / "episodes.csv"])
    pdr, _ = table([R / "rq3-gen-v5" / "base" / "pdr" / "episodes.csv"])
    ft, frefs = table(sorted((OUT / "s0").glob(f"{arm}-b*/episodes.csv")))
    t = base.join(ft.drop(columns=[REF], errors="ignore"), how="left") if not ft.empty else base
    ref = t[REF]
    ok, n = ref_check(frefs, ref)
    mm = ("yes" if (pdr[REF] - ref).abs().max() < 1e-6 else "NO") if REF in pdr.columns else "n/a"
    print(f"-- B2 (s0): MDD-PTECT mean {ref.mean():.1f} k job-s; run-to-run MDD-PTECT = reference on {ok} / {n} "
          f"seed-runs (= rq3-gen-v5 pdr: {mm})")
    fams = {0: family(t, "final")}
    for b in BUDGETS:
        fams[b] = family(t, f"{arm} b{b}")
    e0 = None
    for b, f in fams.items():
        if f is None:
            print(f"   b{b:<3} pending")
            continue
        c = compare(f, ref, rng)
        prefix = "final" if b == 0 else f"{arm} b{b}"
        per, txt = seed_spread(t, prefix, ref)
        e0 = c["pct"] if b == 0 else e0
        ret = "" if b == 0 or not e0 else f"   | retention {c['pct'] / e0:.2f}"
        print(line(f"b{b}", c, txt + ret))
        rows.append({"arm": arm, "cell": "s0", "row": f"b{b}", **c, "seed_pct": " ".join(f"{v:+.2f}" for v in per)})
        if b:
            cp = compare(f, fams[0], rng)
            print(line(f"b{b} vs final (paired)", cp))
            rows.append({"arm": arm, "cell": "s0", "row": f"b{b} vs final", **cp})
    return fams, ref


def criteria(arm, tgt, b2, rng):
    """The pre-declared criteria (README)."""
    fams, ref, goal = tgt
    bf, bref = b2
    print("-- criteria (pre-declared)")
    f24, f0 = fams.get(24), fams.get(0)
    if f24 is None or f0 is None:
        print("   pending (budget 24 not evaluated)")
        return
    c = compare(f24, f0, rng)
    h1 = c["pct"] < 0 and c["p"] < 0.05
    cell = "m100" if arm == "m100" else "s1"
    print(f"   A1 adapts: b24 beats the final policy on {cell} (pooled < 0, Wilcoxon p < 0.05): "
          f"{c['pct']:+.2f}%, p={c['p']:.2g} -> {'PASS' if h1 else 'FAIL'}")
    if arm == "m100" and goal is not None:
        cg = compare(f24, goal, rng)
        h2 = cg["pct"] <= 1.0 and not (cg["pct"] > 0 and cg["p"] < 0.05)
        print(f"   A2 reaches the from-scratch arm: b24 vs T100 within +1 point and not significantly worse: "
              f"{cg['pct']:+.2f}%, p={cg['p']:.2g} -> {'PASS' if h2 else 'FAIL'}")
    for b in BUDGETS:
        if bf.get(b) is None:
            continue
        cb = compare(bf[b], bf[0], rng)
        keep = cb["pct"] <= 1.0 and not (cb["pct"] > 0 and cb["p"] < 0.05)
        print(f"   A3 keeps the B2 edge at b{b}: vs the final policy on B2 within +1 point and not significantly worse: "
              f"{cb['pct']:+.2f}%, p={cb['p']:.2g} -> {'PASS' if keep else 'FAIL'}")


def main(arm):
    rng = np.random.default_rng(0)
    rows = []
    print(f"== eval-ft {arm}: the final policy fine-tuned on {'100 machines' if arm == 'm100' else 'B2 with speed spread 1'}"
          f" (train-ft-{arm}), test seeds 0-39, window tardiness, % vs MDD-PTECT [95% CI] better/worse Wilcoxon")
    tgt = target_cell(arm, rng, rows)
    b2 = b2_cell(arm, rng, rows)
    criteria(arm, tgt, b2, rng)
    pd.DataFrame(rows).to_csv(HERE / f"summary_{arm}.csv", index=False)


if __name__ == "__main__":
    arm = sys.argv[1] if len(sys.argv) > 1 else "het"
    assert arm in ("het", "m100"), arm
    buf = io.StringIO()
    with redirect_stdout(buf):
        main(arm)
    (HERE / f"analysis_{arm}.out").write_text(buf.getvalue())
    print(buf.getvalue())
