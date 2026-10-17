"""
@file analyze.py
@brief rq4-mfail-ccal (2026-10-14): the v5 rules on the 50-machine floor at no failures, failures at B2's c, and
       failures at the calibrated c (same late share as without failures). Per setting: tardiness level and late
       share, every rule vs MDD-PTECT (pooled window tardiness, seed-bootstrap 95% CI, seeds better / worse,
       Wilcoxon), the ranking and its Kendall tau against no failures, the per-instance hindsight best v5 pair, and
       the calibration table (cscale.json). -> analysis.out
"""
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, wilcoxon

HERE = Path(__file__).resolve().parent
REF = "MDD_PTECT:quad:3"
V5 = ["SRT_PECT:quad:3", "SRT_PTECT:quad:3", "MDD_PECT:quad:3", "MDD_PTECT:quad:3", "ATC_PECT:quad:3",
      "ATC_PTECT:quad:3"]
SETTINGS = ["none", "unif", "unif-cal"]


def short(v):
    return v.replace(":quad:3", "").replace("_", "-")


def load():
    data = {}
    for f in sorted((HERE / "out" / "compare").glob("*.json")):
        d = json.loads(f.read_text())
        st = data.setdefault(d["setting"], {"scale": d["scale"], "runs": {}})
        st["runs"].update(d["runs"])
    return data


def cmp(y, b, rng):
    idx = rng.integers(0, len(y), (10000, len(y)))
    bs = 100 * (y[idx].sum(1) / b[idx].sum(1) - 1)
    d = y - b
    p = float(wilcoxon(d).pvalue) if np.any(d) else 1.0
    return (f"{100 * (y.sum() / b.sum() - 1):+6.1f} [{np.percentile(bs, 2.5):+6.1f}, {np.percentile(bs, 97.5):+6.1f}] "
            f"{int((d < 0).sum()):2d}/{int((d > 0).sum()):<2d} p={p:.2g}")


def main():
    rng = np.random.default_rng(0)
    cs = HERE / "cscale.json"
    if cs.exists():
        c = json.loads(cs.read_text())
        print(f"== calibration (seeds 1000-1029, MDD-PTECT): failure-free late share {c['target_late_share']:.3f}, "
              f"mean tardiness {c['none_mean_tard']:.1f} ks; S* = {c['s_star']:.3f} (c range {1.75 * c['s_star']:.2f}-"
              f"{2.5 * c['s_star']:.2f})")
        for s, r in c["table"].items():
            print(f"   S {float(s):4.2f}: failures late share {r['late_share']:.3f}, mean tardiness {r['mean_tard']:.1f} ks")
    data = load()
    if not data:
        print("no comparison outputs yet")
        return
    totals = {}
    for st in SETTINGS:
        if st not in data:
            continue
        runs = data[st]["runs"]
        seeds = sorted(runs, key=int)
        rules = list(runs[seeds[0]])
        t = {v: np.array([runs[s][v]["tard"] for s in seeds]) for v in rules}
        late = sum(runs[s][REF]["late"] for s in seeds) / sum(runs[s][REF]["due_in"] for s in seeds)
        print(f"\n== {st} (c scale {data[st]['scale']:.3f}), {len(seeds)} test seeds: MDD-PTECT mean tardiness "
              f"{t[REF].mean():.1f} ks, late share {late:.3f}")
        hind = np.min([t[v] for v in V5 if v in t], axis=0)
        for v in sorted(rules, key=lambda v: t[v].sum()):
            if v != REF:
                print(f"   {short(v):10s} vs MDD-PTECT {cmp(t[v], t[REF], rng)}")
        print(f"   {'hindsight best v5 pair':10s} {cmp(hind, t[REF], rng)}")
        totals[st] = {v: t[v].sum() for v in rules}
        order = sorted(rules, key=lambda v: t[v].sum())
        print("   ranking: " + "  ".join(f"{short(v)} {100 * (t[v].sum() / t[order[0]].sum() - 1):+.1f}%" for v in order))
    if "none" in totals:
        for st in ("unif", "unif-cal"):
            if st in totals:
                common = [v for v in totals["none"] if v in totals[st]]
                k = kendalltau([totals["none"][v] for v in common], [totals[st][v] for v in common])
                lvl = totals[st][REF] / totals["none"][REF]
                print(f"\n-- {st} vs none: ranking Kendall tau {k.statistic:+.2f} (p {k.pvalue:.2g}, {len(common)} rules); "
                      f"MDD-PTECT tardiness x{lvl:.2f}")


if __name__ == "__main__":
    buf = io.StringIO()
    with redirect_stdout(buf):
        main()
    (HERE / "analysis.out").write_text(buf.getvalue())
    sys.stdout.write(buf.getvalue())
