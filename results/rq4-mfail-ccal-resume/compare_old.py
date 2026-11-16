"""
@file compare_old.py
@brief rq4-mfail-ccal-resume (2026-10-28): the rerun (preempt-resume) next to rq4-mfail-ccal (restart, 10-14): the
       calibrated scale S*, and per setting MDD-PTECT's tardiness and late share and the rule ranking (Kendall tau
       between the two runs). The failure-free setting must be identical (the patch only touches failure handling).
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "rq4-mfail-ccal"
REF = "MDD_PTECT:quad:3"


def load(d):
    data = {}
    for f in sorted((d / "out" / "compare").glob("*.json")):
        x = json.loads(f.read_text())
        st = data.setdefault(x["setting"], {"scale": x["scale"], "runs": {}})
        st["runs"].update(x["runs"])
    return data


def short(v):
    return v.replace(":quad:3", "").replace("_", "-")


def main():
    for d, tag in ((OLD, "restart (rq4-mfail-ccal)"), (HERE, "resume (this run)")):
        cs = d / "cscale.json"
        if cs.exists():
            c = json.loads(cs.read_text())
            print(f"S* {tag}: {c['s_star']:.3f} (c {1.75 * c['s_star']:.2f}-{2.5 * c['s_star']:.2f}); "
                  f"S=1 failures late share {c['table']['1.0']['late_share']:.3f}, mean tardiness "
                  f"{c['table']['1.0']['mean_tard']:.1f} ks (failure-free {c['none_mean_tard']:.1f} ks)")
    old, new = load(OLD), load(HERE)
    bad = False
    for st in ("none", "unif", "unif-cal"):
        if st not in old or st not in new:
            print(f"{st}: missing ({'old' if st not in old else 'new'})")
            continue
        o, n = old[st]["runs"], new[st]["runs"]
        seeds = sorted(set(o) & set(n), key=int)
        rules = list(o[seeds[0]])
        to = {v: sum(o[s][v]["tard"] for s in seeds) for v in rules}
        tn = {v: sum(n[s][v]["tard"] for s in seeds) for v in rules}
        lo = sum(o[s][REF]["late"] for s in seeds) / sum(o[s][REF]["due_in"] for s in seeds)
        ln = sum(n[s][REF]["late"] for s in seeds) / sum(n[s][REF]["due_in"] for s in seeds)
        tau = kendalltau([to[v] for v in rules], [tn[v] for v in rules]).statistic
        print(f"\n== {st} ({len(seeds)} seeds; c scale old {old[st]['scale']:.3f}, new {new[st]['scale']:.3f}): "
              f"MDD-PTECT mean tardiness {to[REF] / len(seeds):.1f} -> {tn[REF] / len(seeds):.1f} ks, late share "
              f"{lo:.3f} -> {ln:.3f}; ranking tau old vs new {tau:+.2f}")
        for v in sorted(rules, key=lambda v: tn[v]):
            print(f"   {short(v):10s} vs MDD-PTECT  restart {100 * (to[v] / to[REF] - 1):+6.1f}%   "
                  f"resume {100 * (tn[v] / tn[REF] - 1):+6.1f}%")
        if st == "none":
            same = all(o[s] == n[s] for s in seeds)
            print(f"   failure-free runs identical to rq4-mfail-ccal: {same}")
            bad |= not same
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
