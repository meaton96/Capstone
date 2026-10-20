"""
@file analyze.py
@brief rq3-size-catalog (2026-10-20): which rule pairs are strong and which are complementary, by floor size. Per floor
       (m15 = ../rq2-portfolio-screen's runs; m30 / m50 / m100 = <cond>/out), with ../rq2-portfolio-screen's own
       proxy and greedy functions (select_portfolios.py), all vs MDD-PTECT, pooled over seeds, < 0 better:
       - the best fixed pairs on test seeds 0-39 (where the v5 pairs rank; the best pair outside v5);
       - the best pair per regime (normal / surge / short fleet), test seeds;
       - portfolio proxies (hindsight: per-instance best, per-block best; observable: one pair per regime) for the
         full 84-pair pool and for v5, fitting 2000-2039 | test 0-39;
       - greedy portfolios (K <= 6) on fitting seeds: "block" (per-block best) and "map" (regime map), with their test
         proxies. The block portfolio is the step-3 candidate for the exact oracle.
       -> analysis.out, summary.csv, <cond>/portfolios.json
"""
import importlib.util
import io
import json
import sys
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
R = HERE.parent
_s = importlib.util.spec_from_file_location("sel", R / "rq2-portfolio-screen" / "select_portfolios.py")
sp = importlib.util.module_from_spec(_s)
_s.loader.exec_module(sp)
FIT = list(range(2000, 2040))
TEST = list(range(40))
FLOORS = {"m15": 15, "m30": 30, "m50": 50, "m100": 100}
V5 = sp.V5
BF = sp.BF


def data(cond, seeds):
    sp.HERE = (R / "rq2-portfolio-screen") if cond == "m15" else (HERE / cond)   # sp.load reads sp.HERE / out
    return sp.load(seeds)


def pct(x, b):
    return 100 * (x / b - 1)


def main():
    print("== rq3-size-catalog: 84 catalog pairs (12 job x 7 machine rules) as fixed pairs, by floor size; vs MDD-PTECT, "
          "pooled; < 0 better. Proxies: inst / block = hindsight best per instance / per regime block; map = one pair per "
          "regime (observable). Fitting seeds 2000-2039 | test 0-39.")
    rows = []
    for cond, m in FLOORS.items():
        fit, test = data(cond, FIT), data(cond, TEST)
        print(f"\n-- {cond} ({m} machines): {len(fit)} / 40 fitting, {len(test)} / 40 test seeds")
        if len(test) < 40 or len(fit) < 40:
            continue
        pairs = next(iter(test.values()))[0]
        tot = {p: sum(v[3][pairs.index(p)] for v in test.values()) for p in pairs}
        ranked = sorted(pairs, key=tot.get)
        rank = {p: k + 1 for k, p in enumerate(ranked)}
        print("   best fixed pairs (test): " + ", ".join(f"{p} {pct(tot[p], tot[BF]):+.1f}" for p in ranked[:8]))
        outside = [p for p in ranked if p not in V5]
        print(f"   v5 ranks: " + ", ".join(f"{p} #{rank[p]}" for p in sorted(V5, key=rank.get))
              + f"; best outside v5: {outside[0]} {pct(tot[outside[0]], tot[BF]):+.1f}% (#{rank[outside[0]]})")
        regs = sorted({r for v in test.values() for r in v[2]})
        per_reg = {}
        for r in regs:
            s = {p: sum(v[1][i, pairs.index(p)] for v in test.values() for i, rr in enumerate(v[2]) if rr == r)
                 for p in pairs}
            top = sorted(pairs, key=s.get)[:3]
            per_reg[r] = top[0]
            print(f"   best per regime, {r:6s}: " + ", ".join(f"{p} {pct(s[p], s[BF]):+.1f}" for p in top))
        wins = Counter(pairs[int(np.argmin(v[1][i]))] for v in test.values() for i in range(len(v[2])))
        nb = sum(wins.values())
        print(f"   per-block winners (test, {nb} blocks): " + ", ".join(f"{p} {c}" for p, c in wins.most_common(6)))
        full, v5f, v5t, fullt = (sp.proxies(fit, pairs), sp.proxies(fit, V5), sp.proxies(test, V5),
                                 sp.proxies(test, pairs))
        print(f"   proxies, full pool : fit map {full['map_%']:+6.2f} block {full['block_%']:+6.2f} inst "
              f"{full['inst_%']:+6.2f} | test map {fullt['map_%']:+6.2f} block {fullt['block_%']:+6.2f} inst "
              f"{fullt['inst_%']:+6.2f}")
        print(f"   proxies, v5        : fit map {v5f['map_%']:+6.2f} block {v5f['block_%']:+6.2f} inst "
              f"{v5f['inst_%']:+6.2f} | test map {v5t['map_%']:+6.2f} block {v5t['block_%']:+6.2f} inst "
              f"{v5t['inst_%']:+6.2f}")
        port = {"v5": {"pairs": V5, "fit": v5f, "test": v5t}, "full": {"pairs": pairs, "fit": full, "test": fullt}}
        for name, key in (("block", "block_%"), ("map", "map_%")):
            print(f"   greedy '{name}' portfolio (fitting, start {BF}):")
            for S, pf in sp.greedy(fit, pairs, BF, key):
                pt = sp.proxies(test, S)
                print(f"     K={len(S)} {'/'.join(S):62s} fit {key[:-2]} {pf[key]:+6.2f} | test map {pt['map_%']:+6.2f} "
                      f"block {pt['block_%']:+6.2f} inst {pt['inst_%']:+6.2f}")
                port[f"{name}{len(S)}"] = {"pairs": S, "fit": pf, "test": pt}
        (HERE / cond).mkdir(exist_ok=True)
        (HERE / cond / "portfolios.json").write_text(json.dumps(port, indent=1))
        b6 = port["block6"]
        rows.append({"floor": cond, "machines": m, "best_pair": ranked[0], "best_pair_%": pct(tot[ranked[0]], tot[BF]),
                     "best_outside_v5": outside[0], "best_outside_v5_%": pct(tot[outside[0]], tot[BF]),
                     "best_per_regime": json.dumps(per_reg),
                     **{f"full_test_{k}": fullt[k] for k in ("map_%", "block_%", "inst_%")},
                     **{f"v5_test_{k}": v5t[k] for k in ("map_%", "block_%", "inst_%")},
                     "block6_pairs": "/".join(b6["pairs"]),
                     **{f"block6_test_{k}": b6["test"][k] for k in ("map_%", "block_%", "inst_%")}})
    if rows:
        t = pd.DataFrame(rows)
        t.to_csv(HERE / "summary.csv", index=False)
        print("\n== by floor size (test seeds): best fixed pair; hindsight proxies of the full pool, v5 and the greedy "
              "block portfolio (K = 6, chosen on fitting seeds)")
        print(t[["floor", "best_pair", "best_pair_%", "best_outside_v5", "best_outside_v5_%", "full_test_block_%",
                 "v5_test_block_%", "block6_test_block_%", "full_test_inst_%", "v5_test_inst_%", "full_test_map_%",
                 "v5_test_map_%"]].round(2).to_string(index=False))


if __name__ == "__main__":
    buf = io.StringIO()
    with redirect_stdout(buf):
        main()
    (HERE / "analysis.out").write_text(buf.getvalue())
    sys.stdout.write(buf.getvalue())
