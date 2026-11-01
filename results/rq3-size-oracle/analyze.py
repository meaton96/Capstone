"""
@file analyze.py
@brief rq3-size-oracle (2026-10-20): the v5 switching-oracle ceiling by floor size. Per floor (m15 = ../rq2-rebase-v5's
       B2 oracle, same oracle and code; m50 / m100 / m200 = out/): the oracle and the hindsight best v5 pair per instance vs
       MDD-PTECT (pooled window tardiness over B2 test seeds 0-39, seed-bootstrap 95% CI, Wilcoxon), switching beyond
       the per-instance pick (oracle vs hindsight best pair), and the oracle's slot use. Reproduction check: MDD-PTECT
       held all episode (the oracle's slot-0 tail) = the grid's MDD-PTECT on the same seed.
       -> analysis.out, summary.csv
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
_s = importlib.util.spec_from_file_location("rq3an", R / "rq3-gen-v5" / "analyze.py")
rq3 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(rq3)
_s = importlib.util.spec_from_file_location("sizesim", HERE / "sizesim.py")
ss = importlib.util.module_from_spec(_s)
_s.loader.exec_module(ss)
V5 = rq3.V5
REF = rq3.REF
FLOORS = {"m15": 15, "m50": 50, "m100": 100, "m200": 200}


def load(cond):
    """{seed: (oracle, {pair: fixed}, schedule)}"""
    out = {}
    if cond == "m15":
        for f in sorted((R / "rq2-rebase-v5" / "out" / "oracle").glob("s*.npz")):
            d = np.load(f, allow_pickle=True)
            s = int(d["seed"])
            if s in rq3.SEEDS:
                out[s] = (float(d["oracle"]), dict(zip([str(p) for p in d["pairs"]], d["tails"][0].tolist())),
                          [str(p) for p in d["schedule"]])
    else:
        for f in sorted((HERE / "out" / cond).glob("s*.json")):
            d = json.loads(f.read_text())
            out[int(d["seed"])] = (d["oracle"], d["fixed"], d["schedule"])
    return out


def main():
    rng = np.random.default_rng(0)
    print("== rq3-size-oracle: v5 switching-oracle ceiling by floor size (6 v5 pairs, 900 s slots, hold-to-end tails), "
          "B2 test seeds 0-39, window tardiness; < 0 better")
    rows = []
    for cond, m in FLOORS.items():
        d = load(cond)
        print(f"\n-- {cond} ({m} machines): {len(d)} / 40 seeds")
        if len(d) < 5:
            continue
        ref = {s: v[1][REF] for s, v in d.items()}
        orc = {s: v[0] for s, v in d.items()}
        inst = {s: min(v[1][p] for p in V5) for s, v in d.items()}
        ok = n = 0
        for s, v in ref.items():
            g = ss.grid_value(cond, s)
            if g is not None:
                n += 1
                ok += abs(v - g) <= 1e-6 * max(1.0, abs(g)) + 1e-4
        print(f"   MDD-PTECT mean {np.mean(list(ref.values())):,.1f} k job-s; = the grid's MDD-PTECT on {ok} / {n} seeds"
              + ("" if ok == n else "   ** MISMATCH **"))
        for lab, x, b in (("switching oracle", orc, ref), ("hindsight best v5 pair", inst, ref),
                          ("oracle vs hindsight best pair", orc, inst)):
            c = rq3.compare(x, b, rng)
            if c is None:
                continue
            saved = np.mean([b[s] - x[s] for s in x if s in b])
            print(f"   {lab:30s} {c['pooled_%']:+6.2f} [{c['ci_lo']:+6.2f}, {c['ci_hi']:+6.2f}] "
                  f"{c['better']:2d}/{c['worse']:<2d} p={c['wilcoxon_p']:.2g}  ({saved:,.1f} k job-s per instance)")
            rows.append({"floor": cond, "machines": m, "what": lab, **c, "saved_per_instance_k": saved})
        use = Counter(p for v in d.values() for p in v[2])
        tot = sum(use.values())
        print("   oracle slot use: " + ", ".join(f"{p} {100 * use[p] / tot:.0f}%" for p, _ in use.most_common()))
        fx = pd.DataFrame({s: v[1] for s, v in d.items()}).T
        print("   fixed pairs vs MDD-PTECT (pooled): " + ", ".join(
            f"{p} {100 * (fx[p].sum() / fx[REF].sum() - 1):+.1f}%" for p in fx.sum().sort_values().index))
    pd.DataFrame(rows).to_csv(HERE / "summary.csv", index=False)


if __name__ == "__main__":
    buf = io.StringIO()
    with redirect_stdout(buf):
        main()
    (HERE / "analysis.out").write_text(buf.getvalue())
    sys.stdout.write(buf.getvalue())
