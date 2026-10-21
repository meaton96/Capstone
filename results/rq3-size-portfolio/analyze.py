"""
@file analyze.py
@brief rq3-size-portfolio (2026-10-20): exact checks of the catalog screen's proxies, B2 test seeds 0-39, window
       tardiness, vs MDD-PTECT held all episode (pooled, seed-bootstrap 95% CI, Wilcoxon); < 0 better.
  3a maps:    the fitted regime map per floor, next to its proxy (../rq3-size-catalog test map %), the v5 switching
              oracle (../rq3-size-oracle; m15: ../rq2-rebase-v5) and the share of that ceiling the map reaches;
              at m100 also vs MDD-TECT held all episode.
  3b oracles: the greedy oracle over each size portfolio vs MDD-PTECT and vs the v5 oracle (paired), slot use, next to
              the catalog's block proxy for that portfolio.
  Fixed-pair references come from the catalog runs (m15: ../rq2-portfolio-screen), whose MDD-PTECT equals the grid's.
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
_s = importlib.util.spec_from_file_location("prun", HERE / "run.py")
pr = importlib.util.module_from_spec(_s)
_s.loader.exec_module(pr)
SEEDS = list(range(40))
REF = "MDD-PTECT"


def fixed(cond, pair):
    d = (R / "rq2-portfolio-screen" / "out") if cond == "m15" else (R / "rq3-size-catalog" / cond / "out")
    out = {}
    for s in SEEDS:
        f = d / f"s{s}.json"
        if f.exists():
            out[s] = json.loads(f.read_text())[pair]["total"]
    return out


def v5_oracle(cond):
    if cond == "m15":
        out = {}
        for f in (R / "rq2-rebase-v5" / "out" / "oracle").glob("s*.npz"):
            d = np.load(f, allow_pickle=True)
            if int(d["seed"]) in SEEDS:
                out[int(d["seed"])] = float(d["oracle"])
        return out
    return {json.loads(f.read_text())["seed"]: json.loads(f.read_text())["oracle"]
            for f in (R / "rq3-size-oracle" / "out" / cond).glob("s*.json")}


def catalog(cond):
    f = R / "rq3-size-catalog" / cond / "portfolios.json"
    return json.loads(f.read_text()) if f.exists() else {}


def line(lab, c, extra=""):
    return (f"   {lab:34s} {c['pooled_%']:+6.2f} [{c['ci_lo']:+6.2f}, {c['ci_hi']:+6.2f}] {c['better']:2d}/{c['worse']:<2d} "
            f"p={c['wilcoxon_p']:.2g}{extra}")


def main():
    rng = np.random.default_rng(0)
    rows = []
    print("== rq3-size-portfolio: exact checks of the catalog's proxies; B2 test seeds 0-39, window tardiness vs "
          "MDD-PTECT held all episode; < 0 better")
    print("\n-- 3a: fitted regime maps (normal / surge / short fleet), exact")
    for cond, m in pr.MAPS.items():
        got = {}
        for f in (HERE / "out" / cond / "map").glob("s*.json"):
            d = json.loads(f.read_text())
            got[d["seed"]] = d["tard"]
        ref = fixed(cond, REF)
        print(f"\n   {cond}: map {m}; {len(got)} / 40 seeds")
        if len(got) < 40 or len(ref) < 40:
            continue
        cat = catalog(cond)
        proxy = cat.get("full", {}).get("test", {}).get("map_%")
        c = rq3.compare(got, ref, rng)
        print(line("map vs MDD-PTECT (exact)", c, f"   | proxy {proxy:+.2f}" if proxy is not None else ""))
        rows.append({"part": "map", "floor": cond, "what": "map vs MDD-PTECT", "proxy_%": proxy, **c})
        orc = v5_oracle(cond)
        if len(orc) >= 40:
            co = rq3.compare(orc, ref, rng)
            share = c["pooled_%"] / co["pooled_%"] if co["pooled_%"] else float("nan")
            print(line("v5 switching oracle vs MDD-PTECT", co, f"   | map reaches {100 * share:.0f}% of it"))
            cm = rq3.compare(got, orc, rng)
            print(line("map vs v5 oracle (paired)", cm))
            rows += [{"part": "map", "floor": cond, "what": "v5 oracle vs MDD-PTECT", **co},
                     {"part": "map", "floor": cond, "what": "map vs v5 oracle", **cm}]
        for p in sorted(set(m.values()) - {REF}):
            fp = fixed(cond, p)
            if len(fp) >= 40:
                cp = rq3.compare(got, fp, rng)
                print(line(f"map vs {p} held all episode", cp))
                rows.append({"part": "map", "floor": cond, "what": f"map vs {p}", **cp})
    print("\n-- 3b: switching oracle over size portfolios, exact")
    for (cond, name), pairs in pr.PORTFOLIOS.items():
        d = [json.loads(f.read_text()) for f in (HERE / "out" / cond / name).glob("s*.json")]
        print(f"\n   {cond} / {name}: {'/'.join(pairs)}; {len(d)} / 40 seeds")
        if len(d) < 40:
            continue
        got = {x["seed"]: x["oracle"] for x in d}
        ref = fixed(cond, REF)
        chk = sum(abs(x["fixed"][REF] - ref[x["seed"]]) <= 1e-6 * abs(ref[x["seed"]]) + 1e-4 for x in d)
        cat = catalog(cond)
        proxy = cat.get("block6", {}).get("test", {}).get("block_%") if name == "block6" else None
        c = rq3.compare(got, ref, rng)
        print(f"   MDD-PTECT held all episode = the catalog's on {chk} / 40 seeds")
        print(line("oracle vs MDD-PTECT", c, f"   | block proxy {proxy:+.2f}" if proxy is not None else ""))
        rows.append({"part": "oracle", "floor": cond, "what": f"{name} oracle vs MDD-PTECT", "proxy_%": proxy, **c})
        orc = v5_oracle(cond)
        if len(orc) >= 40:
            cv = rq3.compare(got, orc, rng)
            print(line("oracle vs v5 oracle (paired)", cv))
            rows.append({"part": "oracle", "floor": cond, "what": f"{name} oracle vs v5 oracle", **cv})
        use = Counter(p for x in d for p in x["schedule"])
        tot = sum(use.values())
        print("   slot use: " + ", ".join(f"{p} {100 * use[p] / tot:.0f}%" for p, _ in use.most_common()))
    pd.DataFrame(rows).to_csv(HERE / "summary.csv", index=False)


if __name__ == "__main__":
    buf = io.StringIO()
    with redirect_stdout(buf):
        main()
    (HERE / "analysis.out").write_text(buf.getvalue())
    sys.stdout.write(buf.getvalue())
