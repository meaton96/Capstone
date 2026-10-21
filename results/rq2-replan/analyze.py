"""
@file analyze.py
@brief rq2-replan (2026-10-21): when should the twin planner re-plan? B2 test seeds 0-39, window tardiness, pooled %
       vs MDD-PTECT (seed-bootstrap 95% CI, wins / losses, Wilcoxon); compute = re-plans per episode and planning seconds.
  1a granularity (exact forecast): % vs decision interval Δ; the ceiling as decisions get finer.
  1b forecast error x interval: % per (level, Δ); the share of the exact ceiling at the same Δ.
  1c event-triggered vs periodic at Δ = 300 s: % and re-plans per trigger; paired vs periodic re-planning every 300 s.
  -> analysis.out, summary.csv (one row per variant), curve.csv (figure data)
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
_s = importlib.util.spec_from_file_location("rq3an", R / "rq3-gen-v5" / "analyze.py")
rq3 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(rq3)
_s = importlib.util.spec_from_file_location("rp", HERE / "run.py")
rp = importlib.util.module_from_spec(_s)
_s.loader.exec_module(rp)


def load(v):
    out = {}
    for f in (HERE / "out" / v).glob("s*.json"):
        d = json.loads(f.read_text())
        out[d["seed"]] = d
    return out


def row(v, d, rng, base=None):
    real = {s: x["real"] for s, x in d.items()}
    ref = {s: x["mdd_ptect"] for s, x in d.items()}
    c = rq3.compare(real, ref, rng)
    level, slot, trig = rp.parse(v)
    r = {"variant": v, "level": level, "interval_s": slot, "trigger": trig, "n": len(d),
         **{f"vs_mdd_ptect_{k}": val for k, val in c.items()},
         "replans_mean": float(np.mean([x["replans"] for x in d.values()])),
         "switches_mean": float(np.mean([x["switches"] for x in d.values()])),
         "plan_seconds_mean": float(np.mean([x["plan_seconds"] for x in d.values()]))}
    if base:
        cb = rq3.compare(real, {s: x["real"] for s, x in base.items()}, rng)
        r.update({f"vs_periodic_{k}": val for k, val in cb.items()})
    return r


def fmt(r):
    return (f"{r['vs_mdd_ptect_pooled_%']:+6.2f} [{r['vs_mdd_ptect_ci_lo']:+6.2f}, {r['vs_mdd_ptect_ci_hi']:+6.2f}] "
            f"{r['vs_mdd_ptect_better']:2d}/{r['vs_mdd_ptect_worse']:<2d}  re-plans {r['replans_mean']:5.1f}  switches "
            f"{r['switches_mean']:4.1f}  planning {r['plan_seconds_mean']:6.0f} s")


def main():
    rng = np.random.default_rng(0)
    data = {v: load(v) for v in rp.variants()}
    done = {v: d for v, d in data.items() if len(d) == 40}
    print(f"== rq2-replan: twin planner v2, v5 pairs, B2 test seeds 0-39; {len(done)} / {len(data)} variants complete")
    rows = {}
    print("\n-- 1a granularity, exact forecast (the greedy switching oracle at decision interval Δ); % vs MDD-PTECT")
    for d in rp.D1A:
        v = f"exact-d{d}"
        if v in done:
            rows[v] = row(v, done[v], rng)
            print(f"   Δ {d:5d} s  {fmt(rows[v])}")
    print("\n-- 1b forecast error x interval; % vs MDD-PTECT (share of the exact ceiling at the same Δ)")
    for lv in rp.L1B:
        parts = []
        for d in rp.D1B:
            v, ve = f"{lv}-d{d}", f"exact-d{d}"
            if v in done:
                rows[v] = row(v, done[v], rng)
                share = ""
                if ve in rows and rows[ve]["vs_mdd_ptect_pooled_%"] < 0:
                    share = f" ({100 * rows[v]['vs_mdd_ptect_pooled_%'] / rows[ve]['vs_mdd_ptect_pooled_%']:.0f}%)"
                parts.append(f"Δ {d}: {rows[v]['vs_mdd_ptect_pooled_%']:+.2f}{share}")
        if parts:
            print(f"   {lv:8s} " + " | ".join(parts))
    print("\n-- 1c event-triggered vs periodic re-planning, switches allowed every 300 s")
    for lv in rp.L1C:
        base_v = f"{lv}-d300"
        base = done.get(base_v)
        if base is None:
            print(f"   {lv}: periodic Δ 300 not complete yet")
            continue
        rows.setdefault(base_v, row(base_v, base, rng))
        print(f"   {lv:8s} periodic every 300 s   {fmt(rows[base_v])}")
        for t in rp.TRIGGERS:
            v = f"{lv}-d300-{t}"
            if v in done:
                rows[v] = row(v, done[v], rng, base)
                r = rows[v]
                print(f"   {lv:8s} {t:6s}                {fmt(r)}  | vs periodic {r['vs_periodic_pooled_%']:+.2f} "
                      f"[{r['vs_periodic_ci_lo']:+.2f}, {r['vs_periodic_ci_hi']:+.2f}] p={r['vs_periodic_wilcoxon_p']:.2g}")
    df = pd.DataFrame(rows.values())
    df.to_csv(HERE / "summary.csv", index=False)
    if not df.empty:
        df[["variant", "level", "interval_s", "trigger", "vs_mdd_ptect_pooled_%", "vs_mdd_ptect_ci_lo", "vs_mdd_ptect_ci_hi",
            "replans_mean", "plan_seconds_mean"]].to_csv(HERE / "curve.csv", index=False)


if __name__ == "__main__":
    buf = io.StringIO()
    with redirect_stdout(buf):
        main()
    (HERE / "analysis.out").write_text(buf.getvalue())
    sys.stdout.write(buf.getvalue())
