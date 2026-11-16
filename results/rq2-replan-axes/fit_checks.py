"""
@file fit_checks.py
@brief rq2-replan-axes, remarks beside README §18's readings of §8's fit (not readings; the verdicts stand on fit.out).
       Reads fit_out/ (the lab run of 11-09) and fits nothing new except the size-slope comparison in (3):
       (1) how close F3 is to its line: M0's held-out intervals, and C(M0) and its coverage over the 2,000 resamples;
       (2) the held-out misses of M0, M1 and M2, and how far M2's predictions spread;
       (3) the size slope of M1 under four treatments (all 35 or the size arm alone; upper bounds censored or taken as
           exact), against axis 6's S_m (§14: size arm, bounds at their value, −0.51);
       (4) what an overshoot costs: the kept share at 1.5 and 2 times Δ*, over the 35 conditions.
@par Usage
@code{.sh}
.venv/bin/python results/rq2-replan-axes/fit_checks.py > results/rq2-replan-axes/fit_checks.out
@endcode
"""
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fit as F  # noqa: E402

OUT = HERE / "fit_out"
read = [r for n in F.task_names() if n.startswith("read") for r in json.loads((OUT / f"{n}.json").read_text())["resamples"]]
point = json.loads((OUT / "point.json").read_text())
loco = list(csv.DictReader(open(HERE / "fit_loco.csv")))
table = pickle.loads((OUT / "table.pkl").read_bytes())
rows = F.rows_of(table, np.arange(F.N_SEEDS))
pm = point["models"]

print(f"(1) F3 against its line (C <= {F.C_MAX}, i.e. a harmonic-mean interval of at least 120 s; coverage >= "
      f"{F.PASS_N} of {F.N_COND})")
d1 = np.array([float(r["d1_rec"]) for r in loco if r["model"] == "M0"])
print(f"   M0's held-out intervals: D1 {d1.min():.0f}-{d1.max():.0f} s, harmonic mean {len(d1) / np.sum(1 / d1):.1f} s; "
      f"Δ_rec / p̄ = 2^{pm['M0']['beta'][0]:.2f} = {2 ** pm['M0']['beta'][0]:.2f} (full fit), margin h 0 in every fold")
c = np.array([r["M0"]["C"] for r in read])
cov = np.array([r["M0"]["cover"] / r["M0"]["n"] for r in read])
print(f"   C(M0) {pm['M0']['C']:.3f}; over {len(read)} resamples: 5% {np.percentile(c, 5):.3f}, median "
      f"{np.median(c):.3f}, 95% {np.percentile(c, 95):.3f}; C <= 0.5 in {np.mean(c <= F.C_MAX):.0%}, coverage at the line "
      f"in {np.mean(cov >= F.PASS_N / F.N_COND):.0%}, both in {np.mean((c <= F.C_MAX) & (cov >= F.PASS_N / F.N_COND)):.0%}")
print(f"   C over the line by {pm['M0']['C'] - F.C_MAX:+.3f}: the rule's harmonic-mean interval is "
      f"{100 * (1 - (len(d1) / np.sum(1 / d1)) / 120.0):.1f}% shorter than 120 s")

print()
print("(2) held-out misses (kept share below 85%) and the spread of the held-out intervals")
for m in ("M0", "M1", "M2"):
    rr = [r for r in loco if r["model"] == m]
    dd = np.array([float(r["d1_rec"]) for r in rr])
    miss = [f"{r['condition']} (D1 {float(r['d1_rec']):.0f} s keeps {float(r['kept']):.1%})" for r in rr if r["keeps"] == "0"]
    print(f"   {m}: D1 {dd.min():.0f}-{dd.max():.0f} s, {int(np.sum(dd <= 60.01))} clipped at 60 s; misses: "
          + ("; ".join(miss) if miss else "none"))

print()
print("(3) the size slope of log2(Δ* / p̄) on log2(m / 15), M1's term, under four treatments")
size = [r for r in rows if r["label"].startswith("size") or r["label"] == "proc k 1"]   # m15 is B2 (listed as proc k 1)
assert len(size) == 5
for what, rr in (("all 35 conditions", rows), ("size arm m15-m100 (5 levels)", size)):
    for how in ("censored (Tobit, as the fit)", "bounds taken as exact"):
        X, y, cens = F.fit_rows(rr, ("m",))
        f = F.tobit(X, y, cens if how.startswith("censored") else np.zeros(len(y), int))
        print(f"   {what:30s} {how:30s} slope {f['beta'][1]:+.2f}, σ {f['sigma']:.2f}")
print("   axis 6's S_m (§14): −0.51 [90% −0.70, −0.16], size arm, bounds at their value (OLS)")

print()
print("(4) what an overshoot costs: kept share at a multiple of each condition's Δ* (D1 clipped to 3600 s)")
for mult in (1.5, 2.0, 4.0):
    k = [float(F.kept(r, min(F.D1_MAX, mult * r["dstar_d1"]))) for r in rows]
    print(f"   {mult:g} Δ*: median {np.median(k):.1%}, range {min(k):.1%}-{max(k):.1%}")
