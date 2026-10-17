# agv-ratio-twin-v5: the fleet knee in the twin on the v5 rules (2026-10-14)

Status: **done** 2026-10-14 16:19 (local, 17 min, 1,134 twin runs; lab `agv-ratio-twin-v5`). Tables: `analysis.out`, `knees.csv`.

## Why
`../agv-ratio-twin` (10-07) fit the free-flow fleet knee, knee5 ≈ 0.087·M^1.29·(util/0.85)^0.53·(op/360 s)^−0.98.
It used SRT_TECT / SRT_ECT on instances without due dates. When the AGV study moved to the v5 rules on 10-14 (user),
the knee had to be measured under the thesis's rule and objective. That knee also sets the fleets of
`../agv-ratio-unity`.

## Method (`twin_point_v5.py`, `sweep.py`, `launch.sh`, `analyze.py`)
- **Copies of `../agv-ratio-scale/twin_point.py` / `sweep.py`.** The instance adds TWK due dates (c ~ U[1.75, 2.5],
  as B2), the default rule is MDD_PTECT, and total tardiness is in the output. The job lists come out the same
  length as before (the m50 / 480 s seed 0 instance still has 466 jobs).
- **Grid:** the same as agv-ratio-twin plus its 10-08 extension.
  - M = 15 / 50 / 100 (layout D, the same floor exports);
  - util 0.7 / 0.85 / 1.0 × mean op 240 / 480 s;
  - 10 fleets (13 at m100 / 240 s);
  - seeds 0-2;
  - MDD_PTECT and MDD_PECT.
- **Knee:** knee5 / knee2 on mean time in system (as before), and knee5 on total tardiness.
- **Fit:** the same log-linear form, against the old SRT fit. `knees.csv` feeds agv-ratio-unity.

## Results

**The fleet-size law does not change with the v5 rules.** The fit over the 18 cells, pooling seeds and both rules
as agv-ratio-twin did:

  **knee5 ≈ 0.080 · M^1.32 · (util / 0.85)^0.54 · (op / 360 s)^−1.01**, median error 4% (max 16%)

against 0.087 · M^1.29 · (util/0.85)^0.53 · (op/360 s)^−0.98 on SRT_TECT / SRT_ECT without due dates. Fitting
each rule alone gives:
- MDD_PTECT: 0.081 · M^1.31 · …^0.54 · …^−0.96;
- MDD_PECT: 0.077 · M^1.33 · …^0.52 · …^−1.06.

**Cell by cell** (knee5 on mean time in system, mean of 3 seeds × 2 rules):

| M | v5 knee5 against the SRT rules |
|---|---|
| 15 | equal |
| 50 | −3% to +9% |
| 100 | 0% to +13% |

The largest change is m100 / util 0.7 / 240 s: 51.7 against 45.8. No knee sits at the largest fleet run.

**On tardiness the knee is later.** knee5 on total tardiness is 10-50% above the time-in-system knee at util 0.85:

| M / mean op | time in system | tardiness |
|---|---|---|
| 15 / 240 s | 4 | 4.7 |
| 50 / 240 s | 21 | 27 |
| 50 / 480 s | 10 | 13 |
| 100 / 240 s | 45 | 53 |
| 100 / 480 s | 25 | 38 |

These are MDD_PTECT values; tardiness is set by the late tail, which keeps falling past the mean-flow knee.

**The travel term at the knee.** MDD_PECT against MDD_PTECT total tardiness at MDD_PTECT's knee fleet is mostly
+1 to +10%, and up to +31% (m100 / 0.7 / 480 s). It is negative in four cells (−1 to −14%). Near the knee, where
AGVs are scarce, the travel-aware machine rule mostly pays.

**For agv-ratio-unity** (util 0.85, MDD_PTECT knee5): m50 240 s → 21, m50 480 s → 10, m100 240 s → 45, m100 480 s
→ 25. The ~1.5× fleets (30 / 16 / 70 / 40) are close to the tardiness knees, so the Unity check covers both.
