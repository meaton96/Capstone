# rq2-twin-scen: which training regime gives the tardiness switching oracle the most headroom? (twin, 2026-10-03)

**Setup.** `run.py`: H15 greedy oracle (6 x 900 s) on the event-based twin, tardiness (late-WIP integral), warm-up
windows, machine failures off, grid c 1.75 / 2 / 2.5 x AGVs 4 / 5 / 7 x load base / utilhi (utilization 1.0-1.8,
lulls 0.4-0.7), seeds 0-39 each (720 tasks, code `587816add471`, 10-03 22:45-23:45). `analyze.py` -> `analysis.out`,
`regimes.csv`. total_% = oracle vs the regime's best-on-average fixed pair; switch_% = vs each seed's own best pair.

**Result.** Every one of the 18 regimes clears the go bar (median >= 3%, >= 60% of seeds above 2%): medians 3.4-13.5%.
- **c:** the percentage rises with c (median over regimes 4.3 / 6.2 / 10.3% at c 1.75 / 2 / 2.5), but the absolute
  saving does not: c = 2 saves the most job-seconds (1,320-2,500 per window) because tardiness at c = 2.5 is about
  half as large (14 vs 25 thousand job-s on base load). Part of the c = 2.5 percentage is a smaller base.
- **Load:** base 6.5% vs utilhi 4.6% median, but utilhi saves more in absolute terms (about 1,700-2,500 job-s).
- **AGVs:** 4 > 5 > 7 (7.2 / 6.2 / 4.6%). The twin has no zone blocking, so the Unity agv4 check is the better guide.
- **Switching alone** (vs each seed's own best pair) is 1-3% everywhere; the rest of total_% is the best pair
  changing from seed to seed (MDD-ECT vs MDD-TECT vs SRT-TECT), which a policy that reads the state can also use.
- **c = 2, 7 AGVs, base** reproduces rq2-twin-due (6.47% vs 6.5% median).
- **Rule use moves with c the way the literature says:** oracle job-rule shares at c = 1.75 SRT 44 / MDD 41 / ATC 9%;
  at c = 2.5 MDD 46 / SRT 25 / ATC 24%. EDD (3-4%) and SPT (1-4%) stay marginal in every regime.

**Recommendation for the training mix:** c uniform over [1.75, 2.5] and load base / utilhi 50:50 (the lever ranges all
pay, and mixing them makes the best rule change between episodes). Confirm in Unity (machine failures, zone
blocking) before training; the twin overstated Unity's gain about 2x at c = 2 (6.5 vs 3.4%).
