# rq2-twin-fleet: switching headroom at a stationary load, with regime and fleet blocks (twin, 2026-10-04)

**Setup.** `run.py` (H15 greedy oracle, tardiness, 900 s slots, random warm-up, failures off, c ~ U[1.75, 2.5]) at the
load calibrated in `../dev-load-calib` (utilization 0.6-1.2 outside lulls: tardiness per hour levels off after ~3 h).
Seeds 0-39 per setting. `analyze.py` -> `analysis.out`, `per_seed.csv`. switch_% = oracle vs the seed's own best
fixed pair (in-episode headroom).

| setting | switch_% median | seeds > 2% | saved / h (job-s) | best-pair tardiness / h |
|---|---|---|---|---|
| single-cal: 1 regime, 1.5 h | 2.38 | 47.5% | 286 | 8,830 |
| long-cal: 1 regime, 6 h | 8.17 | 82.5% | 990 | 13,620 |
| B0: blocks of c + op mean, 6 h | 8.53 | 90.0% | 1,267 | 14,640 |
| B3: blocks with 2-AGV disruptions (1 in 4) | 7.90 | 90.0% | 1,594 | 20,670 |
| B2: normal / surge / 2-AGV blocks | 7.94 | 90.0% | 1,713 | 27,340 |

**Findings.**
1. **The length effect is real, not the runaway backlog of rq2-twin-blocks:** at a stationary load, 6 h episodes give
   an 8.2% median in-episode headroom vs 2.4% over 1.5 h, and 3.5x the job-seconds saved per hour.
2. **Likely mechanism: the 1.5 h window sits in the warm-up transient.** Tardiness per hour ramps for about 3 h
   after the warm-up (dev-load-calib), so a 1.5 h window sees mostly jobs that are not yet late (best-pair tardiness
   per hour 8,830 vs 13,620). Testable cheaply: 1.5 h windows after a >= 3 h warm-up.
3. **Regime and fleet blocks add variety, not headroom:** B0, B2, B3 are all within 0.6 points of long-cal. The
   oracle's choices respond to the fleet (in 2-AGV slots TECT 57-67% vs 50-54%, EDD 13% vs 5-8%) but switch as often
   within a block as at a boundary (26-30% vs 24-28%): rule choice follows the evolving state.
4. **Per-block hindsight selection** explains most of the blocks gain (blocksel 5.0-6.4% of 7.9-8.5%).
5. In the twin transport binds only at 2 AGVs (no zone blocking); Unity binds at ~4, so Unity fleet blocks should use
   3-4 AGVs.

## Control: is the short-window deficit a warm-up transient? No (rq2-twin-warm, 10-04)
`../rq2-twin-warm/run.py`: single-cal with the warm-up drawn from segment starts >= 2 h / >= 3 h (`min_warmup_seconds`,
horizon 30,600 s), seeds 0-39. Switch_% median 1.81% / 2.54% (vs 2.38% normal warm-up, 8.17% for 6 h), although
best-pair tardiness per hour is now 25,000 job-s (more than long-cal's 13,600). Saved per hour 474 / 516 (vs 990 for 6 h).
**So the window length itself matters, not the start state.** Likely an end-of-window effect: tardiness counts only
inside the window, so a rule choice's later consequences are cut off, and in 1.5 h most of them fall after the end.
Implications: measure headroom on long windows (the RL objective discounts over ~3,000 s with bootstrapping, so it
values those consequences), and train on long episodes (no extra cost per step).
