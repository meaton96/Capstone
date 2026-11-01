# rq2-replan: when should the twin planner re-plan? (2026-10-21, user after the advisor meeting)

Status: **done 10-21 22:26 (lab `rq2-replan`, local; 1,640 / 1,640 after rerunning 5 edge-case tasks).**

## Why
The advisor asked why the planner re-plans every 5 minutes and the oracle uses 15-minute slots. Both values were
inherited:
- the oracle's slot length was only ever swept upward (15 / 30 / 60 / 180 min, H15 rules, `../rq2-oracle-slotlen`);
- the planner was compared only at 5 vs 15 minutes (H15, 10-minute arrival error: −4.0% vs −1.6%, `../rq2-head-to-head`).

The user's hypotheses:
1. finer ("continuous") decisions are best in theory but cost compute;
2. the planner only needs to re-plan when the incoming schedule changes or a disruption occurs.

One distinction shapes the design. The planner's interval does two jobs:
- it sets how often the rule pair may change (switch granularity);
- it sets how often the plan is recomputed (re-planning).

The planner is myopic: each re-plan picks only the next pair, scoring candidates as if held to the window end. Even
with a perfect forecast, it therefore has to re-plan in order to switch at all. Re-planning only on schedule changes
would make it hold one pair all episode under a perfect forecast, which is about the per-instance best pair (−4.5%
instead of −9.4%). The hypothesis fits a planner that commits to a whole sequence of pairs, and it fits the error case,
where re-planning corrects drift. Part 1c measures exactly this.

## Design (`run.py`)
- **Planner v2** (`des_twin.planner`): fork the running twin at each re-plan, roll each of the 6 v5 pairs out to the
  window end, keep the best. Harness and frozen code from `../rq4-uncert-screen` (`uc.py`, code/ = ba26f3f5).
  B2 test seeds 0-39, window tardiness vs MDD-PTECT.
- **1a, granularity with an exact forecast:** Δ ∈ {1, 2, 3, 5, 10, 15, 20, 30, 60} min. With the exact future the
  planner is the greedy switching oracle at decision interval Δ.
- **1b, forecast error × interval:** the published schedule of `../rq2-perturbed-v5` (arrival jitter σ 1 / 10 / 30 min,
  10% churn with 10-min jitter, 10% duration errors), Δ ∈ {2, 5, 15, 30} min.
- **1c, event-triggered re-planning:** switches allowed every 5 min, but the plan is recomputed at a boundary only if an
  event happened since the last re-plan; otherwise the current pair is kept (the last plan's hold-to-end choice).
  Triggers:
  - `ev2` / `ev10`: an arrival off its published time by more than 2 / 10 min, an unannounced arrival, or a published
    job overdue by more than the threshold (late or cancelled);
  - `hyb10`: `ev10`, or 30 min since the last re-plan.

  Levels: exact, jit10, jit30, churn10. The first boundary always plans.
- **41 variants × 40 seeds = 1,640 tasks.**

## Checks (10-21, seed 0, scratchpad)
Planner v2 here reproduces the stored v1 runs of `../rq2-perturbed-v5` exactly, with the same tardiness and the same
pair schedule:

| variant here | stored v1 run | window tardiness (k job-s) | v2 / v1 seconds |
|---|---|---|---|
| exact, 15 min | exact (= the greedy oracle) | 79.971484 = 79.971484 | 112 / 281 |
| exact, 5 min | exact-slot300 | 81.378671 = 81.378671 | 340 / 849 |
| jit10, 5 min | jit10-slot300 | 80.482761 = 80.482761 | 340 / 823 |
| churn10, 5 min | churn10-slot300 | 90.981322 = 90.981322 | 348 / 828 |
| proc10, 5 min | proc10-slot300 | 84.108115 = 84.108115 | 344 / 788 |

MDD-PTECT = 95.842006 in all of them.

Event-triggered smoke (seed 0):
- `jit10-d300-ev10` re-planned at 57 of 72 boundaries (80.579);
- `exact-d300-hyb10` re-planned only on its 30-min floor, 12 times (80.772).

## Cost (measured)
About 4.7 s per re-plan with the machine loaded (16-process α sweep running); a task's cost is about proportional
to its number of re-plans. Upper bound with every boundary re-planning: 3,132 re-plans per seed × 40 seeds ≈ 164
core-hours. Event triggers re-plan less. One 1-minute task (exact, seed 0) took 1,574 s: 360 re-plans at 4.4 s.
Seed 0 at 1 / 5 / 15 min: 81.297 / 81.379 / 79.971 k job-s (one seed; finer is not better there).

## Analysis (`analyze.py`)
- **1a:** % vs MDD-PTECT against Δ, with compute (re-plans, planning seconds).
- **1b:** a level × Δ table, and the share of the exact ceiling at the same Δ.
- **1c:** each trigger against periodic re-planning every 5 min (paired), with re-plan counts.

Outputs: analysis.out, summary.csv, curve.csv (figure data, shared with `../eval-v5-slot`).

## Results (10-21 / 10-22)
B2 test seeds 0-39, window tardiness vs MDD-PTECT, pooled (95% seed-bootstrap CI, seeds better / worse); < 0 better.
`analysis.out`, `summary.csv`, `curve.csv`.

**Run note:** 1,635 / 1,640 tasks finished in the first pass (8 h 19 min). 5 tasks at 1 and 2 min failed on an edge case:
at the last re-plan boundary a fork could reach the window end with no agent decision left, and planner v2's tail
function raised StopIteration. Such a tail is now scored as the end state, which is the same for every pair. The fix
only touches tasks that crashed. The 5 reruns finished 10-21 22:26.

### 1a: decision interval with an exact forecast (the greedy switching oracle)

| interval | 1 min | 2 min | 3 min | 5 min | 10 min | 15 min | 20 min | 30 min | 60 min |
|---|---|---|---|---|---|---|---|---|---|
| % vs MDD-PTECT | −11.46 | −11.37 | −11.03 | **−11.24** | −9.68 | −9.42 | −9.12 | −7.99 | −7.77 |
| switches / window | 8.2 | 7.7 | 6.8 | 6.0 | 5.0 | 4.3 | 3.8 | 2.7 | 1.7 |
| planning s / instance | 1,098 | 553 | 356 | 220 | 118 | 80 | 57 | 39 | 21 |

All better than MDD-PTECT on 38-40 / 40 seeds. Paired:
- 5 vs 15 min: −2.01% [−3.52, −0.34], 30 / 8, p 0.0002;
- 15 vs 60 min: −1.78% [−2.86, −0.77], 30 / 7, p 0.0004;
- 1 vs 5 min: −0.25% [−1.20, +0.64], p 0.31;
- 2 vs 5 min: −0.14%, p 0.46.

**Answer to "why 5? why 15?"** The value of finer decisions rises from 60 minutes down to 5 minutes and then
saturates. 1-3 minute decisions add nothing measurable at 2.5-5x the compute. "Continuous" decisions are not better
for this planner. It switches only 6-8 times per 6-hour window even when it may switch every minute; the useful
switching happens at the scale of load regimes, not minutes. 5 minutes is therefore the right interval for this planner
on B2, and 15 minutes (the old oracle slot) gives up about 2 points.

### 1b: forecast error × interval (% vs MDD-PTECT; share of the exact ceiling at the same interval)

| published schedule | 2 min | 5 min | 15 min | 30 min |
|---|---|---|---|---|
| arrivals ± 1 min | −5.04 (44%) | −3.90 (35%) | −4.07 (43%) | −2.14 (27%) |
| arrivals ± 10 min | −2.85 (25%) | −2.67 (24%) | −2.19 (23%) | −1.35 (17%) |
| arrivals ± 30 min | −1.75 (15%) | −2.93 (26%) | −2.77 (29%) | −2.48 (31%) |
| 10% churn (+ 10 min) | −2.39 (21%) | −3.30 (29%) | −2.23 (24%) | −2.94 (37%) |
| durations ± 10% | −5.64 (50%) | −5.78 (51%) | −3.38 (36%) | −1.70 (21%) |

- **With arrival errors of 10 minutes or more, the interval hardly matters** (2 vs 5 min: −0.18%, p 0.48 at ±10 min;
  +1.22%, p 0.79 at ±30 min; 5 vs 15 min at ±10 min: −0.49%, p 0.77). The planner keeps 15-37% of the exact ceiling
  whatever the interval.
- **With duration errors, finer re-planning does help** (2 vs 15 min: −2.35% [−4.41, −0.48], p 0.017): durations are
  revealed as operations finish, so re-planning sooner uses fresher information. Arrival errors are only resolved when
  the job arrives, and re-planning more often does not reveal them sooner.

### 1c: event-triggered vs periodic re-planning (switches allowed every 5 min)

| published schedule | periodic (72 re-plans) | ev2 | ev10 | hyb10 (ev10 or every 30 min) |
|---|---|---|---|---|
| exact | −11.24 | −4.51, 1 re-plan (+7.6 vs periodic, p 2e-12) | same as ev2 | −7.99, 12 re-plans (+3.7, p 2e-5) |
| ± 10 min | −2.67 | −1.58, 65 (+1.1, p 0.10) | **−2.81, 51 (−0.1, p 0.92)** | −2.76, 52 (−0.1, p 0.99) |
| ± 30 min | −2.93 | −1.73, 68 (+1.2, p 0.003) | −1.97, 66 (+1.0, p 0.004) | −1.97, 66 (+1.0, p 0.004) |
| 10% churn | −3.30 | **−3.50, 66 (−0.2, p 0.43)** | −2.37, 58 (+1.0, p 0.06) | −2.37, 58 (+1.0, p 0.06) |

- **With an exact forecast, re-planning only on events fails as predicted:** nothing ever deviates, so the planner plans
  once and holds that pair: −4.5% (= the per-instance best pair) instead of −11.2%. A myopic planner re-plans to switch,
  not to react.
- **With forecast errors, events fire at most boundaries anyway** (51-68 of 72 with these thresholds), so triggers save
  only 6-29% of the re-plans. At best they match periodic re-planning: ±10 min with a 10-min threshold, 29% fewer
  re-plans. At worst they cost about 1 point: ±30 min, where skipping a boundary also removes a chance to switch.
  They never beat periodic re-planning.

### Reading
- For this planner on B2, re-plan every 5 minutes. Finer adds compute without value; coarser loses about 2 points per
  step (5 → 15 → 60 min).
- The user's "re-plan only on change" holds for a planner that commits to a whole sequence of pairs, not for this
  myopic one. Event triggers are a compute saving at best, and a small one, because realistic forecast errors make
  events frequent.
- Forecast quality matters far more than the interval: arrival errors of 10+ minutes cost two-thirds of the gain at
  every interval. That is the same message as `../rq2-perturbed-v5`, now shown across intervals.
- Classic rescheduling work (periodic vs event-driven vs hybrid policies; e.g. Church & Uzsoy 1992, Vieira, Herrmann &
  Lin 2003; to verify before citing) asks the same question for schedules. This is the rule-selection twin version,
  measured against a strong rule. It fits paper C's twin half.
