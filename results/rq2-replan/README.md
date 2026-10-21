# rq2-replan: when should the twin planner re-plan? (2026-10-21, user after the advisor meeting)

Status: **queued 10-21 (user)** as lab `rq2-replan`, a future that starts locally (12 workers, priority 5) when
`rq3-alpha-size` finishes (about 15:20); expected done about 04:00 on 10-22.

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

## Results
Pending.
