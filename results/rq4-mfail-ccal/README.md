# rq4-mfail-ccal: machine failures with due dates calibrated to the failures (50 machines, twin, 2026-10-14, user)

Status: **done** 2026-10-15 00:43 (stage 1 `rq4-mfail-ccal-calib`, array 21894611; stage 2 `rq4-mfail-ccal`, array
21894781, 12 / 12). `analysis.out`.

## Why
- On the 50-machine floor, uniform machine failures (availability 0.90) tripled tardiness at B2's c ~ U[1.75, 2.5]:
  MDD-TECT 158 → ~520 ks (dev-hetero-tune); MDD-PTECT 57.7 → 172.8 ks on seed 0.
- Almost every job is late (the late share on seed 0 goes from 32% to 54%), which takes away what due-date rules work
  with.
- The user (10-14): due dates should be tuned for the failure regime. The fair comparison keeps the late share as it
  is without failures.

## Method (`mfail.py`, `submit_calib.sh`, `submit_compare.sh`, `analyze.py`)
- **Harness:** dev-pect-rule's m50 family (`harness.py` / `hetero.py`: 50 machines, 26 AGVs, load 0.9, random warm-up,
  6 h window, 1.5 h blocks, uniform Weibull failures at 0.90 availability, `calib_models.json`).
- **Code:** frozen v5 code `rq2-rebase-v5/code/env` (6d13412a). Check: seed 0 MDD-PTECT gives 57.742968 without
  failures and 172.838962 with them, exactly dev-pect-rule's, locally and on the cluster.
- **Only the due-date range changes:** (1.75 S, 2.5 S).
- **Late share:** jobs due inside the window that exit after their due date or are still open at the window end,
  over all jobs due in the window.
- **Stage 1 (calib):** S ∈ {1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0}, failures on, MDD-PTECT, calibration seeds 1000-1029,
  plus no failures at S = 1. S* is the linear interpolation where the failure late share equals the failure-free one
  (`cscale.json`).
- **Stage 2 (compare):** on test seeds 0-39, the 6 v5 pairs (quad α 3) + MDD-TECT in three settings: no failures (S 1),
  failures at B2's c (S 1, the old "3×" setting), and failures at S*.
- **Read:**
  - tardiness level and late share per setting;
  - each rule vs MDD-PTECT and the ranking's Kendall tau against no failures;
  - the hindsight best pair;
  - whether the due-date rules (MDD / ATC) regain an edge once due dates are realistic for the failure regime.

## Results

### Stage 1: calibration (seeds 1000-1029, MDD-PTECT; `cscale.json`)

Without failures: late share 27.3%, mean window tardiness 122.2 ks.

| c scale S (c range) | late share with failures | mean tardiness with failures (ks) |
|---|---|---|
| 1.0 (1.75-2.5, B2) | 50.6% | 381.7 (3.1× no failures) |
| 1.25 (2.19-3.13) | 32.0% | 242.0 |
| 1.5 (2.63-3.75) | 20.6% | 148.6 |
| 1.75 | 12.9% | 89.2 |
| 2.0 | 8.0% | 48.9 |
| 2.5 | 2.4% | 10.8 |
| 3.0 | 0.6% | 2.5 |

**S\* = 1.353**, so the failure regime gets c ~ U[2.37, 3.38]. That brings the late share back to 27.3%.
Interpolating, mean tardiness there is about 200 ks, about 1.6× the failure-free floor: the same share of jobs is
late, but each by more.

### Stage 2: comparison (test seeds 0-39, 50 machines)

| setting | c range | MDD-PTECT tardiness | late share | ranking τ vs no failures | hindsight best v5 pair |
|---|---|---|---|---|---|
| no failures | 1.75-2.5 | 158.2 ks | 32.8% | – | −4.6% |
| failures, B2's c | 1.75-2.5 | 474.7 ks (**×3.0**) | 55.1% | +0.62 (p 0.07) | −3.6% |
| failures, calibrated c | 2.37-3.38 | 268.4 ks (**×1.7**) | 30.9% | **+1.00** (p 0.0004) | −5.7% |

**Pairs vs MDD-PTECT** (pooled, seed-bootstrap 95% CI; MDD-PTECT is the best pair in every setting):

| pair | no failures | failures, B2's c | failures, calibrated c |
|---|---|---|---|
| MDD-PECT | +2.0 [+0.6, +3.7] | +1.6 [−0.3, +3.6] | +1.3 [−1.6, +3.8] |
| ATC-PTECT | +2.6 [−2.4, +8.3] | +7.5 [+2.4, +12.0] | +4.8 [−3.0, +10.8] |
| ATC-PECT | +4.6 [−0.1, +10.2] | +9.1 [+4.2, +13.0] | +5.4 [−3.2, +12.1] |
| SRT-PTECT | +8.0 [+5.4, +11.4] | **+2.5** [+0.6, +5.0] | +14.2 [+10.4, +20.9] |
| SRT-PECT | +10.5 [+7.7, +14.0] | **+3.0** [+0.9, +5.7] | +15.3 [+10.8, +22.6] |
| MDD-TECT (H15) | +29.9 | +26.2 | +42.2 |

## Reading
1. **Most of the "failures triple tardiness" result was the due dates.**
   - With B2's c, the failures make 55% of jobs late. The floor then behaves like a flow-time problem: SRT, a
     shortest-remaining-work rule that ignores due dates, climbs from 8-10% behind to 2.5-3% behind. The
     slack-based ATC falls from 2.6-4.6% to 7.5-9.1% behind.
   - That ranking shift (τ 0.62) is an artefact of due dates that ignore 10% downtime.
2. **With due dates calibrated to the failure regime, failures do not change the rule choice at all.**
   - The calibration scales c by 1.35, so the late share is back to 31% against 33%.
   - The ranking of the 7 rules is identical to the failure-free one (τ +1.00). MDD-PTECT stays best, MDD-PECT is
     next, then ATC, SRT and MDD-TECT.
   - Failures still cost ×1.7 tardiness, because late jobs are later.
3. **The switching headroom grows under calibrated failures** (hindsight best pair −5.7% against −4.6% without
   failures). The processing-time rules' edge over the old H15 pair widens (MDD-TECT +42% against +30%).
4. **For the thesis (RQ4, machine failures):** report the calibrated setting as the main result. Machine failures at
   10% downtime cost about 70% more tardiness, but they leave the best rule and the full ranking unchanged once due
   dates reflect the failure regime. Without that calibration, the floor saturates with late jobs and the comparison
   turns into flow time: SRT catches up and ATC falls back. Every due-date study with disruptions should calibrate c
   to the disruption regime.
