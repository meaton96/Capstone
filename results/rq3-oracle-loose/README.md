# rq3-oracle-loose: the switching-oracle ceiling at loose due dates (2026-10-14)

Status: **done** 2026-10-15 00:08 (cluster array 21894598, 34 min, 40 / 40; lab `rq3-oracle-loose`). `analysis.out`.

## Why
This is the ceiling for `../eval-v5-cmix` at due-loose (c ~ U[2.5, 3.5]): how much a policy that knew the future
could gain there. On B2 itself it is −9.4% vs MDD-PTECT (rq2-rebase-v5).

## Method (`run.py`, `submit.sh`, `analyze.py`)
- **Oracle:** rq2-rebase-v5's greedy v5 switching oracle (6 pairs, 900 s slots, hold-to-end tails), with its frozen
  code (`rq2-rebase-v5/code/env` + its `sim.py`).
- **Instances:** built through evaluate.py's own generator call with `--params` at c [2.5, 3.5].
- **Checks:**
  - At B2's own c, MDD-PTECT on seed 0 gives 95.842006 (train-v5-nokl/eval: 95.842007), locally and on the cluster.
  - At loose c, MDD-PTECT on seeds 0 and 7 equals rq3-gen-v5's due-loose exactly (31.1019, 304.9617).
- **Seeds:** B2 test seeds 0-39.

## Results (B2 test seeds 0-39, c ~ U[2.5, 3.5])

| vs MDD-PTECT (mean 69.6 k job-s per instance) | pooled | 95% CI | seeds better / worse | saved per instance |
|---|---|---|---|---|
| switching oracle | **−14.1%** | [−20.6, −9.7] | 40 / 0 (p 2e-12) | 9.8 k job-s |
| hindsight best v5 pair | −8.4% | [−13.8, −4.9] | 35 / 0 | 5.9 k job-s |

**Oracle slot use:**

| pair | share of slots |
|---|---|
| ATC-PECT | 35% |
| MDD-PECT | 24% |
| MDD-PTECT | 17% |
| ATC-PTECT | 14% |
| SRT-PECT | 8% |
| SRT-PTECT | 1% |

## Reading
- **The relative ceiling is half again higher with loose due dates:** −14.1% against −9.4% at B2's c (rq2-rebase-v5).
- **The absolute ceiling is lower:** 9.8 k job-s per instance against about 13.7 k at B2's c (−9.4% of 145.5 k). With
  loose due dates fewer jobs are late (22% against 44%), so there is less tardiness to remove. Each saving is a
  larger share of it.
- **The oracle leans on ATC** (49% of slots, against MDD's 41%). ATC's slack-based look-ahead pays when there is
  slack to manage. This is the job-rule shift rq3-gen-v5 saw: ATC-PTECT is the best fixed pair at load ×0.8.
- **For `../eval-v5-cmix`:** BC's −5.2% at loose c is about 37% of this ceiling; on B2 the realizable share is about
  20% (−1.8 of −9.4).
