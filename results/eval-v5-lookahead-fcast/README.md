# eval-v5-lookahead-fcast: the look-ahead PPO on a perturbed published schedule (2026-10-15, user)

Status: **done** 2026-10-15 22:40 (cluster array 21899301, 54 min, 60 / 60; lab `eval-v5-lookahead-fcast`). `analysis.out`, `summary.csv`.

## Why
eval-v5-lookahead (10-15) found a small, significant look-ahead gain on 200 held-out B2 seeds. The look-ahead family
was −2.02% vs MDD-PTECT against −1.25% for the state-only family (paired −0.78 points, p 0.0005). The features came
from the exact future schedule, so this is an upper bound. The user (10-15) asked for the perturbed-forecast test
planned in `../train-v5-lookahead/README.md`: jitter 1 / 10 / 30 min and churn 10 / 25%, against the twin planner at
the same error.

The planner on a perturbed schedule (`../rq2-perturbed-v5`) drops from −9.4% (exact) to −2 to −4%. The question is
whether the policy's 0.8 points hold up the same way.

## Method (`run_task.py`, `submit.sh`, `analyze.py`)
- **Policies:** the 5 train-v5-lookahead checkpoints, unchanged. Only the 15 look-ahead features change; the policy
  acts in the real instance.
- **Forecast model:** rq2-perturbed-v5's `Forecast` class, imported rather than copied, with the same RNG keys. At a
  given (seed, level) the policy sees exactly the published schedule the planner planned on. Levels:
  - `jit1` / `jit10` / `jit30`: arrival + N(0, σ) with σ 1 / 10 / 30 min. The due date moves with the arrival. Errors
    are drawn once per job; a published arrival already in the past shows as now + 1 s.
  - `churn10` / `churn25`: 10 / 25% of the real jobs are missing from the plan, and as many phantom jobs are in it
    (from B2 instance 70000 + seed). Both levels also have 10 min jitter.
  - `redrawn`: the whole future is replaced by instance 70000 + seed. This is the no-information control, added
    beyond the user's list: it separates "the policy uses the forecast" from "look-ahead training made a better
    policy".
  - `exact`: comes from eval-v5-lookahead (not rerun).
- **Adapter:** a real job leaves the published schedule when it arrives in the twin; a phantom leaves when its published
  arrival has passed. `des_twin.lookahead` reads the schedule through the frozen code's existing published-plan path
  (`twin.unc.plan_future_jobs`, given a stand-in). The frozen code is not edited.
- **Settings:** eval-v5-lookahead's: frozen code `../rq2-rebase-v5/code/env`, floor `dev-popart-smoke/floor7`, B2
  params, random warm-up, 21,600 s window, 900 s slots, deterministic.
- **Seeds:** fitting 2000-2159 and test 0-39 (both held out from PPO training). The planner exists only on the test
  seeds.
- **Score (`analyze.py`):** per level, pooled over the 200 seeds:
  - the look-ahead family vs MDD-PTECT;
  - the family vs the state-only family (eval-v5-final), paired;
  - the family vs its own exact-schedule run, paired;
  - the share of the exact gain it keeps;
  - and the pick s3.

  On the test seeds, it also gives the family next to the planner at the same level, re-planning every 900 s and,
  for jit10 / churn10, every 300 s.

## Checks (local, 10-15)
- **Exact level through the adapter:** reproduces eval-v5-lookahead bit for bit (s3, test seeds 0-2: 92,585.77 /
  22,877.71 / 245,227.89 job-s).
- **Published view vs the planner's `Forecast.episode(t)`:** same jobs, arrivals and due dates within 0.5 s (float32
  rounding). Checked for levels jit1 / jit30 / churn10 / churn25 / redrawn, seeds 0 / 7 / 2003, 8 times each.
- **The features change under the perturbation.** On seed 2, the mean absolute feature change vs exact is 0.10-0.18
  for redrawn and 0.01-0.03 for jit10.
- **Early hint, s3 only:** the policy rarely reacts. Redrawn changes the outcome on 1 of 12 test seeds (seed 8,
  +1.7%), jit10 on 1 of 3, churn25 on 0 of 3. On seed 2, s3 picks the same pair in every slot with exact and with
  redrawn features. If that holds across seeds and checkpoints, the look-ahead gain came from training (the actor and
  critic share the fused features) and not from forecast use at decision time.
- Smoke outputs are in `smoke/`.

## Results

**Pooled over all 200 held-out seeds** (look-ahead family = per-instance mean of the 5 seeds; MDD-PTECT 152.3 k job-s
per instance):

| published schedule | look-ahead family vs MDD-PTECT | vs state-only family (paired) | vs its own exact run (paired) | share of the exact gain kept |
|---|---|---|---|---|
| exact (eval-v5-lookahead) | −2.02 [−2.84, −1.28] | −0.78 [−1.20, −0.39] | – | 100% |
| arrivals ± 1 min | −2.02 [−2.81, −1.27] | −0.78 | +0.00 [−0.02, +0.03], 2 / 5 differ | 100% |
| arrivals ± 10 min | −2.04 [−2.86, −1.31] | −0.81 | −0.02 [−0.07, +0.02] | 101% |
| arrivals ± 30 min | −1.99 [−2.79, −1.22] | −0.75 | +0.03 [−0.04, +0.11] | 98% |
| 10% churn + 10 min | −2.01 [−2.82, −1.26] | −0.77 | +0.01 [−0.04, +0.08] | 99% |
| 25% churn + 10 min | −2.01 [−2.84, −1.22] | −0.77 | +0.01 [−0.07, +0.10] | 99% |
| **redrawn (no information)** | **−2.07** [−2.94, −1.27] | −0.83 | **−0.05** [−0.23, +0.12], p 0.53 | 102% |

The pick s3 stays between −3.46 and −3.57% at every level.

**Test seeds 0-39 vs MDD-PTECT, next to the planner on the same published schedule:**

| published schedule | look-ahead family | planner, re-plan 900 s | planner, re-plan 300 s |
|---|---|---|---|
| exact | −1.42 [−2.91, −0.07] | −9.42 | −11.24 |
| ± 1 min | −1.41 | −4.07 | – |
| ± 10 min | −1.45 | −2.19 | −2.67 |
| ± 30 min | −1.33 | −2.77 | – |
| 10% churn | −1.31 | −2.23 | −3.30 |
| 25% churn | −1.32 | −2.31 | – |
| redrawn | −1.18 | −1.21 (n.s.) | – |

**Training-seed check** (pooled % vs MDD-PTECT on the 200 seeds, one value per trained policy):
- look-ahead s0-s4: −2.37 / +0.29 / −3.05 / −3.48 / −1.49 (mean −2.02, sd 1.49);
- state-only s0-s4: −0.86 / −0.47 / −2.08 / −0.74 / −2.08 (mean −1.25, sd 0.77);
- Welch t p 0.34, Mann-Whitney p 0.31.

## Reading
1. **The look-ahead policy does not use the forecast.** Its score doesn't move under any error, and even a
   completely unrelated future (redrawn) gives −2.07% against −2.02% with the exact one. Paired, the difference is
   −0.05 [−0.23, +0.12]. The features change, by 0.10-0.18 on average for redrawn, but the policy's decisions almost
   never do.
2. **The look-ahead family's 0.78-point edge over the state-only family is training-seed noise.** The paired
   p 0.0005 in eval-v5-lookahead treats the 5 trained policies as fixed. Across training seeds the two families
   overlap (p 0.31-0.34): look-ahead s1 is the worst policy of all ten, and s2 / s3 the best. Whatever made s2 / s3
   good, it is not forecast use at decision time.
3. **The planner and the policy now agree on how much a realistic forecast is worth: little.** At 10-30 min error,
   the planner keeps −2.2 to −2.8% (−2.7 / −3.3% at 300 s re-plans), and the look-ahead policy gets the same −1.3 to
   −1.5% on the test seeds that it gets with no information.
4. **For the thesis (RQ2 / RQ3, information):**
   - Exact future knowledge is worth up to −9.4 to −11.2% (planner, hindsight).
   - At realistic error, about −2 to −3% for the planner.
   - The look-ahead PPO with 15 summary features gets none of that value. Its performance doesn't depend on the
     forecast at all.
   - This replaces eval-v5-lookahead's "the look-ahead helps by 0.8 points", which should not be reported. The
     rq3-lookahead-probe (S6, n.s.) and dev-policy-lookahead (prior + KL) results now agree with the v5 stack.
5. **Next, only if the information question stays in scope:** the README of `../train-v5-lookahead` names a
   future-jobs encoder (the next N arriving jobs as a set) as the next step. Given the planner's −2 to −3% ceiling at
   realistic error, the most it could add is about 1-2 points.
