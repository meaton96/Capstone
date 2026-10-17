# eval-v5-lookahead: the look-ahead PPO against the state-only PPO, the oracle and the planner (2026-10-15)

Status: **done** 2026-10-15 19:54 (cluster job 21898928, 52 min, 10 / 10; lab `eval-v5-lookahead`). `analysis.out`.

## Method (`run_task.py`, `submit.sh`, `analyze.py`)
- **Policies:** the 5 train-v5-lookahead checkpoints, on B2 fitting seeds 2000-2159 and test seeds 0-39. Same frozen
  code and settings as `../eval-v5-final`; evaluate.py computes the exact-future look-ahead features for these
  checkpoints.
- **References on the same seeds:**
  - MDD-PTECT and the state-only 5-seed PPO (eval-v5-final);
  - the greedy switching oracle at 900 s (`../rq2-oracle-bc-v5` labels in `../rq2-rebase-v5/out/oracle`);
  - on the test seeds, the twin planner on the exact schedule re-planning every 300 s (`../rq2-perturbed-v5`).
- **Score:** pooled window tardiness vs MDD-PTECT with CIs, Wilcoxon, the share of the oracle's gain, and look-ahead
  vs state-only paired per instance.
- **Check:** the references already load: oracle −10.7% on the fitting seeds and −9.4% on the test seeds; the
  state-only PPO keeps 14% / 4% of the oracle's gain there.

## Results

**Pooled over all 200 held-out seeds** (fitting 160 + test 40, B2 base c; family = per-instance mean of 5 seeds; MDD-PTECT
152.3 k job-s per instance):

| policy | vs MDD-PTECT | better / worse | k job-s saved per instance | share of the oracle's gain |
|---|---|---|---|---|
| look-ahead PPO, mean of 5 | **−2.02** [−2.84, −1.28], p 3e-9 | 141 / 59 | 3.08 | 19% |
| state-only PPO, mean of 5 (eval-v5-final) | −1.25 [−2.02, −0.51], p 2e-4 | 122 / 78 | 1.90 | 12% |
| switching oracle (900 s, hindsight) | −10.44 | 199 / 1 | 15.89 | 100% |
| **look-ahead vs state-only, paired** | **−0.78** [−1.19, −0.38], p 0.0005 | 114 / 86 | | |

Look-ahead seeds on all 200: s0 −2.4, s1 +0.3 (n.s.), s2 −3.1, s3 −3.5, s4 −1.5%.

**Per seed set** (vs MDD-PTECT; `analysis.out`):

| | fitting 2000-2159 | test 0-39 |
|---|---|---|
| look-ahead mean of 5 | −2.16 [−3.12, −1.29] | −1.42 [−2.90, −0.10] |
| state-only mean of 5 | −1.46 [−2.35, −0.63] | −0.34 [−1.75, +1.02] |
| look-ahead vs state-only, paired | −0.71 [−1.15, −0.27], p 0.006 | −1.08 [−2.06, −0.14], p 0.02 |
| best look-ahead seed | s3 −3.75 | s2 −3.01 |
| oracle / exact-schedule planner | −10.7 / – | −9.4 / −11.2 |

## Reading
> **Superseded 10-15 22:40 by `../eval-v5-lookahead-fcast`.** The policies give the same score with a redrawn
> (unrelated) future (−2.07% vs −2.02%), so they don't use the forecast. Across training seeds, the look-ahead
> family's edge over the state-only family is not significant (Welch p 0.34, Mann-Whitney p 0.31). Do not report
> point 1 below.

1. **(Superseded) Yes, the policy uses the future arrivals, but only a little.** Adding the 15 exact-future summary features moves the
   5-seed family from −1.25% to −2.02% (paired −0.78 points, p 0.0005, the same sign on both seed sets). It captures
   19% of the oracle's gain instead of 12%.
   - This is the first look-ahead gain in the project. On the prior + KL stack (dev-policy-lookahead, 10-08) the policy
     stayed at MDD-TECT, and the S6 classifier probe was n.s. The stack, not the features, blocked it then.
2. **It is an upper bound on forecast value.** The features come from the exact realized schedule. The exact-schedule
   planner reaches −11.2% with the same information, so the summary features carry only a small part of what that
   information is worth.
3. **Seed spread is wide** (+0.3 to −3.5%); s1 learned nothing useful. Report the 5-seed family, with the pick on the
   fitting seeds (s3) as the deployable policy.
4. **About the same size as the due-date training effect.** The state-only policy trained on mixed c (eval-v5-cmix) is
   −2.1% at base c; this one, trained at base c, is −2.0%. The two have not been combined.

## Next
- Perturbed-forecast test (planned in `../train-v5-lookahead/README.md`): the same checkpoints with features from a
  perturbed schedule (rq2-perturbed-v5's forecast model: 1 / 10 / 30 min jitter, 10 / 25% churn), against the planner
  at the same error. The planner drops from −9.4% to −2 to −3% under realistic error, so expect most of the 0.8 points
  to shrink.
- Optional: look-ahead + mixed-c training (both levers at once).
