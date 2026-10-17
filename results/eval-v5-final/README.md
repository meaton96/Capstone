# eval-v5-final: choose the final PPO policy without the test seeds (2026-10-14)

Status: **done** 2026-10-14 21:29 (local, 26 min; lab `eval-v5-final`). `analysis.out`, `selected.txt`.
The first attempt failed after all 20 chunks: the script was then called `select.py`, which shadowed Python's
`select` module. Renamed to `select_final.py` and rerun.

## Why
The final policy for RQ2-RQ4 must be chosen without the test seeds 0-39 (Demo 1 slide 17; THESIS_RESULTS_INDEX
"Final policy selection"). The five v5 PPO seeds (train-v5-nokl s0-s2, train-v5-final s3-s4) differ by about as much
as their effect.

## Method (`run.sh`, `select_final.py`)
- **Policies:** the 5 PPO checkpoints, plus MDD-PTECT and MDD-PECT. Twin, B2 settings of
  `results/scripts/dev_b2_slot_train_eval.sh`. Slot length and discount horizon come from the checkpoints (900 s,
  5,400 s), deterministic. Code: `../rq2-rebase-v5/code/env` (6d13412a, the checkpoints' code).
- **Fitting seeds 2000-2159.** rq2-rebase-v5 chose MDD-PTECT as the best fixed pair on these seeds. The final policy
  is the PPO seed with the lowest pooled window tardiness there. `selected.txt` holds `s<k> <checkpoint>` and is
  read by rq3-gen-v5, rq3-gen-v5-unity and the RQ4 analysis.
- **Test seeds 0-39:** every seed, the pick, and the per-instance mean of the 5 seeds against MDD-PTECT (seed
  bootstrap CI, Wilcoxon).
- **Checks:** s0-s2 and MDD-PTECT must reproduce `../train-v5-nokl/eval` exactly, and s3-s4 must reproduce
  `../train-v5-final/eval`. A local check on 10-14 reproduced seed 0 (MDD-PTECT 95.842, PPO s0 91.369).
- The 20 chunks of 10 seeds take about 5 min each. The run uses 4 processes beside the AGV sweeps (lab workers 0).

> **Final policy changed 10-17 (user):** the thesis's final policy is the PPO trained on mixed due dates,
> train-v5-cmix **s3** (`../eval-v5-cmix/final_policy.txt`). PPO s4 below stays the B2-trained pick and is reported
> next to it. The RQ3 / RQ4 evaluations of the final policy are `../rq3-gen-v5-cmix`, `../rq3-gen-v5-unity-cmix`,
> `../rq3-size-v5`, `../rq3-flex-v5-unity`, `../rq4-agvfail-v5-cmix` and `../rq4-uncert-v5-policy`.

## Results

**Final policy: PPO s4** (`results/train-v5-final_s4/checkpoint.pt`), the lowest pooled tardiness on the fitting
seeds.

**Reproduction:** exact. s0-s2 and MDD-PTECT on the test seeds equal `train-v5-nokl/eval`, and s3-s4 equal
`train-v5-final/eval` (max |diff| 0).

Pooled window tardiness vs MDD-PTECT (seed bootstrap 95% CI, seeds better / worse, Wilcoxon):

| policy | fitting seeds 2000-2159 (n = 160) | test seeds 0-39 (n = 40) |
|---|---|---|
| PPO s0 | −0.93 [−1.85, −0.04], 91/59, p 0.026 | −0.59 [−3.19, +1.75], p 0.54 |
| PPO s1 | −0.61 [−1.84, +0.55], 96/64, p 0.071 | +0.09 [−1.77, +1.77], p 0.81 |
| PPO s2 | −2.31 [−3.31, −1.41], 111/41, p 7e-8 | −1.07 [−3.11, +0.85], p 0.15 |
| PPO s3 | −0.84 [−1.80, +0.05], 96/63, p 0.033 | −0.31 [−2.25, +1.65], p 0.64 |
| **PPO s4 (selected)** | **−2.62** [−4.24, −1.09], 111/49, p 1e-5 | **+0.17** [−2.29, +2.33], 16/24, p 0.38 |
| **PPO mean of 5 seeds** | **−1.46** [−2.36, −0.64], 98/62, p 0.0002 | **−0.34** [−1.76, +1.05], 24/16, p 0.32 |
| MDD-PECT | +0.58 [−0.15, +1.29], 65/90, p 0.036 | −0.47 [−2.68, +1.63], p 0.52 |

## Reading
1. **On the test seeds, the RQ2 answer stays parity.** The pick is +0.17%, the 5-seed mean −0.34%, and every seed is
   within −1.1 to +0.2%. None is significant.
2. **The pick regresses from fitting to test.** s4 is the best of 5 on the fitting seeds (−2.6%) and the worst of 5 on
   the test seeds (+0.2%). With 40 test seeds the seed spread (±1-2%) is as large as the differences between seeds,
   so a single-seed pick is close to arbitrary. Report the 5-seed mean next to the pick.
3. **On the 160 fitting seeds the policies are significantly, if slightly, better than MDD-PTECT.** The 5-seed mean is
   −1.46% [−2.36, −0.64] (p 0.0002), and 4 of 5 seeds have p < 0.05 or close to it.
   - Training draws instance seeds ≥ 10,000, so the fitting seeds are held out for PPO.
   - MDD-PTECT was chosen as the best fixed pair *on these seeds* (in-sample for the baseline), which if anything
     favours the baseline.
   - The test-seed CI of the 5-seed mean, [−1.76, +1.05], contains −1.46. So the two seed sets are consistent with a
     small gain of about 1-1.5% that 40 seeds cannot resolve.
   - This sits at the realizable bound measured elsewhere: the state-only rollout bound is −1.8% [−4.1, +0.3]
     (rq4-uncert-screen) and BC of the oracle −1.0%.
4. **For the thesis (RQ2):** keep the test seeds as the reported benchmark (parity, −0.3% for the 5-seed mean, n.s.).
   Add the 160 fitting seeds as a larger held-out check: a small but significant 1.5% gain, consistent with the
   realizable bound. That reads as "the learned policy realizes about the realizable 1-2%, and no more", not "the
   policy beats the rules". The user should decide how to word it.
5. **Pooled over all 200 held-out seeds** (eval-v5-curve, 10-15): the 5-seed mean is −1.25% [−1.99, −0.53], 122 / 200,
   p 0.00016. Early stopping at 51.2k does not change the test-seed picture (5-seed mean −0.09%).
