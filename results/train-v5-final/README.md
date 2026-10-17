# train-v5-final: PPO seeds 3 and 4 of the v5 stack, for the final policy (2026-10-14)

Status: **done** 2026-10-14 21:02 (cluster CPU job 21892244, 5 h 39 min, 2 seeds × 100k steps; lab
`train-v5-final`). `analysis.out`, `eval/episodes.csv`.

## Why
- Demo 1 slide 17 ("needed to answer the RQs"): the three v5 PPO seeds of `../train-v5-nokl` range from −1.07% to
  +0.09% against MDD-PTECT, so the seed spread is as large as the effect. Picking s2 on the test seeds would be
  in-sample.
- The fix is five seeds on the same stack, one picked on the fitting seeds (`../eval-v5-final`), with all five and
  their mean reported on the test seeds.

## Setup
- Exactly train-v5-nokl's run, with seeds 3 and 4:
  `results/scripts/submit_dev_cluster.sh train-v5-final "3 4" 2 --prior-init offset --discount-horizon-s 5400
  --shared-baseline-group 4 --rollout-length 128 --batch-size 256 --prior-pair "" --prior-kl-coef 0`.
  - dev-nokl-free stack, v5 heads {SRT, MDD, ATC} × {PECT, PTECT};
  - 900 s slot actions on the B2 regime in the twin, tardiness reward, 100k agent steps per seed.
- **Code:** `code/env` is a copy of train-v5-nokl's cluster snapshot (git 6d13412a, env diff empty, `code/STAMP`),
  checked equal to `git archive 6d13412a env`. The same snapshot is `../rq2-rebase-v5/code/env`, which every v5
  evaluation uses.
- **Evaluation in the job:** B2 test seeds 0-39 in the twin, all v5 and H15 pairs (`eval/episodes.csv`,
  `analysis.out` from `results/scripts/dev_b2_slot_analyze.py`).

## Results

**Test seeds 0-39 in the twin:**

| policy | mean tardiness | vs MDD-PTECT | wins / losses vs MDD-PECT |
|---|---|---|---|
| MDD-PTECT | 145.52 | – | – |
| PPO s3 | 145.08 | −0.31% | 24 / 16 |
| PPO s4 | 145.77 | +0.17% | 21 / 19 |
| MDD-PECT | 144.85 | | |

Window tardiness / 1000, mean cost from `analysis.out`. The CIs and the 5-seed comparison are in `../eval-v5-final`.
Both new seeds are at parity, like s0-s2 (−1.07 to +0.09%).

**Training:**
- The critic learns: explained variance 0.80 / 0.81 by the last quarter, as for s0-s2.
- Both policies leave MDD for most slots:

  | seed | ATC-PECT | MDD-PTECT | SRT-PTECT | MDD-PECT | ATC-PTECT |
  |---|---|---|---|---|---|
  | s3 | 41.5% | 29.6% | 17.5% | 11.5% | – |
  | s4 | 67.9% | 6.0% | 16.5% | – | 7.4% |

- Choice tracks the instance (Spearman p(ATC) vs ATC-PTECT's gap: −0.46 / −0.75).
- Same picture as s0-s2: the policies switch a lot and still tie, because the v5 pairs are close substitutes on B2.
