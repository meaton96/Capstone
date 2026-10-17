# train-v5-nokl: PPO on the v5 heads, the dev-nokl-free stack (2026-10-11)

Status: **done** (cluster CPU job 21851362, 5 h 44 min, 3 seeds × 100k steps; evaluated on B2 test seeds 0-39).

## Why

The RQ2 rebaseline on the v5 rules (`../rq2-rebase-v5/`) needs a learned-policy row. On H15 this stack
(dev-nokl-free) beat MDD-TECT on 3/3 seeds. Here the same stack runs on the v5 heads {SRT, MDD, ATC} × {PECT, PTECT}.

## Setup

`results/scripts/submit_dev_cluster.sh train-v5-nokl "0 1 2" 2` with dev-nokl-free's exact flags:
- no action prior, no KL (`--prior-pair "" --prior-kl-coef 0`, `--prior-init offset`);
- 5,400 s discount horizon;
- shared-instance baseline in groups of 4;
- rollout 128 / batch 256;
- 900 s slot actions on the B2 regime (twin);
- tardiness reward;
- 100k agent steps per seed.

Frozen code: `code/env` (env/ synced at submit, 6d13412a). Evaluation: deterministic slot actions on B2 seeds 0-39 in
the twin, with all v5 and H15 fixed pairs (`eval/episodes.csv`).

## Results

**Test seeds 0-39, pooled window tardiness vs MDD-PTECT** (the v5 best fixed pair picked on fitting seeds, rq2-rebase-v5;
seed bootstrap 95% CI, seeds better / worse, Wilcoxon):

| policy | vs MDD-PTECT % | CI | better / worse | p |
|---|---|---|---|---|
| PPO s0 | -0.59 | [-3.19, +1.73] | 22 / 17 | 0.54 |
| PPO s1 | +0.09 | [-1.73, +1.78] | 19 / 21 | 0.81 |
| PPO s2 | -1.07 | [-3.11, +0.80] | 25 / 12 | 0.15 |
| mean of the 3 seeds | -0.52 | [-1.97, +0.85] | 23 / 17 | 0.31 |
| fixed MDD-PECT | -0.47 | [-2.70, +1.61] | 17 / 23 | 0.52 |

The eval's MDD-PTECT equals rq2-rebase-v5's to float32 precision (s0: 95.842007 vs 95.842006).

**Training:**
- The critic learns: explained variance rises to 0.80-0.82.
- The policies are not stuck at the default; they play non-MDD rules on many slots:

  | seed | ATC-PECT | MDD-PTECT | MDD-PECT | other |
  |---|---|---|---|---|
  | s0 | 35% | 24% | 41% | – |
  | s1 | 65% | – | 7% | SRT 27% |
  | s2 | 50% | 32% | 18% | – |

- The choice tracks the instance (Spearman of p(ATC) vs ATC-PTECT's gap: -0.42 to -0.61).
- The `vs_oracle` / `oracle_gain_captured` columns of `analysis.out` use the old H15 oracle and do not apply here. The
  v5 oracle is -9.42% (rq2-rebase-v5).

## Reading

1. **PPO reaches parity with the best fixed v5 pair, not more:** -0.5% on average, every seed within ±1.1%, no CI
   excluding 0. That is the same 0-2% band as every other state-only method on v5:

   | method | vs MDD-PTECT |
   |---|---|
   | observable policies | -1.5 / -0.2% |
   | BC of the oracle | -1.0% |
   | one-step rollout | -1.8% |
   | **PPO** | **-0.5%** |

   The oracle (hindsight) stays at -9.4%.
2. **On H15 the same stack found -2 to -3%. That margin is gone on v5:** the gain it found was largely a missing
   processing-time rule, which PECT now supplies to every policy.
3. **The policies switch a lot and still tie:** ATC-PECT on 35-65% of slots, MDD-PECT and MDD-PTECT on the rest. The
   v5 pairs are close substitutes on B2 (rq2-portfolio-screen: no regime specialists), so the switching costs and gains
   cancel.
