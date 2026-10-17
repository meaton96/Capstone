# eval-v5-curve: fixed-instance learning curve of the v5 PPO seeds (2026-10-14, user)

Status: **done** 2026-10-15 00:23 (cluster arrays 21894569 + 21894688, 15 tasks; lab `eval-v5-curve`).

## Why
The user asked whether the training curves look good. The training-time return curve moves mostly with each
episode's random instance (s3 by fifth of training: −163, −157, −135, −151, −152), so it cannot show learning. This
evaluates the same checkpoints on the same 160 instances.

## Method (`run_task.py`, `submit.sh`, `analyze.py`)
- **Checkpoints:** the 5 v5 PPO seeds (train-v5-nokl s0-s2, train-v5-final s3-s4) at init (untrained) and at
  51,200 steps, in the twin on B2 fitting seeds 2000-2159.
- **Same settings as `../eval-v5-final`** (frozen code, floor, B2 params, 900 s slots, deterministic). Its `fit` rows
  give MDD-PTECT and the final (~100k) checkpoints on the same seeds.
- **Smoke:** the untrained s0 network plays exactly MDD-PTECT's tardiness on seed 2000.
- **Score:** pooled tardiness vs MDD-PTECT with seed-bootstrap CIs, per seed and for the 5-seed mean, at init / 51k /
  final.

## Results (fitting seeds 2000-2159, n = 160, pooled window tardiness vs MDD-PTECT)

| seed | init (untrained) | 51.2k steps | final (~100k) |
|---|---|---|---|
| s0 | +0.14 | −1.72 [−2.81, −0.71] | −0.93 |
| s1 | +5.31 | +0.42 | −0.61 |
| s2 | +2.31 | −3.01 [−4.14, −1.99] | −2.31 |
| s3 | +7.03 | −1.99 [−2.91, −1.14] | −0.84 |
| s4 | +5.18 | −2.43 [−3.87, −1.11] | −2.62 |
| **5-seed mean** | **+3.99** [+2.63, +5.28] | **−1.75** [−2.66, −0.89] | **−1.46** [−2.37, −0.63] |

## Reading
1. **The policies learn, and fast.** Untrained, they are 4% worse than MDD-PTECT (one seed, s0, happens to start on
   it). By 51.2k steps, every seed but s1 is 1.7-3.0% better, significantly.
2. **They then plateau, or drift back a little.** From 51.2k to ~100k:
   - s0, s2 and s3 lose 0.7-1.2 points;
   - s1 gains 1.0;
   - s4 is flat.

   The 5-seed mean goes from −1.75% to −1.46%. Longer training would not help on B2.
3. **The plateau is at the realizable bound.** The 51.2k mean (−1.75%) equals the state-only rollout bound (−1.8%
   [−4.1, +0.3], rq4-uncert-screen): an agent that sees the present state but not the future cannot do much better
   with these actions. That fits the BC capacity check (dev-oracle-bc: the network fits the oracle's training choices
   but not new instances) and the history check (dev-history-signal: no information in lagged state).
   - So neither more training, a larger network nor a temporal layer has room on B2.
   - The headroom left (the −9.4% oracle) needs knowledge of future arrivals.
4. **Early stopping does not help on the test seeds.**

   | test seeds 0-39, vs MDD-PTECT | 51.2k | final |
   |---|---|---|
   | s0 | −1.83 [−3.46, −0.39] | −0.59 |
   | s1 | +1.63 | +0.09 |
   | s2 | −0.67 | −1.07 |
   | s3 | +0.33 | −0.31 |
   | s4 | +0.10 | +0.17 |
   | **5-seed mean** | **−0.09** [−1.79, +1.46] | **−0.34** [−1.78, +1.00] |

   The 51.2k checkpoints led on the fitting seeds but not on the test seeds, so the difference between the two seed
   sets is sampling, not overtraining. Keep the final checkpoints.
5. **Pooled over all 200 held-out seeds** (fitting 160 + test 40; held out for PPO, and MDD-PTECT was chosen on the
   fitting part): the PPO 5-seed mean is **−1.25% [−1.99, −0.53]**, better on 122 / 200 seeds, Wilcoxon p 0.00016.
   Per seed −0.47 to −2.08%. This is the most precise estimate: a small but real gain of about 1%, at the realizable
   bound and far from the −9.4% oracle.
