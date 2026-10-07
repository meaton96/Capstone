# train-due-twin-slot-prior-kl: action prior on MDD-TECT (2026-10-06, branch action-prior)

## Why

The slot policies (train-due-twin-slot, -long) were at parity with MDD-TECT on average, but lost up to +25-35% on the
instances where MDD-TECT was already right and ATC was not. The causes were already ruled down:
- not a scenario property (`../eval-due-twin-slot-long/workup/README.md`)
- not missing congestion information (`../dev-congestion-signal/README.md`)
- not missing history (`../dev-history-signal/README.md`)

So the fix tried was on the learning side: start from the best fixed pair and make leaving it pay for itself.

## What changed (code: `env/models/actor_critic.py`, `env/train.py`, `env/config.py`; tests `env/tests/test_action_prior.py`)

- `--prior-pair MDD-TECT --prior-prob 0.8`: a fresh network starts at the prior p0. The output-layer weights are scaled
  by 0.01 and the bias is set to log p0. Each head puts 0.8 on MDD / TECT and spreads the rest evenly, so the untrained
  policy plays MDD-TECT about 64% of the time (it used to start as a random fixed pair, +52% tardiness).
- `--prior-kl-coef 0.05`: the PPO loss adds 0.05 * KL(pi || p0), summed over both heads. Advantages are normalized per
  batch, so a deviation pays for itself only where its advantage is consistently above about 0.15 std. This is the
  "KL to a default policy" form of regularized RL.
- The entropy bonus (0.01 -> 0.001) is unchanged. Without the flags the code path is the old one; checkpoints keep the
  same format.

## Runs

- **Prior arm:** 5 seeds x 100k agent steps, otherwise identical to train-due-twin-slot (B2 regime, 900 s slots,
  10,800 s horizon). Launcher `run_local.sh`, ~1.9 h per seed, 5 in parallel on the local machine.
- **Baseline arm:** train-due-twin-slot s0-2 (10-05) plus `../train-due-twin-slot-ext` s3-4 (same command, run
  10-06), so both arms have 5 seeds at 100k.
- **Held-out evaluation:** `../eval-due-twin-slot-prior` (twin, B2 seeds 0-39, deterministic, decision logs).

## Result (`../eval-due-twin-slot-prior/analysis.out`)

| arm (5 training seeds) | mean gap to MDD-TECT | instances lost (of 40) | worst instance | gap on the 400k policies' worst 10 | gap elsewhere |
|---|---|---|---|---|---|
| baseline | +1.98% (seeds -0.42 .. +6.21) | 19.4 | +25.5% | +7.5% | -2.0% |
| prior | +0.01% (seeds 0.00 .. +0.07) | 0.4 | +0.1% | +0.01% | 0.00% |

Mann-Whitney over training seeds, prior vs baseline:
- instances lost: p = 0.009
- worst instance: p = 0.010
- gap on the bad quartile: p = 0.010
- mean gap: p = 0.13

**The prior removes the losses entirely, and the gains with them.** Under the deterministic evaluation, four of the
five prior policies play MDD-TECT in 100% of slots; s3 deviates in 0.4% of slots (to SRT-TECT). They reproduce the
fixed rule exactly. The baseline's wins (-2.0% on the instances outside the bad quartile) disappear along with its
losses.

**Underneath, it learned the right direction, too weakly to act on.**
- **Probabilities:** the trained policies keep 0.82-0.87 on MDD and 0.80-0.93 on TECT on average, close to p0.
  p(MDD) varies across slots (s0: 0.70 .. 0.96; s3 dips to 0.31), but never enough to flip the argmax, except in
  s3's few slots.
- **Direction:** the variation is correct. Over instances, the mean p(ATC) is higher where ATC actually beats MDD-TECT
  (Spearman with ATC-ECT's gap to MDD-TECT: -0.25 to -0.39 across the five seeds).

## Reading

- The trade-off between losses and gains is now explicit:
  - free policy: wins about 2% on three quarters of the instances, loses 7-25% on the rest, mean about +2%
  - prior at KL 0.05: matches MDD-TECT everywhere
  - neither beats MDD-TECT on average
- The state signal for "ATC instead of MDD" is weak: AUC <= 0.70 even with oracle labels (dev-congestion-signal). At
  this KL weight and 100k steps, the learned preference never crosses the prior. This is consistent with the earlier
  diagnosis: a learning-signal problem, not a representation problem.
- **Next, in order** (`dev-oracle-bc` sets the ceiling for all of them):
  1. a weaker or decaying anchor: KL 0.01, or 0.05 -> 0 over training
  2. init-only (KL 0), to separate the starting point from the anchor
  3. longer training
  4. a stochastic evaluation, to check whether the small shifts are worth anything when sampled

  If imitating the oracle cannot beat MDD-TECT either, more RL tuning will not get there; if it can, the cloned
  network is also a natural warm start for RL.

## Follow-up: weaker and decaying anchor (10-07; train-due-twin-slot-prior-kl01, -kldecay; ../eval-due-twin-slot-prior-var)

Same setup, 5 seeds each:
- *KL 0.01:* weight 0.01, constant
- *KL 0.05 -> 0:* weight 0.05 decaying linearly to 0 over the 100k steps

One seed of each crashed at start-up from GPU memory contention (10 runs started at once). Both were rerun fresh,
alone (results/train-due-twin-slot-prior-*/run_local.sh now takes SEEDS=...).

| arm (5 seeds) | mean gap to MDD-TECT | instances lost | worst instance | slots not on MDD-TECT | mean p(MDD) |
|---|---|---|---|---|---|
| baseline | +1.98% | 19.4 | +25.5% | most | |
| KL 0.05 | +0.01% | 0.4 | +0.1% | 0-0.4% | 0.82-0.87 |
| KL 0.01 | -0.20% (seeds -0.77 .. 0.00) | 1.8 | +1.7% | 0-12% | 0.77-0.80 |
| KL 0.05 -> 0 | -0.05% (seeds -0.40 .. +0.20) | 3.0 | +1.8% | 0-20% | 0.69-0.94 |

All three prior arms lose fewer instances and less in the worst case than the baseline (Mann-Whitney p ~ 0.01).
None of their means differs from MDD-TECT (p 0.09-0.14 vs the baseline; per seed, Wilcoxon p >= 0.21).

**Reading.** The strength of the anchor is not what keeps the policy on MDD-TECT. Even with no anchor left by the end
(0.05 -> 0), the policies stay near the starting prior, and the few deviations (SRT-TECT, MDD-ECT, EDD-TECT in up to
20% of slots) net out to about zero. Starting at a good fixed rule plus the weak learning signal is enough to keep PPO
there for 100k steps. For comparison, behavior cloning of the oracle (`../dev-oracle-bc/README.md`) gives -1.1 to
-2.6% (2 of 3 significant) with no large losses, so the better use of a prior is the cloned policy, not a single pair.
