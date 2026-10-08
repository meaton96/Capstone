# dev-prior-offset: the action prior as a fixed logit offset (2026-10-07, handoff fix 1, audit M1)

## Why

The 10-06 prior runs (train-due-twin-slot-prior-kl, -kl01, -kldecay) started the policy at p0 by scaling the actor's
output weights by 0.01 and putting log p0 in the bias. The 10-07 audit (`docs/experiments/review_1007/
TRAINING_STACK_AUDIT_1007.md`, M1) measured what that does:
- the policy gradient reaching the shared trunk drops by the same factor: 3.75 -> 0.02, while the value gradient stays
  at 3.31, so the critic alone shapes the features;
- the bias stays at log p0 in every trained prior checkpoint, and the argmax never flips.

So "learned the right direction, too weakly to act on" (train-due-twin-slot-prior-kl/README.md) may come partly from
the init, not from the signal.

## What changed (`env/models/actor_critic.py`, `env/train.py`, `env/config.py`)

- `ActorHead` has a `prior_offset` buffer (saved in checkpoints; zeros for checkpoints before 10-07) added to its
  logits in `forward`, so rollout, update and `evaluate.py` all apply it.
- `--prior-init offset` (the new default) sets it to log p0 on a default-initialized output layer (bias zeroed).
  `--prior-init scale --prior-weight-scale 0.01` reproduces the 10-06 runs.
- A resume with `--prior-pair` from a checkpoint trained without a prior is refused (audit m5).

Gradient check on 64 real B2 slot observations (`scripts/gradcheck.py`, from the audit's script):

| init | policy-gradient norm on the trunk | value-gradient norm | logit spread across states |
|---|---|---|---|
| default (no prior) | 3.75 | 3.31 | 0.066 |
| **offset (new)** | **2.58** | 3.31 | 0.066 |
| scale x0.01 (10-06) | 0.0199 | 3.31 | 0.00066 |

The offset start is about p0: p(MDD) about 0.7 instead of 0.8, because the default layer's mean W.h tilts each logit by
a few tenths of a nat. The prior's rules are still each head's argmax, and the KL term pulls toward p0.

## Run

`bash results/scripts/dev_b2_slot_train_eval.sh dev-prior-offset "0 1" 2 --prior-init offset`. This is
train-due-twin-slot-prior-kl's config: B2, 900 s slots, 10,800 s horizon, KL 0.05, 100k steps, seeds 0-1. Code frozen
in `code/` (`code/STAMP`). Evaluation: B2 seeds 0-39, all 15 pairs, with the three references (MDD-TECT, per-instance
hindsight, the rq2-twin-fleet oracle). Analysis: `results/scripts/dev_b2_slot_analyze.py` (vs the 10-06 prior s0-1).

**Question:** with gradient reaching the trunk, does the argmax leave MDD-TECT on the instances where ATC-ECT wins?

## Result (`analysis.out`, 10-07 16:02; 4 h, 2 seeds in parallel)

**The offset init does not change the outcome: both policies still play MDD-TECT in 100% of held-out slots.**

| | offset (this run) s0 / s1 | scale x0.01 (train-due-twin-slot-prior-kl) s0 / s1 |
|---|---|---|
| held-out B2 0-39 vs MDD-TECT | 0.000% / 0.000% (identical on all 40) | 0.000% / 0.000% |
| slots not MDD-TECT | 0 / 0 of 960 | 0 / 0 |
| p(MDD) over slots: mean (min) | 0.835 (0.798) / 0.843 (0.783) | 0.821 (0.695) / 0.866 (0.775) |
| max p(ATC) | 0.099 / 0.072 | 0.131 / 0.074 |
| Spearman(p(ATC), ATC-ECT's gap to MDD-TECT) | -0.31 / -0.36 | -0.27 / -0.32 |
| actor output weight norm | 1.41 / 1.42 | 0.55 / 0.71 |
| critic explained variance (last quarter) | 0.75 / 0.74 | 0.75 / 0.74 |
| training return by fifth | within 1-2 of the reference in every fifth | |

The three-reference table (same for both checkpoints, since they are MDD-TECT):
- vs the best fixed pair (MDD-TECT): 0.0%;
- vs each instance's hindsight best pair: +5.1%;
- vs the oracle: +12.1%, with 0% of the oracle's gain captured.

The oracle check passed: the 600 fixed-pair episodes match rq2-twin-fleet's within 8e-8. The discounted return
(10,800 s) is -60.25 for MDD-TECT, against -62.7 for SRT-ECT and -69.4 for ATC-ECT, the same order as the window score.

## Reading

- The gradient is restored: the output layer grows to a default-sized norm (1.4 vs 0.55-0.7). The direction of the
  learned preference is the same as before, and slightly stronger (Spearman -0.31 / -0.36 vs -0.27 / -0.32).
- But the policy stays at p0 or above on MDD. The audit's M1 was a real mechanical defect, but **not** what held the
  argmax at the prior. With KL 0.05 and the per-slot signal-to-noise of 0.03-0.05, the learned shift in p(ATC) never
  gets near 0.5.
- What remains to try is the noise side of the handoff: dev-paired-advantage, dev-shared-baseline, dev-slot-horizon,
  dev-critic-lookahead and dev-big-batch, all of which use this offset init so they compare against this run.

(Analysis note: the "logit offset+bias" line in `analysis.out` prints the offset plus the zeroed bias for offset runs,
so it is log p0 by construction; the weight norm and the probabilities are the informative lines.)
