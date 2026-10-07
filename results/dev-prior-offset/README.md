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

## Result

(pending: `analysis.out`)
