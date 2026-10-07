# dev-paired-advantage: same-future difference advantage per slot (2026-10-07, handoff fix 5, credit trace item 1)

## Why

One slot's rule choice moves the discounted return by about 1/20 to 1/40 of its spread over redrawn futures (SNR
0.03-0.05). Comparing the chosen pair with the default pair *on the same future* removes that spread: about 46x less
variance (credit trace 4b). One future predicts the sign elsewhere only about half the time, so the target averages
over several redrawn futures.

## What changed (`env/env_wrappers/paired_slot_env.py`, new; `env/train.py`, `env/env_wrappers/twin_env.py`)

- `PairedSlotEnv` (twin, slot mode) computes, at each slot,
  A = mean over F futures of [G(chosen pair for the slot, then the default) - G(default)], over the slot plus a tail of
  `--paired-tail-s`, discounted with the training gamma.
  - It is exactly 0 when the chosen pair is the default.
  - It estimates Q_d(s, a) - Q_d(s, d): a policy-improvement step over the default pair (MDD-TECT, the prior), not the
    advantage under the current policy's continuation.
- **State reproduction.** The twin cannot be copied (nested generators), so each branch replays the episode from its
  start with the recorded pair of every earlier slot.
  - A redrawn future keeps the jobs that arrived by the slot start and splices in later jobs from another generator
    seed (as `docs/experiments/review_1007/scripts/branch.py`).
  - Each branch checks it reaches the slot start at the same sim time and decision count. A branch that does not keeps
    its GAE advantage, and the mismatch is counted.
- `--paired-advantage --paired-futures F --paired-tail-s H [--paired-default PAIR]` replaces the GAE advantage with A.
  The critic still regresses the GAE returns. Logged: `charts/paired_share`, `paired_nonzero_share`, `adv_var_gae`,
  `adv_var_paired`.

**Checks:**
- On the instance's own future, the branch return equals the real run's discounted return for "the pair, then the
  default" (`env/tests/test_learning_stack_fixes.py`).
- On the B2 floor: 0 mismatches over 28 branches.
- Cost: about 1.4-1.7 s per branch replay, so about 6 s per non-default slot with 2 futures. That is about 100 CPU-h
  per 100k steps at the prior's ~50% non-default rate, above the handoff's 30-60 CPU-h estimate.

## Run

dev-prior-offset config plus `--paired-advantage --paired-futures 2 --paired-tail-s 3600`, **30k steps** per seed
(not 100k, for cost), 6 twin workers per seed, seeds 0-1. Compare with dev-prior-offset at matched steps, and on the
held-out table.

## Result

(pending: `analysis.out`)
