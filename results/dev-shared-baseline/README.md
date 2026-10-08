# dev-shared-baseline: a shared-instance baseline (2026-10-07, handoff fix 4; Mao et al. ICLR 2019)

## Why

Returns spread about 68% of their mean across instances, and the critic sees only the current state, not the
instance's future arrivals (credit trace section 5). Envs that play the same instance differ only in their sampled
actions. The other envs' returns do not depend on this env's action, so subtracting them keeps the policy gradient
unbiased while removing the instance-level noise.

## What changed (`env/rollout_buffer.py`, `env/env_wrappers/twin_env.py`, `env/train.py`)

- `--shared-baseline-group K`: envs i with the same i // K draw the same instance seeds (`VectorizedTwinEnv
  instance_group`). Training checks that every group member really played the same seed as episode n.
- After GAE, each slot step's advantage becomes A_i - mean_{j != i} A_j. The mean is over the group's other envs at
  the same slot of the same instance, matched by (episode ordinal, slot index), not by rollout position.
  `--shared-baseline-mode residual` (default) keeps V in the advantage; `return` uses G_i - mean G_j instead of V. The
  critic's target is unchanged.
- Logged: `charts/shared_baseline_share`, `charts/adv_var_before`, `charts/adv_var_after`.
- Slot mode and twin only (per-decision steps do not line up across envs).

In a 384-step smoke run (untrained critic) the advantage variance dropped from 444 to 11.

## Run

dev-prior-offset with `--shared-baseline-group 4`: 16 envs = 4 instances x 4 envs per rollout, against 16 instances
without it, which is the price of the comparison. Seeds 0-1, 100k steps. Compare advantage variance and the learning
curve with dev-prior-offset.

## Result (10-08 05:17, cluster CPU; `analysis.out`)
- **Held-out B2 seeds 0-39:** both seeds play MDD-TECT in 100% of slots (= MDD-TECT, 0.000%).
- **The shared baseline does what it should for variance:**
  - advantage variance at the first update drops from 2,075 / 3,009 to 26 / 95 (about 30-80×);
  - late in training it is 13-16× lower (1,456 → 108 and 1,416 → 90).
  - The leave-one-out group baseline carries essentially all of it (share 1.00).
- **The policy still doesn't leave the prior:** mean p(ATC) 0.04-0.07, Spearman with ATC-ECT's gap −0.24 / −0.28.
  The prior KL rose to 0.15-0.17 (vs 0.05-0.08 in the reference), so the policy moved more but not enough to flip
  any argmax.
- **Reading:** removing the instance-to-instance variance was not enough. With the MDD-TECT prior at p = 0.8 and KL
  0.05, the remaining per-slot signal still does not change a deterministic choice. Same outcome as dev-prior-offset,
  dev-big-batch and dev-slot-horizon.
