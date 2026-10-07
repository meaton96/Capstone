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

## Result

(pending: `analysis.out`)
