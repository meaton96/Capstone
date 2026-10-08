# dev-critic-lookahead: critic-only look-ahead of known arrivals (2026-10-07, handoff fix 6b; Pinto et al. 2017)

## Why

The critic cannot explain return variance that comes from future arrivals (credit trace section 5). Arrivals are
exogenous, so a value baseline that conditions on them stays unbiased (Mao et al. 2019). The policy never sees them,
so it stays deployable without a forecast. This is the twin-only, critic-only part of the rq3-lookahead obs work.

## What changed

- `env/des_twin/lookahead.py`: 15 features. For each of the next 30 / 60 / 120 min:
  - arrivals per hour
  - offered load
  - bottleneck (per-type) load
  - mean due-date slack of the arriving jobs
  - mean active AGV fraction (the scenario's agvSchedule)
- `--critic-lookahead` (twin): obs key `critic_lookahead`. `ActorCriticConfig.critic_extra_dim` widens only the critic's
  input to [fused features, look-ahead]. The actor and `evaluate.py`'s policies are untouched.
  `models.network.ac_config_for` infers the width from a checkpoint, so old checkpoints load as before.
- Test: the critic's value changes with the features, and the actor's distributions do not.

## Run

dev-prior-offset with `--critic-lookahead`, seeds 0-1, 100k steps.

**Question:** does the critic's explained variance rise (dev-prior-offset is the reference), and does the policy then
learn faster?

## Result (10-08 08:57, cluster CPU; `analysis.out`)
- **Held-out B2 seeds 0-39:** both seeds play MDD-TECT in 100% of slots (= MDD-TECT, 0.000%).
- **The look-ahead critic works:** explained variance 0.90 (s0 0.904, s1 0.902), against 0.75 for the plain critic.
  That is the largest critic improvement of the dev runs (5,400 s horizon 0.84-0.85; shared baseline 0.63-0.66).
  Known future arrivals explain much of the return variance that the current state cannot.
- **The policy does not move:** mean p(ATC) 0.03-0.04, Spearman with ATC-ECT's gap −0.22 / −0.36.
- **Reading:** a better baseline does not create a per-slot signal that the action can exploit. It confirms the credit
  trace: arrival noise dominates the return. Removing it leaves MDD-TECT best from the current state (compare
  dev-paired-advantage).
