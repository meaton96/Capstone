# dev-congestion-signal: does a transport-backlog scalar add information to obs v3? (2026-10-06)

Motivation: the bad-instance workup (`../eval-due-twin-slot-long/workup/README.md`) found that the slot policies lose
where ATC is the wrong family and MDD-TECT is right, and that this is partly predictable from transport congestion.
obs v3 has no explicit transport-backlog scalar. Before adding one (obs schema change, C# + twin + parity), check
whether the realized backlog at a decision says which family to play, beyond what obs v3 already gives the policy.

**Answer: no, not on the twin.** The candidate congestion features carry the same information as obs v3's scalars and
flags. Adding them never improves the out-of-fold prediction (mean AUC change between -0.011 and +0.005 on every target).

## Method (`run.py`, `analyze.py`)

- **States:** B2 seeds 0-39. Each seed's oracle slot schedule (rq2-twin-fleet) is replayed in the twin, and every
  replay reproduces the stored oracle tardiness exactly. At each 900 s slot's first decision (960 states), record:
  - the exact obs v3 global scalars + event flags (`des_twin.ObservationBuilder`), i.e. what the policy's MLP sees
  - candidate congestion features: jobs waiting for an AGV (count, per on-duty AGV, mean / max / summed wait), jobs
    in transit, AGV busy share, AGV idle share over the last slot, queued jobs / work per machine, WIP
- **Targets** (tardiness of ATC-TECT minus MDD-TECT, < 0 = ATC better; also ATC-ECT for the hold target):
  - *hold:* the pair held from slot k to the end of the episode, after the oracle prefix
  - *dev1 / dev4:* the oracle schedule with slot k (1 slot) or slots k..k+3 (1 h) swapped to the pair, then back to
    the oracle. This is the local, decision-relevant difference; *hold* mostly measures future regimes.
- **Models:** logistic / ridge and gradient boosting on three feature sets: obs v3, congestion, and both. Folds are by
  seed (no seed in both train and test), over 10 random partitions. The gain of "both" over "obs" is paired per
  partition.

## Results (`analysis.out`)

1. **obs v3 already carries the congestion features.** Each one's best single obs column:
   - waiting jobs per on-duty AGV: AGVs on duty per machine (|rho| 0.72)
   - waiting count and waits: the idle-AGV flag (0.59-0.60)
   - queued jobs / work per machine: share queued / mean machine load (0.96-0.97)
   - WIP and fleet size: exact copies (1.00)
   - AGV idle share over the last slot: nothing related to the targets (|rho| <= 0.07)

   The models combine the obs columns, so the overlap is larger than these single correlations.
2. **Congestion does relate to the right family, in the expected direction.** More jobs waiting for an AGV, higher
   WIP and fewer AGVs on duty favor MDD-TECT over ATC (Spearman with dev4: waiting +0.22, WIP +0.25, on-duty AGVs
   -0.25). This matches the instance-level finding.
3. **It adds nothing beyond obs v3.** Out-of-fold scores, obs v3 vs obs v3 + congestion:

   | target | AUC obs (logistic / boosting) | gain from congestion, mean (range over 10 partitions) |
   |---|---|---|
   | hold ATC-TECT | 0.667 / 0.683 | +0.003 / -0.003 (-0.026 .. +0.013) |
   | hold ATC-ECT | 0.635 / 0.662 | +0.005 / -0.011 |
   | hold ATC-TECT, first 4 h | 0.561 / 0.584 | +0.005 / -0.004 |
   | dev1 (1 slot) | 0.659 / 0.651 | +0.002 / -0.003 |
   | dev4 (1 h) | 0.689 / 0.696 | +0.003 / -0.002 |

   Regression (Spearman of the prediction with d) agrees: gains between -0.016 and +0.029.
4. **The decision is only moderately predictable from the state at all:** AUC about 0.65-0.70 at best, even with the
   oracle's trajectory and labels. Part of the right choice depends on what arrives next, which no current-state
   feature shows.

## Reading and caveats

- On the twin, the policy is not missing a congestion signal. obs v3 already has the parts: WIP, the idle-AGV flag,
  AGVs on duty, the job-state shares and machine load. The slot policies' ATC-vs-MDD mistakes are a learning problem,
  not missing information: the signal is weak (AUC <= 0.70) and the policy has to learn it from noisy episode
  returns. This supports the action prior / residual on MDD-TECT, and against an obs-v5 transport scalar for the twin.
- **The twin has no zone blocking.** Transport only binds there through the AGV count, so congestion on the twin is
  just "jobs waiting for a free AGV", which the existing scalars capture. In Unity, congestion is also physical
  (zone waits, blocking, stall fallbacks). A backlog or zone-wait scalar could carry information there that obs v3
  does not, and it would bear on the observation research question. The same check on Unity would need per-slot
  state logs from the player (`-decisionlogdir`) and branched episodes. It is far more expensive and was not run.
- Snapshots are taken at decisions, as the policy sees them. Routing decisions fire when an AGV frees up, so the
  instantaneous AGV-busy share reads low at those moments. A time-averaged version (idle share over the last slot)
  did not help either.
- **Reward:** this check does not address congestion in the reward. A potential-based shaping term on the backlog
  would keep the optimum unchanged and only change learning speed.
