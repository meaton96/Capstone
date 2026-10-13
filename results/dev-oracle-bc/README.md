# dev-oracle-bc: can the policy network imitate the switching oracle? (2026-10-06/07)

**Question.** RL on the twin reaches at best parity with MDD-TECT, while the oracle is 10.8% better. Is the limit the
observation and architecture, or RL? Behavior cloning answers it: train the same network supervised on the oracle's
choices and see how close it gets.

## Setup

- **Data** (`collect.py` on the RIT cluster, dev-oracle-bc-collect, 200 tasks, about 12 CPU-min each):
  - the H15 greedy oracle of rq2-twin-fleet (twin, B2, tardiness, 6 h window, 900 s slots), keeping all 15 pairs'
    hold-to-end tardiness at every slot
  - the full obs v3 at each slot start along the oracle trajectory (grid, machine and job tables, scalars; caps
    15 / 256 as in training)
  - training seeds 2000-2159 (3,840 slots); test seeds 0-39 (960 slots), whose recomputed oracle reproduces
    rq2-twin-fleet's schedule and tardiness exactly (40/40), as does every replay
- **Training** (`bc_train.py`):
  - soft targets q(pair) ~ exp(-regret% / 1%), marginalized onto the two heads, so near-ties are not counted as errors
  - early stopping on 15% of the training seeds; 3 BC seeds
  - *full*: the RL policy's architecture, fresh init; *scalars*: an MLP on the global scalars + flags only
  - capacity check: the same without early stopping, 200 epochs (`--tag _noes`, `bc_train_noes.out`)
- **Rollout** (`run.sh` step 3, `rollout/`): the early-stopped full networks run as slot policies in the twin on test
  seeds 0-39 (deterministic), against MDD-TECT and the oracle.

## Results

**Per slot** (`bc_train.out`, `bc_train_noes.out`; test seeds 0-39, means over 3 BC seeds):

| model | train slots matched | test slots matched | job rule matched | regret of chosen pair (hold-to-end) | always MDD-TECT |
|---|---|---|---|---|---|
| full, early stopping (~30 epochs) | 27% | 24% | 46% | 6.7% | 6.6% |
| scalars, early stopping | 24% | 25% | 44% | 7.4% | 6.6% |
| full, 200 epochs (capacity check) | **78%** | 21% | 45% | 7.5% | 6.6% |
| scalars, 200 epochs | 47% | 21% | 42% | 8.4% | 6.6% |

**As policies** (`rollout/analysis.out` + paired statistics; MDD-TECT mean tardiness 172.4, oracle 153.8):

| policy | mean vs MDD-TECT | median | instances lost | worst | 400k policies' bad quartile | other 30 | Wilcoxon p | oracle gap closed (median) |
|---|---|---|---|---|---|---|---|---|
| bc s0 | **-2.6%** | -3.6% | 13 / 40 | +11.0% | -0.7% | -7.7% | 0.019 | 0.27 |
| bc s1 | **-2.2%** | -2.0% | 9 / 40 | +28.7% | +0.4% | -4.7% | 0.021 | 0.14 |
| bc s2 | -1.1% | -1.9% | 12 / 40 | +19.3% | +0.6% | -2.2% | 0.14 | 0.11 |

Slot choices: ATC-TECT and MDD-TECT dominate. s0 plays ATC-TECT 43% / MDD-TECT 26%; s1 and s2 play MDD-TECT 67-78%.

For comparison, the RL slot policies on the same 40 seeds:
- 100k baseline arm: +2.0% mean, 19 instances lost, worst +25%
- best RL policy so far (400k s0): -2.4%, p = 0.046
- action prior at KL 0.05: exactly MDD-TECT

## Reading

1. **Capacity is not the limit.** The full network fits the training instances (78% of oracle pairs) but generalizes
   no better than the scalar MLP (21% on test). The capacity check overfits: its per-slot regret is worse than always
   playing MDD-TECT. Which pair the oracle picks is largely specific to each instance and not carried by the
   slot-start observation in a form that generalizes from 160 instances. This agrees with the per-feature checks
   (dev-congestion-signal, dev-history-signal: AUC <= 0.70). A bigger network or attention layers are not what is
   missing.
2. **Imitation still gives a useful, safe policy.** Cloned from only 160 oracle instances, the early-stopped network
   beats MDD-TECT by 1-2.6% on average (2 of 3 significantly). It matches the best RL policy so far without RL's large
   losses: about 0 on the instances where RL lost 7-25%. The per-slot score understates this: hold-to-end regret is
   dominated by the future, while the rollout sees what a switch actually does.
3. **The policy recovers a quarter of the oracle's advantage at most** (median oracle gap closed 0.11-0.27). The rest
   matches the earlier findings: the oracle's choices follow realized arrivals that no state-based policy can see in
   advance (base-eplen, rq2-oracle-slotlen).

## Next (proposed)

- **BC warm start + RL fine-tune:** start PPO from a cloned network instead of from scratch or a single-pair prior.
  The cloned network can also serve as the prior for the KL anchor (KL to the cloned policy instead of to MDD-TECT).
- **More oracle data:** 160 instances is small for the full network. The cluster produces about 200 instances in
  about 4 h, so 1,000+ is feasible. Data generalized no worse than capacity here, so more data is the cheaper lever.
- Re-do this on the steady-state regime base-eplen recommends (long windows), where the oracle's targets are less
  dominated by per-instance noise.
