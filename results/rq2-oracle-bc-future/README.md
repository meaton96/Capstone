# rq2-oracle-bc-future: can a policy use the full future schedule? (2026-10-15, user)

Status: **done**.
- 1,000 label seeds: 2026-10-16 01:52. Labs `rq2-oracle-bc-future-data` (Slurm 21899517, 1:07) and
  `rq2-oracle-bc-future` (Slurm 21899729 for 27 A100 models + 21899730 for 40 rollout tasks, 1:24). Outputs
  `analysis.out`, `summary.csv`.
- 4,000 label seeds (data scaling, user 10-16): 2026-10-17 03:31. Labs `rq2-oracle-bc-future-data4k` (Slurm
  21901807, 2:37) and `rq2-oracle-bc-future-4k` (Slurm 21902618 for 24 A100 models + 21902619 for rollouts; 13 h,
  mostly A100 queue). Outputs `analysis4k.out`, `summary4k.csv`.

**Answer: yes, a policy can use the full future schedule as a per-job table, and with enough labels the gain is
real.** Cloned from 4,000 oracle-labelled instances, BC with a 128-row table is −2.55% [−3.54, −1.64] vs MDD-PTECT
on 200 held-out seeds (24% of the oracle's gain). That is 3.0 points better than BC without the table: all 5 vs all 5
models, Welch p 0.004, Mann-Whitney p 0.008. An unrelated (redrawn) future takes the gain away (−0.39%, every model
loses 1.5-2.8 points, p 0.0006). The gain holds under 1-30 min arrival error and 10-25% churn (−2.4 to −2.7%), where
the planner keeps −2.2 to −2.8% on the test seeds. With 1,000 seeds the same arms gave only −0.5 / −0.9%. The gain
grew with data and the learning curve has not flattened.

## Why

eval-v5-lookahead-fcast (10-15) showed that PPO with 15 exact-future summary features (`des_twin/lookahead.py`) does
not use them: an unrelated (redrawn) future scores the same as the exact one (−2.07% vs −2.02% vs MDD-PTECT), and its
edge over state-only PPO is training-seed noise. The user pointed out that this tested only summary features, never
the future schedule itself, given to the network the way the live job table is.

This experiment gives the network the next arriving jobs as a per-job table and tests it with behaviour cloning of the
hindsight switching oracle, not PPO. The oracle's per-slot labels look like noise from the state alone (27% pair match,
rq2-oracle-bc-v5): one slot's best pair depends on the realized future (`docs/experiments/review_1007/
CREDIT_ASSIGNMENT_TRACE_1007.md`, §4b). With that future in the input, the labels should become learnable if a network
can read them off the schedule. If BC with the table does not beat BC without it, there is no reason to expect PPO to
do better.

References (B2, pooled % vs MDD-PTECT): switching oracle −9.4% (test seeds) / −10.7% (fitting seeds); planner on the
exact schedule −9.4% (900 s re-plans) / −11.2% (300 s); planner at 10-30 min forecast error −2.2 to −2.8%
(rq2-perturbed-v5); state-only BC of the oracle −1.0% (mean of 3 clones, rq2-oracle-bc-v5; clone s0 −2.25%).

## Method

**Observation (live code, branch `claude/peaceful-pike-a3b399`, commit 7fbf81f8):**
- `env/des_twin/future_table.py`: the next N jobs arriving within 2 h, one row per job, 20 columns: present, arrival
  offset (linear and squashed), rank, cumulative work / machines, op count, work and min work, the first op's shortest
  time and eligible machines, work per machine type (5), has due date, slack at arrival, due − now, relative slack, mean
  eligible machines. The columns are the live job table's static ones (same squash scales) plus the arrival offset;
  they do not depend on N, so the first 64 rows of a 128-row table equal a 64-row table.
- Rows are read the way `des_twin.lookahead` reads the future: through `twin.unc.plan_future_jobs(now, arrived)` when
  the twin has a published plan, else its pending arrivals. So eval-v5-lookahead-fcast's `PublishedFuture` adapter
  feeds perturbed schedules without changes.
- `env/models/future_net.py`: `FutureSchedulingNetwork`, the RL policy network plus a second set encoder over the table
  (the live job table's `SetEncoder`: row MLP, masked mean + max, 128-D), concatenated with the 560-D state embedding
  before the fusion head (688 → 512 → 256). The rq3-lookahead features were appended after the fusion head instead.
- `TwinSchedulingEnv(future_table=(rows, horizon))` adds `obs["future_job_table"]`. Not yet wired into `train.py` /
  `evaluate.py` (only needed for a PPO run).
- Tests: `env/tests/test_future_table.py` (5).

**Frozen code (`code/`, `code/STAMP`):** the two new files, loaded by `common.py` on top of the frozen v5 snapshot
`../rq2-rebase-v5/code/env` (6d13412a, the code of the labels) as `des_twin.future_table` and `models.future_net`. The
frozen snapshot is not edited; every script asserts nothing comes from the live `env/`.

**Data (`collect.py`, cluster, lab `rq2-oracle-bc-future-data`):**
- Training labels on **new B2 seeds 3000-3999** (24,000 slots). The task named the rq2-rebase-v5 labels (fitting seeds
  2000-2159), but those seeds are also evaluation seeds; new seeds keep both evaluation sets held out and give 6× the
  data. Same oracle code path as rq2-rebase-v5 (greedy, 6 v5 pairs, 900 s slots, hold-to-end tails, slot-start
  observation at caps 15 / 256), plus the exact future table (128 rows) at the same capture.
- Evaluation seeds 0-39 and 2000-2159: the stored rq2-rebase-v5 oracle schedule replayed to capture the future tables
  (the exact table and, for a per-slot check, the redrawn one).

**BC (`bc_train.py`, one A100 per model):** rq2-oracle-bc-v5's recipe unchanged: soft labels q(pair) ∝ exp(−regret% /
1%) marginalized onto the two heads, Adam 3e-4, weight decay 1e-4, batch 64, early stopping on 15% of the training
seeds (patience 20).
- Arms: `state` (the RL policy network on the observation), `future64` (+ 64-row table), `future128` (+ 128 rows:
  the 2 h window holds 98 jobs on average, so 64 rows cover about 78 min). 5 BC seeds each.
- Learning curve: state / future64 on the first 250 and 500 training seeds, BC seeds 0-2 (per-slot scores only).
- Per-slot scores on the 4,800 held-out evaluation slots: oracle-pair match, job-rule match, hold-to-end regret of the
  pick; for future arms also the same slots with the redrawn table (how many picks change, and whether they match the
  oracle less often).

**Rollouts (`rollout.py`, cluster CPU):** each clone plays argmax job rule × machine rule per 900 s slot on the 200
held-out seeds, in the same twin and harness as the labels and MDD-PTECT (`sim.Episode`). Future arms run at every
forecast level: exact, and rq2-perturbed-v5's published schedules (the planner's draws at the same seed and level)
jit1 / jit10 / jit30 / churn10 / churn25 / redrawn through the PublishedFuture adapter.

**Analysis (`analyze.py`):** pooled % vs MDD-PTECT (seed bootstrap, Wilcoxon) of each arm's family (per-instance mean
over its BC seeds) and each clone; future vs state (paired); each level vs its own exact run; share of the oracle's
gain; the planner at the same error on the test seeds; and a training-seed-level test: one pooled % per trained
model, Welch t and Mann-Whitney U between arms.

## Checks (local, 10-15)
- `collect.py label 2000` reproduces rq2-rebase-v5's `out/oracle/s2000.npz` exactly (every key; `smoke/checks_label.out`).
- Evaluation seeds: all 200 replays equal the stored oracle value and every stored observation (`collect.py future`).
- The exact table through the PublishedFuture adapter equals the direct table at every slot (max |diff| 0; seeds 0 / 7
  / 2003). Perturbed tables differ from it, redrawn most (mean |diff| 0.15-0.17 vs 0.04-0.05 for jit1;
  `smoke/checks_adapter.out`).
- Rollout smoke (seed 0, 2-epoch clones): runs at all 7 levels, 1.75 s per episode.

**Early read (bc_fit160/, local GPU):** BC trained on rq2-oracle-bc-v5's data (fitting seeds 2000-2159), 3 seeds per
arm, per slot on test seeds 0-39: pair match state 29.7%, future64 27.5%, future128 29.6%; mean regret 6.2 / 5.6 / 6.0%
(always MDD-PTECT: 5.4%). The future clones do react to the table: with the redrawn table in the input 13-35% of their
picks change, but those picks match the oracle about as often. At 160 seeds the table adds no imitation accuracy.

## Results at 1,000 label seeds (10-16)

**Checks.**
- 1,000 / 1,000 label files. Every replay equals its oracle value. The oracle is −10.0% vs MDD-PTECT on the new seeds
  (fitting −10.7%, test −9.4%).
- 27 / 27 models, 200 / 200 rollout files: 15 checkpoints at exact and 10 at each perturbed level, for every seed.
- One stamp everywhere: des_twin a3638018 (the labels' code), 6d13412a, future code b4c9a54b (7fbf81f8).
- A cluster state model (s0) matches the local preview (val pair match 28.1 vs 28.4%; GPU training is not bit-exact).

### 1. Per slot: the table lowers the regret of the pick, not the match rate

On the 4,800 held-out evaluation slots (test + fitting seeds), deterministic argmax pair, mean (sd) over 5 BC seeds:

| arm (1,000 training seeds) | oracle-pair match | job-rule match | hold-to-end regret of the pick |
|---|---|---|---|
| state | 30.6% (1.5) | 59.6% | 4.94% (0.09) |
| future64 | 30.0% (2.3) | 60.3% | 4.66% (0.19) |
| future128 | 30.9% (3.2) | 60.3% | **4.57%** (0.05) |
| same slots, **redrawn** table: future64 / future128 | 28.7 / 29.7% | | 4.98 / 4.98% |
| always MDD-PTECT | | | 5.66% |

- Across BC seeds the regret difference is significant: future64 vs state Welch p 0.026 (Mann-Whitney 0.016);
  future128 vs state Welch p 0.0002 (MW 0.008, every future128 seed below every state seed). The match rate does not
  differ (p 0.66-0.85).
- With the redrawn table, 25-27% of the future clones' picks change, and their regret returns to the state arm's
  (4.98%). So the clones read the table, and the exact table is what helps.
- **Learning curve** (regret, 3 BC seeds at 250 / 500, 5 at 1,000 seeds): state 5.18 → 5.00 → 4.94%, future64 5.47 →
  4.89 → 4.66%. At 250 seeds the table hurts, at 500 it ties, at 1,000 it helps, and the curve is still falling. On
  rq2-oracle-bc-v5's 160 fitting seeds (`bc_fit160/`, early read) it gave nothing (test-slot regret 6.2 / 5.6 / 6.0%,
  n.s.). The 160 labels of the original BC were too few to see this.

### 2. Rollouts: pooled % vs MDD-PTECT over all 200 held-out seeds

Family = per-instance mean over the 5 BC seeds; seed bootstrap 95% CI; Wilcoxon p; switching oracle −10.44%.

| arm, published schedule | vs MDD-PTECT | vs state BC (paired) | vs its own exact run (paired) | share of the oracle's gain |
|---|---|---|---|---|
| state BC (1,000 seeds) | +1.08 [−0.20, +2.54], p 0.20 | – | – | −10% |
| future64, exact | −0.51 [−1.61, +0.55], p 0.002 | **−1.59** [−2.66, −0.58], p 0.005 | – | 5% |
| future64, ± 1 / 10 / 30 min | −0.59 / −0.48 / −0.55 | −1.67 / −1.56 / −1.63 | −0.08 / +0.03 / −0.04 (n.s.) | 5-6% |
| future64, 10 / 25% churn | −0.09 / −0.16 | −1.17 / −1.24 | +0.42 [+0.05, +0.81] / +0.35 | 1-2% |
| future64, **redrawn** | **+1.17** [−0.32, +2.65] | +0.09 (n.s.) | **+1.68** [+0.68, +2.70], p 0.005 | −11% |
| future128, exact | −0.89 [−1.80, +0.02], p 0.0005 | **−1.97** [−2.97, −0.98], p 0.005 | – | 9% |
| future128, ± 1 / 10 / 30 min | −1.00 / −0.87 / −1.00 | −2.08 / −1.95 / −2.07 | −0.11 / +0.02 / −0.10 (n.s.) | 8-10% |
| future128, 10 / 25% churn | −0.70 / −0.88 | −1.78 / −1.96 | +0.19 / +0.02 (n.s.) | 7-8% |
| future128, **redrawn** | −0.19 [−1.48, +1.06] | −1.27 [−1.96, −0.54], p 0.045 | +0.70 [−0.15, +1.64], p 0.35 | 2% |

**Test seeds 0-39 alone, next to the planner on the same published schedule** (vs MDD-PTECT):

| published schedule | state BC | future64 | future128 | planner, 900 s re-plans | planner, 300 s |
|---|---|---|---|---|---|
| exact | +2.37 | +0.06 | +0.14 | −9.42 | −11.24 |
| ± 1 min | | −0.21 | −0.22 | −4.07 | |
| ± 10 min | | +0.03 | −0.31 | −2.19 | −2.67 |
| ± 30 min | | +0.04 | −0.10 | −2.77 | |
| 10% / 25% churn | | +0.58 / +0.76 | +0.31 / +0.14 | −2.23 / −2.31 | −3.30 (10%) |
| redrawn | | +0.78 | +0.32 | −1.21 (n.s.) | |

On the test seeds the future arms beat state BC by 2.2-2.7 points (p 0.003-0.05), but all three arms are within ±2.4%
of MDD-PTECT. rq2-oracle-bc-v5's state BC (trained on the 160 fitting seeds) was −1.04% [−3.30, +1.17] on the same 40
seeds. Clone-to-clone spread on 40 seeds is about ±2 points, so compare arms only within this run.

### 3. Training-seed level (one pooled % vs MDD-PTECT per trained model, 200 seeds)

| arm | BC seeds 0-4 | mean (sd) |
|---|---|---|
| state | +1.41 / +1.51 / +0.59 / −0.50 / +2.39 | +1.08 (1.09) |
| future64, exact | +0.83 / −0.92 / −0.81 / −1.53 / −0.12 | −0.51 (0.90) |
| future64, redrawn | +1.94 / +1.43 / +0.82 / +0.42 / +1.22 | +1.17 (0.58) |
| future128, exact | −1.38 / −1.32 / −1.17 / −0.85 / +0.25 | −0.89 (0.67) |
| future128, redrawn | −1.20 / +0.55 / +0.03 / −0.89 / +0.55 | −0.19 (0.81) |

- future64 exact vs state: Welch p 0.037, Mann-Whitney p 0.056. future128 exact vs state: Welch p **0.012**, MW p
  **0.016**.
- Exact minus redrawn, per model: future64 −1.11 / −2.35 / −1.63 / −1.95 / −1.34 (all 5 negative, paired t p
  **0.0016**); future128 −0.18 / −1.87 / −1.19 / +0.03 / −0.30 (p 0.12).
- Redrawn future64 vs state: p 0.88 (no difference). Redrawn future128 vs state: Welch p 0.07, MW 0.056.

Pairs played (exact): state ATC-PECT 52%, MDD-PTECT 27%, MDD-PECT 17%; future64 ATC-PECT 34%, MDD-PTECT 25%,
MDD-PECT 22%, ATC-PTECT 18%; future128 ATC-PECT 41%, MDD-PTECT 33%, MDD-PECT 21%.

## Reading at 1,000 seeds (points 2 and 5 are superseded by the 4,000-seed run below)

1. **A future-job table is used; the 15 summary features were not.** The look-ahead PPO
   (eval-v5-lookahead-fcast) scored the same with an unrelated future. Here every future64 clone loses 1.1-2.4 points
   when its table is redrawn (p 0.0016), and per slot the redrawn table changes a quarter of the picks and undoes the
   regret gain. This is the first policy in the project whose result depends on the forecast it is given.
2. **The value is small.** With the table, BC beats BC without it by 1.6-2.0 points (training-seed level: future128 p
   0.012-0.016, future64 p 0.04-0.06). Against MDD-PTECT the clones get −0.5% (64 rows) and −0.9% (128 rows), 5-9% of
   the oracle's 10.4%. The CI of future128 just touches 0, and on the 40 test seeds alone both are within noise of
   MDD-PTECT. Part of the edge over state BC is the table repairing a weak state clone (+1.08%, worse than the fixed
   pair) rather than gaining on the fixed pair.
3. **Not all of future128's edge is information.** With 128 rows, an unrelated future keeps −0.19% (1.3 points better
   than state BC, p 0.045 per instance, 0.06-0.07 per model). Exact vs redrawn is only 0.70 points and not
   significant. Training with the table helped the clone in a way that does not need the true future. For 64 rows the
   whole edge is information (redrawn = state BC).
4. **Robust to timing errors, not to churn.** ±1-30 min jitter costs nothing (paired ±0.1). The clones use coarse
   load in the next 1-2 h, not exact arrival times. That is the part of the future a real schedule gets right. Churn
   costs future64 0.35-0.42 points; future128 holds. The planner drops from −9.4% to −2.2 to −2.8% under the same
   errors and stays 2-3 points ahead of the clones.
5. **For the thesis (RQ2 information row):** with the full schedule as input, imitation recovers about 1 point of the
   oracle's gain, robustly under realistic forecast error. The planner keeps 2-3 points at the same error. Neither
   comes close to the exact-schedule −9.4 to −11.2%. Most of the oracle's gain still needs the exact event sequence,
   which only simulation (the planner) extracts. This replaces "future information does not help a learned policy":
   it helps a little, once it is given as a table and enough labels exist.
6. **The learning curve is still falling** (per-slot regret 5.47 → 4.89 → 4.66% at 250 / 500 / 1,000 seeds). More
   oracle labels are the cheapest way to see whether the 1 point grows: about 3 CPU-min per seed on the cluster
   (1,000 seeds took 1 h 07 min of wall time).

## Data scaling: 4,000 label seeds (10-16, user)

The user chose the cheaper step before deciding on PPO: 3,000 more oracle labels (B2 seeds 4000-6999), then the same
BC on all 4,000 seeds (96,000 slots).
- **Data:** `submit_collect4k.sh` (`collect.py list4k`). Same code as the first 1,000 seeds.
- **BC:** `submit_bc4k.sh` → `bc4k/`. The three arms × BC seeds 0-4 on 3000-6999, and a learning-curve point at
  2,000 seeds (3000-4999, BC seeds 0-2). Same recipe; 10-19 min per model on an A100.
- **Rollouts:** `out/rollout4k/`, every level, as before.
- **Analysis:** `analyze.py --variant 4k` → `analysis4k.out`, `summary4k.csv`. It adds the learning curve and the
  1,000- vs 4,000-seed models.
- **Decision rule set beforehand:** propose PPO only if the table's edge over state BC and over MDD-PTECT grows with
  data, and the redrawn control still removes it. All three hold.

**Checks.**
- 3,000 / 3,000 new label files. Every replay equals its oracle value; the oracle is −10.3% vs MDD-PTECT on these
  seeds.
- 24 / 24 models, no errors. 200 / 200 rollout files. One stamp (des_twin a3638018, future code b4c9a54b).
- The A100 array waited about 8 h for resources. Its time limit was lowered from 4 h to 2 h so it could backfill.

### Per slot: the learning curve

All 4,800 evaluation slots, mean (sd) over BC seeds. Hold-to-end regret of the pick; always MDD-PTECT is 5.66%.

| training seeds | state | future64 | future128 | future128, redrawn table |
|---|---|---|---|---|
| 250 | 5.18 (0.12) | 5.47 (0.30) | – | – |
| 500 | 5.00 (0.14) | 4.89 (0.14) | – | – |
| 1,000 | 4.94 (0.09) | 4.66 (0.19) | 4.57 (0.05) | 4.98 |
| 2,000 | 4.79 (0.04) | 4.33 (0.09) | 4.19 (0.22) | 4.75 |
| 4,000 | 4.66 (0.12) | 4.26 (0.30) | **4.07** (0.27) | 4.92 |

- The gap to state widens from 0.4 to 0.6 points, and future128's regret keeps falling.
- The oracle-pair match rate barely moves (state 31.1%, future128 32.6% at 4,000 seeds). The clones get better by
  avoiding bad picks, not by copying the oracle more often.
- At 4,000 seeds the redrawn table changes 32% of future128's picks and raises its regret to 4.92%, the state arm's
  level.

### Rollouts at 4,000 seeds: pooled % vs MDD-PTECT over all 200 held-out seeds

Family = per-instance mean over the 5 BC seeds; seed bootstrap 95% CI; Wilcoxon p; switching oracle −10.44%.

| arm, published schedule | vs MDD-PTECT | vs state BC (paired) | vs its own exact run (paired) | share of the oracle's gain |
|---|---|---|---|---|
| state BC | +0.44 [−1.04, +2.05], p 0.05 | – | – | −4% |
| future64, exact | −2.20 [−3.17, −1.28], p 1e-8 | −2.64 [−3.85, −1.50] | – | 21% |
| future64, ± 1 / 10 / 30 min | −2.26 / −2.49 / −2.68 | −2.71 / −2.94 / −3.12 | −0.06 / −0.30 / −0.48 | 22-26% |
| future64, 10 / 25% churn | −2.35 / −2.11 | −2.79 / −2.55 | −0.15 / +0.09 (n.s.) | 20-23% |
| future64, **redrawn** | **+0.10** [−1.28, +1.46] | −0.34 (n.s.) | **+2.30** [+1.33, +3.33], p 4e-5 | −1% |
| future128, exact | **−2.55** [−3.54, −1.64], p 2e-10 | **−2.99** [−4.35, −1.84], p 1e-5 | – | **24%** |
| future128, ± 1 / 10 / 30 min | −2.54 / −2.51 / −2.68 | −2.98 / −2.95 / −3.12 | +0.01 / +0.04 / −0.13 (n.s.) | 24-26% |
| future128, 10 / 25% churn | −2.36 / −2.42 | −2.80 / −2.86 | +0.19 / +0.13 (n.s.) | 23% |
| future128, **redrawn** | **−0.39** [−1.88, +1.09] | −0.83 [−1.65, +0.02] (n.s.) | **+2.16** [+1.07, +3.37], p 0.0009 | 4% |

- Fitting seeds alone (n = 160): future128 −2.86%, future64 −2.28%.
- Test seeds alone (n = 40): future64 −1.84% [−3.74, −0.32] (p 0.015), future128 −1.23% [−3.12, +0.36]
  (p 0.025), state BC +1.71%.

**Test seeds 0-39, next to the planner on the same published schedule:**

| published schedule | future64 | future128 | planner, 900 s | planner, 300 s |
|---|---|---|---|---|
| exact | −1.84 | −1.23 | −9.42 | −11.24 |
| ± 1 min | −1.62 | −1.06 | −4.07 | |
| ± 10 min | −2.22 | −0.98 | −2.19 | −2.67 |
| ± 30 min | −1.94 | −1.37 | −2.77 | |
| 10 / 25% churn | −1.83 / −1.55 | −0.78 / −0.76 | −2.23 / −2.31 | −3.30 (10%) |
| redrawn | +0.90 | +0.09 | −1.21 (n.s.) | |

**Training-seed level** (one pooled % vs MDD-PTECT per trained model, 200 seeds):

| arm | BC seeds 0-4 | mean (sd) |
|---|---|---|
| state | +1.17 / +2.09 / −0.73 / −0.46 / +0.14 | +0.44 (1.17) |
| future64, exact | −1.98 / −1.94 / −1.94 / −2.54 / −2.59 | −2.20 (0.33) |
| future64, redrawn | +0.39 / +1.11 / −0.04 / −0.53 / −0.45 | +0.10 (0.67) |
| future128, exact | −2.67 / −2.45 / −2.33 / −2.83 / −2.46 | −2.55 (0.20) |
| future128, redrawn | −0.27 / −0.28 / +0.46 / −0.93 / −0.94 | −0.39 (0.58) |

- Exact vs state: future64 Welch p 0.006, future128 p 0.004. Mann-Whitney p 0.008 for both: every future model is
  better than every state model.
- Exact minus redrawn, per model: future64 −2.37 / −3.05 / −1.91 / −2.01 / −2.14 (paired t p 0.0004); future128
  −2.40 / −2.18 / −2.79 / −1.91 / −1.52 (p 0.0006). Redrawn vs state: p 0.59 / 0.21. The whole edge is information.
- 1,000 → 4,000 seeds: future64 −0.51 → −2.20%, future128 −0.89 → −2.55% (MW p 0.008 both), state +1.08 → +0.44%
  (p 0.42).

### Reading (4,000 seeds)

1. **A learned policy can use the future schedule, given as a table and trained on enough labels.** The 64- and
   128-row clones are −2.2 / −2.55% vs MDD-PTECT on 200 held-out seeds. They beat state-only BC by 2.6-3.0 points
   with no overlap between the five models of each arm. A redrawn future puts them back at the state arm's level.
   At 1,000 seeds the same design gave −0.5 / −0.9%. The 15-feature look-ahead PPO (eval-v5-lookahead-fcast) used
   none of its features.
2. **Next to the other learned policies** on the same 200 seeds: state-only PPO family −1.25% (eval-v5-final);
   look-ahead PPO −2.02%, with no forecast use (eval-v5-lookahead-fcast). future128 BC is −2.55%, and unlike the
   look-ahead PPO its gain disappears without the information.
3. **Robust to realistic forecast error.** 1-30 min arrival jitter and 10-25% churn leave the clones at −2.1 to
   −2.7%. 30 min jitter is even slightly better than exact (−0.1 to −0.5, n.s. for future128): the clones read
   coarse load over the next 1-2 h, not exact timing.
   - On the test seeds the planner at 10-30 min error is −2.2 to −2.8% (−2.7 / −3.3% re-planning every 300 s);
     future64 is −1.9 to −2.2% at the same error.
   - The planner on the exact schedule (−9.4 / −11.2%) stays far ahead. Exact event timing is what simulation
     extracts and imitation does not.
   - A cloned network gets most of the planner's realistic-forecast value without simulating at decision time.
4. **The test seeds (n = 40) agree in sign but are weaker:** future64 −1.84% (p 0.015), future128 −1.23% (CI crosses
   0). The 200-seed pool is the reference; the test set alone has the ±2-point clone noise noted above.
5. **Not saturated.** Per-slot regret is still falling at 4,000 seeds (future128 4.57 → 4.19 → 4.07%), and the
   rollout gain more than doubled from 1,000 to 4,000 seeds. Labels cost about 3 CPU-min per seed (3,000 seeds in
   2.6 h of cluster time).
6. **For the thesis (RQ2 information row):**
   - exact schedule: planner −9.4 to −11.2%;
   - realistic forecast: planner −2.2 to −3.3%, and BC on the full schedule −2.2 to −2.7%, robust;
   - summary features: no use (PPO);
   - state only: 0 to −1.8% (BC, PPO, observable policies).

   The future table is the first learned input that carries forecast value. This replaces the 1,000-seed reading
   ("worth little").

## Proposal (not started): PPO with the future table, and more labels

The decision rule is met, so PPO with the table is worth proposing. Two options, cheapest first. **Neither is
started.**

1. **More labels (cheap, likely gain):** 4,000 more oracle labels (seeds 7000-10999, about 3.5 h on the cluster),
   then the same three arms on 8,000 seeds (models about 25-40 min each on an A100, plus the GPU queue). This tests
   whether BC keeps improving. It is also the strongest candidate method so far: a network that imitates the
   planner's hindsight choices from a published schedule.
2. **`train-v5-future` (PPO):** train-v5-lookahead's stack (train-v5-nokl's frozen code, no prior, no KL, B2, 900 s
   slots, 100k steps, seeds 0-4, one 7-CPU job per seed, about 6-10 h) with `--future-table 128` instead of
   `--policy-lookahead`. Two arms:
   - **(a) from scratch:** the clean test of whether RL finds the forecast use;
   - **(b) actor warm-started from the 4,000-seed future128 BC clones:** whether PPO keeps or improves on what BC
     found. This is the more promising arm, given that PPO from scratch has stayed in the state-only band.

   Evaluate as `eval-v5-future` with eval-v5-lookahead-fcast's protocol: 200 seeds, exact + 6 published-schedule
   levels, the redrawn control, per-model Welch / MW against the future128 BC clones (−2.55%) and the state-only PPO
   family (−1.25%).
   - **Code to write first:** `train.py --future-table N` (obs shape, network choice, run settings, BC warm start
     from a `bc_future128_s<k>.pt`); `evaluate.py` loading via `models.future_net.network_for`, with the twin's
     `future_table` turned on from the checkpoint; tests; a `dev-` smoke. The obs key, network and overlay exist.
   - **Timing:** code + smoke about half a day, training about 10 h, evaluation about 1 h. That fits before the
     10-25 freeze if started by about 10-21.

## Files

| file | what |
|---|---|
| `common.py`, `code/` | frozen v5 code + the two new files; published-schedule adapter |
| `collect.py`, `submit_collect.sh`, `submit_collect4k.sh` | training labels (3000-3999, then 4000-6999) and evaluation future tables |
| `bc_train.py`, `gpu_array.sbatch`, `submit_bc.sh`, `submit_bc4k.sh` | BC models (A100 array), then the rollout array (1,000 / 4,000 seeds) |
| `rollout.py`, `analyze.py` | rollouts at every level; analysis → `analysis.out`, `summary.csv` (`--variant 4k` → `analysis4k.out`, `summary4k.csv`) |
| `checks.py`, `smoke/` | local checks |
| `bc/`, `bc4k/`, `bc_fit160/`, `bc_preview/`, `out/` | models and scores (1,000 / 4,000 seeds), early read (160 seeds), local 1-seed preview, data and rollouts (`out/rollout`, `out/rollout4k`) |
