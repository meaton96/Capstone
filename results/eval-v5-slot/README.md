# eval-v5-slot: PPO decision interval (slot length) sweep (2026-10-21, user after the advisor meeting)

Status: **done 10-22 17:23 (8 training runs + evaluation, local). Figure: `slot_sweep.png`.**

## Why
The final policy chooses a rule pair every 15 minutes because that was the slot length its stack inherited. The only
other point ever tested is "at every decision" (H15 era, `../eval-due-twin`): 11-17% worse than the rule, with
incoherent mixing of rules. Nothing in between was swept. In theory finer decisions can only help an optimal
controller; for PPO, more decisions means harder credit assignment and noisier gradients. Is there a best interval,
or a minimum below which learning breaks down? Together with `../rq2-replan` 1a (the switching ceiling at the same
intervals) this gives one figure: ceiling, planner with forecast error, and learned policy against decision interval.

## Design
- **Arms:** slot length X ∈ {60, 120, 180, 300, 600, 900, 1200, 1500, 1800} s. The 900 s arm is the final-policy run
  itself (`train-v5-cmix`, 5 seeds); every other arm is a new run `train-v5-slot<X>` with seeds 0-2.
- **Recipe:** train-v5-cmix's exactly: frozen code 6d13412a, c ~ U[1.25, 3.5], 16 envs, 2 twin workers, shared
  baseline groups of 4, rollout 128, batch 256, 5,400 s discount horizon (per simulated second, so slot-invariant),
  entropy 0.01 → 0.001 over the run, no prior / KL. Only `--slot-seconds` and `--total-timesteps` change
  (`setup_train.sh` writes each run's `code/` copy and `launch.sh`).
- **Equal experience:** each arm gets the cmix run's 48 updates × 900 / X updates of 2,048 steps, so all arms see
  about the same number of simulated episodes (~3,900). Equal decisions would give the 1-minute arm 15x less
  experience.

  | X (s) | 60 | 120 | 180 | 300 | 600 | 900 | 1200 | 1500 | 1800 |
  |---|---|---|---|---|---|---|---|---|---|
  | updates | 720 | 360 | 240 | 144 | 72 | 48 | 36 | 29 | 24 |

- **Known coupling, kept fixed for a clean first pass:** GAE λ is 0.95 per decision (`--gae-lambda-time-s` unset), and
  rollouts are 128 decisions. So shorter slots also have a shorter credit trace in simulated time (about 1,200 s at
  60 s slots vs 18,000 s at 900 s) and a shorter stretch of time per rollout. A follow-up arm with λ fixed in simulated
  time (`--gae-lambda-time-s 900`) would separate decision frequency from credit horizon.
- **Evaluation (`task.py`):** every arm's seeds + MDD-PTECT on B2 test seeds 0-39 (condition `base`, c ~ U[1.75, 2.5])
  through `../rq3-gen-v5/run_task.py`, the harness that scored the final policy, at the arm's own slot length.

## Analysis (`analyze.py`, `plot.py`)
Per arm: the family mean (and each seed) vs MDD-PTECT, pooled over test seeds (seed-bootstrap CI, Wilcoxon), the
spread over training seeds, training wall time. Then the joint figure with `../rq2-replan` 1a / 1b.

## Cost (measured 10-21, local, loaded machine)
Timing check: the recipe for 3 updates at 60 / 300 / 900 s (scratchpad), with 3 such runs plus the 16-process α sweep
on the machine at once.

| slot | simulation / update | inference / update | network update / update | total / update |
|---|---|---|---|---|
| 60 s | 19 s | 9 s | 62 s | 90 s |
| 300 s | 76 s | 9 s | 67 s | 151 s |
| 900 s | 221 s | 9 s | 60 s | 291 s |

The network update costs about the same per update at every slot length (same batch). Simulation per update grows
with the simulated time an update covers. Per run (3 seeds in parallel, ~12 cores):

| slot (s) | 60 | 120 | 180 | 300 | 600 | 900 | 1200 | 1500 | 1800 |
|---|---|---|---|---|---|---|---|---|---|
| hours | 18.0 | 10.4 | 7.9 | 5.9 | 4.4 | (cmix, done) | 3.7 | 3.5 | 3.4 |

About 57 run-hours for the 8 new arms; about 30-35 h wall with two runs at a time once `rq2-replan` is done.

**Actual (10-21 / 10-22):** all 8 runs finished between 05:20 and 15:40 on 10-22, faster than estimated. Wall time per
seed: 15.0-15.6 h (60 s), 8.1-8.3 h (120 s), 5.7-5.8 h (180 s), 4.2 h (300 s), 2.9-3.0 h (600 s), 2.3 h (1200 s), 2.1 h
(1500 s), 2.0-2.1 h (1800 s). Every seed completed with its checkpoint; critic explained variance at the end 0.71-0.97.
The evaluation (`eval-v5-slot`) started 15:40.

## Results (10-22)
B2 test seeds 0-39, window tardiness vs MDD-PTECT, pooled (95% seed-bootstrap CI, test seeds better / worse,
Wilcoxon); < 0 better. `analysis.out`, `summary.csv`, `curve.csv`, figure `slot_sweep.png` (`plot.py`).

**Checks:**
- The 15-min arm reproduces `../eval-v5-cmix` (test / base) exactly: 240 / 240 episodes identical.
- MDD-PTECT is identical in every arm, so the reference does not depend on the slot length.

| slot | PPO family vs MDD-PTECT | training seeds | seed sd | ceiling, exact forecast | share of ceiling |
|---|---|---|---|---|---|
| 1 min | **+6.22** [+3.47, +8.74], 11/29, p 0.002 | +7.1 / +13.9 / −2.3 | 8.2 | −11.46 | — |
| 2 min | +0.32 [−1.73, +2.33], n.s. | +1.8 / +0.7 / −1.5 | 1.7 | −11.37 | — |
| 3 min | −1.69 [−3.43, −0.18], p 0.07 | −1.7 / −1.2 / −2.2 | 0.5 | −11.03 | 15% |
| 5 min | −1.83 [−3.52, −0.18], p 0.04 | −2.1 / −2.2 / −1.3 | 0.5 | −11.24 | 16% |
| **10 min** | **−2.97** [−4.98, −1.48], 29/11, p 8e-5 | −3.0 / −2.7 / −3.2 | **0.25** | −9.68 | **31%** |
| 15 min (final policy's run, 5 seeds) | −1.69 [−2.83, −0.51], p 0.006 | −0.3 / −2.3 / −1.0 / −1.7 / −3.2 | 1.1 | −9.42 | 18% |
| 20 min | −1.52 [−2.81, −0.47], p 0.01 | −2.3 / −1.4 / −0.9 | 0.7 | −9.12 | 17% |
| 25 min | −1.41 [−3.07, +0.02], p 0.03 | +0.7 / −1.1 / −3.8 | 2.3 | — | — |
| 30 min | −1.63 [−3.25, −0.19], p 0.01 | −2.6 / −1.7 / −0.7 | 0.9 | −7.99 | 20% |

Neighbour comparisons for the 10-min arm (family, paired over test instances | training-seed level):
- vs 5 min: −1.16% [−3.01, +0.46], p 0.10 | Welch p 0.04;
- vs 15 min: −1.30% [−3.01, +0.05], p 0.053 | Welch p 0.06;
- vs 20 min: −1.47% [−3.29, −0.09], p 0.16 | Welch p 0.05.

Short slots against 5 min: 1 min +8.20% (p 7e-7), 2 min +2.19% (p 0.03).

### Reading
- **Deciding more often does not help PPO; it breaks down below 3 minutes.** At 1-min slots the family is 6% worse
  than MDD-PTECT and the training seeds range from −2% to +14%. At 2 min it is at parity. This is the same
  instability as per-decision PPO in the H15 era, now with the point where it starts: below about 3 minutes.
- **From 3 to 30 minutes PPO sits on a plateau of −1.4 to −3.0%.** 10 minutes is the best arm (−3.0%, p 8e-5, the
  tightest seed spread, 31% of the ceiling at that interval). Its edge over 5 / 15 / 20 minutes (1.2-1.5 points) is
  borderline with 3 training seeds per arm (p ≈ 0.04-0.16). The 15-minute slot the final policy uses is on the
  plateau, about 1.3 points short of the best arm.
- **The ceiling and the learner pull in opposite directions** (figure). The exact-forecast planner gains most with
  frequent decisions (−11.2% at ≤ 5 min), while PPO gets worse below 3 minutes. Only 15-31% of the ceiling is
  realized anywhere, and most of what is missing is foresight: the planner with ±10-min arrival errors sits at
  −1.4 to −2.9%, about where PPO is.
- **Why short slots fail (hypothesis):** the recipe's GAE λ = 0.95 is per decision and its rollouts are 128 decisions.
  At 1-min slots credit therefore reaches back only about 20 minutes of simulated time (about 5 h at 15 min), and the
  same experience is split into 15x more, noisier decisions. A follow-up arm with λ fixed in simulated time
  (`--gae-lambda-time-s 900`) at 1-2 min would separate "too many decisions" from "too short a credit horizon".
- **Answer to the advisor:**
  - for the planner, 5 minutes (finer adds nothing);
  - for the learned policy, 5-20 minutes all work, about 10 minutes looks best, and below 3 minutes learning becomes
    unstable;
  - "continuous" decisions are not better for either, and are much worse for PPO.
