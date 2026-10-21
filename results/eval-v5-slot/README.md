# eval-v5-slot: PPO decision interval (slot length) sweep (2026-10-21, user after the advisor meeting)

Status: **queued 10-21 (user).** The 8 training runs `train-v5-slot<X>` are lab futures behind `rq3-alpha-size` (8 workers
each, longest first: one alongside `rq2-replan`, two at a time after it); `eval-v5-slot` waits on all 8. Expected done
about 05:00 on 10-23.

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

## Analysis (to write before the evaluation lands)
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

## Results
Pending.
