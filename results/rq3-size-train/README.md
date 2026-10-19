# rq3-size-train: does training on larger or mixed floor sizes give a policy that transfers across size? (2026-10-19, user)

Status: **phase 1 (floor exports, 1000-machine probe) and phase 2 (training) running**; phase 3 (cross-evaluation grid)
planned.

## Why
- The user's proposal claims the policy transfers to a floor of any size. `../rq3-size-v5`: the final policy (trained
  on 15 machines) keeps its edge at 50 machines (−2.1%) but not at 100 (family +2.6% vs MDD-PTECT).
- `../rq3-size-remap`: rewriting the size-dependent inputs does not close that gap. So it is a change in the floor's
  dynamics and rule landscape, not in input scaling.
- The user (10-19) asked:
  - can a policy trained on 15 / 50 / 100 machines go to sizes near its own (15 → 20, 100 → 110) but not an order of
    magnitude away (15 → 100, 100 → 1000)?
  - does a policy trained on a mix of sizes generalize better than one that saw a single size?
- Hypotheses:
  1. performance falls with the size ratio |log(test / train)|, not with absolute size;
  2. mixed-size training stays flat across 15-100;
  3. it degrades less than single-size policies beyond 100.

## Design (user-approved defaults, 10-19)
- **Training arms:** the final policy's stack (train-v5-cmix: frozen v5 code 6d13412a, c ~ U[1.25, 3.5], no prior,
  no KL, shared baseline groups of 4, 900 s slots, 100k steps), 5 seeds each.

  | arm | sizes | envs | lab |
  |---|---|---|---|
  | 15 only | the existing final policy family | 16 | train-v5-cmix (done) |
  | 50 only | 50 | 16 | `train-size-m50` |
  | 100 only | 100 | 16 | `train-size-m100` |
  | mixed | 15, 50, 100: environment e gets size [(e // 4) mod 3], one baseline group per size | 12 | `train-size-mix` |

- **Floors and fleets** (as `../rq3-size-v5`):
  - 15 = B2 itself (floor7, 7 AGVs, B2 load);
  - 50 / 100 = `../agv-ratio-scale` floors, with 0.52 AGVs per machine (26 / 52 on duty, short blocks half), B2
    load × 0.9 and the arrival cap off.

  The observation is padded to the largest floor (100 machine rows, 1,792 job rows); the network is unchanged.
- **Test grid (phase 3):** 15, 20, 30, 50, 75, 100, 150, 200 machines, at the same per-machine scaling
  (round(0.52 M) AGVs). Test seeds 0-39, each arm's 5 seeds, plus MDD-PTECT and the 6 v5 pairs per floor. 1000
  machines is included if `../dev-size-1000-probe` shows the twin can run it in reasonable time.
- **No size-normalized-input arm:** `../rq3-size-remap` showed the inputs are not the bottleneck at 100.

## Code (`train_size.py`, `../scripts/size_train_cluster.sh`)
- `train_size.py` is copied into each run's `code/` next to the frozen `env/`.
  - **Single size:** it sets `--twin` / `--params`, with rq3-gen-v5's fleet wrapper where the floor has more bays
    than AGVs on duty.
  - **Mixed:** it patches `TwinSchedulingEnv.__init__` so env e builds its own floor and generator for size
    [(e // 4) mod 3], and pads the observation rows. The frozen code is not edited; the twin workers fork after the
    patch.
- The cluster jobs (one per seed, `submit_dev_cluster.sh`, `TRAIN_SCRIPT=size_train_cluster.sh`) only train.

## Checks (local, 10-19)
- **Mixed training runs end to end:** 3 envs (15 / 50 / 100), 2 updates, checkpoint saved.
- **Every env gets its own floor:** with group 2, envs 0-5 get sizes 15, 15, 50, 50, 100, 100, with floors of 15 / 50 /
  100 machines and 7 / 26 / 52 AGVs on duty. Observation shapes are uniform (100 × 16 machine table, 1792 × 21 job
  table).

## Phases and lab experiments
1. `dev-size-floors`: exports of the 20-1000 machine floors from the current player (m20 done by hand); a check of the
   new m50 / m100 exports against the v4 ones.
2. `dev-size-1000-probe`: one MDD-PTECT and one final-policy episode at 1000 machines, timed. Waits on the exports.
3. `train-size-m50`, `train-size-m100`, `train-size-mix`: cluster training, running.
4. `rq3-size-grid` (planned): the cross-evaluation grid, after training and the exports.

## Results
Pending.
