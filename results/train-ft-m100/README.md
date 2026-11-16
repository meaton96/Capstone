# train-ft-m100 / train-ft-het: can a short fine-tune adapt the final policy to a very different floor? (2026-10-25, plan §2b)

Status: **both arms trained.** `train-ft-het`: local, 10-26 17:51-18:55, evaluated 10-26 (`../eval-ft`, het arm).
`train-ft-m100`: cluster, jobs 21964586-92, 10-26 03:59 to 10-28 07:01, all 5 seeds 24 / 24 updates (Training, below);
its evaluation (`eval-ft-m100`) started 10-28 07:02.

This README holds the design and checks of both fine-tune arms; `../train-ft-het/README.md` only adds the het specifics.

## Why
- The final policy (`train-v5-cmix` s0-s4, trained on B2 at 15 machines) loses its edge on very different floors:
  - at 100 machines the family is +2.6% vs MDD-PTECT (`../rq3-size-grid`), while T100, trained there from scratch for
    48 updates, is −0.08%;
  - on B2 with machine-speed spread s = 1 it gains only −2.9% (`../eval-alpha-spread`), while the spread law is −15.3%.
- Plan §2b asks whether a short fine-tune adapts it, and how short. The user chose the targets on 10-25 (§9 item 2):
  100 machines and the spread-1.0 floor, not 200.

## Design
- **Start:** `results/train-v5-cmix_s<k>/checkpoint.pt`, k = 0-4 (global step 98,304 = 48 updates of 128 × 16).
- **One resumed run per seed and target, 24 updates long, a checkpoint after every update** (`--save-every 1`). The
  budgets are read off one run:

  | budget (updates) | step | checkpoint | share of a full run (48) |
  |---|---|---|---|
  | 2 | 102,400 | `checkpoint_step102400.pt` | 4% |
  | 5 | 108,544 | `checkpoint_step108544.pt` | 10% |
  | 10 | 118,784 | `checkpoint_step118784.pt` | 21% |
  | 24 | 147,456 | `checkpoint.pt` (= `checkpoint_step147456.pt`) | 50% |

- **Targets:**

  | arm | floor | how | where |
  |---|---|---|---|
  | `train-ft-m100` | 100 machines: `../agv-ratio-scale/floors/m100`, 52 AGVs on duty (short blocks 26), B2 load × 0.9, arrival cap off | `code/train_size.py --sizes 100`, as `../train-size-m100` | cluster, one job per seed (16 twin workers, 21 CPUs, 32g, 16 h limit) |
  | `train-ft-het` | B2 (floor7, 7 AGVs, B2 load) with machine-speed spread s = 1: factors log-uniform in [0.5, 2], capacity-matched per type, B2-tight due dates | `code/hetero.py --spread 1,1` (`../dev-alpha-action`) | local, 5 seeds at once × 3 workers |

- **Everything else is the final policy's stack** (frozen v5 code 6d13412a in `code/env`, identical in both arms):
  c ~ U[1.25, 3.5], tardiness reward, random warm-up, 21,600 s window, 900 s slots, 5,400 s discount horizon, 16 envs,
  rollout 128, batch 256, 4 epochs, lr 3e-4, shared baseline groups of 4 (residual), entropy 0.01 → 0.001, no prior,
  no KL.
- **Code:** `../scripts/ft_train.sh NAME "SEEDS" WORKERS` reads the target from `code/target`. A seed resumes from its
  own latest checkpoint if its run folder has one (a restarted job), otherwise from cmix. The cluster jobs go through
  `../scripts/submit_dev_cluster.sh` (`TRAIN_SCRIPT=ft_train.sh`; `JOB_TIME` added 10-25, default unchanged).
  `pull.sh` brings back the run folders with only the budget checkpoints.

## Checks (10-25, before queueing)

**The resumed network accepts the new floor.**
- `check_obs_schema` compares the observation layout (column meanings and per-row widths), not the row caps.
  `check_action_layout` compares the v5 heads. Both arms run the code cmix was trained with, so both pass.
- Row caps are set per launch from the generated instances (`row_caps_for`). At 100 machines they are 100 machine
  rows and 1,792 job rows (cmix: 15 / 256). The set encoders have no row-count parameters, so the 15-machine weights
  load unchanged.
- T100 trained with the same caps, and `../rq3-size-v5` / `../rq3-size-grid` evaluated T15 on m100 the same way.
- The m100 smoke loaded the cmix checkpoint, ran 2 updates and saved checkpoints recording row caps 100 / 1,792.

**Entropy: the schedule continues, it does not restart** (user accepted 10-25).
- `--total-timesteps` is the cumulative target, 98,304 + 24 × 2,048 = 147,456, and the coefficient is
  0.01 − 0.009 × step / total.
- Computed with the frozen `entropy_coef_at`:

  | where | step | entropy coefficient |
  |---|---|---|
  | cmix, update 1 | 2,048 | 0.00982 |
  | cmix, update 48 (its last) | 98,304 of 100,000 | 0.00115 |
  | fine-tune update 1 | 100,352 | 0.00388 |
  | budget 2 | 102,400 | 0.00375 |
  | budget 5 | 108,544 | 0.00338 |
  | budget 10 | 118,784 | 0.00275 |
  | budget 24 | 147,456 | 0.00100 |

- So the fine-tune starts at 3.4× cmix's final coefficient and decays back to 0.001 by update 24.
- The 2 / 5 / 10-update budgets are snapshots of this 24-update schedule, not runs annealed over their own length.
- The smokes logged 0.001006 / 0.001 for their own total (98,432), which matches the formula, so the code does what
  is computed here.

**Prior / KL match cmix.**
- cmix's checkpoint: `prior_pair ''`, `prior_init offset`, `prior_kl_coef 0`.
- The launcher passes `--prior-init offset --prior-pair "" --prior-kl-coef 0`, as the size launcher did.
- With an empty prior pair the resume skips the prior branch entirely. The smoke checkpoints record prior '' and KL 0.

**Other resume details.**
- **Optimizer:** Adam's state is restored from the checkpoint. lr 3e-4 is constant (no annealing).
- **Instances:** a resume keys its instance stream on the resumed step (`seed_stream` = 98,304), so the fine-tune
  sees new training instances (seeds ≥ 10,000; the test seeds 0-39 are never trained on).
- **PopArt return statistics** come from the checkpoint: mean −38.9, std 38.5, after 48 updates.
  - They move toward the new floor's returns at about 5% per update (β 0.05, bias-corrected).
  - At 100 machines returns are about 2-3× larger (T100's first update: −48 vs cmix's −21.5), so the critic starts
    under-scaled.
  - The residual shared baseline takes most of the critic out of the policy gradient.
  - In the m100 smoke the explained variance stayed positive (+0.23, +0.59).
  - Watch the value loss and explained variance of the first updates in `train_s<k>.out`.
- **Reward:** the frozen `tardiness.json` gives the same archived spec and code as cmix (which used the live copy)
  and T100: identical md5 for `spec.json` and `tardiness.py`.

**Smoke `dev-ft-smoke` (local, scratchpad, 10-25).** Real launcher, 4 envs, rollout 16, 2 twin workers, 2 updates per
target:
- **m100 (seed 0):**
  - resumed at 98,304;
  - 2 updates, the first in 269 s on the loaded machine;
  - `checkpoint_step98368.pt`, `checkpoint_step98432.pt`, `checkpoint.pt` and `DONE_s0` written.
- **het (seed 1):**
  - resumed at 98,304;
  - 2 updates in 16 s;
  - `hetero.json` written.
- **het instances (seed 2, `HETERO_LOG`):** every instance generated in the forked twin workers had s = 1 (12 / 12).

## Cost and placement
- **m100:**
  - An update at 100 machines took 23-26 min in T100 (16 workers; 18.6-20.6 h for 48), so 24 updates ≈ 10 h per
    seed.
  - The 16 h limit leaves margin and backfills sooner than 24 h. A job that hits it keeps its per-update
    checkpoints; resubmitting (`launch.sh`) resumes.
  - The cluster had 72 idle CPUs on 10-25, fragmented (no node with 21 free).
  - **Fallback if the jobs have not started by ~10-27:** run locally after `train-ft-het` (5 seeds × 6 workers, an
    estimated ~15 h on 30 threads). Before that, cancel the Slurm jobs (`lab cancel train-ft-m100`) and re-add it as a
    local run with `ft_train.sh train-ft-m100 "0 1 2 3 4" 6`.
- **het:** 15 machines. cmix took ~195 s per update with 2 workers on the cluster, so 24 updates ≈ 1-1.5 h. All 5
  seeds run at once on the 20 local slots.

## Outputs
- `results/train-ft-<arm>_s<k>/`: `checkpoint_step*.pt` (every update; on the cluster, only the budgets are pulled),
  `checkpoint.pt`, `episodes.csv`, TensorBoard events, `timing.json`; het also `hetero.json`.
- `results/train-ft-<arm>/`: `train_s<k>.out`, `DONE_s<k>`, `code/` (frozen code, `target`, `c_range`, het `spread`,
  `STAMP`).

## Training (m100, 10-28)
- **Complete:** all 5 seeds reached update 24 / 24 (global step 147,456, `progress.json`). The budget checkpoints
  (`checkpoint_step102400/108544/118784.pt`, `checkpoint.pt`) are pulled for every seed.
- **Time:** 9.5-10.0 h of training per seed (34,360-35,992 s; 1.4 steps/s, 93% of it in the twin env), as estimated.
  Finished 10-26 13:42 (s0), 10-27 03:42 (s1), 10-27 21:01 (s2), 10-27 23:08 (s3) and 10-28 07:01 (s4). Each seed
  resumed once, from cmix's step 98,304, so no job hit the 16 h limit and restarted.
- **The critic caught up with the larger returns** (the risk named under Checks):

  | update | value loss (5 seeds) | explained variance | entropy |
  |---|---|---|---|
  | 1 | 1.42-4.08 | +0.21 to +0.40 | 0.80-1.12 |
  | 5 | 0.75-0.95 | +0.60 to +0.75 | 0.86-1.11 |
  | 20 | 0.17-0.32 | +0.54 to +0.79 | 0.86-1.09 |

  The explained variance stayed positive from update 1 and the value loss fell 8-17× by update 20. The entropy did
  not collapse.
- The logged training returns swing with the instance mix (e.g. s0 −248 to −911 per episode across updates), so they
  say nothing about whether the policy improved. That is `../eval-ft`'s job.

## Results
het: `../train-ft-het/README.md` and `../eval-ft/README.md`. m100: evaluation running (`eval-ft-m100`, 10-28).
