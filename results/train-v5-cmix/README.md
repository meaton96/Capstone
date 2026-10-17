# train-v5-cmix: PPO on the v5 stack, trained at due-date allowance c ~ U[1.25, 3.5] (2026-10-14, user)

Status: **done** (see Results).

**Relaunch (10-15 06:40):** the single 5-seed job (27 CPUs, 48 GB, 24 h limit) had an estimated start of 10-17
11:13: no node with 27 free CPUs. It was cancelled and relaunched as one job per seed (7 CPUs, 16 GB, time limit
lowered to 10 h, or 18 h for look-ahead). Jobs train only (`code/train_only`); the evaluation is a separate lab
experiment. `launch.sh` is the per-seed launcher.

## Why
Does one policy trained on a balance of tight and loose due dates adapt to the tightness and beat the best rule where it can? c is redrawn per 1.5 h block, so tightness changes within an episode. The policy sees it through the due-date columns of observation v3 (slack, late shares).

## Setup
- train-v5-nokl's exact stack and frozen code (the 6d13412a snapshot copied into `code/`), seeds 0-4, 100k agent
  steps.
- The only change: the training generator's `due_date_allowance_range` is [1.25, 3.5] instead of B2's [1.75, 2.5].
- How it is passed:
  - `launch.sh` uses `results/scripts/dev_b2_slot_train_eval_cluster_c.sh` via `TRAIN_SCRIPT`;
  - the script reads the range from `code/c_range`, because a JSON `--params` does not survive
    `submit_dev_cluster.sh`'s `sbatch --wrap`.
- A local 512-step smoke on 10-14 confirmed train.py received `"due_date_allowance_range": [1.25, 3.5]` for the cmix
  range.
- **The job's own evaluation stays on B2** (c 1.75-2.5, test seeds 0-39). The tight / base / loose evaluation, with
  picks on the fitting seeds, is `../eval-v5-cmix`.

## Results
**Done** 2026-10-15 11:10 (5 per-seed jobs). Results: `../eval-v5-cmix/README.md`. Family mean of 5 vs MDD-PTECT on 200 held-out seeds: −3.5% at loose c, −2.1% at base c (147 / 200, p 1e-10), −0.6% at tight c; pick s3.
