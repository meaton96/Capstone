# train-v5-lookahead: does PPO use the exact future arrivals? (v5 stack, 2026-10-15, user)

Status: **done** 2026-10-15 19:02 (cluster jobs 21895689-93, one 7-CPU job per seed after 21894861 sat in the queue;
lab `train-v5-lookahead`; 5 seeds × 100k steps, 5.9 h for s0, 9.4-9.7 h for s1-s4).

**Relaunch (10-15 06:40):** the single 5-seed job (27 CPUs, 48 GB, 24 h limit) had an estimated start of 10-17
11:13: no node with 27 free CPUs. It was cancelled and relaunched as one job per seed (7 CPUs, 16 GB, time limit
lowered to 10 h, or 18 h for look-ahead). Jobs train only (`code/train_only`); the evaluation is a separate lab
experiment. `launch.sh` is the per-seed launcher.

## Why
- The thesis says "giving the policy the future schedule does not help". That rests on two tests:
  - rq3-lookahead-probe: supervised models on summary look-ahead features, S6 setting, H15 rules;
  - dev-policy-lookahead (10-08): PPO with the same features, but on the MDD-TECT-prior + KL 0.05 stack, which pinned
    11 of 12 policies to the prior whatever their inputs.
- No policy has been trained with future information on the stack that learns: no prior, no KL, v5 rules
  (train-v5-nokl; eval-v5-curve: −1.75% at 51.2k steps, the realizable bound).
- **Reference points:**

  | method | vs MDD-PTECT |
  |---|---|
  | twin planner on the exact schedule, re-planning every 300 s | −11.2% |
  | hindsight switching oracle | −9.4% |
  | state-only PPO, mean of 5 seeds, 200 held-out seeds | −1.25% |
  | planner on a schedule off by 5-30 min | −2 to −2.8% (rq2-perturbed-v5) |

## Setup
- train-v5-nokl's exact stack and frozen code (the 6d13412a snapshot in `code/`), plus `--policy-lookahead`.
- The 15 features of `des_twin/lookahead.py` go to the actor and the critic. They are computed from the **exact**
  future, over the next 30 / 60 / 120 min: arrivals per hour, offered load, bottleneck load, due-date slack of
  arriving jobs, active AGV fraction.
- Seeds 0-4, 100k steps, B2, 900 s slots.
- **Local smoke (10-15):** 256 steps train and save on the frozen code; the checkpoint records
  `policy_extra_dim 15`.
- The job's own evaluation runs on B2 test seeds 0-39 (evaluate.py turns the features on). The full comparison is
  `../eval-v5-lookahead`.

## Next, depending on the result
- **If the policy uses the look-ahead:** feed it features computed from a perturbed published schedule, using
  rq2-perturbed-v5's forecast model at 1 / 10 / 30 min jitter and 10 / 25% churn. Score it against the planner at
  the same error level. Train on noisy forecasts if it degrades below the state-only policy.
  - Re-running the forward pass on every schedule update solves staleness, but not forecast error. That is why the
    planner, which already re-plans on the newest schedule, drops from −9.4% to −2 to −3%.
- **If it does not:** summary features may be too coarse for the exact event sequence the oracle uses. The next step
  is a future-jobs encoder: the next N arriving jobs as a set, like the job table.

## Results
- All 5 seeds trained to update 48 / 48 and saved `results/train-v5-lookahead_s*/checkpoint.pt`, with no errors.
- Training logs can't tell whether the look-ahead helps. Episode returns are noisy across redrawn instances, and the
  critic fits well (EV about 0.93 from update 15).
- The score is in `../eval-v5-lookahead` (cluster job 21898928, started 19:02). Compare against the base-trained
  family, which was trained at the same c range, not against cmix.
