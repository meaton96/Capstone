# rq4-uncert-v5-policy: the final policy under the twin uncertainty layer (2026-10-17, user)

Status: **queued** (lab `rq4-uncert-v5-policy`, cluster, 96 twin tasks).

## Why
RQ4's scope was open: breakdowns only, or "disruptions and uncertainty". The user (10-17) asked to set up the
uncertainty part. `../rq4-uncert-screen` (10-11) scored the v5 rules, a re-planning planner and the hindsight oracle
on 38 noise cells, but gate G1 stopped training under noise, so no learned policy was ever scored there. This runs the
final policy, the PPO trained on mixed due dates (cmix pick s3), with all PPO families and the 6 v5 pairs in the
screen's cells.

## Method (`task.py`, `submit.sh`, `pull.sh`, `analyze.py`)
- **Harness:** a thin wrapper around `../rq3-gen-v5/run_task.py` (frozen code `../rq2-rebase-v5/code/env` = the
  checkpoints' code, B2, test seeds 0-39) that adds `--uncertainty <cell>`. The noise seed is the instance seed, so
  every policy sees the same noisy world. Policies act on the observed (noisy) state.
- **Cells (24):**
  - realistic / severe for E1, E2, E2d, E3, E4n, O1, O2, O3 and mix (18);
  - the online-estimator cells E2 / E2d / mix × realistic / severe with info=I1 (6).
  - `none` = B2 (rq3-gen-v5 base, eval-v5-cmix test).
- **Groups:**
  - pdr: 6 v5 pairs + MDD-TECT;
  - ppo: B2-trained s0-s4;
  - cmix: s0-s4;
  - loose: s0-s4.

  Each policy group also runs MDD-PTECT.
- **Analysis:** per cell, every pick and family mean vs MDD-PTECT (the screen's noise-free rule NF) and vs the cell's
  best pair, plus MDD-PTECT's degradation from `none`.

## Reproduction check against the screen (local, seed 0, MDD-PTECT; `analyze.py` repeats it on all seeds)
**MDD-PTECT is identical to the screen's MDD-PTECT@3 in 16 of 24 cells, and at `none` on 40 / 40 seeds.** That covers
every E1, E2, E2d, E4n, O1, O2 and O3 cell.

It differs in E3 realistic / severe (172.9 vs 173.5, 279.0 vs 328.4), mix realistic / severe (170.9 vs 171.5, 306.4 vs
355.6) and all 6 info=I1 cells (up to 380.7 vs 450.8).
- **Not the code version:** the screen's own harness gives the screen's values with both snapshots (ba26f3f5 and
  6d13412a).
- **The evaluator path differs:** `evaluate.py` → `TwinSchedulingEnv` builds the world from the scenario's uncertainty
  block, while the screen built the trace directly (`uc.UEpisode`). Both start the window at the first agent decision.
  The routing trigger is not the cause.
- **Not yet explained:** the cause is not found.
- **Consequence:** inside this experiment, policies and pairs share one path, so their comparison is valid in every
  cell. The screen's planner / oracle / level-tuned-rule numbers carry over only to the 16 matching cells.

## Results
Pending.
