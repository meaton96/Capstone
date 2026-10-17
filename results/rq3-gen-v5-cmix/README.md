# rq3-gen-v5-cmix: the RQ3 conditions sweep for the final policy (PPO trained on mixed due dates) (2026-10-17, user)

Status: **queued** (lab `rq3-gen-v5-cmix`, cluster, 16 twin tasks).

## Why
On 10-17 the user chose the PPO trained on mixed due dates (`train-v5-cmix`, c 1.25-3.5; pick **s3** on the fitting
seeds, `../eval-v5-cmix`) as the thesis's final policy, with all other policies' numbers reported next to it.
`../rq3-gen-v5` ran the 11 RQ3 conditions with the B2-trained pick s4. This adds the cmix family (and the
loose-trained family, `train-v5-loose`, pick s2) on the 8 conditions `../eval-v5-cmix` did not cover. base, due-tight
and due-loose on the same test seeds come from `../eval-v5-cmix/test`.

## Method (`task.py`, `submit.sh`, `pull.sh`, `analyze.py`)
- **Harness:** a thin wrapper around `../rq3-gen-v5/run_task.py`, as in `../eval-v5-cmix/cmix_task.py`. It uses the
  same frozen code (`../rq2-rebase-v5/code/env`, 6d13412a), floors, B2 settings, fleet wrapper and condition
  definitions.
- **Groups:**
  - `cmix`: train-v5-cmix s0-s4 + MDD-PTECT;
  - `loose`: train-v5-loose s0-s4 + MDD-PTECT.

  The MDD-PTECT run is a run-to-run check against rq3-gen-v5's pdr group.
- **Conditions:** load-0.8, load-1.2, agv5 / 7 / 9-bay9, lay-c, lay-g, lay-j; B2 test seeds 0-39.
- **Analysis:** per condition, vs the condition's best v5 pair (in-sample) and vs MDD-PTECT, for:
  - the cmix pick s3 and the cmix mean of 5;
  - the loose pick s2 and the loose mean of 5;
  - the B2-trained pick s4 and its mean of 5, and BC (rq3-gen-v5);
  - the hindsight best pair.
- **Check (local smoke, 10-17):** lay-j / cmix on seeds 0-1 gives 12 episodes. MDD-PTECT equals rq3-gen-v5's pdr run
  (93,721.72 / 28,369.25).

## Results
Pending. The analysis already fills base / due-tight / due-loose from eval-v5-cmix: the cmix pick is −1.7 / −1.1 /
−6.7% vs MDD-PTECT there.
