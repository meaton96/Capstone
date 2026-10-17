# rq4-agvfail-v5-cmix: the RQ4 AGV-breakdown half for the final policy (cmix pick s3) (2026-10-17, user)

Status: **queued** (lab `rq4-agvfail-v5-cmix`, cluster, 240 Unity tasks).

## Why
The user (10-17) made the PPO trained on mixed due dates (train-v5-cmix pick s3) the final policy.
`../rq4-agvfail-v5-rules` / `-policy` ran RQ4's AGV breakdowns with the fixed pairs, the B2-trained PPO family (pick
s4) and BC. This adds the cmix family and the loose pick, so RQ4 reports the final policy under breakdowns.

## Method (`submit.sh`, `analyze.py`; task / array scripts are `../rq4-agvfail-v5-rules`'s)
- **Settings:** Unity, B2 test seeds 0-39, AGV breakdowns none / Weibull λ 3000 / 1500 s. The setup, code, player and
  window are as in rq4-agvfail-v5-rules.
- **Groups**, added to the shared `task.sh` on 10-17 (existing groups unchanged):
  - `cmixa`: cmix s0-s2 + MDD-PTECT, a run-to-run check against the rules half;
  - `cmixb`: cmix s3 (final), cmix s4, loose s2. At `none`, its s3 / loose s2 repeat `../eval-v5-cmix-unity`'s base
    runs (run-to-run check).
- **Analysis**, per level vs MDD-PTECT at that level, for:
  - cmix s3, the cmix mean of 5 and loose s2;
  - the B2-trained s4 and its mean of 5, MDD-PECT and the hindsight best pair;
  - plus the degradation from no breakdowns.

## Results
Pending.
