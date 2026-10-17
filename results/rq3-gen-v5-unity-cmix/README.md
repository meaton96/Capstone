# rq3-gen-v5-unity-cmix: the RQ3 Unity subset for the final policy (cmix pick s3) (2026-10-17, user)

Status: **queued** (lab `rq3-gen-v5-unity-cmix`, cluster, 60 Unity tasks).

## Why
The user (10-17) made the PPO trained on mixed due dates (train-v5-cmix pick s3) the final policy.
`../rq3-gen-v5-unity` checked the RQ3 twin answer in Unity on load-1.2, due-tight and lay-g with the B2-trained
pick s4. This runs the cmix pick (and the loose pick s2) on the same conditions and seeds, so the Unity subset covers
the final policy.

## Method (`task.sh`, `array.sbatch`, `submit.sh`, `analyze.py`)
- **Harness:** `../eval-v5-cmix-unity`'s task (group `pc` = MDD-PTECT + loose s2 + cmix s3) on
  `../rq3-gen-v5-unity`'s conditions (`conds.json` is its copy).
- **Settings:** the same code, player and settings: 6d13412a snapshot, `linux_server/` 5e6e0f87 without
  `-pyrules`, random warm-up, 21,600 s window, 900 s slots, deterministic.
- **Seeds:** B2 test seeds 0-19, 3 conditions × 20 seeds = 60 tasks of 3 episodes.
- **Reuse:** the 6 pairs and s4 come from rq3-gen-v5-unity's pa / pb runs. pc's MDD-PTECT checks that reuse seed for
  seed.
- **Analysis:** in Unity and in the twin on the same seeds, the cmix pick, loose pick, B2-trained pick and hindsight
  best pair, each vs MDD-PTECT and vs the best pair.
  - Twin sources: rq3-gen-v5 for the pairs and s4, rq3-gen-v5-cmix / eval-v5-cmix for the cmix and loose families.

## Results
Pending.
