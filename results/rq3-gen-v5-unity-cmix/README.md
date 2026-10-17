# rq3-gen-v5-unity-cmix: the RQ3 Unity subset for the final policy (cmix pick s3) (2026-10-17, user)

Status: **done** 2026-10-17 16:23 (cluster array 21909825, 1 h 33 min, 60 / 60). `analysis.out`.

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
**Checks.** pc's MDD-PTECT equals rq3-gen-v5-unity's run on 20 / 20 seeds in all three conditions.

**vs MDD-PTECT, B2 seeds 0-19** (pooled % [95% CI], better / worse, Wilcoxon p):

| condition | final policy cmix s3, Unity | cmix s3, twin | loose s2, Unity | B2-trained s4, Unity | hindsight best pair, Unity |
|---|---|---|---|---|---|
| load-1.2 | **−4.1** [−7.2, −1.2], 14/5, p 0.009 | −1.8 (p 0.26) | −0.8 | −2.3 (p 0.08) | −4.6 |
| due-tight | −2.8 [−5.5, −0.4], 11/8, p 0.11 | −0.5 | −1.3 | −2.9 (p 0.14) | −5.7 |
| lay-g | **−2.9** [−5.4, −1.2], 15/5, p 0.004 | −3.2 (p 0.18) | −0.5 | −1.7 | −4.3 |

**vs the condition's best pair in Unity:**
- lay-g: −2.9% (p 0.004; the best pair is MDD-PTECT);
- load-1.2: −2.5% (vs MDD-PECT, p 0.13);
- due-tight: −0.05% (vs SRT-PECT).

## Reading
1. **The final policy's twin gains hold in Unity, and in two conditions they are larger.** In Unity cmix s3 beats
   MDD-PTECT by 2.8-4.1% in all three conditions, significant at load-1.2 and lay-g (20 seeds). On the same 20 seeds
   in the twin it is −0.5 to −3.2%, n.s.
2. **It is the only policy significant in Unity here.** The B2-trained s4 is −1.7 to −2.9% (n.s.) and the loose s2 is
   −0.5 to −1.3%.
3. **On the deeper layout G it beats every fixed pair in Unity** (−2.9%, 15 / 20 seeds).

