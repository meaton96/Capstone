# train-ft-het

Fine-tune arm of plan §2b (user decision 10-25): the final policy (`train-v5-cmix` s0-s4) resumed on B2 with
machine-speed spread s = 1 for 24 updates, a checkpoint after every update (budgets 2 / 5 / 10 / 24). Design, code and
checks of both arms: `../train-ft-m100/README.md`. Evaluation: `../eval-ft` (lab `eval-ft-het`).

Status: **trained 10-26** (lab `train-ft-het`, local, 17:51-18:55, ~62 min per seed). All 5 seeds reached step
147,456 (24 updates), and the budget checkpoints for 2 / 5 / 10 / 24 updates (steps 102,400 / 108,544 / 118,784 /
147,456) are present. Evaluation `eval-ft-het` started 18:55.

## Het specifics
- **Floor:** B2 itself (floor7, 7 AGVs, B2 load mix, short-fleet blocks), through `code/hetero.py --spread 1,1`
  (copied from `../dev-alpha-action/code`, md5 75f50351b3c7, the same file `../eval-alpha-spread/code_v5` uses):
  - every instance gets per-machine speed factors log-uniform in [1/(1+s), 1+s] = [0.5, 2], fixed per instance;
  - durations are scaled per machine type so each type keeps B2's capacity;
  - due dates are B2-tight: d = r + c × (work without the speed factors).
- **Jobs, arrivals, op types and warm-up** are those of the plain instance with the same seed.
- **Due dates:** training draws c ~ U[1.25, 3.5] as cmix did. The evaluation uses B2's c 1.75-2.5.
- **s is fixed at 1**, not drawn from a range, so every training instance has spread 1. `hetero.json` in each run
  folder records it. The smoke confirmed s = 1 on every instance the forked twin workers generated.
- **Limit of the action set:** the fine-tune chooses among the v5 pairs, whose machine rules use α = 3. The spread law
  (−15.3% at s = 1) needs α ≈ 0.3 (best fixed α 0.5: −17.2%). So the fine-tune cannot reach the law.
  - What it can reach is set by the v5 pairs at s = 1 (`../eval-alpha-spread` grid, test seeds 0-39, vs MDD-PTECT):

    | v5 pair | % vs MDD-PTECT |
    |---|---|
    | MDD-PECT | −1.31 |
    | SRT-PECT | +2.07 |
    | SRT-PTECT | +2.32 |
    | ATC-PECT | +2.81 |
    | ATC-PTECT | +3.91 |
    | per-instance best static pair (hindsight) | −5.41 |

  - The final policy is already at −2.89%.
  - Switching between pairs can go beyond the static −5.4%; there is no v5 switching oracle at s = 1.
- **No from-scratch v5 arm at s = 1:** unlike m100 (T100), nothing was trained on this floor from scratch with the v5
  heads. The evaluation reads the het curve against the final policy and the per-instance best static v5 pair instead
  (`../eval-ft/README.md`).

## Results
Evaluated 10-26 (`../eval-ft`, het arm): A1 fails (24 updates −0.47 points vs the final policy on s 1, p 0.061; 18% of the way
to the best static v5 pair), A3 passes from 5 updates (B2 retention 0.62 at 24). Details in `../eval-ft/README.md`.
