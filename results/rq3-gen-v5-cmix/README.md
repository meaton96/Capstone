# rq3-gen-v5-cmix: the RQ3 conditions sweep for the final policy (PPO trained on mixed due dates) (2026-10-17, user)

Status: **done** 2026-10-17 15:16 (cluster array 21909802, 29 min, 16 / 16). `analysis.out`, `summary.csv`.

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
**Checks.** MDD-PTECT in every cmix / loose group equals rq3-gen-v5's pdr run on all seeds (33 groups identical).

**vs MDD-PTECT** (pooled % [95% CI], seeds better of 40, * = Wilcoxon p < 0.05; `analysis.out` has the vs-best-pair table):

| condition | final policy (cmix s3) | cmix mean of 5 | loose s2 | B2-trained s4 | B2-trained mean of 5 | hindsight best pair |
|---|---|---|---|---|---|---|
| base | −1.7 [−4.0, +0.4] | **−1.7*** | −1.2 | +0.2 | −0.3 | −4.5* |
| load-0.8 | **−7.0*** | **−4.5*** | −5.4* | −3.1 | −3.6* | −8.8* |
| load-1.2 | **−2.8*** | **−1.7*** | −0.7 | −1.2 | −0.8 | −3.0* |
| due-tight | −1.1* | −0.7* | −0.1 | −0.6 | −0.7* | −4.0* |
| due-loose | **−6.7*** | **−5.5*** | −6.9* | +0.2 | −0.9* | −8.4* |
| agv5-bay9 | −1.1 | **−2.2*** | −0.8 | +0.5 | −0.5 | −5.0* |
| agv7-bay9 | −1.3 | **−1.7*** | −1.2 | +0.1 | −0.2 | −4.5* |
| agv9-bay9 | −1.3 | **−1.9*** | −1.2 | +0.1 | −0.3 | −4.4* |
| lay-c | −0.8 | −1.3 | +0.3 | +2.1 | −0.2 | −3.1* |
| lay-g | −3.0 | **−2.1*** | −0.6 | −1.0 | −0.7 | −4.7* |
| lay-j | **−2.8*** | **−2.2*** | −0.5 | −1.8 | −1.3 | −4.2* |

**vs each condition's best v5 pair (in-sample):** the cmix pick is −2.7* (load-1.2), −6.7* (due-loose), −2.8* (lay-j),
−5.2 [−9.2, −0.8] (load-0.8, 17 / 40, Wilcoxon n.s.), and within −2.4 to +0.5 elsewhere. The cmix mean of 5 is
significant at load-1.2, due-loose, lay-g and lay-j.

**Training-seed check** (pooled % vs MDD-PTECT, one value per trained policy, 5 per family):
- **cmix vs 0:** the cmix family beats MDD-PTECT in 9 of 11 conditions (one-sample t p 0.000-0.041). The exceptions
  are load-0.8 (p 0.06; s0 is +1.2) and due-tight (p 0.06).
- **cmix vs the B2-trained family:** cmix is better at agv5 / agv7-bay9 and lay-g (Mann-Whitney p 0.032), borderline
  at due-loose (0.056) and base / agv9 (0.095), and n.s. elsewhere.

## Reading
1. **The final policy generalizes better than the B2-trained one.**
   - Across the 11 conditions, the cmix family is 0.7-5.5% better than MDD-PTECT, significant at the instance level in
     10 and across training seeds in 9.
   - The B2-trained family is significant in 3 (load-0.8, due-tight, due-loose, all instance-level only).
2. **It beats each condition's own best pair (chosen in hindsight on these seeds) in 3-4 conditions:** load-1.2,
   due-loose, lay-j, plus lay-g for the family. Elsewhere it ties.
3. **Gains are largest where the rule landscape shifts or slack exists:** low load (−7.0% for the pick), loose due
   dates (−6.7%), and the deeper layouts G / J.
4. **For RQ3:** with the final policy, the answer moves from "nowhere except loose due dates" (B2-trained s4,
   rq3-gen-v5) to "a small, consistent gain over the best fixed rule in most conditions, largest at low load and loose
   due dates". Training on varied due dates is what broadens it.
   - Unity checks for the final policy are in `../rq3-gen-v5-unity-cmix` (running) and `../eval-v5-cmix-unity`.
