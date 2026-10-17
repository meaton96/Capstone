# eval-v5-cmix: PPO trained at loose or mixed due dates vs the best fixed rule (2026-10-14, user)

Status: **done** 2026-10-15 12:39 (cluster array 21897089, 37 min, 54 / 54; lab `eval-v5-cmix`). `analysis.out`, `summary.csv`.

## Why
The user (10-14) asked two things:
- Can we show that under loose due dates the learned policy truly beats the rules?
- Should training use a balance of tight and loose due dates?

On B2 the policies are at parity (eval-v5-final). rq3-gen-v5's due-loose has the largest headroom and the only
significant policy gain (BC −5.2%).

## Method (`cmix_task.py`, `submit.sh`, `analyze.py`)
- **Harness:** rq3-gen-v5's (the same frozen code, floor and B2 settings). Conditions due-tight (c 1.25-1.75),
  base (1.75-2.5) and due-loose (2.5-3.5).
- **Seeds:** test 0-39, and the fitting seeds 2000-2159 in four chunks.
- **PPO families**, 5 seeds each, same stack and code:

  | family | trained at c | picked on the fitting seeds by |
  |---|---|---|
  | base (train-v5-nokl + train-v5-final) | 1.75-2.5 | eval-v5-final: s4 |
  | loose (train-v5-loose) | 2.5-3.5 | lowest tardiness at due-loose |
  | cmix (train-v5-cmix) | 1.25-3.5 | lowest mean ratio to MDD-PTECT over the 3 conditions |

- **Also scored:** the 6 v5 pairs + MDD-TECT, BC of the v5 oracle (test, from rq3-gen-v5), the hindsight best v5
  pair, and at due-loose the switching oracle (`../rq3-oracle-loose`).
- **Reuse:** the test-seed pdr / ppo / bc outputs are rq3-gen-v5's (identical instances and code).
- **Score:** pooled window tardiness vs MDD-PTECT and vs the split's best v5 pair, seed-bootstrap CI, Wilcoxon, and
  the mean job-seconds saved per instance. Loose due dates have about half the tardiness of base (70k vs 146k job-s
  per instance for MDD-PTECT), so percentages are of a smaller base.

## Results

**Checks.**
- MDD-PTECT run inside the loose / cmix / ppo groups equals the pdr group on 1,680 / 1,680 seed comparisons.
- Test × pdr / ppo / bc are rq3-gen-v5's outputs.
- The picks were made on the fitting seeds: loose s2, cmix s3, base s4.

**Pooled over all 200 held-out seeds** (fitting 160 + test 40; the seeds are held out for every PPO family). The family
mean is the per-instance mean of its 5 seeds, vs MDD-PTECT:

| condition (MDD-PTECT tardiness per instance) | base-trained (c 1.75-2.5) | loose-trained (c 2.5-3.5) | mixed-trained (c 1.25-3.5) |
|---|---|---|---|
| due-tight (248.6 k job-s) | −0.66 [−1.20, −0.11] | −0.64 [−1.37, +0.06] | −0.62 [−1.12, −0.12] |
| base (152.3 k) | −1.25 [−2.03, −0.53], 122 / 200 | **−2.29** [−3.44, −1.22], 135 / 200 | **−2.10** [−2.88, −1.41], 147 / 200, p 1e-10 |
| due-loose (70.3 k) | +0.29 [−1.29, +1.80] | **−5.28** [−7.06, −3.66], 153 / 200, p 2e-15 | **−3.53** [−4.94, −2.24], 146 / 200 |

**Test seeds 0-39 alone** (vs MDD-PTECT; * = CI below 0 and p < 0.05):

| | due-tight | base | due-loose |
|---|---|---|---|
| base-trained, mean / pick s4 | −0.7 / −0.6 | −0.3 / +0.2 | −0.9 / +0.2 |
| loose-trained, mean / pick s2 | −0.1 / −0.1 | −1.5 / −1.2 | **−6.5* / −6.9*** |
| mixed-trained, mean / pick s3 | −0.7 / −1.1 | **−1.7*** / −1.7 | **−5.5* / −6.7*** |
| BC of the v5 oracle s0 | +0.9 | −2.2 | **−6.0*** |
| hindsight best v5 pair | −4.0* | −4.5* | −8.4* |
| switching oracle | – | (−9.4, rq2-rebase-v5) | **−14.1*** (rq3-oracle-loose) |

**Training-seed check (added 10-15 late).** The p-values above treat the 5 trained policies of a family as fixed
(instance-level). Here each trained policy counts once: pooled % vs MDD-PTECT on the 200 seeds, 5 values per family.

| condition | base-trained s0-s4 | loose-trained s0-s4 | mixed-trained s0-s4 | loose vs base | mixed vs base |
|---|---|---|---|---|---|
| due-loose | −1.6 / +4.1 / −3.2 / +1.3 / +0.8 | −4.6 / −6.1 / −6.1 / −5.6 / −4.1 | −0.9 / −2.4 / −4.9 / −6.1 / −3.3 | **MW p 0.008** (no overlap) | MW p 0.06, Welch p 0.04 |
| base | −0.9 / −0.5 / −2.1 / −0.7 / −2.1 | −2.2 / −1.5 / −2.3 / −2.3 / −3.1 | −0.4 / −1.0 / −2.9 / −3.2 / −3.0 | MW p 0.03 | n.s. (p 0.25-0.31) |
| due-tight | −0.3 / −0.7 / −0.2 / −1.0 / −1.1 | −0.8 / +0.3 / −0.3 / −1.3 / −1.1 | −0.1 / −0.0 / −1.0 / −1.6 / −0.3 | n.s. | n.s. |

Against 0, one-sample t over the 5 policies: loose-trained at due-loose p 0.0002, at base p 0.0008; mixed-trained at base
p 0.02. So:
- **Robust to training seeds:** loose training beats MDD-PTECT at loose c (every seed −4.1 to −6.1%) and beats
  base training there (no overlap).
- **Weaker than the instance-level p suggests:** the mixed-trained policy beats MDD-PTECT at base (p 0.02 over seeds), but
  it is not distinguishable from the base-trained policy there. Reading 3 ("looser c also helps at base") holds for
  loose training (p 0.03), not for mixed training.

## Reading
1. **Yes: under loose due dates a trained policy truly beats the best fixed rule.**
   - The loose-trained family is −5.3% on 200 held-out seeds (153 better). Its pick is −6.9% on the test seeds and
     −5.9% on the fitting seeds. All 5 seeds are significant on both seed sets.
   - In absolute terms that is 3.7 k job-s saved per instance, about 60% of what the hindsight best pair saves
     (5.9 k) and 38% of the switching oracle (9.8 k).
   - The base-trained policy does not reach this: +0.3% at loose c, never having seen loose due dates. The gain is
     learned, not given by the rule set.
2. **Training on a balance of due dates gives one policy that wins wherever there is slack.**
   - The mixed-trained family is −2.1% at base (147 / 200, p 1e-10, significant on the test seeds alone), −3.5% at
     loose, and −0.6% at tight.
   - It gives up some of the loose-only gain (−3.5 vs −5.3%) for robustness across tightness.
3. **Loose and mixed training also helps at B2's own due dates** (−2.3 / −2.1% against the base-trained −1.25%).
   - Situations with slack are where the action matters (rq2-twin-scen: the oracle's percentage rises with c). They
     appear more often in the looser training ranges, which gives PPO more to learn from, and the skill transfers.
4. **With tight due dates no policy and no fixed rule changes much** (all within −0.7%; hindsight best pair −4.0%).
   When most jobs are late, the order hardly matters.
5. **For the thesis:**
   - This turns the RQ2 / RQ3 answer from "parity" into a condition: the learned policy beats the best fixed rule
     where due dates leave slack, by about 5% at loose due dates and about 2% at B2's range if trained on a mix.
   - It stays at parity where most jobs are late.
   - The training distribution of due-date tightness matters as much as the architecture. B2's own range
     (c 1.75-2.5) was, if anything, too narrow to learn from.
   - Report the 200-seed pooled family means as the main numbers, and the test-seed rows as the protocol check.
