# rq3-gen-v5: RQ3 conditions sweep on the v5 rules and the final PPO stack (twin, 2026-10-14)

Status: **done** 2026-10-14 21:58 (cluster array 21894121, 30 min, 33 / 33 tasks; lab `rq3-gen-v5`). Tables in
`analysis.out`, `summary.csv`, `best_pairs.csv`.

## Why
- RQ3 asks under which conditions the policy beats the best fixed pair.
- `../rq3-gen-twin` (10-08) answered it on the H15 pairs with one training seed per method. On v5 the best fixed pair
  moved by about 16% (MDD-PTECT), so those answers do not carry over (Demo 1 slide 17).
- `../dev-pect-rule` already showed the v5 fixed rule beats each condition's best H15 pair in all 12 conditions. What
  is missing is the v5 *policies* against the condition's best *v5* pair.

## Method (`run_task.py`, `submit.sh`, `analyze.py`)
- **Conditions:** the same 11 as rq3-gen-twin, with the same floors (`../rq3-gen-twin/floors`) and the same fleet
  wrapper:
  - base;
  - load ×0.8 / ×1.2;
  - due allowance [1.25, 1.75] / [2.5, 3.5];
  - 5 / 7 / 9 AGVs on duty on the 9-bay D floor;
  - layouts C / G / J.

  B2 test seeds 0-39, deterministic, 900 s slots, tardiness reward, discount horizon 5,400 s (as trained).
- **Groups:**

  | group | policies |
  |---|---|
  | `pdr` | the 6 v5 pairs + MDD-TECT (H15 best on B2) |
  | `ppo` | PPO s0-s4 (train-v5-nokl, train-v5-final) + MDD-PTECT |
  | `bc` | BC of the v5 oracle, rq2-oracle-bc-v5 s0-s2, + MDD-PTECT |

  MDD-PTECT in ppo / bc is a run-to-run check, so `mdd_run_mismatch` must be 0.
- **Code:** `../rq2-rebase-v5/code/env` (6d13412a), the code all the checkpoints were trained and evaluated with.
- **Score:** pooled window tardiness against the condition's best v5 pair, chosen in-sample on the same seeds (this
  flatters the baseline). Seed-bootstrap 95% CI, seeds better / worse, Wilcoxon.
  - Rows: each PPO seed, the PPO 5-seed mean (per-instance mean tardiness), the final policy (`eval-v5-final`
    pick), each BC seed and the BC mean, and the hindsight best v5 pair.
  - The condition's best H15 pair from rq3-gen-twin is listed for context.
- **Smoke (10-14, local, base seed 0):** all 6 v5 pairs equal `train-v5-nokl/eval`, MDD-TECT equals rq3-gen-twin
  (113,602.24), and BC s0 equals the rq2-oracle-bc-v5 rollout (85.112). The agv5-bay9 pdr group also ran.

## Results

**Checks.**
- `mdd_run_mismatch` is 0 in every condition: MDD-PTECT is identical in the pdr, ppo and bc groups.
- `base` reproduces the B2 numbers: the best pair is MDD-PECT, with MDD-PTECT +0.47% and MDD-TECT +19.0%, as in
  rq2-rebase-v5 / train-v5-nokl.
- `agv7-bay9` gives exactly base's best-pair total (5,793.78), as in rq3-gen-twin, so the 9-bay floor does not move
  the rules.

**Best fixed v5 pair per condition** (in-sample on seeds 0-39):

| condition | best v5 pair | MDD-PTECT | MDD-TECT (H15 best) | best H15 pair (rq3-gen-twin) |
|---|---|---|---|---|
| base | MDD-PECT | +0.5% | +19.0% | MDD-TECT |
| load-0.8 | **ATC-PTECT** | +1.9% | +24.7% | ATC-TECT |
| load-1.2 | MDD-PECT | +0.1% | +20.4% | MDD-TECT |
| due-tight | **SRT-PECT** | +0.7% | +13.2% | SRT-TECT |
| due-loose | MDD-PTECT | best | +23.7% | MDD-TECT |
| agv5 / 7 / 9-bay9 | MDD-PECT | +1.6 / +0.5 / +0.9% | +19.1-19.9% | MDD-TECT |
| lay-c | MDD-PTECT | best | +20.0% | MDD-TECT |
| lay-g | MDD-PECT | +0.6% | +18.3% | MDD-ECT |
| lay-j | MDD-PTECT | best | +23.0% | MDD-ECT |

- **The landscape moves where it moved on H15, through the job rule.** ATC is best at low load, SRT with tight due
  dates. The PECT / PTECT machine half is best in every condition.
- **MDD-PTECT stays within 1.9% of the best pair everywhere.** MDD-TECT is 13-25% behind.
- **Fleet size barely matters** (best total 5,757 / 5,794 / 5,734 at 5 / 7 / 9 AGVs): the floor is machine-bound.

**Policies vs the condition's best v5 pair** (pooled %, seed-bootstrap 95% CI, seeds better of 40; * = Wilcoxon
p < 0.05):

| condition | final policy (PPO s4) | PPO mean of 5 seeds | BC mean of 3 seeds | hindsight best v5 pair |
|---|---|---|---|---|
| base | +0.6 [−3.0, +3.9] (21) | +0.1 [−2.1, +2.2] (24) | −0.6 [−3.4, +1.9] (24) | −4.1* |
| load-0.8 | −1.2 [−5.2, +3.7] (15) | −1.8 [−5.5, +2.6] (13) | −2.5 [−6.4, +2.6] (15) | −7.0* |
| load-1.2 | −1.1 [−3.4, +1.0] (25) | −0.7 [−2.2, +0.7] (24) | −0.2 [−2.1, +1.6] (21) | −2.9* |
| due-tight | +0.1 [−2.1, +2.3] (18) | +0.0 [−1.6, +1.5] (23) | +1.3 [−0.7, +3.3] (20) | −3.3* |
| due-loose | +0.2 [−6.4, +5.4] (26) | −0.9 [−5.8, +3.2] (27)* | **−5.2 [−10.6, −0.9] (30)*** | −8.4* |
| agv5-bay9 | +2.1 [−1.3, +5.7] (23) | +1.1 [−1.2, +3.4] (23) | −0.3 [−2.7, +1.7] (25) | −3.5* |
| agv7-bay9 | +0.5 [−3.0, +4.0] (21) | +0.3 [−2.0, +2.5] (23) | −0.5 [−3.5, +1.9] (24) | −4.1* |
| agv9-bay9 | +1.0 [−1.7, +3.5] (23) | +0.5 [−1.5, +2.5] (23) | +0.6 [−1.6, +2.7] (23) | −3.5* |
| lay-c | +2.1 [−1.1, +4.6] (21) | −0.2 [−2.2, +1.3] (25) | −0.3 [−2.8, +1.9] (25) | −3.1* |
| lay-g | −0.5 [−3.2, +2.0] (22) | −0.1 [−2.0, +1.7] (26) | −0.7 [−3.6, +2.0] (23) | −4.2* |
| lay-j | −1.8 [−4.4, +0.9] (27) | −1.3 [−3.3, +0.3] (24) | −0.9 [−3.5, +1.6] (24) | −4.2* |

Per-seed rows are in `analysis.out`. Individual seeds win significantly in a few cells: PPO s2 at lay-c (−2.0%),
lay-j (−1.9%) and due-loose (−5.6%); PPO s0 at lay-c and lay-j; BC s0 at load-0.8 (−6.1%) and due-loose.

## Reading
1. **On the v5 rules, the learned policies beat the condition's best fixed pair in no condition, apart from loose due
   dates.**
   - The final policy and the 5-seed PPO mean are within −1.8 to +2.1% of the best pair in all 11 conditions. No CI
     excludes 0.
   - The one robust gain is with loose due dates. There BC of the oracle is −5.2% [−10.6, −0.9] (30 / 40 seeds), PPO
     s2 −5.6%, and the PPO mean −0.9% (Wilcoxon p 0.04, CI across 0). That is where the H15 policies gained most
     too: avoidable tardiness concentrated in a few jobs per seed.
2. **The rule landscape shifts, and the fixed rule follows it with a parameter, not a switch.** The best pair changes
   at low load (ATC) and with tight due dates (SRT). MDD-PTECT stays within 2% everywhere, so a single fixed rule is
   nearly as good as the per-condition best. The hindsight best pair keeps 3-8% of headroom that no observable policy
   takes.
3. **Against H15 (rq3-gen-twin),** the policies there kept their −2 to −3% in most conditions. Against the v5 pairs
   that margin is gone in every condition except due-loose, for the same reason as on B2: the gain was a missing
   processing-time rule.
4. **For the thesis (RQ3):** with the v5 rules there is essentially no condition where the policy beats the best fixed
   pair; loose due dates are the exception, and only for some seeds or methods. This supports the proposed rewording
   of RQ3 in the claims doc: the conditions change *which rule* is best (the job half) and the rule's parameter
   (rq3-alpha-curve), not whether switching pays. The Unity subset (`../rq3-gen-v5-unity`) checks load-1.2,
   due-tight and lay-g.
