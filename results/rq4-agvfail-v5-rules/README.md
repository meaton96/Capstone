# rq4-agvfail-v5 (rules + policy): RQ4 AGV-breakdown rerun on the v5 rules (Unity, 2026-10-14)

Status:
- **rules half done** 2026-10-14 20:41 (cluster arrays 21892312 + 21893492, 200 / 200 tasks; lab
  `rq4-agvfail-v5-rules`; `analysis.out`);
- **policy half done** 2026-10-15 00:51 (`../rq4-agvfail-v5-policy`, array 21894124, 280 / 280; seed 32 passed with
  `eval_wrap.py`).

## Why
RQ4 asks whether breakdowns change the rule ranking and the policy's edge. Every breakdown result so far is on the
H15 pairs (`../rq4-agvfail-policy`: MDD-TECT stays best, and the policy gains shrink to noise). On v5 the best pair is
MDD-PTECT, 16-20% better than MDD-TECT, so the ranking and the policy rows need the v5 pairs and the final policy
(Demo 1 slide 17). This is also the AGV-breakdown part of the old `base-catalog-h15` plan.

## Method (`task.sh`, `array.sbatch`, `submit.sh`, `analyze.py`; shared with the policy half)
- **Setup:** Unity, B2 test seeds 0-39, the same setup as `../eval-transfer-unity-v5`. Its no-breakdown rows are
  reused as level `none`.
  - Code `../rq2-rebase-v5/code/env` (6d13412a, the checkpoints' code).
  - Player `linux_server/` (5e6e0f87; the cluster copy is glibc-patched; no `-pyrules`).
  - Random warm-up, 21,600 s window, 900 s slots, deterministic.
- **Levels:** AGV Weibull breakdowns with λ = 3000 s (`l3000`) and 1500 s (`l1500`), other fields at
  `scenarios.AGV_FAILURE_DEFAULTS`, as in rq4-agvfail-policy.
- **Rules half:** at l3000 / l1500, `pa` = SRT-PECT, SRT-PTECT, MDD-PECT, MDD-PTECT and `pb` = ATC-PECT, ATC-PTECT,
  MDD-TECT, MDD-PTECT. At `none`, `pn` = the 4 SRT / ATC pairs eval-transfer-unity-v5 did not run, plus MDD-PTECT.
  200 tasks.
- **Policy half:** at l3000 / l1500, `ppoa` = PPO s0-s2 + MDD-PTECT, `ppob` = PPO s3-s4, `bc` = BC v5 s0-s2. At
  `none`, `ppob`. 280 tasks.
- **Run-to-run checks:** MDD-PTECT is repeated in pa / pb / ppoa at each level and in pn against
  eval-transfer-unity-v5 at none.
- **`analyze.py` (reads both halves):** per level, every policy against MDD-PTECT at the same level (pooled, seed
  bootstrap CI, Wilcoxon); the pair ranking and its Kendall tau against no breakdowns; the hindsight best v5 pair;
  the PPO 5-seed and BC 3-seed means; the `eval-v5-final` pick; and each policy's degradation from no breakdowns.
- **Smoke (10-14, local, `smoke/`):** l1500 pb seed 0 gave 47-48 AGV breakdowns per episode. MDD-PTECT 103.8 against
  94.4 without breakdowns (+10%), MDD-TECT 121.9 against 112.5. The local player's MDD-PTECT at none (94.39) equals
  the cluster's, so the glibc patch does not change results.
- **Seed 32 timeout (10-14):** the first array finished 198 / 200. l3000 / pa / s32 and none / pn / s32 failed twice
  with `UnityTimeOutException`. Seed 32's random warm-up is 8,399 s, and Unity simulates it without talking to
  Python. On a cluster node that took about 5.5 min, more than the 300 s `timeout_wait` hard-coded in the frozen
  code. eval-transfer-unity-v5's ppo s32 hit the same timeout on 10-12. Since 18:50, `task.sh` runs the frozen
  evaluate.py through `eval_wrap.py`, which raises the wait to 1,800 s and changes nothing else. The two seed-32
  tasks and the whole policy half run through it.
- **Runtime:** about 12 min per episode on a cluster node (eval-transfer-unity-v5), so about 1 h per task.

## Results (rules half, 10-14)

All 40 seeds at every level. **Run-to-run checks pass:** the repeated MDD-PTECT runs are identical on 40 / 40 seeds
at each level (pa vs pb at l3000 and l1500; pn vs eval-transfer-unity-v5 at none). MDD-TECT matches across players:
mean window tardiness 176.33 / 207.32 / 224.26 at none / l3000 / l1500 is exactly rq4-agvfail-policy's on the v4
player.

**Pairs vs MDD-PTECT at the same level** (pooled window tardiness, seed-bootstrap 95% CI, seeds better / worse,
Wilcoxon):

| pair | none | l3000 (λ 3000 s) | l1500 (λ 1500 s) |
|---|---|---|---|
| MDD-PECT | −0.16 [−1.8, +1.5], 21/18, p 0.72 | +0.95 [−1.1, +3.0], 18/22, p 0.31 | +0.96 [−1.0, +3.0], 12/27, p 0.15 |
| SRT-PECT | +4.3 [+1.9, +7.6] | +5.6 [+3.1, +8.6] | +3.1 [+0.8, +6.0] |
| SRT-PTECT | +6.0 [+3.3, +9.4] | +5.4 [+2.9, +8.7] | +2.9 [+1.0, +5.2] |
| ATC-PTECT | +12.7 [+7.3, +17.7] | +13.8 [+8.4, +18.2] | +14.3 [+7.3, +19.9] |
| ATC-PECT | +14.1 [+8.3, +19.5] | +14.2 [+8.6, +18.7] | +16.1 [+9.3, +21.5] |
| MDD-TECT (H15 best) | +19.7 [+15.3, +25.5] | +16.6 [+11.5, +22.7] | +13.7 [+9.7, +18.3] |
| hindsight best v5 pair | −3.8 [−5.2, −2.6] | −3.7 [−5.4, −2.3] | −4.6 [−6.6, −3.1] |

**Ranking against no breakdowns** (7 pairs): Kendall τ +0.81 (p 0.011) at l3000, +0.62 (p 0.07) at l1500.

**Degradation from no breakdowns** (same seeds):

| pair | l3000 | l1500 |
|---|---|---|
| MDD-PTECT | +20.7% | +33.9% |
| other v5 pairs | +20.1 to +22.1% | +29.9 to +36.2% |
| MDD-TECT | +17.6% | +27.2% |

**Reading:**
1. **Breakdowns do not reorder the v5 rules in any way that matters.** The MDD job rule with either
   processing-time machine rule stays on top at every level. MDD-PTECT and MDD-PECT swap first place, but the
   difference is under 1% and never significant. The SRT pairs stay 3-6% behind and the ATC pairs 13-16%.
   - The only rank changes are swaps within near-tied pairs (PECT vs PTECT for the same job rule).
   - The one cross-family change: MDD-TECT passes the two ATC pairs at λ 1500 s.
   - This is the H15 finding (rq4-agvfail-policy: MDD-TECT stayed best, the top four barely moved) on the v5 rules.
2. **Breakdowns cost about +21% at λ 3000 s and +34% at λ 1500 s** for the best rule. Every v5 pair degrades by about
   the same amount.
   - The H15 pair MDD-TECT degrades least (+17.6 / +27.2%), so its gap to MDD-PTECT narrows from +19.7% to +13.7%. The
     processing-time rule's edge is partly lost when AGVs, not machines, become the constraint.
   - The edge stays large and significant at both levels (2-3 / 37-38 seeds).
3. **The travel term gains slightly under breakdowns.** MDD-PECT goes from 0.16% better than MDD-PTECT to about 1%
   worse once AGVs break down (n.s.), consistent with the twin's result that the travel term pays when AGVs are scarce
   (agv-ratio-twin-v5).
4. **The hindsight switching headroom does not shrink** (−3.8 / −3.7 / −4.6%). On H15 it shrank from −6.3% to −4.7%.
   Whether a policy can use it is the policy half's question.

**For the thesis (RQ4, fixed-rule half):** on the v5 rules, AGV breakdowns raise tardiness by a fifth to a third
without changing which rule to use. The ranking correlates with the no-breakdown ranking at τ 0.6-0.8, and every
change is inside a tie. The processing-time rules stay 14-17% better than the old best pair even at the heavier
breakdown level.

## Results (policy half, 10-15)

All 280 tasks ran, n = 40 everywhere. **Run-to-run checks:** the MDD-PTECT run inside the ppoa group equals the pa
group on 40 / 40 seeds at both levels.

**Policies vs MDD-PTECT at the same level** (pooled window tardiness, seed-bootstrap 95% CI; no row is significant):

| policy | none | l3000 (λ 3000 s) | l1500 (λ 1500 s) |
|---|---|---|---|
| PPO s4 (final policy, picked on fitting seeds) | +0.29 [−2.7, +3.4] | −1.16 [−4.1, +1.5] | −0.54 [−2.9, +1.4] |
| PPO, mean of 5 seeds | −0.13 [−2.3, +2.3] | +0.23 [−1.7, +2.0] | −0.72 [−2.5, +0.7] |
| PPO seeds, range | −1.32 to +0.87 | −1.16 to +1.58 | −0.86 to −0.54 |
| BC of the v5 oracle, mean of 3 | +0.12 | +0.66 | +0.26 |
| hindsight best v5 pair | −3.8 | −3.7 | −4.6 |

**Degradation from no breakdowns:**

| | l3000 | l1500 |
|---|---|---|
| PPO seeds | +19.0 to +23.7% | +31.8 to +34.8% |
| BC seeds | +20.2 to +23.6% | +32.6 to +36.3% |
| v5 pairs | +20.1 to +22.1% | +29.9 to +36.2% |

## Reading (both halves)
1. **The learned policies stay at parity with the best rule under AGV breakdowns.** They neither gain nor lose. The
   final policy and the 5-seed mean are within ±1.2% of MDD-PTECT at every level, and every CI contains 0.
   - The policies never saw breakdowns: the twin has no AGV failure model, so they were trained without failures.
   - They degrade like the fixed pairs (+19-24% / +32-35%). The broken-AGV signal in the observation is neither
     exploited nor misread.
2. **H15 vs v5.**
   - On H15 the question was whether the policies' 1.2-3.0% edge survives breakdowns. It shrank to noise
     (rq4-agvfail-policy).
   - On v5 there is no edge to lose, and none appears. The PPO mean is slightly better at λ 1500 s (−0.7%, 27 / 40
     seeds, p 0.15).
3. **Rules (fixed-rule half above):** no meaningful reordering; MDD-PTECT / MDD-PECT stay best within 1%.
4. **For the thesis (RQ4):** AGV breakdowns cost a fifth to a third more tardiness. They change neither the best
   rule nor the policy-vs-rule comparison: the twin-trained policies keep parity under breakdowns they were never
   trained on. The headroom that hindsight switching would buy stays at 3.7-4.6%.
