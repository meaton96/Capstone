# rq4-agvfail-v5-cmix: the RQ4 AGV-breakdown half for the final policy (cmix pick s3) (2026-10-17, user)

Status: **done** 2026-10-17 18:25 (cluster array 21909846, 3 h 33 min, 240 / 240). `analysis.out`, `summary.csv`.

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
**Checks:**
- cmixa's MDD-PTECT equals the rules half on 120 / 120 (level, seed) pairs;
- at `none`, cmixb's cmix s3 and loose s2 equal eval-v5-cmix-unity's base runs on 40 / 40 seeds.

**vs MDD-PTECT at the same level** (Unity, B2 seeds 0-39; pooled % [95% CI], better / worse, Wilcoxon p):

| policy | no breakdowns | λ 3000 s | λ 1500 s |
|---|---|---|---|
| **final policy cmix s3** | +0.5 [−2.6, +3.6], p 0.65 | −2.0 [−4.6, +0.2], 25/13, p 0.07 | **−2.1** [−4.4, −0.2], 26/13, p 0.02 |
| cmix mean of 5 | −0.8 (p 0.23) | −0.6 | −1.2 (p 0.15) |
| loose s2 | −0.4 | −0.7 | −1.2 |
| B2-trained s4 | +0.3 | −1.2 | −0.5 |
| B2-trained mean of 5 | −0.1 | +0.2 | −0.7 |
| MDD-PECT | −0.2 | +1.0 | +1.0 |
| hindsight best pair | −3.8* | −3.7* | −4.6* |

**Degradation from no breakdowns:**

| policy | λ 3000 s | λ 1500 s |
|---|---|---|
| cmix s3 | +17.7% | +30.4% |
| loose s2 | +20.4% | +32.8% |
| B2-trained s4 | +19.0% | +32.8% |
| MDD-PTECT | +20.7% | +33.9% |

## Reading
1. **In Unity the final policy is at parity without breakdowns, and slightly better with them.**
   - Without breakdowns it is +0.5% (n.s.), as in eval-v5-cmix-unity.
   - With breakdowns it is −2.0% at λ 3000 s (p 0.07) and −2.1% at λ 1500 s (p 0.02).
   - It degrades 3 points less than MDD-PTECT.
2. **The 5-seed family stays within −0.6 to −1.2% (n.s.),** so the breakdown gain belongs mostly to the pick s3. As
   with every single-seed number, report it next to the family.
3. **Breakdowns still do not reorder the fixed pairs** (rq4-agvfail-v5-rules: MDD-PTECT / MDD-PECT within 1%).
4. **For RQ4:** AGV breakdowns cost every rule and policy 18-34%. The final policy keeps parity or a small edge (about
   2% for the pick) and loses the least.

