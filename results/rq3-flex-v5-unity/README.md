# rq3-flex-v5-unity: machine flexibility in Unity for the v5 rules and the final policy (2026-10-17, user)

Status: **done** 2026-10-17 19:06 (cluster array 21910096, 3 h 9 min, 120 / 120; smoke dev-flex-unity-smoke). `analysis.out`.

## Why
RQ3's conditions include machine flexibility. The twin refuses it, because Unity draws each machine's secondary
capabilities from the scenario seed, and the only flexibility result (`rq3-flex`, 10-02) is outdated: flow time,
old rules, pre-audit build. The user (10-17) asked to set it up with the final policy, the PPO trained on mixed due
dates (cmix pick s3), and to report all policies' numbers.

## Method (`task.sh`, `array.sbatch`, `submit.sh`, `analyze.py`)
- **Instances:** Unity, B2 test seeds 0-19 with `--machine-flexibility p --secondary-time-multiplier 1.25`. Each
  machine can also run each other type with probability p, and a secondary type runs 25% slower. Levels:
  `flex-p0.3`, `flex-p0.5`.
- **Settings:** the same code, player and settings as `../eval-v5-cmix-unity` (6d13412a snapshot, `linux_server/`
  5e6e0f87 without `-pyrules`, random warm-up, 21,600 s window, 900 s slots, deterministic).
- **Groups:**
  - fa: SRT-PECT, SRT-PTECT, MDD-PECT, MDD-PTECT;
  - fb: ATC-PECT, ATC-PTECT, B2-trained s4;
  - fc: MDD-PTECT, loose s2, cmix s3.

  fc's MDD-PTECT repeats fa's: a check that the flexible floor is identical per seed.
- **Reference:** no flexibility = B2 in Unity on the same seeds (rq4-agvfail-v5-rules level none, eval-v5-cmix-unity
  base).
- **Analysis:** per level, the best pair and the ranking vs no flexibility (Kendall τ); the picks vs MDD-PTECT and vs
  the best pair; and the cost of flexibility per policy.
- **Smoke first:** one task (flex-p0.5 / fc / s0) on the cluster. The full run waits until it is checked.

## Results
**Checks:**
- fc's MDD-PTECT equals fa's on 20 / 20 seeds at both levels, so the flexible floor is identical per seed.
- The no-flexibility rows reproduce eval-v5-cmix-unity on seeds 0-19.

**Pairs vs MDD-PTECT, and the ranking** (Unity, seeds 0-19):

| level | MDD-PTECT mean (k job-s) | MDD-PECT | SRT-PECT / SRT-PTECT | ATC-PECT / ATC-PTECT | ranking τ vs no flexibility |
|---|---|---|---|---|---|
| none | 140.0 | +0.3% | +5.9 / +6.0% | +15.9 / +11.7% | – |
| p 0.3 | 115.6 (−17%) | **−1.5%** (best) | +3.6 / +7.2% | +17.4 / +17.6% | +0.73 |
| p 0.5 | 106.6 (−24%) | **−2.1%** (best) | +7.2 / +3.5% | +26.2 / +22.7% | +0.73 |

**Policies vs MDD-PTECT** (pooled % [95% CI], p):

| policy | none | p 0.3 | p 0.5 |
|---|---|---|---|
| **final policy cmix s3** | +3.6 [−0.6, +7.1], p 0.52 | −2.0 [−5.8, +0.3], 13/7, p 0.08 | +0.2 [−3.4, +3.4], p 1.0 |
| loose s2 | +0.0 | +2.1 | **+6.5** [+2.7, +8.6], p 0.048 |
| B2-trained s4 | +2.9 | −0.4 | +3.2 (p 0.25) |
| hindsight best pair | −3.4* | −4.3* | −4.8* |

Against each level's best pair (MDD-PECT), cmix s3 is −0.5% (p 0.13) at p 0.3 and +2.3% (n.s.) at p 0.5.

**Change in tardiness from no flexibility** (same seeds):

| policy | p 0.3 | p 0.5 |
|---|---|---|
| cmix s3 | −21.8% | −26.3% |
| MDD-PTECT | −17.4% | −23.8% |
| B2-trained s4 | −20.1% | −23.6% |
| loose s2 | −15.7% | −18.9% |

## Reading
1. **Flexibility helps every rule a lot and shifts the best machine rule toward PECT.**
   - Tardiness falls 17-24% for MDD-PTECT.
   - MDD-PECT becomes the best pair (−1.5 / −2.1% vs MDD-PTECT). The travel term matters less when more machines can
     take a job.
   - ATC falls further behind.
   - The ranking stays similar (τ 0.73).
2. **The final policy transfers to flexible floors at parity with the best fixed pair.** It was trained on typed floors
   only, and it is within −0.5 to +2.3% of the best pair (n.s.). It gains the most from flexibility of all policies
   (−22 / −26%).
3. **The loose-trained policy does not transfer:** +6.5% vs MDD-PTECT at p 0.5 (p 0.048). The final policy's mixed
   training is again the more robust one.
4. **For RQ3:** machine flexibility changes which machine rule is best (PECT over PTECT), not whether a learned policy
   wins. The final policy keeps up with the best pair without retraining. With 20 seeds the differences are within
   noise.

