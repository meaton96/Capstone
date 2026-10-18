# rq3-size-v5: floor size in the twin (50 and 100 machines) for the v5 rules and the final policy (2026-10-17, user)

Status: **done** 2026-10-18 00:36 (m50: array 21909879; m100: 16 ten-seed chunks, array 21912534). `analysis.out`, `summary.csv`.

## Why
RQ3's conditions include floor size, which had no current result. The 10-14 claims doc said "the twin can't run it",
but that holds only for tiled 105-machine floors. Plain 50- and 100-machine floors load in the twin
(agv-ratio-twin-v5, rq4-mfail-ccal). The user (10-17) asked to set this up with the final policy, the PPO trained on
mixed due dates (cmix pick s3), and to report all policies' numbers.

## Method (`task.py`, `submit.sh`, `pull.sh`, `analyze.py`)
- **Harness:** a thin wrapper around `../rq3-gen-v5/run_task.py` (same frozen code, B2 window / slots / reward and
  fleet wrapper). It adds two conditions:
  - **m50:** 50 machines (10 per type), layout D, floor `../agv-ratio-scale/floors/m50` (50 bays). This is
    dev-hetero-tune's validated m50 family: 26 AGVs on duty (short-fleet blocks 13), the B2 load mix × 0.9, arrival
    cap off, due dates c ~ U[1.75, 2.5].
  - **m100:** 100 machines (20 per type), floor `../agv-ratio-scale/floors/m100`, the same per-machine scaling:
    52 AGVs (short 26).
  - **m15** is B2 itself (rq3-gen-v5 base + eval-v5-cmix test base).
- **Groups:**
  - pdr: the 6 v5 pairs + MDD-TECT;
  - ppo: the B2-trained s0-s4;
  - cmix: s0-s4;
  - loose: s0-s4.

  Every policy group also runs MDD-PTECT as a run-to-run check. Every policy was trained on the 15-machine floor
  only, so this measures transfer.
- **Analysis:** per floor, the best pair and the pair ranking vs 15 machines (Kendall τ), and each pick / family mean
  vs MDD-PTECT and vs the floor's best pair.
- **Checks (local smoke, seed 0):**
  - m50 / cmix: 6 episodes in 100 s (cmix s3 −2.9%, s0 +4.6% vs MDD-PTECT).
  - m100 / cmix: 6 episodes in 297 s.
  - No observation-cap warnings; the network's set encoders take 50 / 100 machine rows.

## Results
Pooled window tardiness vs MDD-PTECT on the same floor, test seeds 0-39; pooled % [95% CI], Wilcoxon p.

**Pairs:**

| floor | MDD-PTECT mean (k job-s) | best pair | pair ranking (vs MDD-PTECT) | τ vs 15 machines |
|---|---|---|---|---|
| m15 (B2) | 145.5 | MDD-PECT (−0.5%) | SRT +4.8 / +5.7, ATC +11.3 / +13.1 | – |
| m50 | 158.2 | MDD-PTECT | MDD-PECT +2.0, **ATC-PTECT +2.6 / ATC-PECT +4.6**, SRT +8.0 / +10.5 | +0.20 |
| m100 | 456.0 | MDD-PTECT | SRT-PTECT +5.1, MDD-PECT +5.6, SRT-PECT +11.1, ATC +23.4 / +31.5 | +0.60 |

**Policies** (all trained on 15 machines):

| policy | m15 | m50 | m100 |
|---|---|---|---|
| **final policy cmix s3** | −1.7 [−3.9, +0.4], p 0.18 | **−2.1** [−4.6, +0.1], 29/11, p 0.019 | +0.4 [−0.7, +1.7], p 0.98 |
| cmix mean of 5 | **−1.7***, p 0.006 | −1.3 (p 0.058) | **+2.6** [+1.3, +3.9], p 0.006 |
| loose s2 / mean of 5 | −1.2 / −1.5 | −2.0 / −1.5 (p 0.029) | +5.8* / +4.6* |
| B2-trained s4 / mean of 5 | +0.2 / −0.3 | −1.5 / +0.3 | **+11.5*** / +4.9* |
| hindsight best pair | −4.5* | −4.6* | **−0.7*** |

## Reading
1. **On 50 machines the final policy transfers.** It is −2.1% vs MDD-PTECT, the best pair there (29 / 40 seeds,
   p 0.019), and the family is −1.3% (p 0.058). The B2-trained family does not (+0.3%). This holds even though the
   pair ranking reshuffles: ATC moves up and SRT down, τ 0.20 vs 15 machines.
2. **On 100 machines no policy gains, and the families lose.**
   - The cmix pick is at parity (+0.4%). The cmix family is +2.6%, loose +4.6%, and the B2-trained family +4.9% (s4
     +11.5%).
   - There is also almost nothing to gain: the hindsight best pair is only −0.7% below MDD-PTECT.
   - The 100-machine floor (52 AGVs, 4× B2's tardiness level) is a regime where MDD-PTECT is close to the best
     available choice. A policy trained on 15 machines makes it slightly worse.
3. **The final policy degrades least with size** (−1.7 → −2.1 → +0.4%, pick; −1.7 → −1.3 → +2.6%, family).
   Training on varied due dates again gives the most robust policy, but it does not stretch to a floor 7× larger.
4. **For RQ3 (floor size):**
   - The best rule stays MDD-PTECT on larger floors, while the rest of the ranking shifts.
   - A policy trained on 15 machines keeps a small edge at 50 machines and has none at 100, where the rule choice
     barely matters.
   - Retraining on larger floors is untested.

