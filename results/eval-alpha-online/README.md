# eval-alpha-online: the α policy with the online law, h from the AGVs on duty (2026-10-29, user: "design the online α experiment (h from the on-duty fleet per regime, plus the spread law) with pre-declared criteria, and set it up in the lab")

Status: **done 10-30 06:42** (lab, local; started 04:45, all 538 tasks by 06:41; the first finish was reported
failed by a launcher bug in the final output check, fixed and relaunched, every task skipped). **Q1 FAIL, Q2 PASS,
Q3 PASS, Q4 FAIL, H' FAIL → N: keep the static law + gate as paper B's α policy; the online law is a fit to the
fitted floors, not a shown transfer rule.** The online law alone is strong on the transport-scarce cells (m75 / m100 at
r 0.52 −8.3 / −9.7% vs the static law alone), but the policy keeps only part of it because the offset head halves α in
most full-fleet slots. Criteria pre-declared 10-29, before any result of either stage.

## Why
- Stage 1 (`../rq3-alpha-online`) fits an online law, α_t = min(4, max(0, 3 − 6σ) × h(τ / a_t)), a_t the AGVs on duty
  per machine at the decision.
- This stage puts it in the carried-forward policy (train-alpha-off-het, v7: law × offset head) with
  `../eval-alpha-transport-size`'s size gate (job and machine heads fixed to MDD-PTECT above 50 machines).
- It tests three things against today's static law + gate on the same instances: does the policy keep the online
  law's gain, does it cost on B2, and does the law transfer to held-out floors.

## Design (`task.py`, `onlinelaw.py`, `launch.sh`, `analyze.py`)
- **The online law in the policy:**
  - `task.py` wraps `env_wrappers.twin_env.TwinSchedulingEnv.step`. Before every decision, `self.alpha_law` is set
    to the selected law at the on-duty count of that moment (`onlinelaw.py`, the engine's `_on_duty` rule).
  - The offset head then multiplies it as v7 does. σ is the env's own per-episode σ.
  - The frozen v7 code is not edited, and the policy's inputs do not contain the law.
  - `evaluate.alpha_law` (episode labels only) gives the full-fleet α.
- **Group `og`:** train-alpha-off-het s0-s4, plus two rules:
  - MDD-PTECT (two-element action, α 3), the run-to-run check against the reference's MDD-PTECT;
  - MDD-PTECT@x1, the online law alone in this harness, which must equal the simulator's online-law run.
  - Above 50 machines only the offset head chooses (`../eval-alpha-only-size`'s AlphaOnlyPolicy).
- **Policy cells and references (test seeds 0-39):**

  | cell | harness | static law + gate (reference) | final policy T15 |
  |---|---|---|---|
  | m15_b2 (B2), m15_s1 (15 machines, spread 1) | `../eval-alpha-offset`'s (B2 arguments through the arm's hetero.py, here in process) | its off-het group | `../eval-alpha-spread` final |
  | m50_r052, m75_r052, m100_r052, m100_r035, m100_r080, m200_r052 | `../eval-alpha-transport-size`'s | its `lg` group | `../rq3-size-v5`, `../rq3-size-grid` (m50 / m75 / m100 / m200 r052) |

  - A cell runs only where the online law's pair (α_full, α_short) differs from the static law's (law_online.json,
    cell-mean σ). Elsewhere the two policies are the same by construction.
  - Chunks: 15 machines one task of 40 seeds, 50-100 machines 10 seeds, 200 machines 5 seeds.
- **Laws alone, in the simulator** (`../rq3-alpha-online/run.py ... law`, this stage's launch):
  - the static law, the online law and the map rule (static in full blocks, α 0 in short ones);
  - on the eight policy cells and the five other held-out cells (m75_r035, m150_r080, m50_s1, m100_s0.5, m100_s1).
  - The best grid pair on test seeds of stage 1's six held-out cells (these five and m15_s1) comes from stage 1's
    surfaces.

## Analysis (`analyze.py`, pre-declared 10-29 before any result)
- **Metric:** window tardiness, % vs MDD-PTECT pooled over test seeds 0-39, 95% instance-bootstrap CI, better /
  worse, Wilcoxon ("significant" = p < 0.05). Policies are the family of 5 (per-instance mean).
- **Checks:**
  - og's MDD-PTECT equals the reference group's on every seed;
  - og's MDD-PTECT@x1 equals the simulator's online-law run on every seed;
  - gated cells play only MDD-PTECT on the pair heads.
- **Criteria (changed policy cells only):**
  - **Q1, the policy keeps the gain:** og is better than the static law + gate on at least half of the changed cells
    (Holm over those cells) and significantly worse on none.
  - **Q2, B2:** og is not significantly worse than off-het (the static law on B2) nor than the final policy. PASS by
    construction if the online law does not change B2's pair.
  - **Q3, no loss:** og is not significantly worse than MDD-PTECT on any changed cell.
  - **Q4, a guard on the offset head:** og is not significantly worse than the online law alone on any changed cell.
  - **H', transfer of the online law (law alone, held-out cells):**
    - within +2% of the cell's best test-seed pair;
    - not significantly worse than the static law alone;
    - where the static law is ≥ 2% worse than the best pair, the online law must beat it (p < 0.05).
- **Reported, not gating:** the online law alone vs the static law alone on every cell, and what og plays by fleet
  state (offset shares, mean played α).
- **N, the next step:**
  - Q1, Q2 and Q3 PASS → the online law + gate replaces the static law as paper B's α policy. Next: the Unity check
    of the online law (Python rule mode) on B2 and one large floor (plan §3.2).
  - Q1 FAIL, but the online law alone beats the static law alone (no changed cell worse, at least one significantly
    better) → the policy does not carry the gain. Deploy the law alone above the gate, or retrain the arm with the
    online law in the loop.
  - Q1 FAIL otherwise → the online law adds nothing in the policy; keep the static law + gate.
  - Q2 or Q3 FAIL (Q1 PASS) → apply the online law above the gate only, and report the cost.
  - H' FAIL → the paper presents the online law as a fit to the fitted floors, not as a rule shown to transfer.
- **Expectation, stated before the run (a guess, not a criterion):**
  - The v7 offset head already lowers α in short blocks (×0.5 in ~60% of slots). With the law lowering α there too,
    the policy may undershoot, which Q4 would show.
  - On B2 the gain should be largest. B2's short blocks run 2 of 7 AGVs, and stage 1's dry-run seed had a 21.6% room.

## Checks (10-29, scratchpad, with a stand-in law: one hinge (0.15, 0.45), cells from stage 1's statistics)
- **B2 seeds 0-1** (no short block in either window), checkpoint s0:
  - og's MDD-PTECT = off-het's MDD-PTECT (95.842 / 29.594);
  - MDD-PTECT@x1 = the simulator's online law (and the static law alone, as it must be without a short block);
  - the checkpoint equals its off-het result (94.702 / 32.341).
- **B2 seeds 2-3** (short blocks with 2 AGVs; the stand-in law plays 3 / 1.036):
  - og's MDD-PTECT = off-het's (246.214 / 96.561);
  - MDD-PTECT@x1 = the simulator's online-law run exactly (254.686 / 97.094), and differs from the static law alone
    (246.214 / 96.561), so the law switches with the fleet.
  - The trace shows α 1.036 from the short block's first decision (2 AGVs on duty) and 3 again when the fleet returns.
- **m15_s1 seeds 0-3** (spread 1, σ per episode from the env): og's MDD-PTECT = off-het's on all 4 seeds.
  MDD-PTECT@x1 = the simulator's online law on all 4 (e.g. 293.896 at s2, static 300.899). So the env's σ equals
  the simulator's and the spread factor agrees.
- **m75_r052 seed 2, gated:** MDD-PTECT = `lg`'s (66.089); MDD-PTECT@x1 = the simulator's online law (52.918, static
  51.717); all 24 policy slots play MDD-PTECT on the pair heads (offset ×0.5 in every slot).
- `analyze.py` ran through every section on these outputs (`--min-n 1`, `--online-root` / `--law` at the scratch
  copy). `task.py list` keeps only the changed cells.

## Cost
- Depends on the selected law. With every policy cell changed: m200 8 chunks of ~5.5 h, the 100-machine cells
  4 × 2-3 h each, m75 / m50 4 × ~1 h, 15 machines ~30 min each. About 120 CPU-h, so ~6-8 h on 20 slots.
- Law runs: 13 cells × 40 seeds × 3 laws, ~5 CPU-h.

## Run
`launch.sh [W]` checks G1, then runs `tasks.txt` (`eval ...` = `task.py`, `law ...` = `../rq3-alpha-online/run.py
... law`), W at a time. Finished tasks are skipped; `ALL_DONE` is written when every task has its output.

## Outputs
- `out/<cell>/og_s<a>-<b>/`: `episodes.csv`, `decisions.csv`, `eval.out`, `cond.json` (the law and the cell's
  statistics), `blocks.json`, `alpha_trace.json`, `DONE`;
- the law runs in `../rq3-alpha-online/out/<cell>/law/`;
- `analysis.out`, `summary.csv`, `played.csv`.

## Results (10-30)

### Checks: all pass
- og's MDD-PTECT equals the reference group's on 40 / 40 seeds in every changed cell.
- og's MDD-PTECT@x1 equals the simulator's online-law run (`../rq3-alpha-online/run.py law`) on 40 / 40 seeds in
  every changed cell.
- Gated cells (m75_r052, m100_r080, m100_r052): 0 of 4,800 decisions per cell off MDD-PTECT on the pair heads.
- The online law changes the pair on six policy cells, each from (a, a) to (a, 0): B2 3|3 → 3|0, m15_s1 0.70 → 0.70|0,
  m50_r052 2.60 → 2.60|0, m75_r052 1.88 → 1.88|0, m100_r080 3 → 3|0, m100_r052 1.43 → 1.43|0. m100_r035, m200_r052 and
  m75_r035 play 0|0 under both laws.

### Per cell, % vs MDD-PTECT (window tardiness, test seeds 0-39, family of 5)

| cell | static law alone | online law alone | static law + gate | og: online law + gate | final T15 |
|---|---:|---:|---:|---:|---:|
| m15_b2 (B2) | 0.00 | −0.66 | −1.01 | −0.90 | −1.69 |
| m15_s1 | −16.05 | −16.68 | −16.40 | −16.67 | −2.89 |
| m50_r052 | −0.98 | −3.57 | +0.21 | −0.77 | −1.29 |
| m75_r052 | −0.46 | −8.72 | −2.30 | −8.34 | +1.24 |
| m100_r080 | 0.00 | +1.61 | −0.14 | +2.52 | |
| m100_r052 | −2.90 | −12.36 | −4.25 | −10.23 | +2.60 |

### The pre-declared criteria
- **Q1 FAIL.** og beats the static law + gate after Holm on 2 of 6 changed cells (m75_r052 −6.18%, m100_r052 −6.24%,
  both p ≤ 0.0002); at least 3 were needed. None significantly worse (m100_r080 +2.66% [+0.62, +5.86], 16/18,
  p = 0.12; m50_r052 −0.98%, m15_s1 −0.32%, B2 +0.11%, all n.s.).
- **Q2 PASS.** On B2 og is +0.11% vs off-het (p = 0.4) and +0.81% vs the final policy (p = 0.4).
- **Q3 PASS.** og is not significantly worse than MDD-PTECT anywhere (worst m100_r080 +2.52%, p = 0.44).
- **Q4 FAIL.** og is significantly worse than the online law alone on m100_r052 (+2.44% [+1.53, +3.55], 10/30,
  p = 0.0001); m50_r052 +2.90% is not significant (p = 0.45).
- **H' FAIL, on one cell of six.** The online law alone is within +2% of the best test-seed pair on every held-out cell
  (+0.31 to +1.75%) and beats the static law where it has to on m150_r080 (−11.40%), m100_s0.5 (−5.03%) and m100_s1
  (−2.15%), all p < 0.001. On m50_s1 the static law is +3.15% off the best pair, so the online law had to beat it; it
  is −1.35% [−2.81, +0.02] (20/13, p = 0.081).
- **N: "the online law adds nothing in the policy: keep the static law + gate"** (rule N, third case). The second case
  (law alone beats the static law alone with no changed cell worse) does not apply, because the online law alone is
  +1.61% vs the static law alone on m100_r080 (n.s., p = 0.83); the analysis reads "worse" as a positive mean, as the
  README's criterion is worded ("significantly" only where it says so). Under a not-significantly-worse reading it would
  have been the second case (deploy the law alone above the gate, or retrain the arm with the online law in the loop),
  since m75_r052 (−8.30%) and m100_r052 (−9.74%) are significant. Recorded here, not applied: the pre-declared code
  and wording decide.
- **H' FAIL → paper B presents the online law as a fit to the fitted floors**, with the five passing held-out cells
  reported as such.

### Why the policy loses the law's gain
What og plays (offset head shares ×1 / ×0.5 / ×0.25 / ×1.5 / ×2, the law's α, mean played α):

| cell | full fleet: ×0.5 share | law α | played | short fleet: law α | played |
|---|---:|---:|---:|---:|---:|
| m15_b2 | 0.58 | 3.00 | 2.20 | 0 | 0 |
| m15_s1 | 0.52 | 0.70 | 0.63 | 0 | 0 |
| m50_r052 | 0.67 | 2.60 | 1.62 | 0 | 0 |
| m75_r052 | 0.65 | 1.88 | 1.17 | 0 | 0 |
| m100_r080 | 0.66 | 3.00 | 1.72 | 0 | 0 |
| m100_r052 | 0.66 | 1.43 | 0.87 | 0 | 0 |

The head plays ×0.5 about as often in both fleet states (52-67% of full-fleet slots, 59-70% of short-fleet slots),
so it does not track the fleet. The online law already sets α 0 in the short blocks, so the halving only acts in the
full-fleet blocks, where it brings α down to 57-73% of the law's (90% on m15_s1). This is the undershoot the pre-run
expectation named ("which Q4 would show"). On m75_r052 og keeps nearly all of the law's gain (−8.34 vs −8.72); on
m100_r052 and m50_r052 it loses 2.1-2.8 points.

### Against the expectation stated before the run
- Undershoot by the offset head: yes (Q4).
- "On B2 the gain should be largest": no. The online law alone is −0.66% vs the static law alone on B2 (n.s.), and og
  is +0.11% vs off-het. Stage 1 had already shown the dry-run seed's 21.6% room was one instance (B2's 40-seed best pair
  is 4|2, and O1 plays 3|0).

### What this leaves for paper B
- The α policy stays the static law + gate (`../eval-alpha-transport-size`, `../eval-alpha-offset`).
- The online law alone is the stronger rule on transport-scarce floors without a learned head: m75_r052 −8.72%,
  m100_r052 −12.36% vs MDD-PTECT, and held-out m150_r080 −11.40% vs the static law. It can be reported as a fitted rule
  (stage 1 LOO regret 1.19% vs 3.32%), not as a transfer result and not inside the policy.
- Options this result opens, not started (user's call): deploying the law alone above the gate (would need its own
  pre-declared test, since rule N did not pick it), or retraining the offset head with the online law in the loop.

