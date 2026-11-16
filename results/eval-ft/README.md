# eval-ft: evaluation of the fine-tune arms (2026-10-25, plan §2b)

Status: **done.** het arm 10-26 (`analysis_het.out`): A1 fails, A3 passes from 5 updates. m100 arm 10-28
(`analysis_m100.out`): A1 passes (24 updates −1.05 points vs the final policy, 40% of the way to T100), A2 fails
(+1.60 vs T100), A3 passes at every budget.

Design of the fine-tune runs: `../train-ft-m100/README.md`.

## Question
How much of a very different floor's result does a short fine-tune of the final policy reach, per budget
(2 / 5 / 10 / 24 updates)? And does fine-tuning lose the policy's 15-machine edge on B2?

## Design
- **Arms and budgets:**
  - `m100` = `../train-ft-m100`, `het` = `../train-ft-het`, each with its 5 training seeds;
  - budget b = fine-tune updates: `checkpoint_step<98,304 + 2,048 b>.pt`, and `checkpoint.pt` for b 24;
  - budget 0 is the final policy itself (`train-v5-cmix` s0-s4).
- **Cells** (`task.py`): each budget's 5 checkpoints plus MDD-PTECT, twin, test seeds 0-39, deterministic, frozen v5
  code 6d13412a. B2 settings throughout: tardiness reward, random warm-up, 21,600 s window, 900 s slot actions,
  5,400 s horizon, c ~ U[1.75, 2.5].

  | cell | floor | harness | arms | tasks |
  |---|---|---|---|---|
  | `m100` | 100 machines, as `../rq3-size-v5` / `../rq3-size-grid` | `../rq3-gen-v5/run_task.py` in process, 10-seed chunks | m100 | 4 budgets × 4 chunks |
  | `s1` | B2 with speed spread s = 1, as `../eval-alpha-spread` | `../train-ft-het/code/hetero.py --spread 1,1 --entry evaluate` | het | 4 |
  | `s0` | B2 itself (hetero.py at s 0) | same | both | 4 + 4 |

- **References** (no reruns; MDD-PTECT is evaluated in every task as the run-to-run check):

  | cell | reference | source | value (% vs MDD-PTECT) |
  |---|---|---|---|
  | m100 | final policy (budget 0) | `../rq3-size-v5/m100/cmix*` | +2.60 |
  | m100 | T100, from scratch (48 updates) | `../rq3-size-grid/m100/size100_s*` | −0.08 |
  | m100 | v5 pairs, MDD-TECT; per-instance best static v5 pair | `../rq3-size-v5/m100/pdr*` | MDD-PTECT best; −0.72 |
  | m100 | v5 switching ceiling (quoted) | `../rq3-size-oracle` | −3.3 |
  | m100 | "short fleet → MDD-TECT" (quoted, outside v5) | `../rq3-size-portfolio` | −12.2 |
  | s1 | final policy (budget 0) | `../eval-alpha-spread/out/1/final` | −2.89 |
  | s1 | v5 pairs (= `<pair>@3`); per-instance best static v5 pair | `../eval-alpha-spread/grid/out/m15_s1` | MDD-PECT −1.31; −5.41 |
  | s1 | spread law; best fixed α; train-alpha-het (quoted, outside v5) | `../eval-alpha-spread` | −15.34; −17.22; −15.92 |
  | s0 | final policy (budget 0) | `../eval-v5-cmix/test/base/cmix` | −1.69 |

- **Checks run 10-25:**
  - The s0 path (`hetero.py` at s 0) reproduces `../eval-v5-cmix` base exactly: 240 / 240 episodes, max diff 0.
  - **Dry run** (smoke checkpoints plus `train-v5-cmix_s1`, seeds 0-1, scratchpad): s1 and s0 gave MDD-PTECT and cmix
    s1 exactly as `../eval-alpha-spread` (s1: 349,534 / 351,577; s0: 95,842 / 94,738 job-s on seed 0). An episode
    takes about 4 s.
  - **m100 dry run** (smoke checkpoint plus `train-v5-cmix_s0`, seed 0): MDD-PTECT and cmix s0 equal
    `../rq3-size-v5/m100/cmix_s0-9` exactly (38,230 / 39,169 job-s). The 3 episodes took 300 s on the loaded
    machine, 0.5 GB per process, so an m100 task (6 policies × 10 seeds) is about 2-3 h.
  - **Analysis:** run on the references plus synthetic budget data. It reproduces the grid's m100 numbers (+2.60 /
    −0.08 / −0.72) and eval-alpha-spread's s1 numbers (−2.89, pairs as above).

## Analysis (`analyze.py <arm>`, pre-declared 10-25 before any fine-tune result)
- **Metric:** window tardiness, % vs MDD-PTECT pooled over test seeds 0-39, with a 95% instance-bootstrap CI, test
  seeds better / worse, and a Wilcoxon test.
- **Family:** the mean over the 5 training seeds per instance. Each training seed's own pooled % is shown with a
  one-sample t-test of the 5 against 0 (the training-seed-level test), as in `../rq3-size-grid`.
- **Per arm, target cell (m100 or s1):**
  - every budget 0 / 2 / 5 / 10 / 24 next to the references;
  - each budget paired against the final policy and against the goal;
  - **the sample-efficiency curve,** share(b) = (E_final − E_b) / (E_final − E_goal), with E = family pooled % vs
    MDD-PTECT and an instance-bootstrap 95% CI:
    - m100: goal = T100 (the from-scratch result). The gap is only about 2.7 points, so the share's CI will be wide;
      the paired b vs T100 difference is the primary read.
    - het: goal = the per-instance best static v5 pair (−5.41). There is no from-scratch v5 arm at s 1. A share above
      100% is possible, since switching can beat the best static pair.
- **Per arm, B2 (s0):** every budget vs the final policy, paired, and the retention E_b / E_final (E_final −1.69).
- **Criteria:**
  - **A1, adapts:** at 24 updates the family beats the final policy on the target cell (pooled < 0, Wilcoxon p < 0.05).
  - **A2, reaches the from-scratch arm (m100 only):** at 24 updates the family is within +1 point of T100 and not
    significantly worse (Wilcoxon p ≥ 0.05 or better).
  - **A3, keeps the B2 edge, per budget:** within +1 point of the final policy on B2 and not significantly worse.
    This is the H3 rule of `../eval-alpha-spread`.
- **Read-out for the plan:**
  - the smallest budget whose share reaches 50% and 90% (point estimates);
  - the B2 retention at that budget;
  - for m100, how far the 24-update family is from the v5 switching ceiling (−3.3%), the most a v5 policy can reach
    there.
- **Not compared:** the het arm is evaluated only on s1 / s0, and the m100 arm only on m100 / s0. Each arm is not
  tested on the other's target.

## Run
`launch.sh het|m100 [W]` runs `tasks_<arm>.txt` W at a time (logs in `logs/`). Finished tasks (`out/<cell>/<group>/DONE`)
are skipped, so relaunching resumes. Each evaluate.py process gets a 12 GB address-space cap (`task.py`).

## Outputs
- `out/<cell>/<arm>-b<b>[_s<a>-<b>]/`: `episodes.csv`, `eval.out`, `task.json` or `cond.json`, `DONE`;
- `analysis_<arm>.out`, `summary_<arm>.csv`.

## Results

### het arm (10-26; `eval-ft-het`, 8 tasks, 12 min; `analysis_het.out`, `summary_het.csv`)
Checks: the grid's MDD-PTECT@3 equals evaluate.py's MDD-PTECT (max relative difference 5e-8). MDD-PTECT matches the
reference on 160 / 160 seed-runs per cell, and s0 equals `../rq3-gen-v5`'s B2 pdr run.

**Target cell s1** (B2 with speed spread 1; window tardiness, % vs MDD-PTECT, family of 5, test seeds 0-39):

| budget (updates) | family [95% CI] | test seeds better / worse | vs the final policy, paired | share of the way to the best static v5 pair |
|---|---|---|---|---|
| 0 (final policy) | −2.89 [−4.32, −1.63] | 32 / 8 | | |
| 2 | −3.01 [−4.65, −1.71] | 33 / 7 | −0.12 (p 0.62) | +5% [−12, +29] |
| 5 | −3.17 [−4.79, −1.94] | 34 / 6 | −0.29 (p 0.20) | +11% [−4, +44] |
| 10 | −3.19 [−4.86, −1.90] | 31 / 9 | −0.31 (p 0.28) | +12% [−7, +49] |
| 24 | −3.35 [−5.01, −2.00] | 33 / 7 | −0.47 (p 0.061) | +18% [−0, +58] |

- **References on s1:**
  - best v5 pair held, MDD-PECT: −1.31%;
  - per-instance best static v5 pair (the goal): −5.41%;
  - outside v5: the spread law −15.34%, the best fixed α (MDD-PTECT@0.5) −17.22%, train-alpha-het −15.92%.
- **Every budget is still worse than the goal,** paired (+2.2 to +2.5 points, p 0.005-0.018).

**B2 (s0):** the final policy is −1.69% [−2.87, −0.50].

| budget | family [95% CI] | vs the final policy, paired | retention | training seeds |
|---|---|---|---|---|
| 2 | −0.88 [−1.92, +0.22] | **+0.83** [+0.27, +1.43] (p 0.021) | 0.52 | +0.9 to −2.3 |
| 5 | −1.74 [−3.08, −0.59] | −0.05 (p 0.094) | 1.03 | −0.8 to −3.2 |
| 10 | −1.37 [−2.81, −0.06] | +0.33 (p 0.53) | 0.81 | +0.3 to −3.0 |
| 24 | −1.05 [−2.61, +0.39] | +0.65 (p 0.17) | 0.62 | −0.6 to −1.8 (all < 0, t-test p 0.006) |

**Criteria (pre-declared):**
- **A1, adapts: FAIL.** At 24 updates the family is −0.47 points vs the final policy on s1 (p 0.061).
- **A3, keeps the B2 edge:** FAIL at 2 updates (+0.83, p 0.021), PASS at 5 / 10 / 24 (−0.05 / +0.33 / +0.65, n.s.).
- **Read-out:** no budget reaches 50% of the way to the best static v5 pair (24 updates: 18%). The B2 retention at 24
  updates is 0.62 (n.s. vs the final policy).

**Reading:**
- **A 24-update fine-tune barely moves the policy on the spread floor.** It drifts the right way (−2.89 → −3.35%,
  monotone in the budget), but by half a point, against 2.5 points still left within v5.
- **The bigger limit is the action set, not the weights.** The v5 pairs all use α 3, and the room on s1 lies at low
  α: the spread law and the α head reach −15 to −16%, about five times the fine-tune's result. This is the same finding as
  `../rq3-size-headroom` on large floors: the room is in the rule parameters v5 fixes.
- **B2 is not lost, but it gets noisier.** The 2-update dip (+0.83) recovers by 5 updates. It is not monotone, so it
  looks like update noise at a low entropy (0.0039 → 0.001 over the schedule), not steady forgetting.
- **For the plan:** fine-tuning the v5 policy is not the route to the spread floor. The α-offset arm
  (`train-alpha-off-het` / `-b2`, running 10-26) tests the action-set route with the B2 edge as its guard.

### m100 arm (10-28; `eval-ft-m100`, 20 tasks, 1 h 50 min; `analysis_m100.out`, `summary_m100.csv`)
Checks:
- All 20 tasks are complete (60 episodes per m100 task = 6 policies × 10 seeds, 240 per s0 task).
- Each budget loaded the intended checkpoints (`cond.json` / `task.json`: `checkpoint_step102400/108544/118784.pt`,
  `checkpoint.pt` for b 24).
- MDD-PTECT matches the reference on 160 / 160 seed-runs in both cells, and s0 equals `../rq3-gen-v5`'s B2 pdr run.
- The final policy (b 0, from `../rq3-size-v5`) reproduces the grid's +2.60%.

**Target cell m100** (window tardiness, % vs MDD-PTECT, family of 5, test seeds 0-39; MDD-PTECT mean 456.0 k job-s):

| budget (updates) | family [95% CI] | test seeds better / worse | vs the final policy, paired | vs T100, paired | share of the way to T100 | training seeds |
|---|---|---|---|---|---|---|
| 0 (final policy) | +2.60 [+1.24, +3.85] | 16 / 24 | | | | +0.4 to +4.4 |
| 2 | +2.47 [+1.27, +3.59] | 13 / 27 | −0.13 (p 0.76) | +2.55 (p 2e-5) | +5% [−11, +17] | −0.9 to +4.5 |
| 5 | +2.59 [+1.41, +3.80] | 15 / 25 | −0.01 (p 0.40) | +2.67 (p 1e-5) | +0% [−14, +10] | +0.6 to +5.2 |
| 10 | +1.99 [+0.91, +3.16] | 16 / 24 | −0.59 (p 0.16) | +2.07 (p 4e-5) | +23% [+2, +38] | −1.0 to +4.9 |
| 24 | +1.52 [+0.61, +2.38] | 18 / 22 | **−1.05** [−1.90, −0.33] (p 0.0098) | **+1.60** [+1.21, +1.94] (p 3e-5) | +40% [+19, +56] | −1.2 to +3.9 |
| T100 (from scratch, 48 updates) | −0.08 [−0.94, +0.76] | 23 / 17 | | | | −0.6 to +0.6 |

- **References on m100:**
  - MDD-PTECT is the best v5 pair held. The per-instance best static v5 pair is −0.72%, and the v5 switching
    ceiling is −3.3%.
  - MDD-TECT is −1.91% (n.s.). "Short fleet → MDD-TECT", outside v5, is −12.2%.
- **T100 beats the 24-update family on 32 of 40 test seeds.**
- **Per training seed, the fine-tune moves each by 0.5-1.6 points** (b 0 → b 24): s0 +3.8 → +3.1, s1 +3.4 → +1.9,
  s2 +1.0 → −0.1, s3 +0.4 → −1.2, s4 +4.4 → +3.9.
  - The seeds keep their order, so a seed that starts far from T100 stays far.
  - The 5 seeds still spread 5.1 points at b 24 (T100: 1.2).

**B2 (s0):** the final policy is −1.69% [−2.85, −0.51].

| budget | family [95% CI] | vs the final policy, paired | retention | training seeds |
|---|---|---|---|---|
| 2 | −1.80 [−2.91, −0.68] | −0.11 (p 0.74) | 1.06 | −0.6 to −3.5 |
| 5 | −1.59 [−2.64, −0.58] | +0.11 (p 0.94) | 0.94 | −0.4 to −3.1 |
| 10 | −1.79 [−3.01, −0.60] | −0.10 (p 0.58) | 1.06 | +0.1 to −3.5 |
| 24 | −1.54 [−2.57, −0.52] | +0.16 (p 0.84) | 0.91 | −0.3 to −2.4 (all < 0, t-test p 0.02) |

**Criteria (pre-declared):**
- **A1, adapts: PASS.** At 24 updates the family beats the final policy on m100 by 1.05 points (p 0.0098).
- **A2, reaches the from-scratch arm: FAIL.** At 24 updates it is still 1.60 points worse than T100 (p 3e-5, above the
  +1 point margin).
- **A3, keeps the B2 edge: PASS at every budget** (−0.11 / +0.11 / −0.10 / +0.16, all n.s.).
- **Read-out:**
  - No budget reaches 50% of the way to T100. At 24 updates the share is 40% [19, 56], with B2 retention 0.91.
  - The 24-update family is 4.8 points short of the v5 switching ceiling (−3.3%), and 2.2 points short of the
    per-instance best static v5 pair.

**Reading:**
- **On 100 machines the fine-tune adapts, but slowly.** Nothing moves in the first 5 updates. After that the family
  improves steadily (+2.60 → +1.99 → +1.52 at 10 / 24 updates).
  - At 24 updates it has covered 40% of the way to T100 in half of T100's 48 updates.
  - It costs about 10 h per seed (T100: 18.6-20.6 h).
- **It does not catch up with training from scratch at 24 updates.** The family is still worse than holding
  MDD-PTECT (+1.52%, CI above 0, Wilcoxon p 0.073). The seeds that start worst stay worst, so the start, not just the
  budget, limits the result.
- **B2 is kept cleanly.** Unlike the het arm, no budget dips, and every training seed but one (b 10 s0, +0.1) stays
  below MDD-PTECT on B2. Training on 100 machines does not erase what the policy does on B2.
- **Both arms together (§2b):**
  - Fine-tuning the v5 weights is safe for B2 from 5 updates in both arms.
  - It recovers part of the gap where the gap lies inside v5 (m100: 40% of the way to T100, A1 PASS).
  - It recovers little where the room lies in the rule parameters v5 fixes (s1: 18%, A1 FAIL).
  - Even at m100 the large room is outside v5: the v5 switching ceiling is −3.3%, against −12.2% for "short fleet →
    MDD-TECT" and −13 to −21% for v5 + MDD-TECT (`../rq3-size-headroom`).
- **Caveats:**
  - T100 has no intermediate checkpoints, so there is no from-scratch result at 24 updates. Whether starting from the
    15-machine policy beats starting from scratch at equal cost is not measured here.
  - The share's CI is wide (the gap is only 2.7 points).
