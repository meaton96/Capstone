# eval-ft: evaluation of the fine-tune arms (2026-10-25, plan §2b)

Status: **het arm done 10-26 (`analysis_het.out`): A1 fails, A3 passes from 5 updates.** The m100 arm waits on its
cluster training (ETA ~10-30). Lab:
- `eval-ft-het` waits on `train-ft-het` (local, 8 tasks, ~20 min each);
- `eval-ft-m100` waits on `train-ft-m100` (cluster training, checkpoints pulled; local, 20 tasks).

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

### m100 arm
Pending (`train-ft-m100` on the cluster, ETA ~10-30).
