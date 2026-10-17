# rq4-uncert-v5-policy: the final policy under the twin uncertainty layer (2026-10-17, user)

Status: **queued** (lab `rq4-uncert-v5-policy`, cluster, 96 twin tasks).

## Why
RQ4's scope was open: breakdowns only, or "disruptions and uncertainty". The user (10-17) asked to set up the
uncertainty part. `../rq4-uncert-screen` (10-11) scored the v5 rules, a re-planning planner and the hindsight oracle
on 38 noise cells, but gate G1 stopped training under noise, so no learned policy was ever scored there. This runs the
final policy, the PPO trained on mixed due dates (cmix pick s3), with all PPO families and the 6 v5 pairs in the
screen's cells.

## Method (`task.py`, `submit.sh`, `pull.sh`, `analyze.py`)
- **Harness:** a thin wrapper around `../rq3-gen-v5/run_task.py` (frozen code `../rq2-rebase-v5/code/env` = the
  checkpoints' code, B2, test seeds 0-39) that adds `--uncertainty <cell>`. The noise seed is the instance seed, so
  every policy sees the same noisy world. Policies act on the observed (noisy) state.
- **Cells (24):**
  - realistic / severe for E1, E2, E2d, E3, E4n, O1, O2, O3 and mix (18);
  - the online-estimator cells E2 / E2d / mix × realistic / severe with info=I1 (6).
  - `none` = B2 (rq3-gen-v5 base, eval-v5-cmix test).
- **Groups:**
  - pdr: 6 v5 pairs + MDD-TECT;
  - ppo: B2-trained s0-s4;
  - cmix: s0-s4;
  - loose: s0-s4.

  Each policy group also runs MDD-PTECT.
- **Analysis:** per cell, every pick and family mean vs MDD-PTECT (the screen's noise-free rule NF) and vs the cell's
  best pair, plus MDD-PTECT's degradation from `none`.
  - **Measure (since 10-17):** the E3 / mix cells, incl. mix I1, are on the screen's window-tardiness measure. Every
    policy's value gets the per-seed offset of `screen_offset.csv`, and `summary.csv` marks these rows
    `measure = screen (offset)`. The reason is under "Root cause" below. The offset is the same for every policy, so
    paired differences and tests do not change.

## Reproduction check against the screen (local, seed 0, MDD-PTECT; `analyze.py` repeats it on all seeds)
**MDD-PTECT is identical to the screen's MDD-PTECT@3 in 16 of 24 cells, and at `none` on 40 / 40 seeds.** That covers
every E1, E2, E2d, E4n, O1, O2 and O3 cell.

It differs in E3 realistic / severe (172.9 vs 173.5, 279.0 vs 328.4), mix realistic / severe (170.9 vs 171.5, 306.4 vs
355.6) and all 6 info=I1 cells (up to 380.7 vs 450.8).
- **Not the code version:** the screen's own harness gives the screen's values with both snapshots (ba26f3f5 and
  6d13412a).
- **The evaluator path differs:** `evaluate.py` → `TwinSchedulingEnv` builds the world from the scenario's uncertainty
  block, while the screen built the trace directly (`uc.UEpisode`). Both start the window at the first agent decision.
  The routing trigger is not the cause.
- **Explained (10-17, below):** two independent causes. One is the tardiness measure (E3, mix); the other is when the I1
  bias estimator is read (info=I1). The world, the noise draws, t0 and the routing trigger are the same in both paths.
- **Consequence:** inside this experiment, policies and pairs share one path, so their comparison is valid in every
  cell. The screen's planner / oracle / level-tuned-rule numbers carry over as they are to the 16 matching cells, and
  to the 4 E3 / mix (I0) cells after adding a per-seed offset (cause 1). They do not carry over to the 6 info=I1 cells
  (cause 2).

### Root cause (10-17; `screen_diff.py`, `screen_diff.out`, `screen_offset.csv`, `screen_diff_runs.json`)
`screen_diff.py repro` runs path B (the screen's `uc.UEpisode.play`) on path A's code (`code_realizable` = 6d13412a =
`../rq2-rebase-v5/code/env`) and changes one thing at a time. Seed 0, MDD-PTECT α 3:

| cell | path A | path B | B with A's measure | B with sync at decisions | B with both |
|---|---|---|---|---|---|
| E3 realistic / severe | 172.870 / 279.026 | 173.484 / 328.381 | **172.870 / 279.026** | 173.484 / 328.381 | 172.870 / 279.026 |
| mix realistic / severe | 170.929 / 306.352 | 171.537 / 355.644 | **170.929 / 306.352** | 171.537 / 355.644 | 170.929 / 306.352 |
| E2 I1 realistic / severe | 109.414 / 122.773 | 109.416 / 121.240 | 109.416 / 121.240 | **109.414 / 122.773** | 109.414 / 122.773 |
| E2d I1 realistic / severe | 103.068 / 115.740 | 97.711 / 115.135 | 97.711 / 115.135 | **103.068 / 115.740** | 103.068 / 115.740 |
| mix I1 realistic / severe | 173.211 / 380.701 | 172.315 / 450.805 | 171.706 / 401.512 | 173.819 / 429.993 | **173.211 / 380.701** |

Each change gives path A's value exactly (to < 1e-5, the float32 rounding of the metrics) and leaves the other cells as
they were. The control E1 severe gives 111.544 in every variant.

**Cause 1 (E3, mix): two tardiness measures.**
- **Path A:** `window_tardiness` in `episodes.csv` = `Twin.metrics()["tardiness_sum"]` at the end minus at the first
  agent decision. `metrics()` sums arrived jobs only (`Twin.order`, the Unity sensor's AllJobs), so the measure is a
  delta of cumulative lateness over released jobs.
- **Path B:** `des_twin.planner.window_tardiness`, the sum over every job of the world of
  overlap([due, exit or t1], [t0, t1]).
- **When they differ:** only for a job that arrives after its due date. Without E3 no job does (due = arrival + c ×
  work, c ≥ 1.75), hence the 16 matching cells. Under E3 some do:
  - a job that arrives inside the window after a due date before t0 adds its pre-window lateness (t0 − due) to A on
    arrival;
  - a job that is overdue but has not arrived by t1 adds (t1 − max(due, t0)) to B and nothing to A.
- **Policy-independent:** the gap depends only on arrivals, due dates, t0 and t1. B − A is 49.355 for MDD-PTECT,
  SRT-PECT and ATC-PTECT alike on E3 severe s0 (35.262 on s1), and likewise for E3 realistic and mix severe
  (`screen_diff.out` §2). It equals the warm-up-only formula of `screen_diff.py offset`.
- **Size** (`screen_offset.csv`, 6 cells × 40 seeds): mean B − A is 1.7 in E3 / mix realistic, 25.0 in E3 severe and
  20.5 in mix severe. Path A's level is therefore 0.8-0.9%, 7.1% and 5.1% below the screen's. Of E3 severe's 25.0:
  - +30.6 are overdue orders not yet arrived by t1 (9.7 jobs per seed);
  - −5.7 is pre-window lateness that A books (3.7 jobs).

  A few seeds are negative (A > B). The mix I1 cells have their own offsets, because the I1 warm-up moves t0 by up to
  46 s.

**Cause 2 (info=I1): when the bias estimator is read.**
- **How the estimator updates:** only when reported completions are applied, which happens lazily in
  `ViewCore.sync()`.
- **The stale read:** `ViewCore.bhat` and `ObservedState._ops` read the estimator **without syncing**. So a rule's
  p_hat = p_plan × b_hat (MDD's remaining work, the pre-dispatch estimate in `Twin._start`) uses only the completions
  applied by the last read that did sync (machine loads, live jobs, utilization).
- **Why the paths differ:** path A builds the observation at every agent decision, and those reads sync. Path B builds
  none, so its job-rule halves and pre-dispatch estimates read a stale estimate, whose age depends on the rule's own
  read pattern.

**Which path is right for the thesis.**
- **Measure: path B (the screen).**
  - Window tardiness is the tardiness accrued inside [t0, t1]. An order is late from its due date whether or not its
    material has arrived, and lateness before t0 lies outside the window.
  - Path A's measure leaks pre-window lateness into the window and drops overdue orders that are not released by t1.
  - The gap is policy-independent, so this experiment's paired differences, rankings and Wilcoxon tests in the E3 / mix
    cells are exact. Only levels and percentages shift (percentages by T_A / T_B, e.g. × 0.93 in E3 severe).
  - Adding `screen_offset.csv` per seed to every path-A window tardiness in the E3 / mix cells (incl. mix I1) puts them
    on the screen's measure exactly. Verified for three rules, two seeds, three cells.
- **I1: path A.**
  - Inside the window, path A has the documented semantics: every estimator read sees every booking visible by now.
    Path A equals "full sync from the first decision on" in 36 / 36 runs (6 I1 cells × seeds 0-5, `screen_diff.out`
    §3).
  - Path B equals full sync in 1 / 36 runs. Per seed it is off by 3.2% on average and by up to 10.5%. Pooled over the
    36 runs it is +1.1%, per cell −1.6% to +2.3%, with no consistent sign.
- **Shared caveat (I1):** in both paths the warm-up reads the estimator unsynced, because the warm rule builds no
  observation.
  - Against full sync from t = 0, path A matches in 21 / 36 runs, with mean |deviation| 0.8% and max 9.7%. The state
    at t0 differs.
  - That state is the same for every policy of a seed, so it gives a different starting point, not a bias between
    policies.
- **Code bug and fix (fix applied 10-17 in the live `env/` only; the frozen snapshots are untouched):**
  - Before the fix, `env/des_twin/observed.py` was identical to the frozen copies. `ViewCore.bias_estimate`, `bhat`
    and `ObservedState._ops` now call `sync()` before reading the estimator, as `screen_diff.py`'s "full" monkeypatch
    does.
  - With the fix, path A changes only through the warm-up, and path B moves to path A's values. Runs from the frozen
    snapshots, this experiment and the screen included, keep the old behaviour.

### What in `../rq4-uncert-screen/README.md` is affected
Every screen number uses path B, so the screen's internal comparisons share one measure. What carries over to this
experiment:
- **E3 and mix (I0) rows: correct; comparable after the offset.** These numbers are on the right measure:
  - Table 1 rows E3 / mix realistic / severe: LT means 188.4 / 353.5 / 214.7 / 399.2; NF, planner and oracle vs LT;
  - §2 item 5: E3 severe degradation of LT +143%, oracle +161%, planner +174%. These include the overdue orders not
    yet released, which are the same for every controller;
  - §3: the realizable rows for E3 and mix.

  To set this experiment's numbers next to them, add `screen_offset.csv` to ours, or compare paired differences, which
  the measure does not change.
- **info=I1 rows: run on a stale estimator.** Do not cite them next to this experiment; the screen's I1 conclusions are
  open until a rerun.
  - Table 1, the 6 I1 rows:
    - LT picks and means: 155.5, 161.9, 151.2, 159.3, 214.6, 406.5;
    - NF vs LT: −0.2, +3.2, +0.1, −0.6, +1.0, +1.2;
    - planner vs LT: −7.2, −2.3, −3.1, −4.0, −1.4, +0.5.

    There are no oracle runs in I1 cells.
  - "Why": the I1 figure of the development finding (E2 severe, α 3: +19% at I1). It has the same unsynced reads. The
    I0 (+11%) and I2 (+33%) figures are unaffected, since I2 reads the true bias.
  - §2 item 1: E2 severe I1 as one of the three cells where LT is ahead, and the only Holm survivor (p 0.021, "α 2
    undoes it").
  - §2 item 2, entirely (`I1adaptive_vs_I0LT`, −1.6% to +3.3%).
  - §2 item 3: the planner's 3-10% margin "under ... E2 / E2d at I1" and "the worst cell is mix severe I1 (+0.5%)".
    The planner syncs at its re-plans (`belief_transform`), but the rules it plays between them read stale estimates.
  - §3: the realizable rows E2 I1, E2d I1, mix I1 (bound vs LT and excess over `none`).
  - §4 gate:
    - (i): the E2 severe I1 trigger (bootstrap, Holm p 0.021);
    - (ii): the plain-reading pass at E2 severe I1;
    - (ii)x: the I1 excesses.

    The G1 FAIL verdict still stands, because the 18 I0 cells fail (ii)x on their own.
  - The mix I1 rows also carry cause 1. Their offset is in `screen_offset.csv`, but it alone does not make them
    comparable.
- **Unaffected:**
  - every E1 / E2 / E2d / E4n / O1 / O2 / O3 row at I0;
  - `none`;
  - `none,info=I1`, where b_hat stays exactly 1.

**Decided and done (user, 10-17):**
- **`analyze.py` reports the E3 / mix cells on the screen's measure** (see Method). Smoke check on the path-A values at
  hand (seed 0 of all 6 cells, seed 1 of three): after the offset, E3 / mix (I0) match the screen on every seed, the
  mix I1 cells do not (cause 2), and paired differences are unchanged. `analysis.out` on the current data is
  unchanged.
- **The sync fix is in the live `env/des_twin/observed.py`:** `bias_estimate`, `bhat` and `_ops` sync before reading
  the estimator. The frozen snapshots, and so this experiment and the screen, are untouched.
  - **Regression test:** `env/tests/test_des_twin_uncertainty.py::test_i1_estimate_does_not_depend_on_who_synced_the_view`
    (3 I1 specs; an extra `sync()` per decision must not change the episode). It fails on the old code and passes
    with the fix.
  - **Other tests:** the 33 uncertainty tests and the 111 other twin tests pass.
  - **Exactness:** `../dev-twin-uncertainty/check_exact.py` (zero config, bit-exact vs 95af8606) on the fixed code:
    PASS, 110 / 110 cases.
- **The screen's I1 cells are not rerun (yet).** Until they are, cite them only with the caveat above. This
  experiment's I1 cells need no rerun.

## Results
Pending.
