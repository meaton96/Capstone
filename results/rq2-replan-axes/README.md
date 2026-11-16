# rq2-replan-axes: what sets the best re-plan interval Δ*? (2026-10-29, plan §5b, paper C's main contribution)

Status: **axis 7 approved by the user 11-10 as proposed (§19).**
- **L0, recorded as a result:** under the exact forecast, latency only shifts the re-plan lattice. The forks know when a
  plan acts and the twin is deterministic, so a plan applied T_c late is the zero-latency plan on a lattice shifted by
  T_c (identical on 3 cases). No latency sweep is run.
- **7a done (§20): F7a PASS.** On one cluster core the real-time interval keeps ≥ 85% of the best gain at all 35
  conditions (lowest 92.0%). Only m100's exceeds 60 s: 126 s, keeping 98.4%. Compute starts to bind at about 200
  machines (about 2.5 min serial locally, 28 s with 6 forks).
- **7b, the horizon arm** (§19 implementation): 720 tasks; `launch7.sh` adds the pre-declared 7,200 s extension
  itself. It runs as the local lab entry `rq2-replan-axes-horizon`, about 115 core-h on 20 slots, with readings C7, H1
  and H2 from `analyze7.py`.
  - The first run stopped on 11-11 at 590 of 720: a reboot killed it, and the 11-10 real-time guard had stopped 9 tasks
    at T_c = 0.
  - The guard was fixed on 11-13 (`run7.py` 1dfd1e7f), and its checks pass. The 590 outputs are kept, and the arm
    waits to be requeued for the 130 missing tasks.
- 7c, latency as a factor, goes into axes 4-5's designs.

**§8's fit on axes 1, 2 and 6 done 11-09 17:48** (§18; design §17, approved 11-09; lab `rq2-replan-axes-fit`,
23 tasks in 4 min 31 s). **F1, F2 and F3 all FAIL.**
- **No equation beats one constant ratio condition-out.**
  - M0 re-plans every 0.36 p̄ (about D1 120 s). It keeps ≥ 85% of the best gain at 34 of 35 held-out conditions.
  - Floor size (F1) and the seven covariates (F2) do not improve on it; M2 keeps only 31.
- **Axis 6's size law does not survive censoring.** With its upper bounds treated as bounds, the size slope is +0.07
  (−0.51 with them taken as exact, §14).
- **F3 misses its line by 0.006.** M0's compute is C 0.506 against 0.5 [90% 0.37, 0.63], and both F3 conditions hold
  in 54% of resamples. The verdict stands; §18 says how to state it.
- **Guidance for paper C:** re-plan as often as compute allows, and no coarser than about a third of the mean
  operation time.
- **Next (§18):** the user chose axis 7 on 11-10 (design §19). The other options were axes 4-5 as direction tests and
  an optional held-out check of the constant rule.

**Axis 2 (load × due-date allowance at two regime block lengths) done 11-08 18:28** (§16; design §15, lab
`rq2-replan-axes-load`, Slurm 22040437-44, 8,000 tasks in 4 h 28 min). The 9 failed tasks are one all-zero seed whose
outputs were written before a summary print failed; they change nothing.
- C3 passes.
- **R_ℓ, R_c and R_mix fail at 5,400 s blocks and pass at 1,350 s blocks.** By rules 2-4 the machine busy share u, the
  mean c, and the surge and short-fleet time shares enter §8's fit as covariates.
  - The worst miss is B2's mix without short-fleet blocks, where B2's 300 s keeps 66.4%.
  - R_c's one miss (81.6% at c 3.0) depends on the reference.
  - The two block lengths differ mainly in where the centre sits: B2's plateau is among the longest of its neighbours,
    reg k 0.25's is the shortest.
- Re-planning adds 3.9-19.5 points at every cell. Δ* is an upper bound at 15 of the 22 cells (rule 9).
- At 1,350 s blocks Δ* shortens with machine load (reported slope −2.0 [90% −3.2, −0.9] on log₂ u). At 5,400 s blocks
  there is no load trend, and c shows none at either length.
- D1 60 s keeps at least 92% of the best gain at all 35 distinct conditions of axes 1, 2 and 6. B2's 300 s keeps as
  little as 66%, and D1 120 s as little as 84%.

**Axis 6 (floor size, fleet ratio) done 11-04 02:05** (§14; design §13, lab `rq2-replan-axes-size`, 2,400 tasks, no
failed task; the cluster's patch drain of 10-30 to 11-03 only delayed it).
- C2 passes.
- **S_m moves Δ\***: slope −0.51 [90% −0.70, −0.16] on log₂ m, about half the floor-level rate (Δ* ∝ p̄ / √m), and
  beyond what longer trips alone explain. By rule 4, log m enters §8's fit as a primary variable. Of the five sizes,
  only dropping m100 makes the reading indeterminate.
- **R fails** in both arms (B2's 300 s keeps 75.5% at m100; the fleet arm fails only at its shared m50 level), so by
  rule 6 on-duty AGV utilization enters the fit as a covariate. R120 also fails, narrowly (83.9% at m50).
- Re-planning still adds 4.6-10.2 points at every level. The finest interval run, D1 60 s, keeps ≥ 92% of the best
  gain at every level of axes 1 and 6.
- At 100 machines one serial re-plan takes longer than a 60 s interval (T_c / Δ 1.05).

**Axis 1 (time scale) done 10-29 on seeds 0-39 with D1 60 s added** (§12; seeds 0-19 alone in §11).
- C0 and C1 pass.
- Reading S is indeterminate in both arms at 40 seeds, so by rule 8 transport share and regime length enter §8's fit
  as covariates.
- **R fails in both arms** (worst kept share 83% in proc, 77% in reg, against 85%). By rule 8 both arms therefore count
  as moving Δ*, so rules 3 and 4 both fire: axis 6 next, then axis 2 at two regime block lengths.
- The best gain sits at D1 60 s at 7 of the 9 levels, so Δ* is an upper bound there (rule 9).
- Seed noise in R's reference explains none of the failures, but proc's is marginal (§12 remark).
- The user added rules 8 and 9 on 10-29 20:24, before the 40-seed analysis.

Design, harness and dry run done 10-29. The user approved the §10 proposals on 10-29: X 15%, plan-once baseline, the
proc and reg arms plus the units check, cluster, 20 seeds.

## 1. Question

`../rq2-replan` found that on B2 the twin planner's gain stops growing below 5-minute re-plans (mean operation 240-480 s per block;
exact forecast: −11.2% vs MDD-PTECT at 5 min, −11.5% at 1 min). That is one floor. Plan §5b asks what moves the best
interval Δ* and whether one equation in Δ / p̄ captures it (p̄ = mean operation time). The study ablates one variable at a
time from B2, derives guidance rules, and fits log(Δ* / p̄) on the dimensionless ratios if the data allow it (§8).

Harness: planner v2 exactly as in `../rq2-replan` 1a. At every boundary it forks the running twin, rolls each of the 6 v5
pairs out to the window end with hold-to-end tails, and keeps the best (`plan_triggered`, copied unchanged). Measure:
window tardiness on B2 test seeds. Axis 1 uses the exact forecast. Code is frozen in `code/env`, a `git archive` of
`env/` at 09f53cbb (`code/STAMP`), first on the path; `run.py` asserts that nothing under test comes from the live `env/`.

## 2. Framing: what can and cannot move Δ* / p̄

Everything in the twin that carries time can be scaled by one factor k:
- processing times, arrivals, due dates;
- regime blocks, the AGV schedule, warm-up and window;
- transport: AGV speed and turn rate / k, handshake and pre-dispatch lead × k;
- the tick dt × k.

The rules have no time constants of their own:
- SRT and MDD compare times with times;
- ATC divides slack by k_ATC × p̄ over the candidates;
- PECT / PTECT (quad) add α p² / p̄ to a completion time.

So **scaling every duration by k is a change of units.** The run is the same run in other units, and Δ* / p̄ cannot move.
The dry run confirms this to float rounding (C0, §9). The user's "if Δ* / p̄ stays constant, the guidance is a pure
ratio" is therefore true in the twin by construction. What a real plant adds on top of a change of units is what does not
scale:
- the planner's wall-clock compute and the latency it adds (axis 7);
- reporting delays;
- the ratios that change when only one time scale changes.

By dimensional analysis Δ* / p̄ = f(π₁, π₂, ...) over the dimensionless groups:

| group | B2 value (seed 0) | axis |
|---|---|---|
| transport per operation / p̄ (τ / p̄) | 31 s / 355 s ≈ 0.09 | 1 (proc arm), 6 |
| regime block / p̄ (T_reg / p̄) | 5,400 s / 355 s ≈ 15 | 1 (reg arm), 3 |
| machine utilization u; due-date allowance c | ≈ 0.8; U[1.75, 2.5] | 2 |
| MTTF / p̄, MTTR / p̄ | no failures | 4 |
| forecast error / p̄, churn, duration CV | exact | 5 |
| machines m, AGVs per machine; decisions per Δ (Δ m u / p̄) | 15, 7 / 15 | 6 |
| window and rollout horizon / Δ; compute T_c / p̄ | 21,600 s / Δ; ≈ 3-5 s per re-plan | 7 |

Axis 1 asks this question for the two time scales besides p̄ that B2 has, transport and regime length. Each is scaled on
its own.

## 3. Δ*: definition (pre-declared; X is the user's decision)

- **Gain over plan-once.** g(Δ) = 100 (1 − Σₛ T_s(Δ) / Σₛ T_s(once)), pooled over the level's seeds s, > 0 better.
  - T_s(Δ) is the planner's window tardiness at interval Δ.
  - T_s(once) is plan-once: the same planner with one plan at the warm-up. With the exact forecast that is the
    instance's best fixed v5 pair, held all window (the `<level>-fixed` task).
  - The gain against MDD-PTECT (`../rq2-replan`'s reference) is reported beside it.
  - **Why not MDD-PTECT:** its distance from the best pair changes with the condition. At proc k 4 it is 21% behind the
    best fixed pair on seed 0. The planner's gain against it is then about +20% at every Δ, and a band relative to that
    gain swallows everything re-planning adds (dry run, §9). Plan-once makes the curve measure what re-planning buys.
    In later axes with forecast errors, plan-once is run as the planner's own single plan.
- **Plateau.** The contiguous run of grid points around the best point Δ_b with g(Δ) ≥ (1 − X) g(Δ_b).
- **Δ\*.** The plateau's right edge, interpolated linearly in log Δ to where the curve crosses (1 − X) g(Δ_b) between the
  last grid point inside and the next one. It is the coarsest interval that keeps (1 − X) of the best re-planning gain,
  which is what a practitioner acts on. It is reported in seconds, as Δ* / p̄, and as the grid edge.
- **Plateau width.** Right edge / left edge (the left edge is interpolated the same way). It is censored when the finest
  grid point is inside the band.
- **Flat level.** If g(Δ_b) < 1 point, re-planning buys almost nothing and any Δ will do. No Δ* is defined there.
- **Uncertainty.** Seed bootstrap (2,000 resamples). Resamples are paired across Δ and joint across an arm's levels for
  the slope.
- **p̄.** The mean planned operation time over the eligible machines, for the jobs arriving in the window, averaged over
  the level's seeds. On B2 seed 0 it is 355 s; the fastest-machine mean is 302 s and the realized mean under MDD-PTECT is
  303 s. All three are recorded, so the fit can switch to another definition without rerunning.

**X (proposed: 15%).** The estimator was run on `../rq2-replan`'s 40-seed 1a curve, this study's grid only
(`preview.out`). With the plan-once baseline the B2 curve is +7.2, +6.8, +7.1, +5.4, +5.1, +3.6, +3.4 at D1 120, 180,
300, 600, 900, 1800, 3600 s.

| X | Δ* on B2 | grid edge | bootstrap 90%, 40 seeds | edge = 300 s in, 40 / 20 seeds |
|---|---|---|---|---|
| 5% | 330 s = 0.93 p̄ | 300 | 144-358 s | 52% / 50% (noise in the top plateau moves the edge to 120-180 s) |
| 10% | 384 s = 1.08 p̄ | 300 | 174-432 s | 82% / 66% |
| **15%** | **448 s = 1.26 p̄** | 300 | **319-524 s** | **95% / 82%** |
| 20% | 521 s = 1.47 p̄ | 300 | 387-667 s | 91% / 80% |
| 30% | 950 s = 2.67 p̄ | 900 | 529-1,298 s | 23% / 34% (the band edge lies on the flat 10-20 min shelf) |

- Below 10% the band is narrower than the seed noise among 1-5 min intervals.
- At 30% and above it falls on the shelf between 10 and 20 minutes.
- 15% keeps 85% of what re-planning adds over planning once, and gives the most stable Δ* on B2.

The choice is made on pilot data before any new run, and 10% / 20% are reported as sensitivity on every level.
(Against MDD-PTECT, X 10% plays the same role: 470 s, edge 300 s in 95% / 84%.)

## 4. Interval grid

D1 ∈ {120, 180, 300, 450, 600, 900, 1800, 3600} s, the B2-equivalent interval. The run's interval is Δ = s × D1, with
s = k in the arms that scale the operations' clock (all, proc) and s = 1 in reg. On B2 this is Δ / p̄ ≈ 0.34-10.
- Seven points are `../rq2-replan`'s. 450 s is new: it sits in the 300 → 600 s drop where Δ* lies on B2.
- 60 s is dropped: it is the costliest point (1,098 s of planning per instance) and adds nothing on B2. 1,200 s is dropped
  too: it lies on the shelf.
- **Extension rule:** a level whose Δ* is censored at a grid end gets D1 60 s or 7,200 s added for the same seeds (rule 6).

## 5. Axis 1: time scale (lab `rq2-replan-axes-time`)

| arm | changes (B2 otherwise) | levels k | what moves | role |
|---|---|---|---|---|
| `all` | every duration × k, transport and tick included | 1/8, 10, 32 | nothing (units) | check C0, 5 seeds × D1 300 / 900 |
| `proc` | processing, arrivals, due dates, regime blocks, AGV schedule, warm-up and window × k; transport and tick physical | 0.5, 1, 2, 4, 16 | τ / p̄ ≈ 0.17 → 0.005; AGV busy share 0.76 → 0.03 (seed 0) | reading S / R |
| `reg` | regime blocks × k (B2 regenerated with `regime_block_seconds` 5400 k) | 0.25, 0.5, (1), 2, 4 | T_reg / p̄ ≈ 3.8 → 61; blocks per window 16 → 1-2 | reading S / R |

- **proc range.**
  - Below k 0.5 the 7 AGVs saturate: transport demand per hour doubles with each halving of k, and the AGVs are already
    76% busy at 0.5.
  - Above 16, the 50 Hz float32 countdown starts to distort the floor: Unity's `remaining -= dt` in float32, which the
    twin mirrors, rounds each step to the binade of the remaining time. B2's operations run from 32 to 2,372 s
    (median 285 s, 5 seeds).
  - Measured with the twin's `f32_countdown`: at k 16 operations last ≤ 0.1% longer than nominal up to the 99th
    percentile and 1.3-1.9% longer above it. At k 32 a quarter of them are 0.8-1.8% long.
  - World and forks share the distortion, so the planner's forecast stays exact; only the floor drifts from "B2 × k".
- **proc pairs its instances with B2 exactly.** The same jobs are scaled after generation (common random numbers across
  k). reg regenerates the instances, so its levels pair by seed only. Level k 1 is B2 in both arms, and the tasks are
  shared.
- **What proc changes besides τ / p̄.** As transport gets cheaper, B2's short-fleet blocks stop being a bottleneck. The
  regimes then call for different pairs less often, so re-planning's value itself can fall. The dry run hinted at this at
  k 4 (§9), but 20 seeds do not bear it out (§11). A level where it happened would be reported as flat.
- **Seeds:** B2 test seeds 0-19 (40 if the user prefers, §10).
- **Tasks:**
  - check: 3 k × 2 D1 × 5 seeds = 30;
  - proc: 5 × 8 × 20 = 800, plus 100 fixed-pair tasks;
  - reg: 4 × 8 × 20 = 640, plus 80 fixed-pair tasks;
  - total 1,650.
- **Cost (dry run, box loaded to ~20):**
  - 3-5 s per re-plan, and a task's cost is its number of re-plans (6-180).
  - One (level, seed) over the 8 intervals is about 500 re-plans, about 0.5 core-h; a fixed-pair task takes 5 s.
  - Axis 1 is about 94 core-h: about 7 h wall on the 14 free local slots, or 1-2 h on the cluster (`../rq4-uncert-screen`
    ran 1,360 planner tasks there in 36 min). 40 seeds doubles it.
- **Launch:** cluster `bash submit_cluster.sh axis1` (the lab future as set up), or local `bash launch.sh axis1 14`.
  Both skip finished outputs. The cluster's copies of the instance files (`rq2-oracle-steady/sim.py` and its imports)
  differ from the local ones only in lines B2 does not use (checked 10-29), and C1 would catch any drift.

## 6. Pre-declared criteria and decision rule

**Checks (must pass before any reading):**
- **C0 (units).** Each `all` task at k = 2ⁿ has the pair schedule of proc k 1, and tardiness / k equal to it within 1e-6
  relative. k 10 is reported only: float32 rounding of times that are not exactly scaled may move ties and make one seed's
  run diverge.
- **C1 (regression).** proc k 1 equals `../rq2-replan`'s `exact-d<D1>` (frozen code ba26f3f5) on every overlapping
  (D1, seed): tardiness and MDD-PTECT within 1e-6, same schedule. That is 140 cells.

**Readings, per arm, over its non-flat levels:**
- **S (shape).** The OLS slope of log₂(Δ* / p̄) on log₂ k, with a 90% bootstrap CI.
  - **Invariant** if the CI lies inside ±0.25. Across proc's five octaves that bounds the change of Δ* / p̄ to a factor
    of about 2.4.
  - **Moves Δ\*** if the CI excludes 0. The ratio then enters the fit.
  - **Indeterminate** otherwise.
  - A slope of 0 in proc means Δ* follows p̄; −1 means it stays fixed in seconds (held by transport). In reg, 0 means the
    regime length does not matter, and +1 means Δ* follows T_reg.
- **R (transfer regret).** At each level, B2's plateau edge on the grid (in D1) must keep ≥ (1 − X) of that level's best
  re-planning gain. This is the practical test: is "re-plan every Δ*_B2 / p̄ × p̄" good enough along this axis?

**Decision rule (what runs next):**
1. C0 or C1 fails → stop, fix the harness, rerun the failed part. Nothing else starts.
2. proc and reg both invariant (S) and R passes in both → Δ / p̄ is the study's primary variable. Transport share and
   regime length leave the fit. **Next: axis 2** (load × due-date tightness) on the same grid.
3. reg moves Δ* → T_reg / p̄ enters the fit. Axis 3's remaining question (surge vs short-fleet blocks) folds into axis 2's
   design, run at two block lengths. **Next: axis 2.**
4. proc moves Δ* → transport share enters the fit. **Next: axis 6** (floor size, fleet ratio), which moves τ / p̄ by
   another route and tests whether Δ* follows the per-machine or the floor-level decision rate. Axis 2 follows.
5. Indeterminate in an arm → run that arm on seeds 20-39 before applying 2-4 (about 45 core-h per arm; measured 51-69,
   §12).
6. Δ* censored at a grid end at some level → add D1 60 s or 7,200 s there for the same seeds, then re-read.
7. A flat level is a result ("Δ does not matter here") and leaves the slope. If fewer than 3 levels of an arm are
   non-flat, S is not read for that arm and R decides alone.
8. S still indeterminate in an arm at 40 seeds (rule 5 spent) → the arm's ratio is "not shown to move Δ*". It enters
   §8's fit as a covariate with its own interval, not as a primary variable. For rules 2-4 the arm counts as invariant
   if R passes there, and as "moves Δ*" if R fails. (Added 10-29 by the user, §11.)
9. Best gain still at D1 60 s at some level after rule 6 → stop there; Δ* is reported as an upper bound at that level.
   No D1 30 s. (Added 10-29 by the user, §11.)

## 7. Later axes and what each needs

| axis (plan §5b) | levels (proposal, to settle per axis) | new code |
|---|---|---|
| 2 load × c | u ∈ {0.6, 0.8, 1.0} × c ∈ {1.5, 2.1, 3.0} (pinned per instance instead of drawn); design §15 (load ℓ × c centre at two block lengths, plus regime-mix cells), results §16 | generator parameters only |
| 3 regime | the reg arm of axis 1; surge vs short-fleet blocks separately if rule 3 fires (folded into axis 2's regime-mix cells, §15-§16) | generator parameters only |
| 4 disruptions | machine MTTF / p̄ × MTTR / p̄; AGV failures | **event-driven planner model:** the fork inherits the world's failure trace, so today the planner knows future failures (common random numbers in `branch.py`). It must replace it (`Twin.replace_failures(trace.redraw_after(k, seed))` or no failures) and set the down machine's repair end to an estimate; `redraw_after` keeps the true end of a repair in progress, which is a leak. Preempt-resume failures are in the snapshot (09f53cbb). |
| 5 forecast | arrival error, churn, duration CV / p̄ | durations through events: `../rq4-uncert-*`'s UncertaintyTrace with `belief_transform` (in-progress operations end at the estimate; the I1 estimator is synced in this snapshot). Arrivals: `../rq2-replan`'s Forecast. |
| 6 floor size / fleet | 15-200 machines, AGVs per machine; decisions per Δ (design §13, results §14) | the `../rq3-size-*` floors (built in `run.py` since 10-29); cluster (one full-window re-plan costs ~190 s at 200 machines, §13) |
| 7 planner | candidates (v5 pairs vs pairs × α), rollout horizon H, compute latency; design §19 | latency: the plan computed at t applies from t + T_c (the forks know it); under the exact forecast a shift of the re-plan lattice (§19, L0). `run7.py` adds latency, horizon and parallel forks |

- **Comparators (§5b):** periodic vs event-triggered vs hybrid, and a state-dependent trigger. `plan_triggered` keeps
  `../rq2-replan`'s trigger hook. They run on axes 4-5, where events carry information.
- **Survey item to verify before citing** (`docs/Plans/REPLAN_INTERVAL_SURVEY_2026-10-28.md`):
  - Xie, Zhang & Rose (WSC 2025) report that shorter was always better, down to about 2% of a mean operation time of
    1.8-16.8 h. If that holds, Δ* / p̄ is not a constant across floors.
  - The floor-level decision rate (axis 6) is the obvious candidate. It is one reason axis 6 may deserve to run early.

## 8. Fit (after the axes; design for axes 1, 2 and 6 in §17)

- Regress log(Δ* / p̄) on the log ratios of §2 over all non-flat levels, censored levels as intervals.
- Validate on held-out conditions by the regret of the predicted Δ against each condition's best Δ, the quantity a user
  of the equation pays.
- Report the plateau width beside it, since a wide plateau makes a rough equation good enough.

## 9. Dry run (10-29, `dryrun/`)

23 planner tasks and 6 fixed-pair tasks, all exit 0 (`dryrun/tasks.txt`, `dryrun/run_dry.sh`; analysis
`analyze.py --root dryrun` → `dryrun/analysis.out`, `dryrun/curve.csv`).

**Checks:**
- **C0 (units) PASS.** `all` at k 1/8 and k 32 (D1 900, seed 0) has B2's pair schedule, with tardiness / k within 1.8e-7
  and 2e-8 of B2's. k 10 differs: a different schedule, and tardiness / k 1.2% off on that seed. Float32 rounding of times
  that are not scaled exactly moves a tie; this is reported only.
- **C1 (regression) PASS.** proc k 1 at D1 300 and 900 (seed 0) equals `../rq2-replan` exactly (81.378671 and 79.971484;
  MDD-PTECT 95.842006), with the same schedule. So the new snapshot (09f53cbb) reproduces the old code (ba26f3f5) with
  failures and uncertainty off.
- **Plan-once.** The best fixed pair of seeds 0 and 1 equals `../rq2-replan`'s single-plan run (`exact-d300-ev2`)
  exactly (89.187018, 22.398315).
- **reg.** reg k 0.25 builds and runs: 16 blocks of 22.5 min in the window; on seed 0 the best fixed pair is MDD-PECT.

**Gain curve.** Re-planning gain over plan-once, %, pooled over seeds 0-1:

| D1 (s, B2-equivalent) | 120 | 180 | 300 | 450 | 600 | 900 | 1800 | 3600 |
|---|---|---|---|---|---|---|---|---|
| **proc k 4** (dry run; Δ = 4 D1) | +4.97 | +5.13 | +1.60 | +1.63 | +1.57 | +1.14 | +0.89 | +0.74 |
| [proc k 4, % better than MDD-PTECT] | +23.04 | +23.17 | +20.32 | +20.34 | +20.29 | +19.94 | +19.74 | +19.62 |
| B2, the same 2 seeds (`../rq2-replan` = proc k 1 by C1) | +7.16 | +7.16 | +7.16 | – | +5.85 | +8.29 | +7.54 | +7.54 |
| B2, 40 seeds (`../rq2-replan`) | +7.18 | +6.82 | +7.05 | – | +5.41 | +5.14 | +3.64 | +3.42 |

**Reading (a check of the harness, not a result: 2 seeds):**
- The harness produces the gain curve at a scaled level, with the plan-once baseline, Δ* and the descriptors
  (`dryrun/analysis.out`).
- Measured against MDD-PTECT, the k 4 curve is nearly flat at +20-23%, because plan-once alone is +19% there. Over
  plan-once, re-planning adds about 5% at the two finest intervals and 1-1.6% from 300 s up. On B2 it adds about 7% down
  to 300 s. This looked like the §5 effect, but 20 seeds do not bear it out: proc k 4 adds +7.0 at 120 and 300 s (§11).
- Two seeds cannot locate Δ*. B2's two-seed curve peaks at 900 s, its 40-seed curve does not. The Δ* values in
  `dryrun/analysis.out` (from 1-2 seeds and, at k 1, three intervals) are meaningless for that reason.

**Cost:**
- Box at load 20-30 (the dry run plus other sessions): 3.0-5.1 s per re-plan.
- A task costs its re-plans times that: 6 re-plans take about 30 s, 180 take about 11 min (`proc-k4-d120` s0: 675 s).
- A fixed-pair task takes 4-6 s.

## 10. Decisions for the user

**Decided 10-29 (user): every proposal as written below.** X 15%, plan-once baseline, axis 1 with the proc and reg arms
plus the units check, on the cluster, seeds 0-19 with the rule-5 extension. Item 6 stands as written. Started 10-29 18:06.
A login-node smoke test of `proc-k1-fixed` s0 on the cluster matched the local run exactly (89.187018).

1. **Plateau band X.** Proposed 15%: keep 85% of the best re-planning gain. It is the most stable on B2's pilot curve
   (§3 table). 10% and 20% are the alternatives; both are reported as sensitivity whatever is chosen.
2. **Gain baseline** (part of the Δ* definition; the sketch implied rq2-replan's MDD-PTECT). Proposed plan-once (§3),
   because MDD-PTECT's distance from the best pair changes with the condition. MDD-PTECT stays as a reported column.
3. **Which axis first.** Proposed: axis 1 as set up, with the proc and reg arms together plus the units check (1,650
   tasks, about 94 core-h). Without reg, a constant Δ* / p̄ in proc cannot tell p̄ from the regime length, since proc
   scales both. Alternatives:
   - proc alone, the user's example taken literally (930 tasks, about 52 core-h);
   - axis 6 (floor size) first, because the survey's WSC 2025 item hints that Δ* / p̄ falls on large floors. It is
     larger and needs the cluster.
4. **Local vs cluster.** Proposed: the cluster (free, 1-2 h wall). Local is about 7 h on the 14 free slots, and longer
   while other sessions load the box (it was at 20-30 during the dry run).
5. **Seeds.** Proposed: 20 (0-19) with the pre-declared extension to 40 when a reading is indeterminate (rule 5).
   At 20 seeds B2's Δ* has a 90% interval of 234-594 s, against 319-524 s at 40 (X 15%, `preview.out`). 40 from the
   start doubles the cost.
6. **Remaining §5b "to settle before any run" items, proposed as written above:**
   - p̄ = planned mean over the eligible machines (the alternatives are recorded);
   - exact forecast on axis 1 (forecast quality is axis 5);
   - slope margin ±0.25 per octave; flat threshold 1 point;
   - grid D1 120-3600 s with the extension rule.

**To start after the decisions:** `lab set rq2-replan-axes-time --waits-note ""` (cluster, as set up). For a local run,
first `lab set rq2-replan-axes-time --location local --workers 14 --launch 'bash results/rq2-replan-axes/launch.sh axis1
14' --pull ''`. Changing X or the baseline needs no rerun (`analyze.py` constants). Changing the seeds or the arms needs
`SEEDS` / `LEVELS` in `run.py`, and `--done-count` in the lab.

## 11. Axis 1 results, seeds 0-19 (10-29, lab `rq2-replan-axes-time`)

1,650 tasks on the cluster (Slurm 21978927 + 21978928, 18:06-19:45), all completed, 118 core-h. The analysis is
`analysis_s0-19.out`, `curve_s0-19.csv` and `dstar_s0-19.csv`: copies of the lab's 19:45 output, because the extension
rewrites `analysis.out`, `curve.csv` and `dstar.csv`. The header's "1830 task outputs" counts B2 (proc k 1) twice,
once per arm.

**Checks:**
- **C0 (units) PASS.** All 20 `all` tasks at k 1/8 and k 32 have B2's pair schedule, with tardiness / k within 5e-7
  relative. At k 10 all 10 tasks differ (another schedule; tardiness / k −3.4% to +7.9% per seed). That is reported
  only, as declared.
- **C1 (regression) PASS.** proc k 1 equals `../rq2-replan`'s `exact-d<D1>` on all 140 overlapping cells (7 intervals
  × 20 seeds).

**Gain curves.** Re-planning gain over plan-once (%, pooled over seeds 0-19, > 0 better), by B2-equivalent interval
D1 (s). Δ = D1 at k 1 and in reg, Δ = k D1 in proc.

| level | p̄ (s) | AGV busy | 120 | 180 | 300 | 450 | 600 | 900 | 1800 | 3600 | Δ* (s) | Δ* / p̄ [boot 90%] | plan-once vs MDD-PTECT |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| proc k 0.5 | 163 | 0.61 | +3.83 | **+3.87** | +3.56 | +3.19 | +2.97 | +2.49 | +2.36 | +2.09 | 202 | 1.24 [0.76-1.84] | +2.5% |
| **B2** (k 1) | 326 | 0.36 | **+7.46** | +7.23 | +5.98 | +5.83 | +5.01 | +5.28 | +3.85 | +3.83 | 259 | 0.80 [0.54-1.78] | +6.3% |
| proc k 2 | 652 | 0.20 | **+7.64** | +7.19 | +6.57 | +6.47 | +5.09 | +4.06 | +2.81 | +2.00 | 806 | 1.24 [0.50-1.51] | +7.4% |
| proc k 4 | 1,303 | 0.11 | **+7.04** | +6.38 | +7.03 | +5.03 | +5.15 | +3.12 | +1.57 | +1.20 | 1,484 | 1.14 [0.46-1.91] | +9.1% |
| proc k 16 | 5,213 | 0.03 | **+7.71** | +6.70 | +5.97 | +4.99 | +4.97 | +4.66 | +3.44 | +2.78 | 3,202 | 0.61 [0.46-1.13] | +7.9% |
| reg k 0.25 | 332 | 0.38 | **+10.26** | +8.78 | +7.66 | +6.22 | +7.37 | +6.86 | +6.29 | +3.51 | 184 | 0.56 [0.46-0.92] | +4.4% |
| reg k 0.5 | 331 | 0.37 | **+8.55** | +8.25 | +7.46 | +7.42 | +8.03 | +6.18 | +5.59 | +4.57 | 709 | 2.14 [0.68-2.36] | +4.7% |
| reg k 2 | 328 | 0.35 | **+7.19** | +6.73 | +5.90 | +5.37 | +5.49 | +4.91 | +3.86 | +3.03 | 263 | 0.80 [0.53-1.08] | +6.6% |
| reg k 4 | 331 | 0.35 | +7.75 | **+8.68** | +7.64 | +8.15 | +6.15 | +6.16 | +5.14 | +3.38 | 503 | 1.52 [0.73-1.61] | +2.9% |

Bold marks each level's best point. Transport per operation stays at 32-33 s throughout. Against MDD-PTECT
(`analysis_s0-19.out`, in brackets), every level gains 4.6-15.5%.

**Readings (pre-declared, §6):**
- **S, proc: INDETERMINATE.** The slope of log₂(Δ* / p̄) on log₂ k is −0.147 [90% −0.346, +0.125]. The interval
  crosses both −0.25 and 0. Rule 5 applies.
- **S, reg: INDETERMINATE.** The slope is +0.148 [−0.063, +0.319]. The interval crosses both 0 and +0.25. Rule 5
  applies.
- **R: PASS in both arms.** B2's plateau edge on the grid (D1 180 s) keeps at least 87% of the best re-planning gain at
  every proc level (worst k 16) and at least 86% at every reg level (worst k 0.25). The worst regrets are 13% and 14%,
  against X 15%.
- **No level is flat.** The best gains are 3.9-10.3 points.

**What 20 seeds already show.** These points are not readings; the pre-declared verdicts above stand.
- **Δ* follows the operation time, not a clock in seconds.** Across proc, p̄ grows 32-fold (163 → 5,213 s) and Δ*
  16-fold (202 → 3,202 s). The slope of log₂ Δ* in seconds is 1 − 0.147 = 0.85 [0.65, 1.13]. "Transport holds Δ* fixed
  in seconds" (slope −1 for Δ* / p̄) is far outside the interval. What 20 seeds cannot do is bound the rest within
  ±0.25 per octave.
- **The regime length moves Δ\* weakly at most.** "Δ* follows T_reg" (slope +1) is far outside [−0.06, +0.32]. The
  shortest blocks (22.5 min, reg k 0.25) give the largest gain (+10.3 at D1 120 s) and the narrowest plateau (Δ* 0.56 p̄,
  width 1.5×). reg k 0.5 has the widest plateau (Δ* 2.14 p̄) because of one high point at 600 s (+8.03).
- **Re-planning keeps its value when transport gets cheap.** The dry run's two-seed hint (proc k 4 adding only about
  1.6% from 5 min) was noise: on 20 seeds proc k 4 adds +7.0 at both D1 120 s and 300 s. The fixed pair that fits best
  does shift. Plan-once's lead over MDD-PTECT grows from +2.5% at k 0.5 to +9.1% at k 4. At k 16, ATC-PECT is the best
  fixed pair on 15 of 20 seeds. The transport-heaviest floor gains least: proc k 0.5 (AGVs 61% busy) gets +3.9.
- **Δ* / p̄ lies between 0.56 and 2.14 at every level** (median 1.14). B2's 40-seed value from `../rq2-replan` is
  1.26 p̄.

**R is fragile at 20 seeds.** R measures each level against B2's grid edge on the same seeds.
- On seeds 0-19 that edge is D1 180 s. On seeds 0-39 it is 300 s: `../rq2-replan`'s 40 seeds, which C1 reproduces
  exactly, put it at 300 s in 95% of the bootstrap resamples (`preview.out`).
- At D1 300 s the 20-seed curves keep 75% of the best gain at reg k 0.25, 77% at proc k 16 and 80% at B2 itself, and
  82-100% elsewhere. So if the 40-seed curves look like the 20-seed ones, R at 300 s fails at the two extreme levels.
- Neighbouring grid points carry about ±1 point of seed noise at 20 seeds. B2 at D1 300 s gives +5.98 on seeds 0-19 and
  +8.01 on seeds 20-39. That is the size of the band itself (X g(Δ_b) ≈ 1.1 points). R is read again on 40 seeds.

**How rule 6 applies here.**
- **No Δ\* is censored.** At every level the curve crosses the band between two grid points, so the right edge is
  interior and D1 7,200 s is not needed.
- **What sits at the grid's fine end is the plateau.** It reaches D1 120 s at all 10 levels. The best point is D1 120 s
  at 8 of them; at proc k 0.5 and reg k 4 it is D1 180 s, with 120 s inside the band.
- **Δ\* is measured from the band (1 − X) g(Δ_b).** A finer grid point can only leave g(Δ_b) as it is or raise it, and
  a higher band can only move the right edge left. So the grid's Δ* is an upper bound until the fine end is known.
  R's kept shares are upper bounds too, and R passes by only 1-2 percentage points. At reg k 0.25 a gain of +10.33 at D1
  60 s, 0.07 points above the D1 120 s value, would already fail it.
- **So rule 6 is applied.** Its purpose is a Δ* that does not depend on where the grid stops, and that applies to this
  case as much as to a censored edge. D1 60 s is added at every level on seeds 0-39. The analyzer now reports this case
  separately from a censored Δ*.
- **B2 shows what to expect.** `../rq2-replan` ran 60 s on all 40 seeds. On seeds 0-19 the 60 s point lies below 120 s
  (+7.02 against +7.46), so B2's best point and Δ* do not move. Over 40 seeds it is 0.1 point above (+7.28 against
  +7.18), which moves Δ* from 448 s to 431 s.

### Extension (lab `rq2-replan-axes-time-ext`, 10-29 19:52-23:17; results in §12)

- **Tasks.** The `axis1-ext` set in `run.py`, 1,980 tasks (Slurm 21980807 + 21980808):
  - rule 5: proc and reg on seeds 20-39, on the 8-point grid with their fixed-pair tasks (1,620 tasks);
  - rule 6: D1 60 s at the 9 proc and reg levels on seeds 0-39 (360 tasks; reg k 1 is proc k 1).
- **C1 grows with it.** proc k 1 on seeds 20-39 and at D1 60 s must equal `../rq2-replan` on 180 more cells, 320 in all.
- **Cost, measured on axis 1** (4.3-6 s per re-plan on the cluster):
  - rule 5 takes about 117 core-h: proc 65, reg 52, above the 45 per arm that §6 quoted;
  - rule 6 takes about 170 core-h: proc about 94, reg about 76. A D1 60 s task makes 360 re-plans, twice the costliest
    grid point (about 28 min per task, about 50 min at most, against a 3 h limit).
  - **So rule 6 costs about twice the 45 core-h per arm that rule 5 quotes**, and about 1.5 times the measured rule-5
    cost.
  - Total about 287 core-h, 3-4.5 h wall at axis 1's concurrency (72 tasks at a time on average, 112 at most); lab ETA
    10-30 00:30.
- **Outputs** go to the same `out/` (done count 3,630 = 1,650 + 1,980). On finish the lab pulls and runs `analyze.py`.
  It reads everything (seeds 0-39, D1 60-3600 s) and rewrites `analysis.out`, `curve.csv` and `dstar.csv`; the 20-seed
  record stays in `*_s0-19.*`. The new outputs carry a new `run_py_md5`: only `task_sets` changed, and `code/env` is
  unchanged.

### Decided before the 40-seed analysis is read (user, 10-29 20:24)

§6 did not cover two cases. To keep the reading pre-declared, each needed a rule before the extension finished. The
user adopted both proposals as written ("Adopt both README §11 proposals"), at 10-29 20:24. The extension was running
then (1,650 of 3,630 outputs pulled, all of them axis 1's), and no 40-seed analysis had been run. They are §6 rules 8
and 9.
- **a. S still indeterminate at 40 seeds.** Rule 5 is then spent. Rule (adopted):
  - the arm's ratio counts as "not shown to move Δ*". It enters §8's fit as a covariate with its own interval, not as
    a primary variable;
  - R decides the practical question. For rules 2-4 an indeterminate arm counts as invariant if R passes there, and as
    "moves Δ*" if R fails.
- **b. Best gain still at D1 60 s at some level.** Rule (adopted): stop there and report Δ* as an upper bound at that level.
  D1 30 s would make 720 re-plans per task (about 1 core-h each, about 340 core-h for all levels at 40 seeds).
  `../rq2-replan`'s B2 curve varies by less than 0.5 points between 1 and 3 min.

## 12. Axis 1 results, seeds 0-39 with D1 60 s (10-29, lab `rq2-replan-axes-time-ext`)

**Run.**
- The extension's 1,980 tasks (§11) ran on the cluster: Slurm 21980807 + 21980808, 10-29 19:52-23:17, 3 h 25 min wall.
  All completed.
- It cost 289 core-h:
  - rule 5, 120 core-h (proc 69, reg 51);
  - rule 6, 169 core-h (proc 95, reg 74; the longest D1 60 s task took 44 min).
- Axis 1 in all: 3,630 tasks, about 407 core-h.
- The analysis (`analysis.out`, `curve.csv`, `dstar.csv`) reads every output: seeds 0-39 on D1 60-3600 s.
  - Its header's "4030 task outputs" counts B2 twice.
  - Its "README §11, decision a / b" lines are §6 rules 8 and 9.

**Checks:**
- **C0 (units) PASS**, unchanged (the same 30 tasks).
- **C1 (regression) PASS** on all 320 overlapping cells: `../rq2-replan`'s 8 intervals, D1 60 s included, × 40 seeds.

**Gain curves.** Re-planning gain over plan-once (%, pooled over seeds 0-39, > 0 better), by B2-equivalent interval
D1 (s), as in §11.
- Bold marks each level's best point.
- ≤ marks a Δ* that is an upper bound by rule 9 (best gain at the finest point run).
- Δ* is in the run's own seconds (Δ = k D1 in proc).
- Transport per operation stays at 32-33 s.

| level | p̄ (s) | AGV busy | 60 | 120 | 180 | 300 | 450 | 600 | 900 | 1800 | 3600 | Δ* (s) | Δ* / p̄ [boot 90%] | plan-once vs MDD-PTECT |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| proc k 0.5 | 163 | 0.63 | **+4.90** | +4.47 | +4.32 | +4.06 | +3.87 | +3.90 | +3.39 | +2.73 | +2.45 | 122 | ≤ 0.75 [0.44-1.85] | +2.6% |
| **B2** (k 1) | 326 | 0.37 | **+7.28** | +7.18 | +6.82 | +7.05 | +6.01 | +5.41 | +5.14 | +3.64 | +3.42 | 420 | ≤ 1.29 [0.54-1.59] | +4.5% |
| proc k 2 | 652 | 0.20 | **+7.17** | +6.58 | +6.23 | +6.01 | +5.69 | +5.25 | +4.31 | +2.71 | +1.80 | 497 | ≤ 0.76 [0.42-1.42] | +5.9% |
| proc k 4 | 1,303 | 0.11 | **+7.84** | +7.34 | +6.83 | +7.00 | +6.28 | +6.11 | +5.01 | +3.35 | +2.30 | 1,450 | ≤ 1.11 [0.42-2.02] | +6.1% |
| proc k 16 | 5,214 | 0.03 | +7.56 | **+8.22** | +7.85 | +7.05 | +6.97 | +5.97 | +6.12 | +4.22 | +3.09 | 6,678 | 1.28 [0.56-1.66] | +5.8% |
| reg k 0.25 | 335 | 0.38 | +10.01 | **+10.04** | +8.81 | +7.70 | +6.92 | +7.03 | +7.12 | +5.79 | +3.63 | 204 | 0.61 [0.46-0.90] | +9.2% |
| reg k 0.5 | 334 | 0.36 | **+9.77** | +9.27 | +8.28 | +7.55 | +7.75 | +7.80 | +6.55 | +5.91 | +4.78 | 178 | ≤ 0.53 [0.42-1.03] | +4.6% |
| reg k 2 | 327 | 0.36 | **+7.29** | +7.17 | +7.17 | +5.76 | +5.10 | +5.20 | +4.89 | +4.22 | +3.53 | 257 | ≤ 0.78 [0.58-0.94] | +6.7% |
| reg k 4 | 332 | 0.36 | **+8.38** | +7.12 | +7.67 | +6.97 | +7.43 | +5.81 | +5.31 | +4.98 | +3.20 | 120 | ≤ 0.36 [0.28-1.59] | +4.8% |

Against MDD-PTECT (`analysis.out`, in brackets) every level gains 5.0-18.3%.

**Readings (pre-declared, §6, with rules 8 and 9):**
- **S, proc: INDETERMINATE.** The slope of log₂(Δ* / p̄) on log₂ k is +0.114 [90% −0.226, +0.282]. The upper end lies
  0.03 outside the ±0.25 margin.
- **S, reg: INDETERMINATE.** The slope is −0.096 [−0.247, +0.370].
- **R: FAIL in both arms.** B2's plateau edge on the grid is now D1 300 s (B2's Δ* 420 s = 1.29 p̄, itself an upper
  bound).
  - Kept shares in proc: 83% (k 0.5), 97% (B2), 84% (k 2), 89% (k 4), 86% (k 16).
  - Kept shares in reg: 77% (k 0.25), 77% (k 0.5), 79% (k 2), 83% (k 4).
  - 6 of the 8 levels besides B2 keep less than 85%.
- **No level is flat** (best gains 4.9-10.0 points), and no Δ* is right-censored.

**How the rules apply.**
- **Rule 9.** The best gain is at D1 60 s at proc k 0.5, 2 and 4, reg k 0.5, 2 and 4, and B2. The extension stops
  there, and Δ* at those levels is an upper bound (≤ in the table).
  - At proc k 16 and reg k 0.25 the curve turns down at 60 s, inside the band, so Δ* is measured there.
  - For R a finer point cuts both ways. It can only lower a level's own kept share, but by raising B2's band it could
    move B2's edge finer, which raises the kept shares.
- **Rule 8, S.** S is indeterminate in both arms with rule 5 spent. So transport share (τ / p̄, proc) and regime length
  (T_reg / p̄, reg) are each "not shown to move Δ*". Each enters §8's fit as a covariate with its own interval, not as a
  primary variable.
- **Rule 8, R.** R fails in both arms, so for rules 2-4 both arms count as "moves Δ*". Rule 2 does not apply.
- **Rule 3 (reg).** T_reg / p̄ enters the fit. Axis 3's remaining question (surge vs short-fleet blocks) folds into
  axis 2's design, run at two block lengths.
- **Rule 4 (proc).** τ / p̄ enters the fit. Next is axis 6 (floor size, fleet ratio), and axis 2 follows.
- **Together, as written:** axis 6 first, then axis 2 at two regime block lengths. Neither is designed or queued.
  The next axis needs the user's approval.

**Remark: how much of R's failure could be seed noise in its reference.** This is a check of the reading, not a reading;
the rules stand as decided (`r_noise.py` → `r_noise.out`). R divides the gain at B2's edge by the level's best grid
point. The maximum of 9 noisy points is biased upward, which lowers the kept share. Three references without that bias,
or with less of it, give the same failures:
- **Best point minus its bootstrap bias** (4,000 paired seed resamples). The bias is −0.01 to +0.24 points. Removing it
  raises the kept shares by at most 3 points (reg k 2: 79% → 82%), and no level crosses 85%.
- **Cross-fit.** Choose the best point on 20 seeds and measure it on the other 20 (2,000 splits, both ways). This
  reference carries no selection bias; it is low if anything. Worst kept share: 82% in proc (k 0.5), 78% in reg
  (k 0.25).
- **A fit that does not rise with Δ** (gain assumed not to grow with the interval). Worst kept share: 83% in proc,
  77% in reg.

How far each failing level is from passing: its reference would have to be lower by the amounts below. One grid point
carries a standard error of 0.56-1.10 points at 40 seeds, and the paired difference between D1 60 and 300 s carries
0.27-0.80.

| level | reference must drop by (points) |
|---|---|
| proc k 2 | 0.10 |
| proc k 0.5 | 0.12 |
| reg k 4 | 0.19 |
| reg k 2 | 0.51 |
| reg k 0.5 | 0.89 |
| reg k 0.25 | 0.99 |

- **reg's failure is not an artefact of the reference.**
  - At three levels it needs 2-8 times the estimated bias.
  - It fails on the 8-point grid without D1 60 s as well (77%, 81%, 80%).
  - A rerun on resampled seeds passes reg's R in 0-3% of resamples.
- **proc's failure is marginal.**
  - 0.10-0.12 points is about twice the estimated bias, and a fraction of one standard error.
  - In seed resamples k 0.5 and k 2 keep ≥ 85% in 30-43% of cases, and k 16 in 53-58%: proc sits on the threshold
    rather than below it.
  - It also comes entirely from the D1 60 s points that rule 6 added. On the 8-point grid k 0.5 and k 2 both keep
    91%, and proc's R passes.
  - So rule 4's "proc moves Δ*" rests on a shortfall of about 0.1 point at two levels.
- **B2's own edge is a larger noise lever than the reference.**
  - Across resamples it lands on D1 300 s in 65%, 450 s in 25% (where most kept shares fall further) and 120-180 s
    in 9%.
  - At 180 s, seeds 0-19's edge, proc would pass and reg would fail only at k 0.5 (84.7%).

**What 40 seeds show.** These points are not readings; the verdicts above stand.
- **Δ\* follows the operation time.** Across proc the slope of log₂ Δ* in seconds is 1 + 0.114 = 1.11 [0.77, 1.28].
  "Transport holds Δ* fixed in seconds" (slope 0) lies far outside, as at 20 seeds.
- **The regime length moves Δ\* weakly at most.** "Δ* follows T_reg" (+1) lies far outside [−0.25, +0.37].
- **The interval that transfers is finer than B2's edge.**
  - B2-equivalent D1 120 s keeps ≥ 85% at every level but reg k 4 (84.9%).
  - D1 180 s keeps ≥ 85% at every level but reg k 0.5 (84.7%).
  - D1 300 s loses 11-23% of the best gain at the 8 levels besides B2 (more than 15% at 6 of them), which is 0.8-2.3
    points of tardiness.
  - Δ* / p̄ lies at 0.36-1.29 (median 0.76), and 7 of the 9 values are upper bounds.
- **Short regime blocks gain most** (reg k 0.25 +10.0, reg k 0.5 +9.8). The transport-heaviest floor gains least
  (proc k 0.5 +4.9).
- **The per-level Δ\* did not settle with 40 seeds.**
  - From 20 to 40 seeds it moved by up to a factor of 4: reg k 4 1.52 → 0.36 p̄, reg k 0.5 2.14 → 0.53, proc k 16
    0.61 → 1.28.
  - Both slope intervals got wider, not narrower: proc 0.47 → 0.51, reg 0.38 → 0.62.
  - First cause: the D1 60 s points raised the band at 7 levels.
  - Second cause: the contiguous plateau's edge jumps a whole grid gap when a point near the band crosses it. At
    reg k 4, D1 120 s sits just below the band while 180 s and 450 s are inside it.
  - Rule 5 assumed that more seeds narrow S. Here they did not. §8's fit will need the bootstrap intervals, not the
    point values.

## 13. Axis 6 design: floor size and fleet ratio (drafted 10-29, approved by the user 10-30 as proposed; lab `rq2-replan-axes-size`, ran on the cluster 11-01 12:38 to 11-04 02:05; results in §14)

**Why now.** At axis 1's 40-seed reading rules 3 and 4 both fired (§12): axis 6 next, then axis 2 at two regime block
lengths. Axis 6 asks plan §5b's item 6: does Δ* follow the per-machine decision rate, or the floor-level one?
- **Per-machine rate:** Δ* / p̄ stays constant across sizes.
- **Floor-level rate:** the number of decisions per interval, Δ m u / p̄, stays constant, so Δ* ∝ p̄ / m.

The axis also moves the transport share τ / p̄ by two routes that proc did not use: longer trips, and scarcer AGVs.

**Arms.**

| arm | changes | levels | what moves (seed 0 under MDD-PTECT, unless noted) | readings |
|---|---|---|---|---|
| `size` | m machines (see below for the floor and fleet) | m 15 (B2; axis 1's 40 seeds reused), 20, 30, 50, 100 | τ / p̄ 0.09 → 0.10 → 0.12 → 0.16 → 0.21; jobs per window 302 → 1,402 | S_m, R, R120 |
| `fleet` | AGVs on duty per machine r, at m = 50 | r 0.35, 0.52 (= size m50), 0.80 | AGVs on duty 18 / 26 / 40 (short blocks 9 / 13 / 20); busy share on duty 0.82 / 0.62 / 0.42 (`../rq3-size-headroom`, 20 seeds) | R, R120 |

- **The size arm's floor and fleet:**
  - the floor-size grid's floor: m bays, m / 5 machines per type;
  - B2's load × 0.9, arrival cap off;
  - round(0.52 m) AGVs on duty, half that in short-fleet blocks.
- **Instances are the floor-size grid's own** (`../rq3-size-grid`). They are built as `../rq3-size-headroom` builds
  them, but with this folder's frozen code.
  - Within the fleet arm only the fleet changes, so the jobs are the same at every r.
  - Across sizes the instances are not paired.
- **Operations keep B2's clock.** p̄ is 349-369 s at every size (seed 0), so Δ = D1 and the grid in Δ / p̄ is axis 1's.
- **m15 is B2 itself:** a 7-bay floor, B2's load, 7 AGVs (0.47 per machine). It is built differently from the grid's
  floors, so S_m is also read over m20-m100 alone, as a sensitivity.
- **Unchanged from axis 1 (§1, §3):** planner v2 over the 6 v5 pairs, the exact forecast, X 15%, and the gain over
  plan-once.
- **Grid:** D1 60-3600 s, 9 points, with 60 s from the start (the user, 10-29). Rules 6 and 9 hold as in axis 1:
  7,200 s is added if a level is right-censored, and there is no 30 s (best at 60 s → Δ* is an upper bound there).

**Screen from runs that already exist** (no new compute).
- The greedy switching oracle at 900 s slots is the planner at D1 900 s with the exact forecast. On B2 seeds 0-4 they
  agree to the last digit (`../rq2-rebase-v5` oracle tails against `out/proc-k1-d900`).
- `../rq3-size-oracle` and `../rq3-size-headroom` ran that oracle on the grid's floors, on the same instances (dry run
  below) under frozen code 6d13412a.
- Below is the re-planning gain over plan-once at D1 900 s on 20 seeds, ± bootstrap SE where computed.

| floor | r 0.35 | r 0.52 | r 0.80 | r 0.52, MDD-TECT in the set |
|---|---|---|---|---|
| m15 (B2) | – | +5.3 ± 1.0 | – | – |
| m50 | +6.0 | +5.6 ± 1.2 | +6.2 | +8.2 |
| m100 | +0.6 | +2.5 ± 0.6 | +3.2 | +3.1 |
| m200 | +0.4 | +0.9 ± 0.2 | +1.4 | +0.5 |

What the screen means for the design:
- **Re-planning's value falls with size.** At 15 min it is 5-6 points up to 50 machines, 2.5 at 100 and 0.9 at 200. On
  B2 the 900 s point holds 71% of the best gain. If that held at m200, its best gain would sit at about the flat line of
  1 point.
- **Seed noise does not fall with size.** The SE is 19-24% of the gain at every size, so big floors need as many seeds
  as B2.
- **MDD-TECT mostly lowers plan-once.** It is the best fixed pair at 100-200 machines; it adds to what re-planning buys
  only at m50 (+2.6 points).
- **The fleet moves re-planning's value at 100 machines** (0.6 → 3.2), not at 50 (5.6-6.2).

**Dry run** (10-29, `dryrun6/`, seed 0, local, box at load about 20):
- **Instances.** MDD-PTECT equals the grid's own value exactly in all 10 conditions: m20, 30, 50, 75, 100, 150 and
  200, and fleet r 0.35 and 0.80. All 6 fixed pairs equal `../rq3-size-oracle` at m50 and m100.
- **Regression.** The edited `run.py` reproduces axis 1's `proc-k1-fixed` and `proc-k1-d3600` on seed 0 exactly.
- **Planner end to end.** D1 3600 s ran on m30, m50, m100 and fleet r 0.35 (6 re-plans each, outputs and descriptors
  written).
- **Cost.** One re-plan over the full window (c_W: 6 rollouts) costs 4.8-5.8 episodes.

  | | B2 | m20 | m30 | m50 | m100 | m200 | fleet r 0.35 | fleet r 0.80 |
  |---|---|---|---|---|---|---|---|---|
  | one episode (s) | 1.8 | 2.1 | 3.2 | 5.8 | 12.2 | 37.2 | 5.0 | 6.6 |
  | c_W (s) | 9.7 | ~12 | 17.6 | 30.2 | 58.6 | ~190 | 29.0 | ~36 |
  | full grid, one seed (h) | 1.18 | 1.4 | 2.1 | 3.7 | 7.1 | ~23 | 3.5 | 4.4 |

  - A level's 9 intervals cost about 434 c_W plus 24 episodes: n re-plans at interval Δ roll (n + 1) / 2 full windows.
  - On B2 this model gives 1.18 h per seed; the cluster measured 1.12-1.15 h. `../rq3-size-grid` saw the cluster up
    to 1.6 times slower per core at m100.
- **New descriptor `agv_busy_on_duty`:** busy AGV-seconds over on-duty AGV-seconds. The old `agv_busy_share` counts
  parked off-duty AGVs as idle. Every machine on a grid floor has a bay, so on those floors it reads about half the
  real share.
- **τ / p̄ by size** (seed 0): 0.09 (B2), 0.10 (m20), 0.12 (m30), 0.16 (m50), 0.19 (m75), 0.21 (m100), 0.28 (m150),
  0.32 (m200). At m50 the fleet barely changes τ (57-59 s): the fleet arm moves AGV scarcity, not trip length.

**Pre-declared criteria (approved 10-30).**

Checks:
- **C2 (instances).** MDD-PTECT at every (level, seed) equals the grid's own value within 1e-6, wherever one exists:
  - m50 and m100, seeds 0-39 (`../rq3-size-oracle`);
  - the fleet levels, seeds 0-19 (`../rq3-size-headroom`);
  - m20 and m30 (`../rq3-size-grid`).
  If C2 fails, stop (rule 1).
- **C1 carries over through m15:** B2's tasks are axis 1's and are not rerun.

Readings:
- **S_m (size).** The slope b of log₂(Δ* / p̄) on log₂ m over the non-flat size levels, with a 90% seed bootstrap.
  - **Per-machine rate:** the CI lies inside ±0.25. Then Δ / p̄ carries across sizes.
  - **Floor-level rate:** the CI lies inside −1 ± 0.25. Then the decisions per interval, Δ m u / p̄, carry.
  - **Moves Δ\*:** otherwise, if the CI excludes 0.
  - **Indeterminate:** otherwise.
  - Upper-bound levels (rule 9) enter at their bound. That pulls b toward 0, so "moves" and "floor-level" are
    conservative verdicts.
  - Sensitivity: the same slope over m20-m100, the grid's floors only.
- **What S_m can resolve.** From B2's Δ* (≤ 420 s), the two rates predict:

  | | m20 | m30 | m50 | m100 |
  |---|---|---|---|---|
  | floor-level rate | 315 s | 210 s | 126 s | 63 s |
  | per-machine rate | ~420 s | ~420 s | ~420 s | ~420 s |

  The per-machine prediction carries B2's 40-seed spread of 0.36-1.29 p̄ at every size. The grid's 60 s point reaches
  the floor-level prediction up to m100.
- **Attribution: decision rate or transport?** From m15 to m100 the size arm also raises τ / p̄ 2.4-fold (1.3
  octaves).
  - Axis 1's proc arm bounds τ / p̄'s own effect: its slope on log₂(τ / p̄) is −0.11 [−0.28, +0.23] (§12; the sign
    flips because proc lowers τ / p̄).
  - Over the size arm that allows at most ±0.13 per octave of m. A size slope beyond that is read as the decision
    rate, not transport.
  - Caveats: proc covered τ / p̄ 0.005-0.17 and the size arm 0.09-0.21. proc made operations slower against fixed
    transport, whereas size lengthens the trips.
- **R (size and fleet).** As in axis 1: B2's grid edge, D1 300 s, keeps ≥ 85% of each level's best gain.
- **R120 (new; size and fleet).** D1 120 s keeps ≥ 85% at every non-flat level.
  - Axis 1 found this after the fact (§12). Here it is pre-declared, on conditions it was not found on.
  - It is reported beside R and feeds §8's guidance, not the decision rule.
- **The fleet arm is read by R alone.** It has 3 levels over 1.2 octaves, so a ±0.25 margin per octave would need Δ*
  fixed within 1.23 times across the arm. S is reported, not read; R decides, as rule 8 would.
- **Flat levels** (best gain < 1 point) follow rule 7. If fewer than 3 size levels are non-flat, S_m is not read and
  R decides.
- **Compute, reported per level and interval:** the single-core time of one re-plan over the interval, T_c / Δ. At
  D1 60 s it is about 0.08 on B2, 0.25 at m50, 0.5 at m100 and 1.5 at m200, the real-time limit that axis 7 studies.

**Decision rule after axis 6 (approved 10-30).**
1. **C2 fails:** stop, fix, and rerun the failed part.
2. **S_m per-machine:** m leaves §8's fit, and the guidance stays in Δ / p̄ across sizes.
3. **S_m floor-level:** the fit's size variable becomes the decisions per interval, Δ m u / p̄. Axis 2 then reads load
   through that variable.
4. **S_m moves Δ\* otherwise:** log m enters the fit as a primary variable.
5. **S_m indeterminate:** as rule 8, log m enters the fit as a covariate, and R decides whether the guidance changes
   with size.
6. **The fleet arm's R:** if it fails, on-duty AGV utilization enters the fit as a covariate; if it passes, the fleet
   leaves the fit.
7. **In every case axis 2 runs next,** at two regime block lengths (rule 3), with the size variable as items 2-5 leave
   it.

**Seeds and cost** (core-h at local speed; the cluster matched it on B2 and was up to 1.6 times slower at m100):

| part | tasks per seed | core-h, 20 seeds | core-h, 40 seeds |
|---|---|---|---|
| size m20, m30, m50, m100 | 4 × (9 + 1) | 28 + 43 + 74 + 143 = 288 | 576 |
| fleet r 0.35, 0.80 at m50 | 2 × (9 + 1) | 71 + 88 = 159 | 318 |
| **total** | 60 (1,200 / 2,400 tasks) | **~450** | **~890** |
| m200 (option) | 9 + 1 | ~450 | ~900 |

- **The longest task** is D1 60 s at m100: about 3 h on one core, or 5 h on the cluster's slower cores. The 3 h limit
  in `submit_cluster.sh` would cut it off. Two ways around it:
  - raise the limit to 12 h for these arrays (proposed);
  - or run each re-plan's 6 forks on 6 cores (`fork_map` max_parallel 6), after checking that it equals the serial run.
- **Wall time** at axis 1's concurrency (70-110 tasks at a time): about 5-8 h at 20 seeds, 9-15 h at 40.

**Decisions (the user approved every proposal on 10-30: "Approve README §13 as proposed: v5 pairs, sizes
m15/20/30/50/100 without m200, fleet arm at m50, 40 seeds, the criteria and decision rule as written"):**
1. **Candidate set.**
   - Proposed: the 6 v5 pairs, as in axis 1. m15 is then reused, and one planner runs across all axes.
   - Alternative: v5 + MDD-TECT. It is the best fixed pair at 100-200 machines. It costs 17% more and m15 would need a
     rerun, but in the screen it adds re-planning gain only at m50.
2. **Sizes.** Proposed: m15 (reused), 20, 30, 50 and 100. That spans 2.7 octaves, with the advisor's realistic range
   (50-100 machines, 10-21) at the top. For m200:
   - (a) leave it out (proposed): the screen puts it near the flat line, and it would double the cost;
   - (b) a probe: 10 seeds at D1 60, 300 and 3600 s (about 115-185 core-h), with the full grid only if D1 60 s gains
     ≥ 1 point;
   - (c) the full grid: about 450 core-h at 20 seeds, 900 at 40.
3. **Floor of the fleet arm.**
   - Proposed: m50, with r 0.35 / 0.52 / 0.80. All three levels are non-flat in the screen; 159 core-h at 20 seeds.
   - Alternative: m100, where the fleet moves re-planning's value most (0.6 → 3.2 at 15 min). But r 0.35 is likely
     flat there, and it costs about twice as much (about 330 core-h). R decides the arm either way.
4. **Seeds.**
   - Proposed: 40 from the start, about 890 core-h. Axis 1 spent rule 5 in both arms, the extra 20 seeds did not
     narrow S (§12), and the screen's noise is no smaller on big floors.
   - Alternative: 20 seeds plus rule 5, as in axis 1. That is about 450 core-h, then about 450 more if S_m is
     indeterminate.
5. **Criteria:** the S_m bands (±0.25 around 0 and around −1), R and R120, the fleet arm read by R alone, and the
   decision rule above.

**Implementation (10-30), ready to run.**
- **Harness.** `run.py` builds both arms' instances (`AxEpisode` arms `size` and `fleet`; variants `size-m<M>-d<D1>`
  and `fleet-r<r>-d<D1>`) and records `agv_busy_on_duty`.
  - Task sets `size` (1,600 tasks), `fleet` (800) and `axis6` (2,400): 4 + 2 levels × (9 intervals + the fixed-pair
    task) × seeds 0-39.
  - `run.py list` now orders every set costliest first (`est_cost`: re-plans × the level's re-plan cost), so the long
    m100 tasks start first. `launch.sh` and `submit_cluster.sh` no longer re-sort.
- **Submit.** `submit_cluster.sh` gives the `size`, `fleet` and `axis6` arrays a 12 h limit (3 h otherwise) and no
  longer ships `dryrun6/`. The four floor exports on the cluster are byte-identical to the local ones (md5, 10-30).
- **Analysis.** `analyze.py` adds, once axis 6 outputs exist:
  - check C2 (every planner task's MDD-PTECT equals its fixed-pair task, and that equals the grid's own value);
  - the gain curves with τ / p̄ and the on-duty AGV share;
  - S_m (all levels, and m20-m100 as the sensitivity) with the attribution bound;
  - the fleet slope, reported only;
  - R, R120, the rule 9 upper bounds and T_c / Δ.
  The bootstrap resamples seeds jointly across levels, because one seed gives related instances at every size (the
  same regime block sequence).
- **Tests.**
  - Axis 1's `analysis.out`, `curve.csv` and `dstar.csv` come out byte-identical, since the axis 6 sections print
    only when axis 6 outputs exist.
  - A synthetic run (axis 1 levels relabelled as size and fleet levels, in the scratchpad) exercises every new
    path.
  - On `dryrun6/` C2 passes: m30, m50 and m100 seed 0 equal the grid, and the planner tasks equal their fixed-pair
    tasks.
- **Lab.** `rq2-replan-axes-size` is a cluster future: launch `bash results/rq2-replan-axes/submit_cluster.sh axis6`,
  axis 1's pull and `analyze.py`.
  - The done-glob `out/*-[mr][0-9]*/s*.json` counts only axis 6's outputs (2,400).
  - It waits on a waits-note. To start: `lab set rq2-replan-axes-size --waits-note ""`, then
    `lab requeue rq2-replan-axes-size` (a future without waits_on is not queued by clearing the note alone).
- **Expected:** about 890 core-h at local speed (up to 1.6 times that on slower cluster cores) and 9-15 h wall.

## 14. Axis 6 results (11-04, lab `rq2-replan-axes-size`)

**Run.** 2,400 tasks on the cluster from 11-01 12:38 to 11-04 02:05, with no failed task. 168 tasks (the costliest,
417 core-h) were done by 11-02 01:44. Then 58 of the 63 tier3 nodes drained for a security patch until about
11-03 21:00, which only delayed the run. The analysis is in `analysis.out` (axis 6 sections at the end), `curve.csv`
and `dstar.csv`. The checks beside the readings are in `axis6_checks.py` → `axis6_checks.out`. They are remarks; the
decision rule stands on `analysis.out`.

**Checks.**
- **C2 passes.** All 200 instances with a grid value equal the grid's own MDD-PTECT, and all 2,160 planner tasks equal
  their fixed-pair task: 0 mismatches.
- **C1 carries over through m15,** since axis 1's tasks are reused.
- **The screen is reproduced.** On seeds 0-19 the planner at D1 900 s matches the 10-29 screen within 0.05 points at
  all five levels the screen had: +5.28 / 5.3 (m15), +5.55 / 5.6 (m50), +2.48 / 2.5 (m100), +6.00 / 6.0 (r 0.35)
  and +6.17 / 6.2 (r 0.80).
  - Seeds 20-39 lower some of these points. r 0.35 is +3.85 on all 40 seeds.

**Gain curves** (gain over plan-once in points, 40 seeds; ≤ marks an upper bound, best gain at D1 60 s, rule 9):

| level | τ / p̄ | AGV busy on duty | jobs per window | plan-once vs MDD-PTECT | 60 | 120 | 180 | 300 | 450 | 600 | 900 | 1800 | 3600 | Δ* / p̄ [90%] | grid edge |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| m15 (B2) | 0.099 | ≈ 0.44 ¹ | 287 | +4.5% | +7.28 | +7.18 | +6.82 | +7.05 | +6.01 | +5.41 | +5.14 | +3.64 | +3.42 | ≤ 1.29 [0.55, 1.57] | 300 |
| m20 | 0.108 | 0.44 | 363 | +8.1% | +7.84 | +7.22 | +7.18 | +6.55 | +6.29 | +5.46 | +6.40 | +5.52 | +5.04 | ≤ 0.83 [0.50, 1.44] | 180 |
| m30 | 0.138 | 0.47 | 541 | +9.6% | +5.95 | +5.84 | +5.76 | +5.54 | +6.30 | +4.99 | +5.06 | +4.07 | +2.81 | 1.70 [0.83, 2.83] | 450 |
| m50 = r 0.52 | 0.180 | 0.58 | 905 | +4.6% | +10.19 | +8.56 | +8.58 | +8.39 | +8.04 | +6.97 | +5.97 | +5.51 | +4.15 | ≤ 0.35 [0.30, 1.46] | 60 |
| m100 | 0.242 | 0.69 | 1,805 | +0.7% | +4.47 | +4.55 | +3.93 | +3.44 | +2.92 | +2.98 | +2.59 | +1.97 | +1.77 | 0.58 [0.48, 0.75] | 180 |
| r 0.35 | 0.192 | 0.79 | 905 | +1.0% | +6.56 | +5.54 | +6.23 | +5.65 | +4.91 | +4.32 | +3.85 | +3.38 | +2.10 | ≤ 0.36 [0.29, 1.23] | 60 |
| r 0.80 | 0.173 | 0.38 | 905 | +9.1% | +7.23 | +6.66 | +6.60 | +6.42 | +5.24 | +5.10 | +5.19 | +3.54 | +3.73 | ≤ 1.01 [0.31, 1.38] | 300 |

¹ Axis 1 ran before `agv_busy_on_duty` existed, and the analyzer prints B2's share over all AGVs (0.37). B2's
short-fleet blocks park AGVs too: on seeds 0-7 (`run.py describe proc-k1 <seed>`) the on-duty share is 0.44 against
0.38 over all AGVs.

- **No level is flat.** Re-planning adds 4.6-10.2 points over plan-once at its best interval.
  - m100 has the smallest gain, as the screen predicted, but it is far from the 1-point line.
  - The screen's 2.5 points at m100 was mostly the interval: D1 900 s keeps 57% of m100's best gain.
- **Plan-once's lead over MDD-PTECT changes from 0.7% to 9.6%,** which is why the gain is measured over plan-once.
- **Where the best interval sits.**
  - 5 of the 7 levels have their best gain at D1 60 s, so Δ* is an upper bound there.
  - m30's best is a bump at 450 s; from 60 to 450 s its curve stays within 12% of that best.
  - m100's best is at 120 s.

**Reading S_m: Δ\* moves with floor size, at about half the floor-level rate.**
- **The slope** of log₂(Δ* / p̄) on log₂ m is −0.511 [90% −0.701, −0.164] over m15-m100. On the grid floors alone
  (m20-m100) it is −0.487 [−0.837, −0.140].
  - The CI excludes 0 and lies outside both bands (per-machine ±0.25, floor-level −1 ± 0.25), so Δ* moves.
  - By rule 4, log m enters §8's fit as a primary variable.
- **A slope near −0.5 means Δ* ∝ p̄ / √m.** Doubling the floor shortens the best interval by about 30%, where the
  floor-level rate would halve it.
  - The decisions per Δ* are not constant either: 31-108 across the levels (`axis6_checks.out` (4)).
- **Attribution.** τ / p̄ rises from 0.099 to 0.242, 0.49 octaves per octave of m. Axis 1's proc bound (|slope| ≤ 0.282
  per octave of τ / p̄) lets transport alone move the size slope by at most ±0.14.
  - The CI ends at −0.164, beyond that bound, so the slope is more than longer trips alone.
  - The margin is narrow (0.02), and at m100 the bound is extrapolated: τ / p̄ is 0.24 there, while proc reached 0.20.
- **What carries the slope** (`axis6_checks.out` (3)).
  - Leaving out any single size keeps the reading at "moves", except m100.
  - Without m100 the slope is −0.86 [−1.14, +0.42], which is indeterminate.
  - m100 is the one level whose Δ* is both well resolved ([0.48, 0.75] p̄) and not an upper bound.
- **The upper bounds cut both ways here.** §13 expected them on the large floors and argued that they pull b toward 0.
  They sit at m15, m20 and m50 instead.
  - Lower true values at m15 and m20 would flatten the slope; a lower true value at m50 would steepen it.
  - The sign of the slope is supported; its size is not pinned.
- **The point estimates are not monotone.** m30 (1.70 p̄) lies above m15 (1.29 p̄), because m30's flat curve sets its
  edge at the 450 s bump. The per-level CIs overlap widely.

**Reading R: B2's interval does not carry across sizes or fleets.**
- **Size arm: FAIL.** D1 300 s keeps 96.8 / 83.5 / 87.9 / 82.3 / 75.5% of the best gain at m15 / 20 / 30 / 50 / 100.
  - The failure at m100 holds under every reference: 76-78% bias-corrected, cross-fit and non-increasing.
  - 300 s keeps 85% at m100 in only 3% of resamples (`axis6_checks.out` (2)).
- **Fleet arm: FAIL,** with 86.1 / 82.3 / 88.7% at r 0.35 / 0.52 / 0.80. Rule 6 applies: on-duty AGV utilization
  enters §8's fit as a covariate.
  - The failing level is r 0.52, which is size m50. The arm's own two levels pass under every reference (86-92%).
  - The kept share shows no trend in r.
- **R120: FAIL, narrowly.** D1 120 s keeps 98.6 / 92.1 / 92.7 / 83.9 / 100% (size) and 84.4 / 83.9 / 92.2% (fleet).
  - At m50 the miss holds under every reference (83.5-84.7%). At r 0.35 the bias-corrected and cross-fit references
    pass (86-87%).
  - This was the pre-declared test of axis 1's post-hoc finding. "D1 120 s keeps 85%" does not carry as a rule:
    across all 18 levels of axes 1 and 6, 120 s keeps at least 83.9% and 180 s at least 84.2%.
- **The finest interval run, D1 60 s, keeps at least 92% of the best gain at every level of axes 1 and 6.**

**Fleet arm** (m50, the same jobs at every r).
- Δ* / p̄ is 0.36 / 0.35 / 1.01 at r 0.35 / 0.52 / 0.80; the first two are upper bounds at the 60 s grid edge.
- The slope on log₂ r is +1.27 [−0.60, +1.54], reported only.
- A scarce fleet does not cost re-planning its value at 50 machines (6.6-10.2 points), as in the screen.

**Compute.** T_c / Δ at D1 60 s is 0.08 (m15), 0.11, 0.19, 0.40 and 1.05 (m100); at m100 it is 0.51 at 120 s.
- At 100 machines one serial re-plan takes longer than a 60 s interval.
- The 6 forks of a re-plan are independent, so running them in parallel would cut the time up to 6-fold (not measured).
- Axis 7's latency question binds from about 100 machines.

**Post hoc (not a reading): AGV load separates the levels** (`axis6_checks.out` (4)).
- Over the 7 distinct size and fleet levels, the 3 with an on-duty AGV share of at least 0.58 (m50, m100, r 0.35) have
  Δ* ≤ 0.58 p̄. The 4 with at most 0.47 (m15, m20, m30, r 0.80) have Δ* ≥ 0.83 p̄. The correlation of the logs is −0.78.
- Within the fleet arm, where the jobs are the same, Δ* falls from 1.01 to 0.35 p̄ as the on-duty share rises from 0.38
  to 0.58.
- So floor size may act on Δ* partly through AGV load, which rule 6 already brings into the fit.
- Two cautions:
  - axis 1's reg arm moves Δ* over 0.36-1.29 p̄ at a nearly fixed share (about 0.37 over all AGVs);
  - proc k 0.5, at an all-AGV share of 0.63, has 0.75 p̄.

**How the rules apply** (§13 decision rule):
1. **Rule 1:** C2 passes, so the readings stand.
2. **Rules 2-3:** S_m is neither the per-machine rate nor the floor-level rate.
3. **Rule 4:** S_m moves Δ*, so log m enters §8's fit as a primary variable.
4. **Rule 6:** the fleet arm's R fails, so on-duty AGV utilization enters the fit as a covariate. The failure sits at
   the arm's shared m50 level.
5. **Rule 7:** axis 2 runs next, at two regime block lengths. It runs on B2's floor, and its interaction with size is
   left to the fit (§15).
6. **Rule 9:** Δ* is an upper bound at m15, m20, m50, r 0.35 and r 0.80.

**What axis 6 shows.**
- **Re-planning's value holds up to 100 machines at short intervals:** 4.6 points at m100. The screen's fall with size
  at 15 minutes came mostly from the interval being too coarse on large floors.
- **The best interval shortens with floor size,** Δ* ∝ m^−0.5 (CI −0.70 to −0.16). That is more than longer trips
  explain and less than the floor-level decision rate predicts. AGV load is a candidate mediator (post hoc).
- **No fixed interval in Δ / p̄ carries across axes 1 and 6.** B2's 300 s loses up to 25% at 100 machines and 120 s
  up to 16%. The finest interval run keeps at least 92% everywhere, so re-planning as often as the grid allows
  costs no level more than 8% of its best gain (cf. the survey item in §7).
- **At 100 machines the finest interval needs parallel forks** to run in real time.

## 15. Axis 2 design: load × due-date allowance at two regime block lengths (drafted 11-04, approved by the user 11-04 as proposed; lab `rq2-replan-axes-load`, ran on the cluster 11-08 14:00 to 18:28; results in §16)

**Why now.** Axis 6's rule 7 (§13) runs axis 2 next in every case, at two regime block lengths (rule 3 of §6). The
size variable is as axis 6 left it: log m is primary (rule 4), so axis 2 runs on B2's floor, and its interaction with
size is left to §8's fit. Plan §5b item 2 asks for load and due-date tightness, which the literature finds interact
with the interval.

**Cells.** B2's generator on B2's floor (15 machines, 7 AGVs), changed in three ways (`dryrun2/screen.py`):
- **Load ℓ.** Every target utilization (normal, surge, short-fleet and lull) is multiplied by ℓ.
  - The AGV arrival cap is kept, as B2's design has it, so the regime stays machine-bound.
  - In short-fleet blocks the cap binds, so ℓ moves those blocks little.
- **Due-date allowance c.** Each block draws its c from [c − 0.375, c + 0.375], B2's width, with c 1.5, 2.125 or 3.0.
  c 2.125 is B2's own [1.75, 2.5].
- **Regime block length:** 5,400 s (B2) and 1,350 s (axis 1's reg k 0.25).

| factor | levels | what moves (dry run, seeds 0-4) |
|---|---|---|
| ℓ | 0.7, 1.0, 1.3 | machine busy share 0.53 / 0.67-0.72 / 0.79; jobs per window 206-210 / 266-295 / 324-339 |
| c | 1.5, 2.125, 3.0 | plan-once tardiness at ℓ 1 and 5,400 s blocks: 881 / 449 / 125 (1000 job-s, 5 seeds) |
| block | 5,400 s, 1,350 s | 4-5 or 16 blocks per window |

- **A full 3 × 3 factorial at both block lengths:** 18 cells.
  - The two centre cells (ℓ 1, c 2.125) are axis 1's B2 and reg k 0.25, and are reused.
  - In the dry run both centres reproduce axis 1 exactly: the 6 fixed pairs and the planner at D1 900 s, on 5 seeds.
- **Regime mix** (axis 3's remaining question, which rule 3 folds in): 4 cells. At the centre, B2's mix is run without
  its surge blocks (`nosurge`) and without its short-fleet blocks (`noshort`), at both block lengths.
- **Pairing.** The jobs are paired across c: only the due dates change (70 of 70 groups in the dry run). They are not
  paired across ℓ, mixes or block lengths.
- **Unchanged from axes 1 and 6:** planner v2 over the 6 v5 pairs, the exact forecast, X 15%, the gain over plan-once,
  D1 60-3600 s (9 points), and rules 6 and 9.
- **The fit's load variable** is the measured machine busy share, not ℓ.

**Dry run** (11-04, `dryrun2/`, seeds 0-4, local, idle box): plan-once tardiness (1000 job-s, summed over the 5 seeds)
and the planner's gain over plan-once at D1 900 s (`dryrun2/summary.out`).

| ℓ, c | 5,400 s blocks: plan-once | gain at D1 900 s | 1,350 s blocks: plan-once | gain at D1 900 s |
|---|---|---|---|---|
| 0.7, 1.5 | 373 | +2.1 | 197 | +4.8 |
| 0.7, 2.125 | 155 | +3.9 | 53 | +5.7 |
| 0.7, 3.0 | 35 | +6.9 | 13 | +16.4 |
| 1.0, 1.5 | 881 | +3.4 | 1,079 | +4.2 |
| 1.0, 2.125 (B2; reg k 0.25) | 449 | +6.1 | 569 | +7.8 |
| 1.0, 3.0 | 125 | +2.9 | 153 | +21.9 |
| 1.3, 1.5 | 1,785 | +5.0 | 2,174 | +3.1 |
| 1.3, 2.125 | 1,116 | +4.0 | 1,440 | +4.5 |
| 1.3, 3.0 | 464 | +6.4 | 718 | +7.9 |
| 1.0, 2.125, `nosurge` | 271 | +2.9 | 349 | +10.6 |
| 1.0, 2.125, `noshort` | 507 | +4.8 | 698 | +3.1 |

- **No cell is flat at 900 s.** On 5 seeds the SE is about 40% of the gain, so the screen rules out flat cells and
  nothing more.
- **Low-tardiness corner.** At ℓ 0.7 and c 3.0 the window's tardiness is small: 13-35 thousand job-s over 5 seeds,
  against 449 on B2. A few late jobs carry the gain there, so its seed noise will be larger.
- **Loads above ℓ 1.3 add little.** The busy share is 0.74 at ℓ 1.15 and 0.79 at ℓ 1.3, because the arrival cap and
  saturation compress it. ℓ 0.85 and 1.15 were screened too (fixed pairs only, `summary.out`).
- **Cost.** One re-plan over the full window (c_W) takes 4.0-5.5 s at ℓ 0.7, 7.2-7.7 s at ℓ 1, 8.5-8.9 s at ℓ 1.3 and
  6.1-7.2 s for the mix cells. These are idle-box times: B2 is 7.3 s here against the 9.7 s of the 10-29 dry run on a
  loaded box.

**Readings (proposed).**
- **S cannot be read on these factors.** The measured u spans 0.58 octaves and c spans one, so the ±0.25-per-octave
  band would need Δ* fixed within 1.1-1.2 times across a factor. As in the fleet arm, R decides and the slopes are
  reported for the fit.
- **C3 (instances).** Every planner task's MDD-PTECT equals its cell's fixed-pair task within 1e-6, and the c levels
  of each (ℓ, block, seed) share one job set. If C3 fails, stop.
- **R_ℓ and R_c, per block length.** The centre cell's own grid edge must keep ≥ 85% of the best gain at the factor's
  other levels, along the line through the centre.
  - The edge is 300 s at 5,400 s blocks (B2) and 180 s at 1,350 s blocks (reg k 0.25, §12).
  - R_ℓ tests ℓ 0.7 and 1.3 at c 2.125; R_c tests c 1.5 and 3.0 at ℓ 1.
  - Why the centre's own edge, not B2's: at 1,350 s blocks B2's edge already fails at the centre (77%, §12), so it
    would test the block length again.
- **R_mix, per block length:** the same test at `nosurge` and `noshort`.
- **Reported beside them:**
  - R (B2's 300 s) and R120 at every cell;
  - Δ* with its bootstrap;
  - the slopes of log₂(Δ* / p̄) on log₂ u and on log₂ c per block length, with a joint seed bootstrap;
  - the corner cells, for §8's interaction terms;
  - T_c / Δ.
- **Flat cells** (best gain < 1 point) follow rule 7.

**Decision rule after axis 2 (proposed).**
1. **C3 fails:** stop, fix, and rerun the failed part.
2. **R_ℓ:** if it passes at both block lengths, load leaves §8's fit. If it fails at either, u (the measured machine
   busy share) enters the fit as a covariate.
3. **R_c:** the same for c, with the mean c as the variable.
4. **R_mix:** if it passes at both block lengths, the regime type leaves the fit (block length stays, rule 8). If it
   fails, the shares of surge and short-fleet time enter as covariates.
5. **Next, in every case: §8's fit** on axes 1, 2 and 6. It covers all non-flat levels, with upper bounds as
   intervals, and is validated by the leave-one-condition-out regret of the predicted Δ.
   - Axes 4 and 5 need the event-driven planner model (§7). They are designed after the fit shows what is left to
     explain.

**Seeds and cost.** Core-h are at cluster speed: B2's 9 intervals take 1.15 h per seed on the cluster, scaled by each
cell's c_W.

| part | new cells | core-h, 20 seeds | core-h, 40 seeds |
|---|---|---|---|
| ℓ × c at 5,400 s blocks | 8 | ~175 | ~350 |
| ℓ × c at 1,350 s blocks | 8 | ~175 | ~350 |
| regime mix | 4 | ~85 | ~170 |
| **total** | 20 (4,000 / 8,000 tasks) | **~440** | **~870** |
| lines through the centre only (no corners), both block lengths | 8 | ~180 | ~360 |

- **The longest task** is D1 60 s at ℓ 1.3: about 27 minutes on one idle local core, about 35 on the cluster. That is
  inside the 3 h limit.
- **Wall time** at axis 1's concurrency (70-110 tasks at a time): about 5 h at 20 seeds, 10 h at 40.

**Decisions (the user approved every proposal on 11-04: "Approve as proposed"):**
1. **Levels.**
   - Proposed: ℓ 0.7 / 1.0 / 1.3, and c centred at 1.5 / 2.125 / 3.0 with B2's width. This reuses B2 and reg k 0.25.
   - Alternative: c pinned to one value per instance (§7's sketch). It costs two more cells and loses the reuse.
2. **Second block length.**
   - Proposed: 1,350 s. B2's interval transfers worst there (77%, §12), and Δ* is resolved best ([0.46, 0.90] p̄).
   - Alternative: 21,600 s, one regime per window. u and c then become per-instance constants, but re-planning has no
     regime changes to follow.
3. **Factorial.**
   - Proposed: the full 3 × 3, since its corners give §8 the interactions the literature reports.
   - Alternative: the two lines through the centre, about 360 core-h less.
4. **Regime-mix cells.** Proposed: include them (+170 core-h). Alternative: leave axis 3's question open.
5. **Seeds.** Proposed: 40, as on axes 1 and 6 (about 870 core-h). Alternative: 20 (about 440).
6. **Criteria and decision rule** as above. In particular:
   - R against the centre's own edge;
   - §8's fit next, rather than axis 4 or 5.

**Implementation (11-04), ready to run.**
- **Harness.** `run.py` gets an `ax2` arm (`scenario_ax2`, the builder of `dryrun2/screen.py`). Its variants are
  `ax2-l<ℓ>-c<c>-b<block>[-<mix>]-d<D1>`, and `ax2-...-fixed` for the fixed-pair task, which also records `jobs_hash`.
  - Task set `axis2`: 20 cells × (9 intervals + the fixed-pair task) × seeds 0-39 = 8,000 tasks, costliest first (the
    dry run's c_W per cell). The two centres are not in it.
  - The `axis1`, `axis1-ext` and `axis6` lists hold the same tasks as before. `axis6`'s order is unchanged; axis 1's
    lists were launched before the costliest-first ordering existed.
- **Analysis.** `analyze.py` adds, once axis 2 outputs exist:
  - check C3 (below);
  - the gain curves with the machine busy share, the on-duty AGV share and the plan-once tardiness;
  - R_ℓ, R_c and R_mix per block length, with the decision lines of rules 2-4;
  - R and R120 at every cell, the rule 9 upper bounds and right-censoring, and T_c / Δ;
  - the reported slopes on log₂ u and log₂ c per block length: one regression over the block's 3 × 3 cells, with a
    joint seed bootstrap.
  - Axis 2 draws on its own bootstrap stream, so axes 1 and 6 keep theirs.
- **C3 as implemented.** It has three parts:
  - every planner task's MDD-PTECT equals its cell's fixed-pair task;
  - the c levels of each (ℓ, block, mix, seed) share one job set;
  - the tasks equal the local dry run (`dryrun2/out`: the 6 fixed pairs and D1 900 s on seeds 0-4, 200 tasks).
  The third part catches a cluster copy that builds other instances.
  - The cluster's copies of the B2 settings (`rq2-twin-fleet/run.py`) and the floor (`dev-popart-smoke/floor7`) are
    byte-identical to the local ones (md5, 11-04).
  - `rq2-oracle-steady/sim.py` and `rq2-twin-due/run.py` differ there only in lines `run.py` does not use: an unused
    `HEAD_PAIRS` list, and a floor path the twin planner does not read.
  - `submit_cluster.sh` ships `run.py`, `analyze.py` and `code/` at submit time, as for axes 1 and 6, and no longer
    ships `dryrun2/`.
- **Tests** (scratchpad roots, nothing under `out/`):
  - **Regression.** On the current `out/`, `analyze.py`'s `curve.csv` and `dstar.csv` come out byte-identical, and so
    does `analysis.out` apart from the root folder name in its first line.
  - **Builder.** `run.py`'s `ax2` tasks equal `dryrun2/` exactly on 4 cells × 2 seeds (fixed pairs, jobs hash, D1
    900 s). `run.py describe ax2-l1-c2.125-b5400` and `-b1350` equal axis 1's proc-k1 and reg-k0.25 on seed 0.
  - **Synthetic.** Axis 1 and 6 levels relabelled as the 20 cells exercise every new path. C3 flags two planted faults
    (one MDD-PTECT, one job set) and the relabelled data's mismatch with the dry run.
  - **Real dry check.** 4 cells × 2 seeds × (fixed + D1 900, 1800, 3600 s) give C3 PASS, with 16 tasks equal to
    `dryrun2/`.
- **Lab.** `rq2-replan-axes-load` is a cluster future: launch `bash results/rq2-replan-axes/submit_cluster.sh axis2`,
  axis 1's pull and `analyze.py`.
  - The done-glob `out/ax2-*/s*.json` counts only axis 2's outputs (8,000).
  - It waits on a waits-note. To start: `lab set rq2-replan-axes-load --waits-note ""`, then
    `lab requeue rq2-replan-axes-load`.
- **Expected:** about 870 core-h at cluster speed and about 10 h wall. The 3 h Slurm limit holds, since the longest
  task takes about 35 minutes.

## 16. Axis 2 results (11-08, lab `rq2-replan-axes-load`)

**Run.** 8,000 tasks on the cluster (Slurm 22040437-44) from 11-08 14:00 to 18:28, 4 h 28 min against the 10 h
expected. 7,991 tasks completed. The 9 that failed are all seed 5 of ℓ 0.7, c 3.0 at 1,350 s blocks, one per D1.
- On that seed every policy has zero tardiness: the planner, MDD-PTECT and the best fixed pair.
- Each of the 9 tasks wrote its output. Only the summary line printed after the write failed, because it divides by
  the zero MDD-PTECT tardiness (the `return` of `run.py`'s planner task).
- The analysis pools tardiness over seeds, so the seed counts as it is and nothing is rerun.
- The print is left unguarded for now. Every output stamps `run.py`'s md5, so the guard waits for the next harness
  change (axes 4-5's planner model, §7).

The analysis is in `analysis.out` (the axis 2 sections at the end), `curve.csv` and `dstar.csv`. The checks beside the
readings are in `axis2_checks.py` → `axis2_checks.out`; they are remarks, and the decision rule stands on
`analysis.out`.

**Checks.**
- **C3 passes.**
  - All 7,200 planner tasks equal their fixed-pair task's MDD-PTECT.
  - All 240 (ℓ, block, mix, seed) groups with two or more c levels share one job set.
  - All 200 tasks of the local dry run (`dryrun2/out`) are reproduced exactly.
- **C1 carries over to the two centres,** since they are axis 1's B2 and reg k 0.25.

**Gain curves** (gain over plan-once in points, 40 seeds; u = machine busy share; plan-once tardiness in 1000 job-s,
summed over the 40 seeds; ≤ marks an upper bound, best gain at D1 60 s, rule 9):

| cell | u | AGV busy on duty | plan-once tardiness | plan-once vs MDD-PTECT | 60 | 120 | 180 | 300 | 450 | 600 | 900 | 1800 | 3600 | Δ* / p̄ [90%] | grid edge |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **5,400 s blocks** | | | | | | | | | | | | | | | |
| ℓ 0.7, c 1.5 | 0.52 | 0.36 | 2,592 | +4.9% | +5.61 | +4.75 | +5.11 | +4.89 | +4.58 | +4.11 | +4.81 | +3.93 | +3.55 | ≤ 0.36 [0.27, 4.44] | 60 |
| ℓ 0.7, c 2.125 | 0.52 | 0.36 | 905 | +8.8% | +11.19 | +10.51 | +10.51 | +9.51 | +8.25 | +7.27 | +5.18 | +5.29 | +4.74 | ≤ 0.92 [0.61, 1.18] | 180 |
| ℓ 0.7, c 3.0 | 0.52 | 0.36 | 179 | +20.6% | +15.67 | +16.24 | +16.11 | +14.06 | +13.85 | +12.51 | +11.72 | +11.23 | +9.32 | 1.39 [0.72, 1.96] | 450 |
| ℓ 1, c 1.5 | 0.68 | 0.46 | 9,306 | +3.2% | +5.65 | +5.35 | +5.17 | +4.93 | +4.75 | +4.28 | +4.25 | +2.96 | +2.36 | ≤ 1.23 [0.50, 2.25] | 300 |
| ℓ 1, c 2.125 (B2) | 0.68 | (0.46) ¹ | 5,558 | +4.5% | +7.28 | +7.18 | +6.82 | +7.05 | +6.01 | +5.41 | +5.14 | +3.64 | +3.42 | ≤ 1.29 [0.74, 1.57] ² | 300 |
| ℓ 1, c 3.0 | 0.68 | 0.46 | 2,524 | +9.8% | +8.85 | +8.57 | +7.58 | +7.23 | +7.77 | +5.70 | +6.44 | +3.63 | +2.73 | ≤ 0.60 [0.37, 1.66] | 180 |
| ℓ 1.3, c 1.5 | 0.78 | 0.52 | 18,765 | +1.8% | +3.94 | +3.71 | +3.18 | +3.46 | +3.64 | +3.13 | +2.65 | +2.19 | +1.81 | ≤ 0.48 [0.39, 1.95] | 120 |
| ℓ 1.3, c 2.125 | 0.77 | 0.52 | 13,420 | +2.5% | +6.83 | +6.03 | +6.39 | +5.50 | +4.80 | +4.91 | +3.83 | +2.38 | +2.12 | ≤ 0.76 [0.30, 1.10] | 180 |
| ℓ 1.3, c 3.0 | 0.78 | 0.52 | 7,811 | +4.1% | +6.47 | +6.27 | +5.55 | +5.29 | +4.76 | +3.80 | +3.86 | +2.84 | +2.07 | ≤ 0.60 [0.34, 1.44] | 180 |
| `nosurge` | 0.64 | 0.47 | 4,667 | +4.6% | +8.14 | +7.37 | +6.71 | +6.78 | +5.37 | +5.66 | +4.39 | +3.73 | +2.55 | ≤ 0.49 [0.30, 1.22] | 120 |
| `noshort` | 0.71 | 0.41 | 4,826 | +8.5% | +6.99 | +6.22 | +5.91 | +4.65 | +5.09 | +4.84 | +3.97 | +2.97 | +1.82 | ≤ 0.53 [0.30, 0.69] | 120 |
| **1,350 s blocks** | | | | | | | | | | | | | | | |
| ℓ 0.7, c 1.5 | 0.52 | 0.35 | 1,678 | +8.7% | +6.77 | +6.60 | +6.95 | +6.91 | +6.32 | +5.71 | +5.54 | +4.73 | +3.56 | 1.66 [1.20, 2.83] | 450 |
| ℓ 0.7, c 2.125 | 0.52 | 0.35 | 387 | +24.0% | +8.62 | +8.46 | +8.55 | +7.79 | +6.78 | +7.04 | +6.34 | +4.57 | +3.16 | ≤ 1.10 [0.71, 2.89] | 300 |
| ℓ 0.7, c 3.0 | 0.52 | 0.35 | 50 | +39.9% | +18.25 | +18.15 | +18.78 | +19.47 | +18.37 | +17.60 | +16.91 | +15.05 | +11.38 | 3.12 [1.17, 5.69] | 900 |
| ℓ 1, c 1.5 | 0.71 | 0.46 | 7,781 | +5.1% | +5.64 | +4.97 | +4.90 | +4.78 | +3.86 | +3.79 | +3.46 | +2.54 | +2.58 | ≤ 0.84 [0.28, 1.11] | 180 |
| ℓ 1, c 2.125 (reg k 0.25) | 0.71 | (0.46) ¹ | 4,029 | +9.2% | +10.01 | +10.04 | +8.81 | +7.70 | +6.92 | +7.03 | +7.12 | +5.79 | +3.63 | 0.61 [0.46, 0.88] ² | 180 |
| ℓ 1, c 3.0 | 0.71 | 0.46 | 1,325 | +20.3% | +14.10 | +13.18 | +12.51 | +12.23 | +11.75 | +12.10 | +9.93 | +9.04 | +5.83 | ≤ 1.11 [0.32, 2.18] | 300 |
| ℓ 1.3, c 1.5 | 0.79 | 0.51 | 15,156 | +3.3% | +4.06 | +3.93 | +3.90 | +3.37 | +3.34 | +2.77 | +2.58 | +2.04 | +1.46 | ≤ 0.82 [0.42, 1.41] | 180 |
| ℓ 1.3, c 2.125 | 0.79 | 0.51 | 9,806 | +4.8% | +7.44 | +7.42 | +6.70 | +6.38 | +5.62 | +5.51 | +4.82 | +4.06 | +2.75 | ≤ 0.91 [0.48, 1.22] | 300 |
| ℓ 1.3, c 3.0 | 0.80 | 0.51 | 4,665 | +10.2% | +8.38 | +8.39 | +7.38 | +6.90 | +6.37 | +5.75 | +5.74 | +3.29 | +2.94 | 0.70 [0.47, 1.22] | 180 |
| `nosurge` | 0.66 | 0.46 | 1,993 | +13.6% | +13.50 | +13.70 | +13.22 | +12.53 | +10.54 | +10.96 | +9.57 | +8.38 | +5.76 | 1.09 [0.81, 1.26] | 300 |
| `noshort` | 0.72 | 0.41 | 3,336 | +12.1% | +6.53 | +6.23 | +6.69 | +6.03 | +5.65 | +5.37 | +6.03 | +4.34 | +3.07 | 1.30 [0.78, 3.81] | 300 |

¹ The centres ran on axis 1, before `agv_busy_on_duty` existed. Their c 1.5 and c 3.0 neighbours, with the same jobs,
have 0.46 (§14 measured 0.44 on B2's seeds 0-7).
² The centres' bootstrap here is axis 2's stream, so the intervals differ a little from §12 and §14 ([0.55, 1.57] for
B2, [0.46, 0.90] for reg k 0.25).

- **No cell is flat** (rule 7). Re-planning adds 3.9-19.5 points over plan-once at its best interval.
- **No cell is right-censored** (rule 6), so no D1 7200 s is needed.
- **Δ* is an upper bound at 15 of the 22 cells** (rule 9). The best gain sits elsewhere only at ℓ 0.7, c 3.0 (both
  block lengths), and at five 1,350 s cells: ℓ 0.7 c 1.5, reg k 0.25, ℓ 1.3 c 3.0 and the two mixes. Most of these
  bests are at 120 or 180 s.
- **The relative gain is largest where there is least tardiness to save.**
  - It mostly rises as c loosens and as load falls. At ℓ 1 and 1,350 s blocks it is 5.6 / 10.0 / 14.1 points at c 1.5 /
    2.125 / 3.0.
  - In absolute terms the order reverses. At its best interval the planner saves about 740 thousand job-s at ℓ 1.3,
    c 1.5 and 5,400 s blocks (3.9% of 18,765), about 405 on B2, and about 29 at ℓ 0.7, c 3.0 (the low-tardiness
    corner, also 5,400 s blocks).
  - Plan-once's lead over MDD-PTECT moves the same way, from 1.8% to 39.9%.

**Readings R_ℓ, R_c and R_mix: the centre's interval carries at 1,350 s blocks and not at 5,400 s.**

| block | line | the centre's grid edge keeps | reading | under other references ³ | line passes in resamples |
|---|---|---|---|---|---|
| 5,400 s (edge 300 s) | R_ℓ | 85.0% (ℓ 0.7), 80.4% (ℓ 1.3) | FAIL | ℓ 1.3: 80.4-82.6% | 17% |
| | R_c | 87.2% (c 1.5), 81.6% (c 3.0) | FAIL | c 3.0: 81.6-87.2% | 19% |
| | R_mix | 83.3% (`nosurge`), 66.4% (`noshort`) | FAIL | `noshort`: 66.3-67.3% | 0% |
| 1,350 s (edge 180 s) | R_ℓ | 99.2% (ℓ 0.7), 90.1% (ℓ 1.3) | PASS | ℓ 1.3: 90.1-92.0% | 76% |
| | R_c | 87.0% (c 1.5), 88.8% (c 3.0) | PASS | c 1.5: 87.0-90.3% | 47% |
| | R_mix | 96.5% (`nosurge`), 100% (`noshort`) | PASS | `nosurge`: 96.5-99.7% | 100% |

³ The worst cell's kept share against the bias-corrected best point, a cross-fit best point (chosen on 20 seeds,
measured on the other 20) and a non-increasing fit, as in §12 and §14 (`axis2_checks.out` (1)).

- **The misses at 5,400 s blocks.**
  - The ℓ 1.3 and `noshort` misses hold under every reference.
  - The c 3.0 miss depends on the reference: it passes against the bias-corrected (85.2%) and cross-fit (87.2%) best
    points.
  - `noshort` is the largest miss in axes 1, 2 and 6. B2's 300 s keeps 66.4% there, against 75.5% at m100 (§14).
- **The passes at 1,350 s blocks.** R_mix's pass is robust and R_ℓ's mostly so (76% of resamples). R_c's is not:
  c 1.5 sits at 87.0% and the line passes in only 47% of resamples.
- **Why the two block lengths differ: where the centre sits among its neighbours.** A reference interval carries to
  any cell whose plateau reaches at least as far.
  - At 5,400 s blocks B2's plateau is one of the longest of the nine 3 × 3 cells (1.29 p̄; only ℓ 0.7, c 3.0 has
    more). Its 300 s grid edge lies beyond the edges of six of the other eight.
  - At 1,350 s blocks reg k 0.25's plateau is the shortest of the nine (0.61 p̄), and every other cell's grid edge is
    at 180 s or beyond.
  - So the axis does not show that load and c matter more when regimes are long. It shows that an interval tuned on
    one condition transfers only as far as that condition's plateau happens to reach.
  - For comparison only: at 5,400 s blocks a 180 s reference would keep 85.6-93.9% along R_ℓ and R_c, but still only
    82.4% and 84.5% at the two mixes (`axis2_checks.out` (1)).

**Reported beside them.**
- **R (B2's 300 s): FAIL.** It keeps less than 85% at 10 of the 22 cells. The range is 66.4-100%; it is 76.7% at
  reg k 0.25, as in §12.
- **R120: FAIL at one cell,** ℓ 0.7 c 1.5 at 5,400 s blocks (84.7%). It keeps 88.2-100% at the other 21.
  - Across axes 1, 2 and 6, D1 120 s keeps at least 83.9% (m50, §14).
- **D1 60 s keeps 93.7-100% of the best gain at all 20 new cells** (`axis2_checks.out` (3)).
  - Across the 35 distinct conditions of axes 1, 2 and 6 it keeps at least 92.0% (lowest at proc k 16).
  - The best gain is at 60 s itself at 25 of the 35.
- **Slopes of log₂(Δ* / p̄)** over each block's 3 × 3 cells, with upper-bound cells at their bound. They are reported,
  not read, since u spans only 0.52-0.80.

  | block | on log₂ u | on log₂ c |
  |---|---|---|
  | 5,400 s | −0.43 [90% −2.71, +1.21] | +0.41 [−1.44, +1.05] |
  | 1,350 s | −1.97 [−3.16, −0.90] | +0.36 [−0.55, +1.30] |

  - **At 1,350 s blocks Δ\* shortens with machine load.** A slope of −2 over u 0.52-0.80 makes Δ* at ℓ 1.3 about 0.43
    times that at ℓ 0.7.
    - The CI excludes 0 with any one cell left out (upper ends −0.27 to −1.20).
    - Without the low-tardiness corner the slope is −1.21 [−2.98, −0.27] (`axis2_checks.out` (2)).
    - The upper bounds sit mostly at ℓ ≥ 1 (4 of 6 cells, against 1 of 3 at ℓ 0.7). Lower true values there would
      steepen the slope.
  - **At 5,400 s blocks there is no load trend** within the noise.
    - ℓ 0.7, c 1.5's Δ* (0.36 p̄) comes from one point: 120 s falls 0.01 points under the 15% band, while 180, 300
      and 900 s are inside it. Its CI reaches 4.44 p̄.
    - Without that cell the u slope is −1.62 [−2.75, +0.36].
  - **c shows no trend at either block length.**
- **The regime mix moves Δ\* in opposite directions at the two block lengths.**
  - At 5,400 s blocks, dropping either regime type shortens Δ*: ≤ 0.49 [0.30, 1.22] p̄ (`nosurge`) and ≤ 0.53
    [0.30, 0.69] p̄ (`noshort`), against B2's ≤ 1.29 [0.74, 1.57].
  - At 1,350 s blocks it lengthens it: 1.09 [0.81, 1.26] and 1.30 [0.78, 3.81] p̄, against 0.61 [0.46, 0.88].
  - The jobs are not paired across mixes.
  - `noshort` has no parked AGVs (on-duty share 0.41, equal to the all-AGV share). `nosurge` lowers u to 0.64-0.66.
- **The low-tardiness corner** (ℓ 0.7, c 3.0; `axis2_checks.out` (2)).
  - Plan-once tardiness is 179 and 50 thousand job-s over 40 seeds, against 5,558 on B2.
  - 5 seeds have zero tardiness under the best fixed pair, and the top 5 seeds carry 61-70% of the total.
  - Its relative gains are the largest of the axis (16.2 and 19.5 points), and its Δ* intervals the widest ([0.72, 1.96]
    and [1.17, 5.69] p̄).
- **Compute:** T_c / Δ at D1 60 s is 0.044-0.091 on every cell.

**How the rules apply** (§15 decision rule):
1. **Rule 1:** C3 passes, so the readings stand.
2. **Rule 2:** R_ℓ fails at 5,400 s blocks, so u (the measured machine busy share) enters §8's fit as a covariate.
3. **Rule 3:** R_c fails at 5,400 s blocks, so the mean c enters as a covariate.
   - The failure sits at one cell (c 3.0) and depends on the reference, and the c slopes straddle 0. The fit may find
     no c term.
4. **Rule 4:** R_mix fails at 5,400 s blocks, so the surge and short-fleet time shares enter as covariates. Block
   length stays (rule 8).
5. **Rule 5:** §8's fit on axes 1, 2 and 6 comes next.
   - 25 of its 35 distinct conditions have Δ* as an upper bound, so the fit is mostly a fit to intervals (§8).
   - Axes 4 and 5 wait for the fit.
6. **Rules 6, 7 and 9:** no cell is right-censored and none is flat; Δ* is an upper bound at 15 of the 22 cells.

**What axis 2 shows.**
- **Re-planning's value holds across load, due-date allowance and regime mix:** 3.9-19.5 points over plan-once at every
  cell. The relative gain is largest where there is least tardiness to save; the absolute saving is largest on the
  heaviest, tightest floor.
- **A tuned interval carries only as far as its plateau reaches.** B2's 300 s fails along all three lines at 5,400 s
  blocks (down to 66.4% without short-fleet blocks). Reg k 0.25's 180 s carries along all three at 1,350 s blocks,
  because its plateau is the shortest of its neighbours.
- **Δ\* shortens with machine load at 1,350 s blocks,** with a reported slope of −2.0 on log₂ u; there is no trend at
  5,400 s blocks. The due-date allowance shows no trend at either length. Removing a regime type moves Δ* in opposite
  directions at the two lengths.
- **Across all 35 distinct conditions of axes 1, 2 and 6, no grid interval longer than the finest one carries.** B2's
  300 s keeps as little as 66% and 120 s as little as 84%, while D1 60 s keeps at least 92% everywhere.
  - So "re-plan as often as compute allows" is the guidance that holds so far.
  - On 15 machines that costs at most 9% of one core per 60 s interval. At 100 machines it needs parallel forks (§14).

## 17. §8's fit on axes 1, 2 and 6: design (drafted 11-09, approved by the user 11-09 as proposed; lab `rq2-replan-axes-fit`, ran 11-09 17:44-17:48; results in §18)

**Why now.** §15 rule 5 runs §8's fit next, in every case. Axes 4 and 5 wait for it, because the fit says what is left
to explain. This section turns §8's three lines into a procedure with pre-declared readings. The only thing computed so
far is the fit's input table (`fit_inputs.py` → `fit_inputs.out`). It holds covariates and the Δ* values already in
`analysis.out`, so no reading has been looked at.

**Data: the 35 distinct conditions of axes 1, 2 and 6.**
- The aliases count once: B2, reg k 0.25 and m50 (= r 0.52). The `all` arm is a units check and is left out.
- None of the 35 is flat or right-censored, so all of them enter.
- Δ* is measured at 10: proc k 16, reg k 0.25, m30, m100 and six axis 2 cells (ℓ 0.7 c 3.0 at both block lengths;
  ℓ 0.7 c 1.5, ℓ 1.3 c 3.0, `nosurge` and `noshort` at 1,350 s). At the other 25 it is an upper bound (rule 9). The
  fit is therefore mostly a fit to intervals.
- The response is y = log₂(Δ* / p̄), with Δ* in the run's own seconds (Δ = k D1 in proc) and p̄ as §3 defines it.

**Covariates.** Each covariate is the mean over the condition's 40 seeds. "Moved at" counts the conditions that differ
from B2 by more than 10% (ratios) or 0.10 (shares); `fit_inputs.out` (3) has the full table.

| term | coding | brought in by | range over the 35 | moved at |
|---|---|---|---|---|
| floor size m | log₂(m / 15) | axis 6 rule 4 (primary variable) | 15-100 | 6 |
| τ / p̄ | log₂ | axis 1 rule 8 | 0.006-0.24 | 9 |
| T_reg / p̄ (regime block) | log₂ | axis 1 rule 8 | 4.0-65 | 14 |
| mean c of the window's blocks | log₂ | axis 2 rule 3 | 1.50-3.01 | 12 |
| on-duty AGV share | share | axis 6 rule 6 | 0.03-0.79 | 13 |
| machine busy share u | share | axis 2 rule 2 | 0.52-0.80 | 10 |
| surge time share | share | axis 2 rule 4 | 0-0.32 | 2 |
| short-fleet time share | share | axis 2 rule 4 | 0-0.32 | 2 |

- **No new runs are needed for the inputs.**
  - Axis 1 ran before the on-duty AGV share was recorded. It is rebuilt exactly from the recorded all-AGV share and the
    instance's AGV schedule (busy AGV-seconds over on-duty AGV-seconds). On all 1,040 fixed-pair tasks of axes 2 and 6,
    where both values exist, the rebuilt share matches the recorded one within 6.1e-7.
  - The surge share and the mean c come from the instance's regime blocks. No simulation is run.
- **Ratios in log₂, utilizations and time shares as they are** (proposed, decision 2).
  - Congestion grows with the share itself. A log scale would give proc's step from 3% to 6% AGV busy the weight of the
    step from 40% to 80%.
  - Over u 0.52-0.80 the two codings differ little. §16's slopes were on log₂ u.
- **τ / p̄ and the AGV share are partly collinear.** They correlate 0.82 as coded, and 0.94 with the share in log₂.
  - Together their variance inflation is 4.8-4.9; with either one alone, every covariate stays at 1.0-2.1.
  - They separate only through the fleet arm (AGV share 0.38-0.79 at τ / p̄ 0.17-0.19) and axis 2's load cells.
- **c is orthogonal to the rest** (variance inflation 1.0). It moves only along axis 2's c cells.
- **The mix shares move only at the 4 mix cells.** §16 found their effect reversing with block length.
  - Condition-out, a held-out mix cell is predicted mainly from its twin at the other block length.
  - These two terms may therefore lower M2's condition-out score. That is part of what the test measures.

**Models.** Each model is a censored normal regression (Tobit):
- y_i = x_iᵀβ + ε_i, with ε_i ~ N(0, σ²), fitted by maximum likelihood;
- an upper-bound condition contributes P(y_i ≤ its bound), a measured one its density;
- every condition counts once, unweighted (§8: "over all non-flat levels"). Seed noise enters through the bootstrap.

| model | terms | role |
|---|---|---|
| M0 | intercept | the constant-ratio rule: one Δ / p̄ everywhere |
| M1 | M0 + log₂(m / 15) | axis 6's primary variable alone |
| M2 | M1 + the seven covariates above | everything the decision rules brought in |
| M1 + one covariate (7 models) | | reported: which covariate carries what |
| M2 with τ / p̄ or the AGV share alone | | reported: the collinear pair |
| M3 | M2 + u × log₂ c + u × log₂(T_reg / p̄) | reported: the load × due-date interaction the literature reports, and §16's load × block contrast |

**From a model to a re-plan interval (the rule a user applies).**
- The recommended interval is Δ_rec = p̄ · 2^(xᵀβ̂ − h).
  - h ≥ 0 is a safety margin in octaves, set on the training conditions: the smallest h at which ≥ 90% of them keep
    ≥ 85% of their best gain at their own Δ_rec (decision 3).
  - For M0 the rule is a single Δ / p̄, tuned the same way.
- **Kept share at any Δ:** K_i(Δ) = g_i(Δ) / g_i(Δ_b), with the gain interpolated linearly in log Δ between grid points,
  as Δ*'s definition does (§3).
  - The kept share is read off the measured curve, so censoring does not enter the score.
  - Every condition's left edge is censored at 60 s (`dstar.csv`), so every Δ between D1 60 s and Δ* keeps ≥ 85%.
- **Clipping.** Δ_rec is clipped to the grid, D1 60-3600 s. A recommendation finer than 60 s is scored at 60 s, where
  every condition keeps ≥ 92%.
- **Compute share:** C_i = 60 s / Δ_rec,i in B2-equivalent D1. This is the planner's compute relative to re-planning
  every 60 s at that condition: the re-plans scale as 1 / Δ, and each rolls to the window end.

**Validation: condition-out (no new runs).**
- For each condition i: fit on the other 34, set h on those 34, predict Δ_rec,i, and score K_i(Δ_rec,i) on i's own curve.
- Over the 35 held-out predictions, each rule gets:
  - a **coverage**, the number of conditions keeping ≥ 85%;
  - a **compute**, C = the mean of C_i (1 for "always 60 s"; C ≤ 0.5 is a harmonic-mean interval of at least 120 s).
- This is §8's "regret of the predicted Δ against each condition's best Δ", put in the two numbers a user pays: how
  often the rule loses more than 15% of the gain, and what it costs to run.
- §8's raw regret alone would favour "always 60 s", which keeps ≥ 92% everywhere. The compute share is what an equation
  can save.
- **Reference points** from `fit_inputs.out` (4), untuned and in-sample. A constant D1 keeps ≥ 85% at:

  | constant D1 | keeps ≥ 85% at | compute C |
  |---|---|---|
  | 60 s | 35 of 35 | 1 |
  | 120 s | 31 | 0.50 |
  | 180 s | 30 | 0.33 |
  | 300 s | 17 | 0.20 |

**Seeds and bootstrap.**
- Point estimates use all 40 seeds of every condition. The 9 failed tasks of §16 wrote their outputs and count.
- **The bootstrap is joint across the 35 conditions.** One resample of the 40 seed indices serves every condition,
  because seeds link the instances: proc pairs exactly with B2, axis 2 pairs across c, and axis 6's sizes share the
  regime block sequence.
- Each resample recomputes, for every condition:
  - the curve, Δ* and the rule 9 flag;
  - the kept shares and the covariate means.
  It then refits every model and reruns the condition-out validation.
- **Rules 6 and 7 inside a resample.** A condition that comes out flat leaves that resample's fit (rule 7). One that
  comes out right-censored enters as right-censored (rule 6); none is at the point estimate.
- **Resamples:** 2,000 for M0, M1 and M2 (the read models), on a new stream; 200 for the reported variants.
- **Cost:** about 1-3 core-h of analysis, no simulation. It would run as a local lab entry.

**Pre-declared readings.**
- **F1 (the size law carries condition-out).** PASS if both hold:
  - M1's rule needs less compute than M0's, with the 90% bootstrap interval of C(M1) − C(M0) below 0;
  - M1's rule keeps ≥ 85% at ≥ 30 of the 35 held-out conditions.
- **F2 (the covariates carry condition-out).** The same test for M2 against M1, or against M0 if F1 fails.
- **F3 (coarser than the finest interval is safe).** The selected rule must keep ≥ 85% at ≥ 30 of 35 held-out
  conditions with C ≤ 0.5, half the compute of re-planning every 60 s.
  - The selected rule is the last model to pass F1-F2, or M0 if neither passes.
- **Why these thresholds.** The margin targets 90% (31 of 34 in training). 30 of 35 held out allows for the optimism of
  tuning on the other 34. C ≤ 0.5 is the compute of the 120 s rule, the coarsest grid interval that keeps 85% at 31 of
  35 in sample.

**Reported beside them (not readings).**
- **Coefficients** of M1 and M2 with 90% intervals, set against the axis readings: the size slope −0.51 (§14) and the
  u slope −2.0 at 1,350 s blocks (§16).
- **σ̂,** the residual scatter in octaves: what an equation leaves unexplained, and axes 4-5's starting point.
- **Condition-out coverage and compute** of every model rule, including the singles, the collinear pair and M3.
- **What an overshoot costs:** the median kept share at 2 Δ*. It turns σ̂ into regret.
  - The plateau width (§8) is reported as its lower bound, Δ* / 60 s, since every left edge is censored.
- **Real-time feasibility:** Δ_rec against the serial re-plan time T_c at each condition. T_c / Δ at 60 s is ≤ 0.09 on
  15 machines and 1.05 at m100 (§14).
- **Sensitivities:**
  - X 10% and 20%, with Δ* re-derived from the same curves;
  - the upper bounds treated as exact (as §14's S_m did);
  - p̄ as the fastest-machine mean.

**What each outcome means.**

| outcome | paper C states | axes 4-5 |
|---|---|---|
| F1 or F2 PASS, F3 PASS | an equation Δ_rec(p̄, m, ...) with its margin, validated condition-out. The held-out check below comes before claiming it across floors | read against the equation: does its interval still keep 85% under failures and forecast error? MTTF / p̄, MTTR / p̄ and error / p̄ enter a refit |
| F1 and F2 FAIL, F3 PASS | one fraction of p̄ carries across the 35 conditions. Size, load and regimes move Δ*, but not enough to predict it condition-out beyond the margin | read against the constant-ratio rule |
| F3 FAIL | no interval coarser than half the finest carries condition-out. The guidance is "re-plan as often as compute allows", and compute binds from about 100 machines (§14) | axis 7's compute latency becomes the sharper question; axes 4-5 test only direction (do events favour coarser or finer re-planning?) |

**Held-out check on combined conditions (new runs; decision 5).**
- Condition-out validation predicts along the axes, since the 35 conditions vary one factor at a time from B2 plus
  axis 2's factorial. It cannot test combinations such as a large floor at high load with short regime blocks.
- A held-out check would run new conditions that combine factors, drawn as a space-filling design inside the fitted
  ranges after the fit is frozen. Each would be scored by the same coverage and compute.
- **It needs a combined builder in `run.py`** (m, fleet, ℓ, c, block, mix and proc k in one arm). Each single factor
  must reproduce its existing cell (checks like C2 and C3). The new `run.py` md5 would also carry §16's guarded print.
- **Cost, at 40 seeds and 9 intervals:**

  | floor | core-h per condition |
  |---|---|
  | B2's floor | about 46-55 |
  | m30 | about 85 |
  | m50 | about 150 |
  | m100 | about 300-450 (cluster cores up to 1.6 times slower, §13) |

  - An 8-condition set without m100 (4 on B2's floor, 2 at m30, 2 at m50) is about 650-700 core-h, roughly 5-10 h
    wall on the cluster.
  - The longest task, D1 60 s at m50, takes about 1.5 h on one core (up to about 2.5 h on slower cluster cores),
    inside the 3 h limit. m100 conditions would need §13's 12 h limit.

**Decisions (the user approved every proposal on 11-09: "Approve as proposed"; each proposal first):**
1. **Response and censoring.**
   - Proposed: Δ* / p̄ as §3 defines it, with upper bounds as left-censored in the likelihood.
   - Alternatives: treat the bounds as exact, which biases the fit toward longer, unsafe intervals; or fit only the 10
     measured conditions, too few for eight covariates.
2. **Model terms and coding.**
   - Proposed: M0, M1 and M2 read, with all eight covariates in M2 as the decision rules require. Ratios in log₂, shares
     as they are. The singles, the collinear pair and M3 are reported.
   - Alternative: a lean M2 with one transport term (the AGV share) and without the two mix shares, the 4 mix cells
     then checked as residuals. That is fewer parameters for 10 measured points, but leaves out terms rules 4 and 8
     brought in.
3. **Safety target.**
   - Proposed: 90% (the margin keeps 85% of the gain at 9 training conditions in 10), with ≥ 30 of 35 held out as the
     pass line.
   - Alternatives: 80% (coarser intervals, more misses) or 95%.
4. **Pass lines F1-F3 as written:** the compute difference with its bootstrap interval below 0, ≥ 30 of 35, C ≤ 0.5.
5. **Held-out check.**
   - Proposed: decide after the fit. Run it if F1 or F2 passes; if only F3 passes, a smaller set on B2's floor (about
     200 core-h) can test the constant rule.
   - Alternatives: design it now and run it beside the fit (about 650-700 core-h), though the conditions could then not
     target the fit's predictions; or skip it, and the paper's claim rests on condition-out validation alone.
6. **Where the fit runs.** Proposed: a local lab entry `rq2-replan-axes-fit` (analysis only, about 1-3 core-h).

**Implementation (11-09; the lab ran it unchanged on 11-09, results in §18).**
- **`fit.py`.** Its steps are `table`, `tasks`, `run <task>`, `report` and `test`.
  - `table` writes `fit_out/table.pkl`: per-seed arrays of the 35 conditions, with `fit_inputs.py`'s covariates (that
    script now has its helpers importable; its output is byte-identical).
  - The 23 tasks:
    - `point`: all 40 seeds, every model in detail, the reported quantities and the sensitivities;
    - `read-00` to `read-19`: M0-M2, 100 resamples each;
    - `var-00`, `var-01`: the variants, 100 resamples each.
    Each task writes `fit_out/<task>.json` with the md5s of `fit.py` and the table, and is skipped if its output
    exists.
  - `report` checks that every output comes from one `fit.py` and one table, then writes `fit.out` and `fit_loco.csv`
    and prints F1-F3.
  - Tobit by maximum likelihood with scipy 1.15 (L-BFGS-B, analytic gradient, standardized columns).
  - The resamples are `default_rng(1709)`'s 2,000 rows of 40 seed indices; the variants use the first 200.
- **Numerical guards (not in the design; counted in `fit.out`).**
  - σ ≥ 0.01 octave, and |coefficient| ≤ 50 octaves per SD of its column. They keep the likelihood bounded where the
    measured conditions alone would fix the line.
  - A fit is counted as converged if L-BFGS-B succeeds, or, after one restart, if its projected gradient is below 1e-4.
- **`launch_fit.sh [workers]`** builds the table, runs the unfinished tasks on `workers` processes (one BLAS thread
  each), and exits non-zero while any task is unfinished. A rerun resumes.
- **Tests (`fit.py test`, synthetic data only, all pass).**
  - With nothing censored the fit equals OLS (8.9e-16).
  - The analytic gradient matches finite differences (2.3e-8).
  - Known coefficients and σ are recovered within 0.025 under 68% upper bounds and under 34% lower bounds; treating the
    bounds as exact biases the intercept by +0.38.
  - The kept share equals g / max g at the grid points, and y uses Δ = k D1 in proc-like rows.
  - The margin reaches the target and is the smallest on its grid.
  - End to end, a synthetic 35-condition table with a size slope of −0.5 gives M1 −0.52, and the report prints F1-F3.
  - A finished task is skipped.
- **Launcher test.** On a synthetic table in a scratch folder (`FIT_OUT`), it wrote all 23 outputs in 75 s on 8
  workers (8.3 CPU-min), and the rerun skipped everything in 0.8 s.
- **Real-data smoke** (table in a scratch folder; only times and fit diagnostics printed):
  - the table builds in 103 s;
  - one resample takes 0.2 s for M0-M2 and 0.6 s for the variants;
  - 0 of 576 fits failed to converge or reached a guard.
- **Lab fields for `rq2-replan-axes-fit` (local):**
  - launch `bash results/rq2-replan-axes/launch_fit.sh 4`;
  - done when `results/rq2-replan-axes/fit_out/*.json` counts 23 (or the launcher exits 0);
  - 4 slots;
  - analysis `.venv/bin/python results/rq2-replan-axes/fit.py report` (writes `fit.out` and `fit_loco.csv`, prints
    F1-F3);
  - about 11 CPU-min, about 5 min wall on 4 slots.

## 18. §8's fit results (11-09, lab `rq2-replan-axes-fit`)

**Run.** All 23 task outputs were written on 11-09, 17:44-17:48 (4 min 31 s, launcher exit 0).
- They share one `fit.py` (md5 75feba5c, the current file) and one table. The table was built from the 11-08 18:28
  `analysis.out` (md5 6e8f8a5c, unchanged).
- **Fit diagnostics.** At the point estimate: 468 fits, none unconverged or at a guard. In the bootstrap: 288,000 fits,
  none unconverged; σ sat at its floor 3 times and a coefficient at its bound 62 times (0.02%).
- The report is `fit.out`, and the held-out predictions are in `fit_loco.csv`. The remarks beside the readings are in
  `fit_checks.py` → `fit_checks.out`; the verdicts stand on `fit.out`.

**The three read models.** Each is fitted on all 35 conditions. Held out = condition-out (each condition predicted from
the other 34); C = the mean of 60 s / Δ_rec.

| model | σ (octaves) [90%] | coefficients [90%] | held out: keeps ≥ 85% | compute C [90%] |
|---|---|---|---|---|
| M0 | 1.49 [1.02, 1.84] | intercept −1.47 [−1.78, −0.77]: Δ_rec = 0.36 p̄ | 34 of 35 | 0.506 [0.373, 0.634] |
| M1 | 1.48 [0.99, 1.80] | log₂(m / 15) −0.09 [−1.12, +0.35] | 34 of 35 | 0.508 [0.382, 0.634] |
| M2 | 0.72 [0.71, 1.35] | all eight 90% intervals include 0 | 31 of 35 | 0.576 [0.444, 0.666] |

- **The margin h is 0 in every fold of every read model.** With 25 of the 35 values as upper bounds, the censored fit
  puts its line well below the measured values. The line alone already keeps 85% at ≥ 90% of the training conditions.
- **M0 is one constant, about D1 120 s.** Its held-out intervals run 107-129 s, moving with the condition left out.
  - At three conditions where a fixed 120 s missed narrowly (r 0.35, reg k 4, ℓ 0.7 c 1.5 at 5,400 s blocks), M0 lands
    at 127-129 s.
  - Their curves dip at the 120 s grid point and rise toward 180 s, so they keep 85.7-86.2% there.
  - The coverage count moves with grid-point noise of this size: 34 held out here, against 31 for a fixed 120 s in
    sample.

**Readings (pre-declared, §17).**
- **F1 FAIL.** M1 needs no less compute than M0: C(M1) − C(M0) is +0.002 [90% −0.082, +0.066], at 34 of 35. M1 beats
  M0 at the line in 40% of resamples.
- **F2 FAIL** (against M0, since F1 failed). C(M2) − C(M0) is +0.070 [−0.066, +0.179], and M2 keeps 85% at only 31 of
  35. It beats M0 at the line in 8% of resamples.
- **F3 FAIL** (on M0, the selected rule). Its coverage, 34 of 35, passes; its C of 0.506 misses the 0.5 line. Both
  hold together in 54% of resamples.

**How F3's miss should be stated.**
- The verdict stands as pre-declared: FAIL.
- **The miss is small against the noise** (`fit_checks.out` (1)).
  - M0's harmonic-mean interval is 118.6 s, 1.1% short of the 120 s that C 0.5 stands for.
  - Over the 2,000 resamples C(M0) runs 0.37-0.63 (median 0.480) and is ≤ 0.5 in 54%. Its coverage holds at the line
    in 95%.
- **So F3 sits on its line.** The data show neither that a constant rule safely halves the compute of re-planning
  every 60 s, nor that it does not.
- **What is shown:** a constant 0.36 p̄, about twice the finest interval run, keeps 85% at 34 of 35 held-out
  conditions. The finest interval keeps ≥ 92% at all 35.
- **In the paper:** state F3 as failed at its line, with C and its interval. Do not state it as "coarser intervals
  are unsafe".

**Where the models miss** (`fit_checks.out` (2)).
- **M0 and M1 miss only m50.** There D1 128 s keeps 84.0%, and M1's 136 s also 84.0%. It is the condition where 120 s
  missed in sample too (83.9%, §14).
- **M2 misses where it extrapolates long.** Its held-out intervals spread over D1 60-958 s, with 9 clipped at 60 s.
  - The two large misses: r 0.80 (737 s keeps 71.2%) and ℓ 0.7, c 2.125 at 1,350 s blocks (737 s, 77.6%).
  - The two near misses: m50 (163 s, 84.1%) and ℓ 0.7, c 1.5 at 5,400 s blocks (122 s, 84.9%).
- **M2 overfits.** Its σ in sample (0.72) is half M0's, but it has 9 coefficients for 10 measured conditions, and held
  out it does worse. Some point estimates fall outside their own bootstrap intervals (log₂ m +0.90 against
  [−0.92, +0.83]), a sign that the fit is not stable.

**Reported beside them.**
- **Single covariates** (M1 plus one; 200 resamples). Machine and AGV load carry the clearest signs: higher load,
  shorter Δ*. §14 saw this post hoc for the AGV share, and §16 for u at 1,350 s blocks. None improves on M0 held out.

  | added to M1 | coefficient [90%] | held out: keeps | C |
  |---|---|---|---|
  | log₂ τ / p̄ | −0.39 [−0.47, −0.00] | 34 | 0.529 |
  | log₂ T_reg / p̄ | −0.74 [−0.74, −0.04] | 32 | 0.554 |
  | log₂ c | +1.38 [−0.73, +1.92] | 32 | 0.529 |
  | AGV share on duty | −4.87 [−5.77, −1.14] (−0.49 per 0.1) | 31 | 0.506 |
  | u | −5.89 [−10.32, −0.29] (−0.59 per 0.1) | 34 | 0.519 |
  | surge share | −0.44 [−4.40, +9.23] | 33 | 0.520 |
  | short-fleet share | −3.03 [−5.96, +9.81] | 33 | 0.526 |

- **The collinear pair and M3.** M2 without τ / p̄ keeps 32 at C 0.561, M2 without the AGV share 33 at 0.577, and M3
  32 at 0.602. None improves on M0.
- **The size slope under censoring** (`fit_checks.out` (3)):

  | data | upper bounds | slope on log₂ m |
  |---|---|---|
  | size arm, m15-m100 | taken as exact (§14's S_m) | −0.51 |
  | size arm, m15-m100 | censored | +0.07 |
  | all 35 conditions | taken as exact | −0.25 |
  | all 35 conditions | censored (M1) | −0.09 [−1.12, +0.35] |

  - Three of the five sizes are upper bounds (m15, m20, m50). §14 already noted that their true values could flatten
    or steepen the slope.
  - Treated as bounds, they leave only m30 and m100 measured, and the slope is not pinned in either direction.
  - So "Δ* ∝ m^−0.5" should not be stated as a law. §14's R reading stands (B2's 300 s keeps 75.5% at m100), and so
    does m100's measured Δ*, a short 0.58 p̄.
- **What an overshoot costs.** At 1.5, 2 and 4 times Δ*, the kept share has a median of 79.8%, 75.3% and 64.0% (ranges
  71-94%, 65-90% and 47-85%). Each doubling beyond Δ* costs about 10 points of the gain.
  - The plateau is at least Δ* / 60 s wide: median 4.55, range 1.91-17.16.
- **Real time.** Under M0 fitted on all 35, one serial re-plan takes a median 3.6% of the interval. It takes 20% at
  m50 and r 0.35, and 53% at m100.
  - The constant rule runs in real time on one core up to 100 machines. D1 60 s does not at m100 (T_c / Δ 1.05, §14).
- **Sensitivities** (point estimate; coverage of 35, and C):

  | | M0 | M1 | M2 |
  |---|---|---|---|
  | X 10% (keep 90%) | 35, 0.742 | 33, 0.747 | 30, 0.720 |
  | X 20% (keep 80%) | 34, 0.268 | 33, 0.271 | 30, 0.476 |
  | upper bounds taken as exact | 31, 0.346 | 31, 0.367 | 27, 0.338 |
  | p̄ = fastest-machine mean | 34, 0.502 | 34, 0.508 | 32, 0.577 |

  - **The band sets the constant.** Keeping 90% needs about D1 81 s; keeping 80% allows about 224 s, and there M0
    would pass F3.
  - **Taking the bounds as exact would have been unsafe.** It recommends about 173 s and misses 4 conditions. The
    censored fit is the safer one, as §17 expected.
  - The fastest-machine p̄ changes nothing material.

**How the outcome table applies (§17).**
- All three readings fail, so the third row applies. No rule at half the compute of the finest interval, or less, is
  shown to carry condition-out. The guidance is "re-plan as often as compute allows", and compute binds from about
  100 machines.
  Axis 7's compute latency becomes the sharper question, and axes 4-5 test only direction.
- With F3 on its line, the row's first sentence carries the qualification above. A constant 0.36 p̄ keeps 85% at 34 of
  35 at about half the compute, and the half-compute line itself is not resolved.
- **No held-out check is triggered.** By §17 decision 5 it runs only if F1 or F2 passes, or, as a smaller set, if only
  F3 passes.

**What it means for paper C.**
- **The main contribution becomes a guidance rule and a negative result, not an equation.**
- **The guidance: re-plan as often as compute allows.**
  - D1 60 s keeps ≥ 92% of the best gain at all 35 conditions.
  - A constant 0.36 p̄, about 2 minutes on 5-6 minute operations, keeps 85% at 34 of 35 held-out conditions at about
    half the compute.
  - Each doubling beyond Δ* costs about 10 points of the gain. Erring fine is cheap; erring coarse is not.
- **The negative result: across the variables studied, no fitted equation predicts Δ* condition-out better than one
  constant.** The variables are floor size, transport, regime length, load, due-date allowance and regime mix.
  - Every one-at-a-time R reading failed, so a tuned interval does not carry.
  - The directions (higher load, shorter Δ*) are real but too weak to predict with.
  - §14's size law is withdrawn as a law: it rests on taking upper bounds as values.
- **Compute becomes the binding part of the guidance.** At 100 machines the constant rule needs 53% of a core, and
  D1 60 s more than one core (§14).
- **The paper's claim and outline need recasting around this** when paper C is outlined (plan §5b, "Where it goes").

**What it means for axes 4-5.**
- **They test direction only.** Do machine and AGV failures (axis 4) or forecast error (axis 5) favour finer or coarser
  re-planning than the constant rule?
- **They are read against M0's 0.36 p̄ and the finest interval,** not against a fitted equation.
- **They still need the event-driven planner model** (§7) before any run.
- Since the question is direction, their grids could be smaller than D1 60-3600 s; that is for their design.

**Next: the user's decision.** (11-10: the user chose axis 7; design §19, which finds item 1's arm, as described
here, would measure only the lattice's phase.)
1. **Axis 7's compute-latency arm** (proposed first).
   - The guidance is now "as often as compute allows", so what compute allows is the open quantity.
   - The arm would apply each plan T_c late (the forks know it, §7) and measure the gain at fine intervals on 50-100
     machines, with serial and parallel forks.
   - It needs a harness change in `run.py`, which would also carry §16's guarded print, and a design for approval.
2. **Axes 4-5 as direction tests** (proposed second). They need the event-driven planner model (§7), the larger code
   change.
3. **Optional: a held-out check of the constant rule** on combined conditions. §17 costs it at about 200 core-h on B2's
   floor, or 650-700 core-h with m30 and m50.
   - Decision 5 does not trigger it.
   - It would test whether 0.36 p̄ carries off the axes. The paper's quantitative guidance now rests on that rule.

## 19. Axis 7 design: compute, latency and the rollout horizon (drafted 11-10, approved by the user 11-10 as proposed; 7a results in §20; 7b is lab `rq2-replan-axes-horizon`, local)

**Why now.** The user chose axis 7 on 11-10. §18 leaves the guidance "re-plan as often as compute allows", with
compute binding from about 100 machines. Compute limits re-planning in two ways:
- **the rate:** a plan must act before the next one starts, so Δ must exceed T_c, the wall time of one re-plan (6 forks,
  each rolled to the window end);
- **the latency:** a plan computed from the state at t can act only from t + T_c.

§18 proposed measuring the second: apply each plan T_c late at fine intervals on 50-100 machines, with serial and
parallel forks. The dry run below shows that under the exact forecast this measures nothing new, so the design changes
the arm.

**Under the exact forecast, latency only shifts the re-plan lattice** (check L0 below).
- The forks know the latency: each one plays the pair in force until the plan acts, then its candidate. The twin is
  deterministic, so each fork reaches the world's exact state at t + T_c.
- So plan k, computed at t_k and acting at t_k + T_c, scores every candidate exactly as a zero-latency plan computed
  at t_k + T_c would, and picks the same pair. By induction, a run with latency T_c is the zero-latency run on the
  lattice shifted by T_c (its first interval is Δ + T_c).
- Every condition of axes 1, 2 and 6 uses the exact forecast. There a latency below Δ changes the run only through the
  lattice's phase, which has no expected direction. All that is left is the first interval's extra T_c, at most
  T_c / window of the run (0.3% at T_c = 60 s).
- So §18's arm would measure how the gain depends on the lattice's phase, not what latency costs. Under the exact
  forecast, "as often as compute allows" means back to back (Δ just above T_c), and only the rate binds.
- Latency costs something only when the plan's information goes stale during T_c: a failure, a duration off its
  estimate, or an arrival off its forecast between t and t + T_c, which the forks did not see. That is axes 4-5's
  information structure.
- run.py's planner already has a small latency. It plans at warm-up + kΔ and switches at t0 + kΔ, where t0, the
  window's first agent decision, comes 2-26 s after the warm-up in the dry run. By L0 this is harmless.

**What axis 7 then asks (proposed).**
- **7a. Where the rate binds** (no new runs). This uses T_c per condition: serial on the cluster's cores from the stored
  outputs, and with 6 parallel forks from the dry run. With the stored gain curves, it gives each condition's
  real-time interval and the share of its best gain that interval keeps.
- **7b. The rollout horizon, the lever on T_c** (the run). Today each fork rolls to the window end: 6 h at the
  window start and 3 h on average. A horizon H cuts T_c roughly in proportion.
  - Does a planner that looks only H ahead keep its gain?
  - At equal compute, is a fine interval with a short horizon better than a coarse interval with the full one?
  - This is the lever a slower twin needs. des_twin rolls out B2 about 24,000 times faster than real time on one core
    (6 forks × 6 h in 5.3 s), and about 3,000 times at 100 machines. A higher-fidelity simulator, a larger floor or
    a slower computer moves T_c up by the same factor.
- **7c. Latency under stale information:** a factor inside axes 4-5's designs, not a run now (below).

**The harness (`run7.py`; `run.py` unchanged).**
- **Where the code lives.** `run.py` is left as it is (md5 1f09ef96), so every stored output's stamp still names the
  file on disk. `run7.py` imports its instances, episodes and stamps, and adds `run7_py_md5` to each stamp.
  - The axes 1-6 edits each changed `run.py`'s md5. Only the latest version is on disk, since `results/` is not in
    git.
  - §16's guarded print is in `run7.py`'s summary line; `run.py`'s stays unguarded.
- **Latency.** Plan k ≥ 1 is computed at t_k = warm-up + kΔ from the state then, and acts from a_k = t0 + kΔ + T_c.
  - run.py's planner is the case T_c = 0.
  - Plan 0 acts at t0, as if computed during the warm-up. Under the exact forecast that is the same plan.
  - Real time: one plan's compute ends before the next one starts, T_c ≤ Δ. `run7.py` refuses other values. So no two
    plans compute at once, and no plan uses a result that is still being computed.
  - **A late first decision** (t0 − warm-up > Δ − T_c; 68-122 s on 3 seeds at D1 60 s) makes plan k act after plan
    k + 1 has started. That is not overlapping compute. Plan k's pair and the time it acts are known when plan k + 1
    forks, and its forks play the pairs in force until then. run.py does the same at T_c = 0.
    - `run7.py` asserts only causality, that no plan acts before it is computed: (t0 − warm-up) + T_c ≥ offset.
    - L0's argument needs nothing more, and the 11-13 check repeats L0 on the latest seed (below).
    - The 11-10 guard, T_c + (t0 − warm-up) < Δ, was too strict: it stopped 9 7b tasks at T_c = 0 (§19 implementation).
- **T_c is set, not measured,** as a parameter in sim seconds. Runs are then reproducible on any node, and every plan's
  wall time is recorded beside it. 7a uses measured T_c.
- **Serial vs parallel forks** (`+par<P>`: `branch.fork_map` with P children at once). This changes only the wall time:
  L2 reproduces the serial run exactly. With 6 candidate pairs the speed-up is at most 6.
- **Horizon** (`+h<H>`). Each fork plays its candidate until t_k + H and scores the window tardiness accrued by then,
  including jobs still in the system up to t_k + H. Without H it plays to the window end, as run.py does. Within H of
  the window end the two are the same.
- **Shifted lattice** (`+off<s>`, for L0 only): re-plans k ≥ 1 move to t_k + offset.
- Variants are a run.py planner variant plus options joined by `+`, e.g. `size-m50-d1800+lat600`, so they never collide
  with run.py's names.

**Dry run (11-10, `dryrun7/`, seeds 0-1, local, 12 tasks at once on an otherwise idle 32-core box).**
`dryrun7/run_dry.sh`; tasks in `dryrun7/tasks.out`, timings in `dryrun7/time.out`.
- All 13 outputs stamp `run.py` 1f09ef96. 12 stamp `run7.py` b6204829, the current file. The first smoke task
  (`proc-k1-d3600+lat0`) stamps f8874314, the version before `compare` learned to skip the descriptor axis 1 lacks.
- The 12 tasks first landed in `out/`, because the launcher passed no root. They were moved to `dryrun7/out/`
  unchanged, and `out/` holds no `+` folder.
- **L1, zero latency = run.py: PASS** on 5 stored outputs. `run7.py compare` checks the tardiness, MDD-PTECT,
  schedule, re-plans, switches, decisions and descriptors (those both outputs have).
  - The 5 are proc k 1 at D1 3600 and 900 s, proc k 4 at D1 300 s (Δ = 1,200 s), size m50 at D1 3600 s, and ℓ 1.3,
    c 3, 1,350 s blocks at D1 1800 s.
- **L2, 6 parallel forks = serial: PASS** (proc k 1, D1 900 s, identical).
- **L0, latency T_c = zero latency on the lattice shifted by T_c: PASS** in all 3 cases. The pair schedule, the
  tardiness and the score of every plan are identical:

  | case | tardiness, latency T_c | same, shifted lattice | stored, no latency | effect of the latency |
  |---|---|---|---|---|
  | B2 seed 0, D1 900 s, T_c 300 s | 76.638549 | 76.638549 | 79.971484 | −4.2% (5 switches vs 4) |
  | m50 seed 0, D1 1800 s, T_c 600 s | 54.725699 | 54.725699 | 54.067792 | +1.2% |
  | B2 seed 1, D1 300 s, T_c 150 s | 22.219750 | 22.219750 | 22.219750 | 0.0% (another schedule, same tardiness) |

  - On one seed, shifting the lattice moves tardiness by up to 4% either way. That is part of the grid-point noise §18
    saw, such as the dips at 120 s.
- **Horizon smoke:** B2 seed 0 at D1 900 s with H 3,600 s ran end to end. Tardiness 79.678 against 79.971 with the
  full horizon, 13 switches against 4, and a third of the planning time (23 s against 71 s locally).
- **Wall time of one re-plan** (6 forks, seed 0, local core; the cluster's serial times are about 2 × the stored mean
  per re-plan at D1 60 s):

  | floor | full horizon, serial | 6 forks | speed-up | H 1 h, serial | H 1 h, 6 forks | full, mid-window | cluster core, serial |
  |---|---|---|---|---|---|---|---|
  | m15 (B2) | 5.3 s | 1.3 s | 4.0 × | 1.1 s | 0.3 s | 3.1 s | ≈ 10 s |
  | m50 | 20.9 s | 3.8 s | 5.5 × | 5.8 s | 1.1 s | 10.6 s | ≈ 48 s |
  | m100 | 44.2 s | 7.9 s | 5.6 × | 11.8 s | 2.0 s | 18.9 s | ≈ 125 s |
  | m200 | 153.2 s | 27.9 s | 5.5 × | 28.3 s | 5.1 s | 78.8 s | – (not run) |

  - 6 parallel forks give the same fork results as serial at every floor and point (`par_equal`), 4.0-5.6 times faster.
  - A 1 h horizon costs 0.18-0.28 of a full-window plan, against a rollout share of 0.17. Each plan has a fixed
    overhead (the forks and their first ticks), which favours fewer, longer plans.
  - On B2's floor (axes 1 and 2) the cluster's serial time is 5-11 s at every condition, 13-48 s at m20-m50 and on the
    fleet arm, and 125 s at m100.

**7a, the rate limit (computed from stored outputs after approval).**
- **Real-time interval** Δ_rt of a condition: the cluster core's serial window-start re-plan time, estimated as 2 × the
  mean per re-plan of the condition's D1 60 s tasks, and expressed in D1. The cluster's cores are the slowest measured,
  1.8-2.8 times the local core. With 6 forks Δ_rt is divided by the floor's measured speed-up (4.0-5.6).
- **F7a (pre-declared): on one core, Δ_rt keeps ≥ 85% of the best gain at all 35 conditions.**
  - The kept share is read as in `fit.py`: the gain interpolated linearly in log D1, with Δ_rt below 60 s read at
    60 s.
  - The outcome is close to known. Only m100 has Δ_rt above 60 s (about 125 s), and m100's best gain sits at 120 s
    (§14).
  - Reported beside it: the real-time interval per floor and core count, and Δ_rt as n_cand × H / (S × P) for a
    simulator that runs S simulated seconds per wall second on one core. This is the form a practitioner can carry to
    another twin.
- **m200** gets a real-time interval only (no gain curve, §13 decision 2). On the local core that is about 150 s
  serial and 28 s with 6 forks. So on one core this planner re-plans no faster than about 2.5 min at 200 machines.

**7b, the horizon arm (proposed run).**

| | levels |
|---|---|
| floors | m15 (B2, proc k 1), m50, m100: the axis 6 levels where T_c grows |
| horizon H | 900, 1,800 and 3,600 s (2.5-10 p̄; 0.17-0.67 of B2's 5,400 s regime block); 7,200 s added where 3,600 s fails H1 |
| intervals | D1 60 s (the finest) and 300 s (B2's old edge) |
| seeds | 0-39, paired with the stored full-horizon runs (axes 1 and 6: D1 60-3600 s on all 40 seeds) |
| forecast | exact; no latency (by L0 it only shifts the lattice) |
| tasks | 3 floors × 3 H × 2 D1 × 40 seeds = 720 |

- **Compute** is measured as rollout volume: simulated fork-seconds per window, Σₖ 6 min(H, window end − t_k). It is
  deterministic and the same on every node. The wall time of every plan is recorded and reported beside it.
- **Equal compute.** A full-horizon planner at interval Δ rolls out 3W(n + 1) fork-seconds (W the window, n = W / Δ).
  So the H-planner at D1 60 s matches the full horizon at Δ_eq, read on the stored curves by interpolation in log D1.
  Δ_eq is about 760 s at H 900, 380 s at H 1,800 and 200 s at H 3,600.

**Readings (proposed).** g(Δ, H) is the gain over plan-once pooled over seeds (§3), and g(Δ, ∞) the stored
full-horizon gain. The bootstrap is 2,000 joint resamples of seeds across floors, as in §13.
- **H1 (what the horizon costs).** K_H = g(D1 60, H) / g(D1 60, ∞) ≥ 0.85 at all three floors (point estimate; its
  90% interval is reported).
  - The shortest H that passes is the horizon the planner needs, H_min.
  - **Extension rule:** if H 3,600 s fails at a floor, H 7,200 s is added there at both intervals.
- **H2 (how to spend compute), read at H 1,800 s.** d = g(D1 60, 1,800) − g(Δ_eq ≈ 380 s, ∞).
  - **PASS** (spend compute on the interval): the 90% interval of d lies above 0 at ≥ 2 of 3 floors.
  - **FAIL** (spend it on the horizon): below 0 at ≥ 2 of 3.
  - **Indeterminate** otherwise: at equal compute the two are equivalent within the noise.
- **Reported:** K_H at every H and both intervals, and d at H 900 and 3,600 s and at D1 300 s (Δ_eq about 2,040 s at
  H 1,800 and 1,020 s at H 3,600; H 900 s falls beyond the grid). Also the measured wall time against the volume, and
  the switches per window.
- **Check before the readings (C7):** each new task's MDD-PTECT equals its stored full-horizon task's. The instances
  are the same, so this catches harness drift.

**What each outcome means.**

| outcome | paper C | axes 4-5 |
|---|---|---|
| H1 passes at 900 or 1,800 s | Compute gets a lever: a horizon of H_min cuts the mean rollout per plan 3-6 times at the same interval (6-12 times at the window start). The guidance becomes "re-plan as often as compute allows, looking H_min ahead" | they may run at H_min for about a third of the cost (their full-horizon references stay) |
| H1 passes only at 3,600 or 7,200 s | The horizon matters below about one regime block. The lever is modest (2-3 times) | full horizon |
| H1 fails at 7,200 s | The planner needs most of its horizon. Parallel forks are the lever, then a coarser interval at about 10 points per doubling (§18) | full horizon |
| H2 PASS / FAIL / indeterminate | At equal compute: re-plan more often with a shorter look-ahead / keep the look-ahead and re-plan less often / either | none directly |
| F7a PASS (expected) | Compute does not bind on one core up to 100 machines here. The guidance becomes a statement about the twin's speed, Δ_rt = n_cand × H / (S × P) | none |
| F7a FAIL at a condition | That condition needs parallel forks to re-plan at the 85% interval in real time | none |

**7c, latency in axes 4-5 (to settle in their designs).** By L0, any latency cost there is the cost of stale
information. Proposed for each design:
- a latency factor T_c ∈ {0, Δ / 2} at one fine interval;
- read as whether latency changes the direction finding, and by how many points.

T_c there is set as a share of Δ rather than measured. The measured T_c at 15 machines (5-11 s) is too small against
Δ to show anything.

**Cost** (core-h; the cost model below, with 25% added for the per-plan overhead the dry run measured):

| part | local core | cluster core | wall |
|---|---|---|---|
| 7a | 0 (stored outputs, dry run) | – | minutes |
| 7b, 720 tasks | about 115 | about 300 | about 6 h on the lab's 20 local slots, or about 3 h on the cluster |
| H 7,200 s extension, per floor that needs it | 9 (m15), 35 (m50), 74 (m100) | 1.8-2.8 times that | |

- **The cost model:** per floor and seed, 118 full-window re-plans (c_W): 55 + 29 + 15 at D1 60 s and 11 + 6 + 3 at
  300 s for H 3,600 / 1,800 / 900 s. c_W is 5.3 / 20.9 / 44.2 s locally and about 10 / 48 / 125 s on the cluster's
  cores.
- The longest task is m100 at D1 60 s with H 3,600 s: about 50 min locally.

**Decisions (the user approved every proposal on 11-10: "Approve as proposed"; the first option in each):**
1. **The latency arm.**
   - Proposed: do not run §18's latency sweep. Record L0 as a result: under the exact forecast, latency is a phase
     shift of the lattice, verified on 3 cases. Latency then enters axes 4-5 as a factor (7c).
   - Alternative: run the sweep anyway, at T_c = Δ/4, Δ/2 and 3Δ/4 on D1 60, 120 and 300 s on m50 and m100, 20 seeds
     (about 330 core-h locally, 850 on the cluster). It would measure the gain's phase sensitivity.
   - Cheaper alternative: a phase check on B2's floor (D1 120 and 300 s, offset Δ/2, 40 seeds, about 8 core-h
     locally). It would tell whether §18's dips at 120 s are phase noise.
2. **7a, the rate limit:** computed from the stored outputs with F7a as written. No runs.
3. **7b, the horizon arm.**
   - Proposed: H 900, 1,800 and 3,600 s at D1 60 and 300 s on m15, m50 and m100, with the 7,200 s extension rule.
   - Alternatives:
     - D1 60 s only (saves about 17%), losing the coarse-interval comparison;
     - skip 7b and go to axes 4-5. The guidance then has no compute lever but parallel forks.
   - A candidate-set arm (fewer pairs, T_c ∝ candidates) is possible but not proposed. Cutting candidates changes what
     the planner can choose, not only its cost.
4. **Seeds.** Proposed: 40, paired with the stored references. Alternative: 20 (half the cost; K_H is a paired ratio,
   so it is less noisy than Δ*).
5. **Where.** Proposed: local through the lab: about 115 core-h and about 6 h on 20 slots. The box was idle on 11-10,
   and the local core is 2-3 times faster than the cluster's at m100. Alternative: the cluster, about 300 core-h at its
   speed and about 3 h of wall time.
6. **Criteria as written:**
   - F7a's 85% line at all 35 conditions;
   - H1's 85% line at all three floors, with the extension rule;
   - H2 read at H 1,800 s, PASS / FAIL at ≥ 2 of 3 floors;
   - C7;
   - rollout volume as the compute measure.

**Implementation (11-10, after the approval), ready to run.**
- **Outputs.** `run7.py` writes to `out7/<variant>/s<seed>.json`. Nothing writes to `out/`, so `analyze.py`'s input
  stays axes 1, 2 and 6. The dry run's outputs are in `dryrun7/out/`.
- **Task sets.**
  - `run7.py list horizon` gives the 720 tasks, costliest first by rollout volume × the level's re-plan cost.
  - `run7.py list horizon-ext <floor> ...` gives the 7,200 s tasks, 80 per floor.
  - `--todo` lists only missing outputs, and `--root` points at another folder.
  - Task names are `<level>-d<D1>+h<H>`, e.g. `size-m100-d60+h3600`, with serial forks and no latency.
- **Launcher `launch7.sh [workers]`.** The extension runs inside the launcher, so there is one lab entry:
  1. it runs the base tasks;
  2. once all 720 exist, `analyze7.py extension` names the floors where H 3,600 s fails H1, and their 7,200 s tasks
     run;
  3. it then writes `out7/DONE`, the done check, with the task count and the extended floors.
  - Each step skips finished outputs, so a rerun resumes. It exits 1 while any task the rule calls for is missing.
- **Analysis `analyze7.py`.**
  - It prints F7a, then C7, the K_H table with 90% intervals, H1 with H_min and the extension call, the equal-compute
    table with H2, and wall time against volume and switches. It writes `analysis7.out`.
  - `analyze7.py frontier` prints F7a alone (`frontier7.out`, §20). With 7b incomplete it says so and reads nothing.
  - The bootstrap uses stream 1910, 2,000 joint resamples.
- **Tests (11-10, in the scratchpad; the study's `out/` and `out7/` untouched).** The scratch `out7/` had 798 synthetic
  outputs: stored full-horizon runs with T_H = once − K (once − T_full), so that g_H = K g_full. Two cheap real tasks
  were left for the launcher to run.
  - The launcher ran the two real tasks (B2, D1 300 s, H 900 s, seeds 0-1).
  - It named m100 for the extension, as the synthetic K of 0.80 at 3,600 s called for, and wrote `DONE`.
  - A rerun ran nothing and rewrote `DONE`.
  - `analyze7.py` recovered every synthetic K exactly (95%, 90%, 80%). C7 passed on all 800, including the two fresh
    tasks.
  - With no `out7/` it reports 7b as incomplete.
- **Memory:** one m100 task peaks at 95 MB (the cluster ran these at 4 GB), so 20 at once is no concern.
- **Stamps:** the 7b tasks stamp `run.py` 1f09ef96, and `run7.py` 0c6c571f (11-10) or 1dfd1e7f (11-13 guard fix,
  below). `launch7.sh` is b65a6da2 and `analyze7.py` 930bd3da. The two `run7.py` versions run the same planner, and
  `analyze7.py` accepts exactly those two and flags any other.
- **First run, 11-11; fix, 11-13.** The lab entry ran from 11-10. It was marked failed on 11-11 at 09:30 with 590 of
  720 outputs, for two separate reasons.
  - **A reboot at 09:18 killed the run.** Most of the 130 missing tasks are only that: B2 at D1 300 s with H 900 and
    1,800 s, m50 at D1 300 s with H 900 s, and one B2 task at D1 300 s with H 3,600 s. Writes are atomic, and no
    partial output or `.tmp` file was left.
  - **The 11-10 real-time guard stopped 9 tasks at T_c = 0:** B2 seeds 8 and 20 and m50 seed 9, at D1 60 s, all three
    H. On those seeds the window's first decision comes 67.8, 122.1 and 70.8 s after the warm-up, after the first
    60 s re-plan.
    - The guard demanded T_c + (t0 − warm-up) < Δ. That conflates when a plan acts with how long it computes.
    - Fixed on 11-13 (`run7.py` 1dfd1e7f) as described under Latency above: T_c ≤ Δ, plus causality. Nothing else in
      the file changed, so a run the old guard let through executes the same code.
  - **Checks of the fix** (`dryrun7/fix1113/`: `run_checks.sh`, then `compare_checks.py` → `checks.out`): **all
    PASS.**
    - A: at latency 0, the three stopped seeds at D1 60 s equal run.py's stored outputs in every compared field.
    - B: three of the 9 stopped tasks run now, and their MDD-PTECT equals the stored full-horizon task's (C7).
    - C: three finished 7b tasks (one per floor) rerun with the new file equal `out7/` in every compared field, plus
      the plan scores, plan times and t0.
    - D: L0 holds on the latest first decision. B2 seed 20 at D1 120 s with T_c 30 s equals the lattice shifted by
      30 s, down to every plan score (90.624414; 89.653011 with no latency).
  - **The 590 finished outputs are kept.** The requeued launcher skips them and runs the 130 missing tasks.
- **Lab fields** (the parent session creates the entry): `rq2-replan-axes-horizon`, local, 20 slots;
  - launch `bash results/rq2-replan-axes/launch7.sh 20`;
  - done glob `results/rq2-replan-axes/out7/DONE`, count 1;
  - analysis `.venv/bin/python results/rq2-replan-axes/analyze7.py`;
  - about 6.5 h of wall time, 3-4 h more if the extension runs at m100, about 6 h more at all three floors.

## 20. 7a results: the rate limit (11-10, from stored outputs)

**Run.** `analyze7.py frontier` → `frontier7.out`, computed after the approval. Inputs: `fit_out/table.pkl` (the 35
conditions' curves and per-seed re-plan times at D1 60 s) and `dryrun7/time.out`. No new simulation.

**F7a PASS.** On one cluster core, re-planning at the real-time interval keeps ≥ 85% of the best gain at all 35
conditions. The lowest is 92.0% (proc k 16).
- **Only m100's real-time interval exceeds 60 s** (126 s in D1). It keeps 98.4% there, since m100's best gain is at
  120 s. With 6 forks the interval is 22 s, read at 60 s: 98.1%.
- **Every other condition re-plans faster than 60 s on one core** (5-48 s), so its kept share is the 60 s point's:
  92.0-100%. The conditions below 100% are those whose best gain lies at 120-450 s.
- **Over fit.py's 2,000 resamples,** all 35 keep ≥ 85% in 67.7%, against 67.8% for D1 60 s at every condition. The
  resample spread is the seed noise of the finest point; compute adds nothing to it.

**The real-time interval per floor** (window-start re-plan, full horizon, 6 forks):

| floor | local core, serial | local, 6 forks | H 1 h, serial | H 1 h, 6 forks | cluster core, serial | S (local) |
|---|---|---|---|---|---|---|
| m15 (B2) | 5.3 s | 1.3 s | 1.1 s | 0.3 s | 9.6 s | 24,270 |
| m50 | 20.9 s | 3.8 s | 5.8 s | 1.1 s | 48.0 s | 6,207 |
| m100 | 44.2 s | 7.9 s | 11.8 s | 2.0 s | 125.8 s | 2,934 |
| m200 | 153.2 s | 27.9 s | 28.3 s | 5.1 s | – | 846 |

S is the simulated seconds one core rolls out per wall second: 6 forks × 6 h over the serial time.

**What it means.**
- **In this twin, compute does not bind up to 100 machines.** One cluster core re-plans every 1-2 min at 100 machines
  and keeps 98% of the best gain. A local core does it in 44 s, and 6 forks in 8 s.
- **Compute starts to bind at about 200 machines on one core:** about 2.5 min serial locally, 28 s with 6 forks. §13
  left m200 out, so there is no gain curve to read it on.
- **For paper C, the compute guidance is a statement about the twin's speed.** The finest real-time interval is
  Δ_rt ≈ n_cand × H / (S × P): 6 candidates, rollout horizon H, S simulated seconds per wall second per core, P cores.
  S falls from about 24,000 at 15 machines to about 850 at 200 in des_twin. A twin with a smaller S, such as a
  higher-fidelity simulator or a slower computer, reaches the binding range on smaller floors. 7b measures the one
  lever that may cost plan quality, H.
- **Together with L0:** under the exact forecast, latency only shifts the lattice, and the rate binds only at large
  floors or with a slow twin. So "re-plan as often as compute allows" carries no hidden penalty up to 100 machines
  here.

## Files

| file | what |
|---|---|
| `run.py` | harness: arms, instances, planner loop, tasks (`run.py <variant> <seed> [root]`, `run.py list <set>`, `run.py describe <arm>-k<k> <seed>`) |
| `analyze.py` | gain curves, Δ*, bootstrap, C0 / C1, readings S / R → `analysis.out`, `curve.csv`, `dstar.csv`; `--preview` → `preview.out`; `--root dryrun` |
| `launch.sh`, `submit_cluster.sh` | local (lab, xargs) and cluster (twin_array.sbatch) launchers of a task set |
| `code/` | frozen `env/` at 09f53cbb (`STAMP`) |
| `dryrun/` | the dry run: tasks, runner, outputs, `analysis.out`, `curve.csv` |
| `dryrun6/` | axis 6's dry run (§13): `time_size.py` → `time_size.out` (instance check, episode time, descriptors per level), `tasks.out` and `out/` (planner and fixed-pair tasks on seed 0) |
| `preview.out` | the Δ* estimator on `../rq2-replan`'s curve (X and seed-count sensitivity) |
| `analysis_s0-19.out`, `curve_s0-19.csv`, `dstar_s0-19.csv` | axis 1 on seeds 0-19 (the lab's 10-29 19:45 analysis, §11) |
| `analysis.out`, `curve.csv`, `dstar.csv` | the analysis of everything under `out/`: axes 1, 6 and 2, seeds 0-39, D1 60-3600 s (the lab's 11-08 18:28 run; axis 1's sections are unchanged from 10-29 23:17, §12; axis 6's from 11-04 02:05, §14; axis 2's are §16) |
| `r_noise.py`, `r_noise.out` | §12 remark: R against references with less selection bias (bootstrap-corrected, cross-fit, non-increasing fit) |
| `axis6_checks.py`, `axis6_checks.out` | §14 remarks: the planner at D1 900 s against the 10-29 screen, seed noise in R and R120's reference, S_m with one size left out, Δ* against the on-duty AGV share (post hoc) |
| `axis2_checks.py`, `axis2_checks.out` | §16 remarks: seed noise in the reference of R_ℓ / R_c / R_mix, the low-tardiness corner and the slopes with one cell left out, D1 60 s against the best gain at all 35 distinct conditions of axes 1, 2 and 6 |
| `fit_inputs.py`, `fit_inputs.out` | §17 design input: the 35 conditions with Δ* and the fit's covariates (axis 1's on-duty AGV share rebuilt from the AGV schedule, checked against the recorded share on axes 2 and 6), the design's ranges, correlations and variance inflation, and the kept share of every fixed grid interval. It fits nothing |
| `fit.py`, `launch_fit.sh` | §17's fit: `fit.py table / tasks / run <task> / report / test`; the launcher (lab `rq2-replan-axes-fit`) builds the table and runs the 23 tasks, skipping finished ones |
| `fit_out/` | the fit's table (`table.pkl`) and task outputs (`point.json`, `read-00..19.json`, `var-00..01.json`); written by the lab run |
| `fit.out`, `fit_loco.csv` | the fit's report (coefficients, condition-out coverage and compute, readings F1-F3, reported quantities) and its per-condition held-out predictions per model; written by `fit.py report` |
| `fit_checks.py`, `fit_checks.out` | §18 remarks: F3 against its line over the resamples, the held-out misses per model, the size slope with upper bounds censored or taken as exact, the kept share beyond Δ* |
| `run7.py` | axis 7 harness (§19): run.py's planner with latency `+lat`, shifted lattice `+off`, rollout horizon `+h` and parallel forks `+par`; `run7.py <variant> <seed> [root]`, `run7.py time <level> <seed> [P ...]` (single re-plan wall times), `run7.py compare <a> <b>` (the regression fields) |
| `analyze7.py`, `launch7.sh` | axis 7 (§19): F7a, C7, H1, H2 and the reported items → `analysis7.out` (`frontier` for F7a alone, `extension` for the floors needing H 7,200 s); the 7b launcher (lab `rq2-replan-axes-horizon`: base tasks, then the extension rule, then `out7/DONE`) |
| `frontier7.out` | §20: F7a, the real-time interval per condition and per floor (`analyze7.py frontier`, 11-10) |
| `out7/`, `analysis7.out`, `tasks_horizon*.txt` | 7b's task outputs `out7/<level>-d<D1>+h<H>/s<seed>.json` and `out7/DONE`, its analysis, and the task lists of the last launch (written by the lab run) |
| `dryrun7/fix1113/` | 11-13 checks of `run7.py`'s real-time guard fix (§19): `run_checks.sh` → `out7/`, `tasks.out`; `compare_checks.py` → `checks.out` (A-D all PASS) |
| `dryrun7/` | axis 7's dry run (§19): `run_dry.sh`, `tasks.txt` / `tasks.out` (L1, L2, L0 and a horizon task), `time.out` (re-plan wall times, serial and 6 forks, full horizon and 1 h), `out/` |
| `dryrun2/` | axis 2's dry run (§15): `screen.py` (cells, fixed pairs, the planner at one D1) → `out/`, `summarize.py` → `summary.out`; `out/` is also C3's reference for the cluster run |
| `out/` | task outputs `out/<variant>/s<seed>.json` (axis 1, its extension, axis 6, then axis 2) |
| `tasks_<set>.txt` | the task list of the last launch of a set |
