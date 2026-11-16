# rq4-mfail-ccal-resume: machine failures with calibrated due dates, rerun under preempt-resume (2026-10-28, user)

Status: **done 10-28** (lab `rq4-mfail-ccal-resume-calib` 22:17-22:26, job 21977012; `rq4-mfail-ccal-resume`
22:26-22:37, job 21977020; cluster). The change landed on main 10-28 (09f53cbb). **Supersedes rq4-mfail-ccal**
(restart model) for the thesis: the claim holds, its numbers change (Results).

## Why

The failure model changed on 10-28 from restart to preempt-resume (`results/dev-failure-resume/README.md`): the job in
process stays on the failed machine and finishes the rest after the repair, instead of being re-routed and starting
its operation again. rq4-mfail-ccal is the thesis's machine-failure result (`Rule_Design.tex` §rules-conditions:
"machine failures at calibrated due dates leave the ranking unchanged, τ 1.00"; RQ4), so it is rerun in full. Failures
cost less under resume, so the calibrated due-date scale S* (×1.353 under restart) moves; both stages run again.

## Setup

rq4-mfail-ccal's, unchanged except the failure model (its README has the details): dev-pect-rule's m50 family (50
machines, 26 AGVs, load 0.9, 6 h window, 1.5 h blocks, uniform Weibull failures at availability 0.90).
- Stage 1 (calib): S ∈ {1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0}, MDD-PTECT, seeds 1000-1029; S* where the failure late
  share equals the failure-free one.
- Stage 2 (compare): the 6 v5 pairs + MDD-TECT, test seeds 0-39, at no failures / failures at B2's c / failures at S*.
- **Code:** `code/env` = rq4-mfail-ccal's frozen v5 snapshot (`results/rq2-rebase-v5/code/env`, 6d13412a) plus only
  `code/failure_resume.patch` (the engine and rules-comment diff of branch `failure-resume`). `mfail.py` points
  `PECT_CODE` at it.

## Checks before launch

- `mfail.py check-restart` (10-28, local): the patched code with `failure_restart` reproduces rq4-mfail-ccal's
  calibration seeds 1000 and 1001 exactly (window tardiness, late and due counts); under resume the same seeds give
  383.4 vs 404.2 ks and 258.2 vs 309.0 ks. **PASS.**
- `mfail.py check` (run by `submit_calib.sh` on the cluster): the failure-free seed 0 must still equal dev-pect-rule
  (57.743 ks); the failure seed prints its resume value. 10-28 on the cluster: m50-none 57.742968 = dev-pect-rule
  **OK**; m50-unif seed 0 155.95 ks under resume vs 172.84 restart.
- The patch equals the engine / rules diff landed on main (`git diff -- env/des_twin/engine.py env/des_twin/rules.py`,
  10-28), so this snapshot and main's twin model failures the same way.
- `compare_old.py` (in the analysis): the failure-free setting must be identical to rq4-mfail-ccal.

## Outputs

`out/calib/`, `cscale.json`, `pick.out`, `out/compare/`, `analysis.out` (`analyze.py`, as rq4-mfail-ccal) and
`compare_old.out` (`compare_old.py`: S*, tardiness, late share and ranking, restart vs resume).

## Results

### Stage 1, calibration (10-28, seeds 1000-1029, MDD-PTECT, `pick.out`)
- **S\* = ×1.252 under resume, against ×1.353 under restart.** The target, the failure-free late share 0.273
  (mean tardiness 122.2), is the same as rq4-mfail-ccal's, as it must be: failures off is unchanged.
- At B2's c (×1.0) failures cost less under resume: mean tardiness 292.2 against 381.7 under restart (−23%), late
  share 0.453 against 0.506. The gap shrinks as due dates loosen (×2.0: 27.3 against 48.9; ×3.0: 2.0 against 2.5).

| S | late share resume | restart | mean tardiness resume | restart |
|---|---|---|---|---|
| 1.0 | 0.453 | 0.506 | 292.2 | 381.7 |
| 1.25 | 0.274 | 0.320 | 175.4 | 242.0 |
| 1.5 | 0.162 | 0.206 | 98.3 | 148.6 |
| 1.75 | 0.100 | 0.129 | 54.9 | 89.2 |
| 2.0 | 0.057 | 0.080 | 27.3 | 48.9 |
| 2.5 | 0.018 | 0.024 | 6.6 | 10.8 |
| 3.0 | 0.006 | 0.006 | 2.0 | 2.5 |

### Stage 2, comparison (10-28, test seeds 0-39, `analysis.out`, `compare_old.out`)
Tardiness % vs MDD-PTECT pooled over the 40 seeds (95% bootstrap CI), restart from rq4-mfail-ccal beside it.

| rule | none | failures at B2's c, resume | restart | failures at S\*, resume (×1.252) | restart (×1.353) |
|---|---|---|---|---|---|
| MDD-PTECT (mean tardiness, ks) | 158.2 | 381.9 | 474.7 | 240.1 | 268.4 |
| MDD-PECT | +2.0 | +0.1 | +1.6 | +1.1 | +1.3 |
| ATC-PTECT | +2.6 | +5.0 | +7.5 | +3.0 | +4.8 |
| ATC-PECT | +4.6 | +3.6 | +9.1 | +4.0 | +5.4 |
| SRT-PTECT | +8.0 | +2.3 | +2.5 | +11.2 | +14.2 |
| SRT-PECT | +10.5 | +2.9 | +3.0 | +12.5 | +15.3 |
| MDD-TECT | +29.9 | +26.9 | +26.2 | +38.7 | +42.2 |
| hindsight best v5 pair | −4.6 | −4.4 | −3.6 | −6.4 | −5.7 |
| late share | 0.328 | 0.509 | 0.551 | 0.317 | 0.309 |
| Kendall τ vs none | | +0.52 (p 0.14) | +0.62 | **+1.00 (p 0.0004)** | +1.00 |
| MDD-PTECT tardiness vs none | | ×2.41 | ×3.00 | ×1.52 | ×1.70 |

- **Checks:** failures off is identical to rq4-mfail-ccal on all 40 seeds and 7 rules (`compare_old.py`: True). The
  cluster's `mfail.py check` reproduced the failure-free 57.742968 before stage 1.
- **The claim holds under resume.** With due dates calibrated to the failure regime (S\* ×1.252, late share 0.317 vs
  0.328 failure-free), the ranking of the 7 rules is identical to no failures (τ +1.00) and MDD-PTECT is best.
  Failures cost ×1.52 tardiness there, against ×1.70 under restart.
- **At B2's c** failures cost ×2.41 (restart ×3.00) and 51% of jobs are late (restart 55%). MDD-PTECT stays best, tied
  with MDD-PECT (+0.1%); the flow-time rules move up as before (SRT-PTECT +2.3% vs +8.0% without failures), and the
  ranking against no failures is τ +0.52 (restart +0.62). Restart and resume rank the rules almost the same (τ +0.90);
  the ATC pairs lose less under resume (+3.6 / +5.0 against +9.1 / +7.5).
- **What changes in the text:** the numbers, not the finding. S\* ×1.35 → ×1.25, failure cost ×1.7 → ×1.5 at
  calibrated c and ×3.0 → ×2.4 at B2's c, 55% → 51% late, τ 0.62 → 0.52, hindsight best pair −5.7% → −6.4%.
  Thesis: `Rule_Design.tex` l. 29 (τ 1.00: unchanged) and its failure subsection; `Platform.tex` l. 228 still describes
  the restart model.
