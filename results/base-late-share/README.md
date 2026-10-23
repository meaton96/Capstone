# base-late-share: share of late jobs on B2 under MDD-PTECT (2026-10-22, user: thesis RERUN-V5)

Status: **done 10-22 (lab `base-late-share`, local, about 4 min).**

## Why
The thesis (Background, due dates) says the allowance c was set as Sels et al. (2012) do, by the share of jobs that
finish late: "c = 2 left about half of the jobs late under the rule pair SRT-ECT". That calibration was made on the
earlier randomized instances with 5,400 s windows (`../rq2-twin-due`: SRT-ECT about 77 / 46 / 15% late at c = 1.5 / 2 /
3). The benchmark is now B2, with c ~ U[1.75, 2.5], and the reference pair is MDD-PTECT. The user wants the sentence
replaced with the late share under the current rule on the current benchmark.

## Design (`run.py`)
- **Instances:** B2 itself: 15 machines, 7 AGVs (short-fleet blocks 2), the B2 load mix in 1.5 h regime blocks, random
  warm-up, 6 h scoring window. Seeds 0-39 (test) and 1000-1039: 80 instances.
- **Due dates:** total work content, d = r + c · TWK.
  - `B2`: c ~ U[1.75, 2.5], drawn per regime block (the benchmark as used everywhere);
  - fixed c ∈ {1.5, 1.75, 2, 2.25, 2.5, 3}, B2 otherwise unchanged.
- **Rules, each held all episode:**
  - MDD-PTECT (the reference pair);
  - SRT-ECT (the rule of the old calibration);
  - SPT-ECT (Sels et al.'s calibration rule).
- **Late share (as `../rq4-mfail-ccal`):** jobs due inside the window [t0, t1) that exit after their due date or are
  still open at t1, over all jobs due in the window, pooled over instances. Also reported: the share among jobs that
  exit in the window, and the mean window tardiness.
- **Frozen code:** `../rq2-rebase-v5` (6d13412a + sim.py), B2 via `../rq3-size-oracle/sizesim.py` (condition `base`).
- **Check (10-22):** MDD-PTECT on seed 0 gives window tardiness 95.842006, B2's known value. (A two-seed smoke at B2's
  c gave MDD-PTECT 39.1%; that was not representative, see the results.)

## Results (10-22)
Pooled over 80 instances (seed-bootstrap 95% CI); the test seeds 0-39 alone are within 1.5 points of these.
`analysis.out`, `summary.csv`.

| due dates | MDD-PTECT | SRT-ECT | SPT-ECT | MDD-PTECT mean window tardiness (k job-s) |
|---|---|---|---|---|
| **B2: c ~ U[1.75, 2.5] per block** | **51.8%** [48.5, 54.9] | 48.5% [45.7, 51.2] | 55.6% | 152.7 |
| c = 1.5 | 80.7% | 81.1% | 84.2% | 251.6 |
| c = 1.75 | 68.6% | 66.7% | 72.3% | 206.4 |
| **c = 2** | **56.9%** [53.8, 60.0] | 54.2% [51.4, 57.0] | 60.6% | 168.9 |
| c = 2.25 | 47.7% | 44.0% | 50.7% | 136.2 |
| c = 2.5 | 39.1% | 36.0% | 42.9% | 107.9 |
| c = 3 | 26.3% | 25.0% | 30.7% | 66.7 |

Among only the jobs that leave in the window, the shares are within about 2 points of these.

### Reading
- **The thesis's "about half of the jobs late" still holds for the current benchmark and reference rule.** Under
  MDD-PTECT on B2, 51.8% of the jobs due in the window are late (50.0% on the test seeds); at c = 2 exactly, 56.9%.
- **The rule changes the share only slightly; the allowance sets it.**
  - Across c = 1.5-3 the three rules stay within 3-7 points of each other: SRT-ECT lowest, SPT-ECT highest.
  - MDD-PTECT has 3 points more late jobs than SRT-ECT but 21% less tardiness (152.7 vs 193.5 k job-s): it lets more
    jobs be a little late and fewer very late, as a due-date rule should.
- **Over B2's range the share runs from 69% (c = 1.75) to 39% (c = 2.5)**, so B2's due dates mix tight and moderate
  blocks around the 50% point that Sels et al. (2012) aim for with c = 1.9 under SPT.
- **The old calibration transfers:** on the earlier instances SRT-ECT had 46% late at c = 2 (`../rq2-twin-due`); on B2 it
  has 54%. B2 is a little tighter than those instances at the same c.

### Suggested thesis wording (Background, due dates)
"This thesis sets c in the same way: on the benchmark instances (Section ...), with c drawn between 1.75 and 2.5, about
half of the jobs that fall due in the scoring window are late under the reference pair MDD-PTECT (51.8%; 56.9% at
c = 2), and a similar share under SRT-ECT (48.5%)."
