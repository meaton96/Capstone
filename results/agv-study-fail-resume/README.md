# agv-study-fail-resume: the AGV study's failures-on cells under preempt-resume (2026-10-28, user)

Status: **done 10-30 06:24** (lab, local; started 10-29 08:06, 22 h 18 min including the user's pause 10-29
21:18-23:09): 29 of 30 cells. The hold cell `agv5_b2tf_s1` at 20 AGVs hit the 20 h wall timeout. Release stays clean
with failures, its knee is at 10 AGVs, and resume lowers its tardiness in all 15 cells (mean −6.5%). The claims of
`Physical_Floor.tex` §exp-agv-failures hold as written; the numbers to quote are in Results.

## Why

The failure model changed on 10-28 from restart to preempt-resume (`results/dev-failure-resume/README.md`). The AGV
study's machine-failure section (`Physical_Floor.tex` §exp-agv-failures: release stays clean with failures, knee near
10 AGVs, the cost of failures grows with the fleet) rests on agv-study-rerun's 30 failures-on cells, which ran with
restart. The other 446 runs of the study are failure-free and stand.

## Setup

agv-study-rerun's `agv-v5-fail-rel` / `agv-v5-fail-hold` cells, unchanged except the player (`launch.sh`, from its
`launch_layouts_fail.sh`): `agv5_b2tf_s0-s2` (b2t's jobs with machine failures: Weibull k 1.5, λ 7,200 s, repairs
~620 s, Unity's native samplers), layout D, 5 / 7 / 10 / 15 / 20 AGVs, MDD_PTECT, `-baselinedrain -parking lane
-allowunsafefleet -destrace`, 20 h timeout. Release first (`agv-study-fail-resume-rel`, 15 cells), then hold
(`agv-study-fail-resume-hold`, 15 cells; under restart the 15-AGV hold cells took up to ~17 h). Output
`linux_server/Results/agv-study-fail-resume-{rel,hold}/`. `launch.sh` refuses to start unless `linux_server/` holds
the resume player (Simulation.dll `22891261...`), and writes `PROVENANCE.txt` and `DONE`.

## Analysis

`analyze.py` (agv-study-rerun's loader) → `analysis.out`: tardiness by fleet for release / hold, restart vs resume;
failures, deadlocks, collisions and stalls; the cost of failures under release against the failure-free b2t runs on
the same jobs (seeds 0-1), restart vs resume. Under restart (agv-study-rerun): release 1,965 / 890 / 588 / 566 / 584
ks at 5-20 AGVs, hold ×18 / ×6 vs release at 15 / 20 AGVs, cost of failures +6-8% at 5 AGVs and +26-210% at 7-20.

## Results (10-30)

### Checks

- Player and inputs: `PROVENANCE.txt` has Simulation.dll `22891261...` (the resume player), git HEAD 09f53cbb, the
  same `agv5_b2tf_*` scenarios as agv-study-rerun.
- Release: 15 of 15 cells, 0 collisions, deadlocks, stalls and incomplete runs, as under restart.
- Hold: 14 of 15 cells. `agv5_b2tf_s1/D/agv20` was killed at 1,200 min of wall time (the queue's 20 h timeout counts
  wall time, and the run was paused 1 h 51 min on 10-29, so it had about 18.1 h of compute; under restart the same cell
  took 461 min). Not rerun: every other hold run at 15-20 AGVs is incomplete under both models, so one more incomplete
  hold run would not change what this section reports. The hold 20-AGV resume mean below is over seeds 0 and 2.

### Release: the section's claims

Total tardiness (1,000 job-s), mean over seeds 0-2, and resume against restart (mean of the per-seed changes):

| AGVs | restart | resume | resume vs restart |
|---:|---:|---:|---:|
| 5 | 1,965 | 1,905 | −2.8% |
| 7 | 890 | 829 | −6.7% |
| 10 | 588 | 524 | −9.7% |
| 15 | 566 | 524 | −7.0% |
| 20 | 584 | 547 | −6.1% |

All 15 cells are lower under resume (−0.7 to −17.0%, mean −6.5%), as expected when interrupted work is no longer
thrown away. The knee is at 10 AGVs (524 ks at 10 and 15, 547 at 20); under restart the curve flattened between 10
and 15 (588 → 566), so "knee near 10 AGVs" holds under both.

Cost of the failures against the failure-free b2t runs on the same jobs (agv-v5-rel, seeds 0-1):

| AGVs | inst 0 restart | inst 0 resume | inst 1 restart | inst 1 resume |
|---:|---:|---:|---:|---:|
| 5 | +7.5% | +6.8% | +5.7% | +3.7% |
| 7 | +42.7% | +36.2% | +26.0% | +18.4% |
| 10 | +61.7% | +51.1% | +126.2% | +92.3% |
| 15 | +51.8% | +47.0% | +179.3% | +177.0% |
| 20 | +57.2% | +50.1% | +209.8% | +196.9% |

Resume cuts the cost of failures by 1-34 points (most at 10 AGVs) and keeps its shape: small at 5 AGVs (+4-7%), where
transport is the bottleneck and a failure mostly overlaps waiting for an AGV, and large from 7 AGVs on (+18-197%),
where machines are the bottleneck and every repair delays the queue behind it. "The cost of failures grows with fleet
size" holds; instance 0 levels off at about +50% from 10 AGVs, instance 1 keeps growing to +197%.

### Hold

| AGVs | restart | resume |
|---:|---:|---:|
| 5 | 2,097 | 2,053 |
| 7 | 1,000 | 946 |
| 10 | 715 | 755 |
| 15 | 9,945 | 5,444 |
| 20 | 3,382 | 15,814 (2 seeds) |

At 5-10 AGVs hold under resume is 0-7% below restart on 8 of 9 cells; seed 2 at 10 AGVs is +29% (7 stalls against 1).
At 15-20 AGVs every hold run is incomplete under both models (resume 208-2,159 stalls per run, restart 157-1,485), and
tardiness spreads 13-fold across seeds under resume (1,047-20,007 ks) and 400-fold under restart (55-23,272 ks). The
restart / resume difference there is which seeds fall into a stall cascade, not the failure model. Hold against
release is ×10 at 15 AGVs and ×29 at 20 under resume (×18 and ×6 under restart); quote the breakdown (every run
incomplete, hundreds of stalls), not the multiple. Hold runs that cascade also collect more failures (up to 192 per
run against 38-67 under release) because they last longer in calendar time.

### For the thesis

`Physical_Floor.tex` §exp-agv-failures says that with machine failures release stays clean with its knee near 10 AGVs
while the cost of failures grows with fleet size. That holds under resume without change. The numbers to quote now
come from here: release 1,905 / 829 / 524 / 524 / 547 ks at 5 / 7 / 10 / 15 / 20 AGVs, cost of failures +4-7% at 5
AGVs and +18-197% at 7-20. They replace agv-study-rerun's restart numbers for the failures-on cells (+6-8% and
+26-210%, `results/agv-study-rerun/README.md`); the study's 446 failure-free runs stand.
