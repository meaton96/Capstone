# dev-agv105-probe: how fast does the player run a 105-machine floor? (2026-10-15, user)

Status: **done** 2026-10-16 01:42 (lab `dev-agv105-probe`, 2 h 58 min; both cells finished). `analysis.out`.

## Why
agv-study-rerun's 105-machine stream ran four linked b2t cells (49 / 70 / 84 AGVs, 5,893 jobs, run to empty) for
30 h. All four hit `--timeout-hours 30` and were killed with no output. Batch mode writes nothing until a cell ends,
so nobody knew how far they got. The 15-layout and failures-on cells queued behind them never started.

The user (10-15) stopped that stream, moved layouts and failures-on to `agv-study-layouts-fail`, and asked for a
short timed probe before deciding what to do with the 105-machine cells (6 isolated, 12 linked):
- run them on the cluster with a longer limit;
- shorten the horizon;
- or drop them from the v5 study.

## Method (`make_scenarios.py`, `launch.sh`, `analyze.py`)
- **Scenarios:** the first hour of arrivals of two agv-study-rerun scenarios, run to empty:
  - `agv5_b2t_t7open_arr1h_s0` at 84 AGVs: linked, pooled. This is the heaviest of the killed cells. Its first hour
    has 1,186 of 5,893 jobs, including the opening burst.
  - `agv5_b2m_t7iso_arr1h_s0` at 21 AGVs: 7 isolated tiles, the lightest 105-machine cell (402 of 2,021 jobs).
- **Flags:** the same as launch.sh's 105-machine cells (`-baselinedrain -parking lane -allowunsafefleet
  -reservation releasePrevious`, MDD_PTECT), plus `--loglevel Medium`. Medium stamps every job exit in sim.log with
  wall time, so progress is visible even if a cell times out.
- **Run:** 3 h timeout, one worker per cell, beside the 18 AGV-study players. That load is the realistic one; a cell
  alone would run faster.
- **Analysis:** jobs exited vs wall time, and sim makespan / wall time if a cell finishes. From that it projects the
  full cell. The baseline is a 15-machine b2t cell of agv-study-rerun: 22,104 sim-s in 48 wall-min = 7.7 sim-s per
  wall-s, under the same kind of load.

## Results (beside 18 other players; 15-machine reference 7.7 sim-s per wall-s)

| cell | jobs | wall time | makespan | sim-s per wall-s | job exits per wall-hour | collisions / stalls |
|---|---|---|---|---|---|---|
| b2m, 7 isolated tiles, 21 AGVs (first hour) | 402 | 0.53 h | 7,140 s | **3.7** | about 800 | 0 / 0 |
| b2t, linked, 84 AGVs (first hour) | 1,186 | 2.97 h | 7,835 s | **0.73** | 350-390, steady | 0 / 0 |

**Full-cell projection:**
- **b2m-105** (2,021 jobs, 20,854 s of arrivals): at 3.7 sim-s per wall-s the arrivals take about 1.6 h, and the
  cell is done in about 2-4 h. That is feasible locally, isolated and linked (linked is pooled, so somewhat slower).
- **b2t-105** (5,893 jobs, 21,390 s of arrivals):
  - The arrivals alone take at least 8.1 h at 0.73 sim-s per wall-s. At the probe's 370 exits per hour, all
    5,893 jobs take about 16 h.
  - The full cell holds far more jobs on the floor than this probe: 6 h of arrivals with surges up to
    utilization 1.6. Per-frame cost grows with that, and the four killed cells did not finish in 30 h.
  - Expect more than 30 h per cell locally. That is not feasible for 9 cells before the 10-25 freeze, not even on
    the 20 local slots.

## Reading
1. **The player is not stuck on 105 machines, just slow.** The linked floor at 84 AGVs moves jobs steadily with no
   collisions and no stalls. It simulates about 10× slower than a 15-machine cell, and the isolated b2m floor about
   2× slower.
2. **The killed cells were slow, not deadlocked** (as far as the first hour shows).
3. **Options for the 105-machine part of the AGV study** (user's call):
   - run b2m-105 as designed (3 isolated + 6 linked cells, a few hours each);
   - for b2t-105, shorten the horizon. One 1.5 h load block of arrivals, run to empty, is about 4-5 h per cell at
     84 AGVs, so the 9 cells fit;
   - or run full b2t-105 on the cluster with a multi-day limit, if the partition allows it;
   - or drop b2t-105.
