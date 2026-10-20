# rq3-size-oracle: the v5 switching-oracle ceiling at 50 and 100 machines (2026-10-20, user)

Status: **running locally since 10-20 ~14:30 (lab `rq3-size-oracle`, 80 tasks, 16 at a time).** Step 1 of the size
rule sweep; step 2 is `../rq3-size-catalog`; step 3 (the exact oracle for a size-specific portfolio) waits on step 2.

## Why
- The user (10-20): maybe the rules that give switching headroom change with the number of machines.
- The grid (`../rq3-size-grid`) already shows the fixed-rule landscape moving with size: ATC pairs best at 20-30
  machines and worst at 150-200; the per-instance best v5 pair gains 8-10% over MDD-PTECT at 20-30 machines but 0.7%
  at 100. Switching headroom (choosing a pair per slot) had only been measured at 15 machines (−9.4%,
  `../rq2-rebase-v5`).
- CLAUDE.md asks for the switching headroom on the regime a run trains on *before* the run. That was skipped for
  `train-size-m50` / `-m100` (queued 10-19). This is that check, late: it is the yardstick for reading those arms
  (no headroom = nothing to learn; headroom not reached = failure to learn).

## Method (`sizesim.py`, `run.py`, `analyze.py`, `launch.sh`)
- **Oracle:** rq2-rebase-v5's greedy switching oracle, unchanged: 6 v5 pairs, 900 s slots, at each slot the pair
  whose hold-to-end tail is best, frozen code (`../rq2-rebase-v5/code/env`, 6d13412a) and its frozen `sim.py`. Each
  instance replays the episode 144 times (24 slots × 6 pairs).
- **Floors** (`sizesim.py`): the grid's conditions, m50 and m100 (agv-ratio-scale floors, 26 / 52 AGVs on duty, short
  blocks half, B2 load × 0.9, arrival cap off). The adapter points the oracle's simulator at the floor and builds each
  instance through evaluate.py's own generator call with rq3-gen-v5's fleet wrapper, as the grid does.
- **Checks:**
  - MDD-PTECT through the adapted simulator equals the grid's MDD-PTECT: m15 / m30 / m50 / m100 / m200 seed 0, m30 s3,
    m50 s1 / s6, m100 s1 / s3 / s5, m200 s1, all within float32 rounding of evaluate.py's metrics (relative ~1e-8).
    `launch.sh` reruns the check (seeds 0-2) per floor before the oracle starts; `analyze.py` checks all 40 seeds.
  - The analysis on rq2-rebase-v5's 15-machine oracle reproduces −9.42% (40 / 40 seeds match the grid).
- **Cost:** the oracle's simulator plays rules directly, without building the policy's observations, so it is much
  faster than evaluate.py: ~4.5 s per m50 episode and ~15 s per m100 episode (vs ~21 s / ~92 s through
  evaluate.py), ~50 s at m200. About 31 core-hours in all.
- **Seeds:** B2 test seeds 0-39.

## Analysis
Per floor: the oracle and the hindsight best v5 pair vs MDD-PTECT (pooled, seed-bootstrap CI, Wilcoxon), switching
beyond the per-instance pick (oracle vs hindsight best pair), the oracle's slot use, and the fixed pairs.

15-machine reference (rq2-rebase-v5, same analysis): oracle −9.42% [−11.33, −7.61] (40/0), hindsight best pair
−4.51%, oracle vs hindsight best pair −5.14%; slot use MDD-PECT 29%, MDD-PTECT 20%, ATC-PTECT 17%, ATC-PECT 17%,
SRT-PECT 9%, SRT-PTECT 7%.

## Results
Pending.
