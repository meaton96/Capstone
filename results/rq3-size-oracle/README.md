# rq3-size-oracle: the v5 switching-oracle ceiling at 50, 100 and 200 machines (2026-10-20, user)

Status: **50 / 100 machines done 10-20 16:29 (lab `rq3-size-oracle`, local, 1 h 59 min, 80 / 80);
200 machines (`rq3-size-oracle-m200`, `launch_m200.sh`, 40 tasks, ~80 core-hours) added by the user 10-20, queued
after `rq3-size-catalog`.** Step 1 of the size
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

## Results: 50 and 100 machines (10-20)
B2 test seeds 0-39, window tardiness, vs MDD-PTECT held all episode; < 0 better. MDD-PTECT here equals the grid's on
40 / 40 seeds at every floor. `analysis.out`, `summary.csv`.

| floor | MDD-PTECT mean (k job-s) | switching oracle | hindsight best v5 pair | switching beyond the per-instance pick | oracle saving per instance |
|---|---|---|---|---|---|
| m15 (rq2-rebase-v5) | 145.5 | **−9.42%** [−11.33, −7.61], 40/0 | −4.51%, 32/0 | −5.14%, 39/0 | 13.7 k job-s |
| m50 | 158.2 | **−10.27%** [−13.77, −7.78], 40/0 | −4.57%, 31/0 | −5.97%, 40/0 | 16.2 k job-s |
| m100 | 456.0 | **−3.29%** [−4.65, −2.20], 38/0 | −0.72%, 20/0 | −2.59%, 38/0 | 15.0 k job-s |

All p < 1e-4 (Wilcoxon). 95% seed-bootstrap CIs in brackets.

**Oracle slot use** (share of 900 s slots):

| floor | MDD-PTECT | MDD-PECT | ATC-PTECT | ATC-PECT | SRT-PTECT | SRT-PECT | switches per episode |
|---|---|---|---|---|---|---|---|
| m15 | 20% | 29% | 17% | 17% | 7% | 9% | |
| m50 | 22% | 11% | 31% | 25% | 4% | 7% | 6.0 |
| m100 | 39% | 8% | 23% | 15% | 11% | 5% | 5.5 |

Runtime: ~14 min per instance at m50, ~32 min at m100 (max 48 min).

### Reading
- **Switching headroom does not shrink at 50 machines; it is slightly larger than at 15** (−10.3% vs −9.4%), and
  more of it comes from switching within the episode (−6.0% beyond the per-instance best pair, against −5.1%). So
  `train-size-m50` trains on a regime with at least as much to learn as the final policy had.
- **At 100 machines the relative ceiling falls to −3.3%,** but the oracle saves about as much tardiness per instance as
  on the smaller floors (15.0 k job-s, against 13.7 / 16.2). The percentage shrinks because the floor's tardiness is
  three times larger (456 k job-s), not because there is less to gain in absolute terms. Almost all of it is
  switching within the episode (−2.6% of the −3.3%); picking one pair per instance in hindsight gains only −0.7%.
  `train-size-m100` has a real but small target: on B2 the learned policies realized about a fifth of the oracle
  (−1.8 of −9.4), which here would be about −0.7%.
- **The rules that pay change with size, as the user suspected.** At 15 machines the oracle leans on MDD (50% of slots);
  at 50 machines on ATC (56%), although both ATC pairs are worse than MDD-PTECT when held all episode (+2.6% /
  +4.6%); at 100 machines MDD and ATC share it (47% / 38%), although held all episode the ATC pairs are 23-31% worse. ATC is a situational rule on big floors: very costly as a fixed choice, valuable in some slots.
  The machine rule moves too: PECT gets 55% of slots at 15 machines, 43% at 50, 28% at 100 (PTECT the rest).
- **For the training arms:** the ceiling is the yardstick. T50 has the same room as the final policy had at 15
  machines; T100 has a third of it in relative terms. If T100 ends near 0%, that is close to what the ceiling allows,
  not necessarily a failure to learn.
- **Next:** `rq3-size-catalog` (running) asks whether rules outside v5 add complementarity at size;
  `rq3-size-oracle-m200` follows it.
