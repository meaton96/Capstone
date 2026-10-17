# agv-study-rerun: the AGV physical study on the v5 rules and tardiness (RQ1 / N4, 2026-10-14)

Status: **done** 2026-10-17 07:14 (all v5 sweeps; `analysis.out`, `runs.csv`). It was split into three lab runs on 10-15 22:40:
- `agv-study-rerun`: release (256 / 256) and hold (125 / 128; 3 cells hit the 20 h timeout), 12 workers; done 10-16 20:36.
- `agv-study-layouts-fail`: the 44 layout and 30 failures-on cells (`launch_layouts_fail.sh`), 6 workers; 74 / 74, done 10-16 23:34.
- `agv-study-105` (`launch_105.sh`, 10-16 01:46 on 2 workers, restarted 18:45 on 7; 18 / 18, done 10-17 07:14): the 105-machine cells, decided after
  `../dev-agv105-probe`. The full b2t-105 cells run at 0.73 sim-s per wall-s and take more than 30 h each.
  - b2m-105 runs as designed: 3 isolated, 6 linked.
  - b2t-105 runs as regime **b2tb**, the first 1.5 h load block of the same instance (`make_b2tb.py`; 1,685 of 5,893
    jobs, opening burst included), run to empty, at the same fleets and rules (9 cells).
  - All 18 log job exits (`--loglevel Medium`).
  - Report b2tb as "one load block", not as the 6 h b2t cell.

**10-15: the 105-machine stream was stopped.** The four linked b2t cells (49 / 70 / 84 AGVs, 5,893 jobs) ran 30 h and
were killed by `--timeout-hours 30` with no output. Batch mode logs nothing until a cell ends. The layout and
failures-on cells queued behind them never started. The user stopped the stream (the two 49-AGV linked cells were
killed at start), kept the hold sweep, and moved layouts and failures-on to their own run.

**History.** A first version (10-14 15:39) reused twin-gap-reg and ran flow time with SRT_TECT / SPT_ECT on the old
batch scenarios. The user stopped it after 25 min, with 0 runs finished. The point of the rerun is to measure the AGV
system under the rules and objective the thesis now uses, not only on the current build. Its partial outputs were
removed.

## Why
- The AGV study (fleet, layouts, reservation protocol, tiles) was removed from the thesis on 10-08. All of it was
  pre-audit (`docs/experiments/outdated/E1_E2_findings.md`, `E4_E6_..._0928.md`).
- It also used flow time and rules that are no longer in the action space: SPT_SRWT, LRT_MMUR.
- Demo 1 slide 17 lists the rerun as needed for RQ1. Thesis `sec:exp-agv-study` NEEDED (1) asks for:
  - protocol × fleet;
  - a reduced layout comparison;
  - the tiled / linked floor;
  - the objective and rules chosen to match the rest of the thesis;
  - how often stalls end in a teleport.

## Instances (`agv5_scenarios.py` → `linux_server/BatchConfigs/Scenarios/agv5_*.json`)
B2 (the RL benchmark) with the two parts that clash with a fleet sweep removed:
- **No 2-AGV short-fleet block,** because fleet size is the swept variable.
- **No random warm-up.** Batch runs start empty and run until every job has finished, so tardiness is never censored.

What stays the same:
- 1.5 h load blocks drawn from normal (utilization 0.6-1.2, weight 2) and surge (1.0-1.6);
- TWK due dates with c ~ U[1.75, 2.5];
- 6 h of arrivals;
- 15 machines, failures off.

Regimes:

| regime | what | twin knee5 (mean time in system) |
|---|---|---|
| `b2m` | machine-bound: B2's operation times (mean op 240-480 s) and B2's arrival cap at 7 AGVs | 3-5 AGVs |
| `b2t` | transport-bound: mean op 90-150 s, arrival cap off (same jobs at every fleet). AGVs are 80% busy at 7 | 10-15 AGVs |
| `b2tf` | b2t with machine failures on: Weibull k 1.5, λ 7200 s, repairs ~620 s. Same jobs as b2t | - |
| `b2m` / `b2t` 105 | 7 tiles of D. The tiling block is in the scenario (no `-tiles` flag); isolated = job scope tile, linked = open + pooled AGVs | - |

**Calibration** (`calib_twin.py` → `calib_twin.out`, twin DES-1k, MDD-PTECT, seeds 0-1, fleets 2-25). The
candidate b2t op ranges gave:

| mean op | twin knee5 | AGV busy at 7 AGVs |
|---|---|---|
| 60-120 s | 15 | 0.94 |
| 90-150 s | 10-15 | 0.80 |
| 120-240 s | 7-10 | 0.60 |

90-150 s was picked: AGVs bind at B2's fleet of 7, and the knee falls inside the swept range, so 15-25 AGVs is where
Unity congestion can matter. In the twin, MDD-PECT (no travel term) is 1-7% worse than MDD-PTECT at ≤ 7 AGVs and
0-6% better at ≥ 10.

## Sweeps (`launch.sh`; outputs in `linux_server/Results/<sweep>/`)

| sweep | protocol | grid | runs |
|---|---|---|---|
| `agv-v5-rel` | release | D J C G × b2m s0/s1, b2t s0/s1 × 2 3 5 7 10 15 20 25 AGVs × MDD_PTECT, MDD_PECT | 256 |
| `agv-v5-hold` | hold | D J × the same scenarios, fleets and rules | 128 |
| `agv-v5-layouts` | release | A B E F H I K L M N O × b2m s0, b2t s0 × 7, 15 AGVs × MDD_PTECT | 44 |
| `agv-v5-fail-rel` / `-fail-hold` | release / hold | b2tf s0-s2 × D × 5 7 10 15 20 AGVs × MDD_PTECT | 15 + 15 |
| `agv-v5-tiles` | release | 105 isolated: b2m at 21 / 35 / 49 AGVs (3 / 5 / 7 per tile), b2t at 49 / 70 / 84 (7 / 10 / 12 per tile) × MDD_PTECT | 6 |
| `agv-v5-linked` | release | the same floors linked × MDD_PTECT, MDD_PECT | 12 |

- **105 machines at 15 AGVs per tile:** refused on the linked floor (first launch 16:32). The player stops with exit
  code 4: the parking-lane exit corridors run into the seam bridges and the zone graph check fails. 12 per tile (84)
  is the largest; the old E4S linked runs also used it. The b2t 105-machine cells therefore use 7 / 10 / 12 per tile.
- **Rules:** MDD_PTECT (the thesis's best fixed pair) and MDD_PECT (the same pair without the travel term). The
  contrast asks whether the travel term matters more when AGVs are scarce, congested, or cross tiles.
- **Settings:**
  - player `linux_server/` (5e6e0f87, without `-pyrules`), lane parking, `-baselinedrain`;
  - `-allowunsafefleet` (it only lifts the protocol's fleet bound: hold ≤ 8, release ≤ 15 per 15 machines);
  - `-destrace` on the single-tile floors, with the twin pairing at the end (`twin/`). These runs are also held-out
    data for the N1 congestion model, now on the v5 rules.
- **Provenance:** `PROVENANCE.txt` records git HEAD and the Simulation.dll, des_twin and scenario md5s.
- **Runtime:** two streams of about 28 h each (release + hold on 12 workers; the 105-machine, 15-layout and
  failures-on cells on 4).

## Smoke (`smoke.sh`, lab `dev-agv5-smoke`, 10-14 16:00-16:32)
Three Unity batch cells (`linux_server/Results/dev-agv5-smoke/`), each paired with the twin on its export:

| cell | Unity | twin DES-1k | checks |
|---|---|---|---|
| b2m s0, D, 7 AGVs, MDD_PTECT, release | mean flow 1,786.9 s; 232/232 jobs with due dates, mean tardiness 242.6 s, 37.5% tardy | 1,821.5 s (−1.9%), decisions 1,358 = 1,358 | 0 collisions, 0 censored |
| b2t s0, D, 10 AGVs, MDD_PECT, **hold** | 990.0 s; `reservation_protocol` holdPrevious | 923.1 s (+7.2%), decisions 3,390 = 3,390 | 0 deadlocks, 0 collisions |
| b2tf s0, J, 7 AGVs, MDD_PTECT, release | 38 machine failures; 1,157.5 s | 947.7 s; the twin ran **without** failures (3,390 vs 3,499 decisions) | 0 collisions |

- Due dates load (the export's `des_jobs.json` carries `due`), and results.csv logs tardiness.
- The v5 rules run in batch mode, and the CLI protocol override applies.
- The twin does not replay batch-mode machine failures, so the failures-on sweeps are left out of the twin pairing.
- **Speed:** 27-30 min per 6 h cell at 7-10 AGVs, about 14 sim-s per wall-s. Batch mode paces frames by wall clock; RL
  mode is about 2.9× faster but writes no collision, stall or AGV logs.

## How to read (`analyze.py` → `analysis.out`, `runs.csv`)
The objective is **total tardiness** (1000 job-s), uncensored because every run goes to empty. Deadlocked or
timed-out runs are flagged as `incomplete`. Tables:
1. **Protocol × fleet** on D / J: tardiness, deadlocks, stalls, teleports; the knee per protocol (5% on tardiness, and
   on flow as a check).
2. **Layouts:** D / J / C / G against fleet and the knee per layout; all 15 layouts at 7 / 15 AGVs as % against D.
3. **Failures on:** release vs hold, and the cost of failures on the same jobs.
4. **105 machines:** isolated vs linked tiles, with the share of cross-tile moves.
5. **Safety and liveness per sweep,** including the share of stalls that end in a teleport (the "physical" caveat).
6. **The travel term:** MDD-PECT vs MDD-PTECT by regime and fleet.
7. **Unity vs the free-flow twin:** the gap by fleet band, which is the v5 version of the twin-gap table.

## Results

### Release + hold (`agv-study-rerun`, done 10-16 20:36): 381 of 384 runs
**Missing runs.** Three hold cells hit the queue's 20 h timeout and wrote no result, all b2t_s1 on D: 25 AGVs with
both rules, and 20 AGVs with MDD_PECT. Count them as hold failures. In table 1, hold at b2t / D / 20-25 is a mean over
the seeds that finished. Another 16 hold runs finished with jobs left unfinished (`incomplete`; their tardiness is a
lower bound): b2m D 20 (2) / 25 (4), b2t D 15 (4) / 20 (3) / 25 (2), J 25 (1). No release run is incomplete.

**1. Protocol × fleet (MDD-PTECT, total tardiness in 1000 job-s, mean over the 2 instance seeds).**

| regime / layout | fleet | 7 | 10 | 15 | 20 | 25 |
|---|---|---|---|---|---|---|
| b2m / D | release | 43.9 | 44.6 | 42.8 | 44.2 | 45.2 |
| | hold | 44.7 | 46.9 | **598** | **2,837** | **2,812** |
| b2m / J | release | 47.0 | 43.2 | 46.9 | 41.2 | 43.7 |
| | hold | 46.2 | 44.5 | 40.6 | 42.7 | 44.5 |
| b2t / D | release | 732 | 281 | 236 | 230 | 236 |
| | hold | 893 | 516 | **4,842** | **16,265** | **18,788** |
| b2t / J | release | 535 | 236 | 219 | 217 | 233 |
| | hold | 632 | 284 | 252 | **1,403** | **16,219** |

- **Release is clean at every fleet,** 2-25 AGVs, all four layouts, both regimes: 0 collisions, 0 deadlocks, 0
  incomplete runs, 21 stalls in 4 of 256 runs. Past the knee, tardiness is flat.
- **Hold breaks down at large fleets.** On D it falls apart from 15 AGVs (b2t: ×20 at 15, ×70-80 at 20-25), and on J
  from 20. There are 36,035 stalls in 32 of 125 runs, and 24% of them end in a teleport to parking.
- **Hold is worse even below the breakdown,** on the transport-bound floor: b2t / D +22% at 7 AGVs and +83% at 10.
  On the machine-bound floor the protocols tie up to 10 AGVs.
- **Knee (5% on total tardiness), release:** b2m 3-10 AGVs (machine-bound; little changes past 5), b2t 15 on every
  layout, the same as the twin's 10-15.

**2a. Layouts D / J / C / G × fleet (release).**
- b2m: the layouts are within noise (41-50 k) from 3 AGVs on. This is machine-bound, as in AGVR.
- b2t: **J is best at every fleet.** It is 10-27% below D at 2-10 AGVs (534.7 vs 731.8 at 7, 235.6 vs 281.4 at 10),
  and 1-7% below at 15-25. C is worst at 25 AGVs (309 vs 236).

**2b. All 15 layouts at 7 / 15 AGVs (instance seed 0, % vs D).**
- b2t at 7 AGVs:
  - J −7.0%, B +1.2%;
  - A / F / C / G / H / I / E +3 to +9%;
  - the two-way-perimeter layouts K-O +13 to +40% (O worst).
- b2t at 15 AGVs: every layout is within −2.3 to +6.9% of D.
- b2m: within about 10% at both fleets, except O (+26% at 7).
- So the layout matters only when AGVs are scarce, and the two-way perimeters (K-O) cost the most.

**5. Safety.**
- 0 collisions and 0 deadlocks in every sweep (minimum distance 1.8-2.1 m).
- The stalls are all in hold: 36,035 (8,595 teleports). Under release there are 21 (19 teleports). The teleport
  caveat ("physical") applies to hold at 15+ AGVs, not to release.

**6. The travel term (MDD-PECT vs MDD-PTECT).**
- Under release the two are within about ±5% everywhere. The travel term does not matter once the protocol keeps
  traffic moving.
- Under hold, PTECT is far better where hold breaks down (b2t at 15 / 20 AGVs: PECT +214% / +306%). Steering jobs to
  nearby machines reduces the gridlock.

**7. Unity vs the free-flow twin (DES-1k: Unity's own motion, no interaction).**
- The mean |gap| in mean flow time is ≤0.1% for every release cell and every layout cell.
- Under hold it is 7.9% (b2m / D) and 15-33% (b2t / D and J) at 20-25 AGVs.
- So under release the AGVs on a 15-machine floor barely interact. The twin reproduces Unity, and AGV interaction
  only matters under hold at large fleets.

### Layouts + failures-on (`agv-study-layouts-fail`, done 10-16 23:34): 74 / 74
Layouts are in 2b above. The last fail-hold cell (b2tf_s1, 15 AGVs) took about 17 h and finished inside the 20 h
timeout.

**3. Machine failures on** (b2tf, seeds 0-2, D, MDD-PTECT; total tardiness 1000 job-s, mean over seeds):

| fleet | 5 | 7 | 10 | 15 | 20 |
|---|---|---|---|---|---|
| release | 1,965 | 890 | 588 | 566 | 584 |
| hold | 2,097 | 1,000 | 715 | **9,945** | **3,382** |
| hold stalls (mean) | 0 | 0 | 0.3 | 847 | 606 |

- **Release stays clean with failures on:** 0 collisions, 0 deadlocks, 0 stalls. The knee stays at about 10 AGVs.
- **Hold with failures breaks down at 15 AGVs, as without failures:** ×18 / ×6 vs release at 15 / 20, with 600-850
  stalls per run.
- **What failures cost under release** (same jobs, seeds 0-1): +6-8% at 5 AGVs, +26-43% at 7 and +52-210% at 10-20.
  The cost grows with the fleet, because with ample AGVs the machines are the bottleneck, and failures hit the
  machines. At 5 AGVs transport dominates, and a failed machine matters less.

### 105 machines (`agv-study-105`, done 10-17 07:14): 18 / 18
7 tiles of D, MDD-PTECT unless noted. Total tardiness is in 1000 job-s. "isolated" = jobs and AGVs stay in their
tile; "linked" = open job scope and pooled AGVs, so jobs may cross the seams.

**4. Isolated vs linked tiles.**

| regime | AGVs per tile (fleet) | isolated | linked, MDD-PTECT | linked, MDD-PECT | cross-tile share (linked, PTECT / PECT) |
|---|---|---|---|---|---|
| b2m (machine-bound, full 6 h) | 3 (21) | 508 | **5,475** | 8,827 | 0.55 / 0.85 |
| | 5 (35) | 446 | **197** | 298 | 0.69 / 0.86 |
| | 7 (49) | 484 | **106** | 134 | 0.71 / 0.85 |
| b2tb (transport-bound, first 1.5 h block) | 7 (49) | 1,663 | 5,361 | 6,954 | 0.50 / 0.85 |
| | 10 (70) | 1,332 | 4,271 | 5,372 | 0.62 / 0.86 |
| | 12 (84) | 1,314 | 4,388 | 5,284 | 0.66 / 0.85 |

- **Linking tiles helps a lot on a machine-bound floor with enough AGVs.** b2m linked at 5-7 AGVs per tile is 56-78%
  below isolated: pooling jobs across tiles balances the machines. But with only 3 per tile, linking is 11× worse,
  because more than half the moves cross a seam and the scarce AGVs cannot carry them.
- **Linking hurts on a transport-bound floor at every fleet tested:** b2tb is 3-4× worse linked than isolated, even at
  12 AGVs per tile, the most the linked floor accepts.
- **The travel term matters on linked floors** (unlike on single tiles under release): MDD-PTECT is 20-38% below
  MDD-PECT, and it keeps jobs local (cross-tile share 0.50-0.71 vs 0.85).
- **Clean:** 0 collisions, 0 deadlocks and 0 stalls in all 18 runs.
- **Runtime limit:** the full 6 h b2t-105 cells run at 0.73 sim-s per wall-s and take more than 30 h each
  (`../dev-agv105-probe`). So b2t on 105 machines is the first 1.5 h block only.
