# agv-ratio-unity: does Unity congestion move the twin's fleet knee? (v5 rules, 2026-10-14)

Status: **done** 2026-10-15 01:27 (local, 4 workers, 8 h 55 min, 16 / 16; lab `agv-ratio-unity`). `analysis.out`,
`agv-ratio-unity_runs.csv` (DES-1k pairing). The first version (SRT_TECT, no due dates) was cancelled after 25 min with
0 runs done, when the AGV work moved to the v5 rules.

## Why
- The twin's free-flow fleet knee is a lower bound. Thesis `sec:exp-agv-study` NEEDED (2) and Demo 1 slide 17 ask
  for the Unity congestion check, which has never been run.
- On the v5 rules the knee comes from `../agv-ratio-twin-v5`: MDD-PTECT with due dates.

## Method (`make_scenarios.py`, `launch.sh`, `analyze.py`)
- **Instances:** the v5 twin sweep's own instances, built exactly as its `twin_point_v5.py` builds them. Randomized
  generator, M/5 machines per type, arrival cap off, 6 h, failures off, TWK due dates c ~ U[1.75, 2.5].
- **Cells:** M = 50 / 100, util 0.85, mean op 240 / 480 s, seeds 0 / 1, MDD_PTECT, layout D, release, lane parking.
- **Fleets:** for each cell, the twin's knee5 (mean over 3 seeds, rounded up to a fleet the twin ran) and the twin
  fleet nearest 1.5× it. `make_scenarios.py` reads them from `../agv-ratio-twin-v5/knees.csv` and writes
  `cells.txt`.
- **Player:** `linux_server/` (5e6e0f87), `-destrace`, so each run is also paired with DES-1k on its own export.
- **Read:**
  - the premium: Unity against the twin at the same fleet, on mean time in system and on total tardiness;
  - the knee test: Unity at the twin knee within 5% of Unity at 1.5× the knee means the twin knee holds in Unity.
    Otherwise congestion moves the knee up;
  - AGV busy and route-wait fractions, stalls and teleports.

## Results (util 0.85, MDD-PTECT, layout D, release)

**Safety:** 0 collisions, 0 deadlocks, 0 stalls, 0 teleports and 0 censored jobs in all 16 runs, up to 100 machines
and 70 AGVs.

| M | mean op | seed | fleet: twin knee5 / ~1.5× | Unity vs twin sweep, mean time in system | Unity at the knee vs at 1.5× (Unity / twin) | same, total tardiness (Unity / twin) | Unity vs DES-1k on its own export | AGV busy at the knee |
|---|---|---|---|---|---|---|---|---|
| 50 | 240 s | 0 | 21 / 30 | +1.5 / +1.1% | +1.7 / +1.3% | +15 / +7% | +3.4 / +2.6% | 0.72 |
| 50 | 240 s | 1 | 21 / 30 | +2.7 / +2.5% | +1.9 / +1.7% | +20 / +27% | +4.2 / +4.2% | 0.75 |
| 50 | 480 s | 0 | 10 / 16 | −1.1 / −0.7% | +1.1 / +1.5% | +8 / −11% | −0.3 / +0.2% | 0.62 |
| 50 | 480 s | 1 | 10 / 16 | −1.5 / −0.1% | +2.8 / +4.2% | +13 / +17% | +1.0 / +1.1% | 0.76 |
| 100 | 240 s | 0 | 45 / 70 | **+21.3** / +3.5% | **+19.9** / +2.3% | **+200** / +17% | **+27.2** / +5.5% | 0.75 |
| 100 | 240 s | 1 | 45 / 70 | **+7.0** / +2.8% | **+7.1** / +2.9% | **+60** / +33% | **+12.9** / +4.8% | 0.59 |
| 100 | 480 s | 0 | 25 / 40 | −3.4 / −2.5% | +0.5 / +1.3% | −19 / +24% | +0.5 / +0.2% | 0.64 |
| 100 | 480 s | 1 | 25 / 40 | −2.6 / −1.4% | +1.3 / +2.5% | +1 / +244% | +1.5 / +1.6% | 0.70 |

The "vs twin sweep" columns compare against the agv-ratio-twin-v5 run at the same fleet, seed and rule, on an M-bay
floor; Unity's floor has one bay per AGV, which is why some differences are negative. The DES-1k column replays each
Unity run's own export (same floor and jobs).

## Reading
1. **The twin's fleet knee holds in Unity in 6 of 8 cells:** both loads at 50 machines and the long operations
   (480 s) at 100 machines.
   - Unity at the twin knee is within 0.5-2.8% of Unity at 1.5× the knee, as in the twin.
   - Unity's mean time in system is within ±3.5% of the twin's.
   - Congestion is small there: trips are 2-8% longer than free flow, and zone waits are 0.7-3 s per leg.
2. **It fails where transport is densest:** 100 machines with 240 s operations, about 1,500 trips per hour.
   - At the twin's knee (45 AGVs), Unity's mean time in system is 7-21% above the twin's, and 7-20% above Unity's
     own value at 70 AGVs.
   - Trips are 13-16% longer than free flow, and zone waits are 4-5 s per leg.
   - The real knee there lies between 45 and 70 AGVs. The free-flow fit (≈ 0.080·M^1.32·…) underestimates the fleet
     by a congestion premium that appears only at about 100 machines with short operations.
3. **On tardiness the fleet need is higher in both simulators, and congestion amplifies it.** In the twin, total
   tardiness at the knee is 7-33% above that at 1.5× the knee in most cells (the tardiness knee is 10-50% later,
   agv-ratio-twin-v5). In Unity at 100 machines / 240 s it is 60-200% above. The late tail is what congestion hits.
4. **For the thesis (N4 / RQ1, AGV system half):**
   - A fast free-flow DES is a good fleet-sizing tool up to about 50 machines, and on floors with long operations.
   - On large floors with heavy transport (100 machines, short ops), the physical simulation is needed: the
     free-flow knee is 7-21% optimistic on flow and much more on tardiness.
   - This is the regime N1 (a congestion-corrected twin) targets.
   - Collision-free, deadlock-free and teleport-free at every point.
