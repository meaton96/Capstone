# rq3-alpha-transport: a size / transport term for the α law (2026-10-28, user: "design the size / transport term in the law, with the pair heads restrained above ~50 machines, with pre-declared criteria, and set it up in the lab")

Status: **done 10-28** (lab `rq3-alpha-transport`, 20:19-22:38, 840 tasks). Stage 1 of 2. Criteria pre-declared
10-28, before any result. **Gate F1 PASS** with L2 (x = travel / op time per AGV per machine), so stage 2
(`../eval-alpha-transport-size`) started 22:38.

## Why
- **`../eval-alpha-only-size` (10-28)** settled why the v7 law + offset policy (off-het) gains nothing at 200 machines.
  - With the pair fixed to MDD-PTECT, the offset head alone gives −4.7% vs MDD-PTECT there. That is still 6.9 points
    short of fixed α 0.75, so rule N said both: restrain the pair heads, and add a size / transport term to the law.
  - The head plays about the same α (≈ 1.8) at every size. On homogeneous floors the v7 law, max(0, 3 − 6σ), is α 3
    whatever the size. So nothing in the policy brings α down on large floors.
- **What is known about α on large floors:**
  - `../rq3-alpha-size` (15-100 machines, 0.52 AGVs per machine): α* falls only from about 2-4 to about 1. A size
    term τ = travel / op time missed its pre-declared margin (0.81 of the 1.0 points of leave-one-cell-out regret).
    α* was lowest in short-fleet blocks.
  - `../rq3-size-catalog`: MDD-TECT (α 0) is −19.0% vs MDD-PTECT at 200 machines.
  - `../rq3-size-headroom` (fleets 0.35 / 0.52 / 0.80 AGVs per machine): MDD-TECT pays where transport binds. At 100
    machines the fleet decides (−13% / −2% / +22%). At 200 machines the ~700 m trips decide (−12 to −20% at every
    fleet).
- **So the term should track how hard transport binds,** which depends on trip length and fleet. This stage fits it on
  fixed-α curves that span both: 15-200 machines, and fleets of 0.35 to 0.80 AGVs per machine.

## Design (`run.py`, `launch.sh`, `analyze.py`)
- **Simulator:** the frozen v5 simulator (`../rq2-rebase-v5`, 6d13412a + sim.py), MDD-PTECT at α ∈ {0, 0.25, 0.5,
  0.75, 1, 1.5, 2, 3, 4} held all episode (α 0 = TECT exactly), window tardiness. B2 window, homogeneous floors
  (σ 0), the size grid's per-machine scaling (B2 load × 0.9, arrival cap off).
- **Cells:** `m<M>_<fleet>`.
  - r052 is the grid's condition (round(0.52 M) AGVs on duty, short blocks half).
  - r035 / r080 are the same floor with round(0.35 / 0.80 M) AGVs, as `../rq3-size-headroom`.
  - m15_b2 is B2 itself.

  | role | cells | seeds | source |
  |---|---|---|---|
  | fit (14) | m15_b2, m20 / m30 / m50 / m75 / m100 r052 | fitting 2000-2039 | curves `../rq3-alpha-size` (s0 cells), AGV utilization rerun here |
  | | m150 / m200 r052; m50 / m100 / m200 r035 and r080 | fitting | here |
  | held out (2) | m75_r035, m150_r080 | test 0-39 (statistics from fitting seeds) | here |
  | stage-2 reference | m200_r052, m100_r035, m100_r080 | test 0-39 | here (m50-m100 r052 test curves: `../rq3-alpha-size`) |

- **Floor statistics per cell** (means over its fitting instances; all knowable from the floor or a plant KPI):
  - τ = mean travel time from a machine (or the input) to a candidate machine of each type / mean op time
    (`../rq3-alpha-size` run.stats);
  - a = AGVs on duty per machine (full fleet);
  - u = AGV utilization over the window under MDD-PTECT at α 3 (busy / on-duty AGV-seconds).

## Analysis (`analyze.py`, pre-declared 10-28)
- **Candidate laws** (homogeneous floors, so v7's spread factor is 3):
  - L0: α 3 everywhere (the v7 law, no size term);
  - L1 / L2 / L3: α = 3 h(x), h(x) = clip((x1 − x) / (x1 − x0), 0, 1), with x0 ≥ B2's x, so the law is exactly v7's on
    B2. The variable is L1 x = τ, L2 x = τ / a (travel per op per AGV), or L3 x = u.
  - Off the homogeneous grid the law reads α = min(4, max(0, 3 − 6σ) × h(x) × offset). The product with the spread
    factor is a design choice; it is not tested here (σ 0 everywhere).
- **Regret** of a cell at α: the % its pooled fitting-seed tardiness at α exceeds that at its best grid α. A non-grid
  α is interpolated linearly between the two neighbouring grid values; the law plays its α exactly in stage 2.
- **Fit:** grid search over (x0, x1 − x0), minimizing the mean regret over the 14 fit cells (equal weight). Ties within
  0.01 points go to the gentlest slope, then the median knot.
- **Selection:** leave-one-cell-out regret (refit without the cell, score the cell). Take the lowest, then the simplest
  within 0.5 points, in the order L1 (geometry) < L2 (geometry + fleet) < L3 (a measured KPI).
- **Gate F1 (go for stage 2):** the selected law's leave-one-cell-out regret is at least 1.0 point below L0's.
  `law.json` records the law, F1, and each cell's x, h, α and fitting-seed best α. Stage 2's launcher refuses to run
  without F1 PASS.
- **Pipeline test (10-28, synthetic curves planted on the 8 new cells with L2, x0 0.20 / x1 0.40, plus the real
  reused curves):** recovered x0 0.205 / x1 0.407, selected L2 (L3 tied: its synthetic u was linear in τ / a), F1 PASS.

## Checks (10-28, scratchpad, before queueing)
- **util reruns reproduce `../rq3-alpha-size`:** m100_r052 s2000 and m15_b2 s2000 at α 3 are identical, and so are
  the statistics.
- **The r035 loader reproduces `../rq3-size-headroom`:** m100_r035 s0 at α 3 / α 0 equals its MDD-PTECT / MDD-TECT
  (39.471 / 38.053). The window AGV utilization is 0.741, against 0.735 as its slot mean.
- **Simulator = harness:**
  - m200_r052 s0 at α 3 equals `../eval-alpha-off-size`'s MDD-PTECT (158.078);
  - `../rq3-alpha-size`'s m100 seeds 0 / 1 at α 1.5 / 0.75 equal `../eval-alpha-off-size`'s alpha group;
  - a law-α run equals the harness's forced-×1 alpha-only policy (`../eval-alpha-transport-size` checks).
- **Timing:** one m200 episode takes ~32 s, one m100 r035 episode ~7 s.

## Cost
840 tasks: 400 curves on fit seeds, 200 curves on test seeds, 240 util runs. About 25-30 core-h, so ~2 h on 20 slots;
the longest task is an m200 r035 curve, ~10-15 min.

## Run
`launch.sh [W]` runs `tasks.txt` (`run.py list`), W at a time; finished seeds are skipped.

## Outputs
- `out/<cell>/s<seed>.json`: tardiness per α (total and per regime block), floor statistics, AGV utilization;
- `out/<cell>/law/s<seed>.json`: stage 2's law runs;
- `analysis.out`, `alpha_curves.csv`, `cells.csv`, `fit.json`, `law.json`.

## Results (10-28, `analysis.out`, `law.json`, `fit.json`, `alpha_curves.csv`)

**Checks.** All 840 tasks are complete. Each fit cell has its statistics from 40 fitting instances and its curve on 40
fitting seeds. The util reruns of the six reused cells equal `../rq3-alpha-size` at α 3 on 40 / 40 seeds.

**The α curves** (fitting seeds, % vs α 3, MDD-PTECT; α\* = best grid α):

| cell | τ | a | τ / a | AGV util | α\* | α 0 | α 1 | α 2 | α 4 |
|---|---|---|---|---|---|---|---|---|---|
| m15_b2 | 0.046 | 0.47 | 0.099 | 0.48 | 2 | +24.7 | +4.4 | −1.5 | −1.5 |
| m50_r052 | 0.100 | 0.52 | 0.191 | 0.59 | 1.5 | +28.6 | +3.4 | −0.1 | +1.2 |
| m100_r052 | 0.124 | 0.52 | 0.239 | 0.70 | 1 | +7.8 | −2.1 | −2.1 | +2.3 |
| m150_r052 | 0.161 | 0.52 | 0.310 | 0.79 | 0 | −13.4 | −6.9 | −2.9 | +2.1 |
| m200_r052 | 0.187 | 0.52 | 0.360 | 0.89 | 0 | −15.8 | −8.2 | −3.5 | +2.1 |
| m50_r035 | 0.100 | 0.36 | 0.276 | 0.80 | 0.5 | +4.3 | −1.3 | −0.4 | +2.4 |
| m50_r080 | 0.100 | 0.80 | 0.124 | 0.39 | 4 | +39.1 | +5.7 | +1.4 | −0.3 |
| m100_r035 | 0.124 | 0.35 | 0.355 | 0.91 | 0 | −13.2 | −6.7 | −1.9 | +1.7 |
| m100_r080 | 0.124 | 0.80 | 0.155 | 0.47 | 3 | +30.2 | +5.1 | +1.1 | +2.4 |
| m200_r035 | 0.187 | 0.35 | 0.534 | 0.98 | 0 | −15.2 | −6.7 | −2.6 | +1.6 |
| m200_r080 | 0.187 | 0.80 | 0.234 | 0.64 | 1 | +1.7 | −2.9 | −1.7 | +1.7 |

(m20 / m30 / m75 r052 in `analysis.out`: α\* 4, 4 and 2.)

- **The fleet decides as much as the size.** At 100 machines the same floor wants α 0 with 35 AGVs, α 1 with 52 and
  α 3 with 80 (α 0 is −13.2%, +7.8% and +30.2% against α 3). At 200 machines α 0 or near it wins at every fleet but the
  largest (m200_r080 α\* 1, a flat −2.4..−2.9% from α 0.5 to 1.5).

**Fits** (mean regret over the 14 fit cells, in sample / leave-one-cell-out):

| law | x | x0 | x1 | in sample | leave-one-out |
|---|---|---|---|---|---|
| L0 | α 3 everywhere (v7) | | | 5.80% | 5.80% |
| L1 | τ | 0.103 | 0.137 | 1.78% | 2.71% |
| L2 | τ / a | 0.175 | 0.297 | 0.51% | **1.26%** |
| L3 | AGV utilization at α 3 | 0.477 | 0.865 | 0.85% | 1.29% |

- **Selection:** L2 has the lowest leave-one-out regret; L3 is within 0.5 points but is the less simple law under the
  pre-declared order, so **L2 is selected**.
- **Gate F1:** 1.26% against L0's 5.80%, 4.5 points better (needs ≥ 1.0). **PASS.**
- **L1 fails on the fleets.** τ alone cannot tell m100_r035 (α\* 0) from m100_r080 (α\* 3), which share a floor:
  its leave-one-out regret is 12.7% and 11.3% there. Dividing by AGVs per machine fixes both (0.0%).
- **Where L2 misses:** m50_r035 (leave-one-out 5.7%; without it the knot moves and the law plays α 0 where the curve
  is flat from 0.5 to 2, α 0 +4.3%) and m150_r052 (3.5%; without it the law keeps α > 0 there). Both sit on the slope
  of the hinge, so the knots are pinned by few cells.

**The selected law:** α = min(4, max(0, 3 − 6σ) × h × offset), h = clip((0.297 − x) / (0.297 − 0.175), 0, 1),
x = (mean travel time to a candidate machine / mean op time) / (AGVs on duty per machine). It is v7's law for x ≤ 0.175
(B2 0.099) and α 0 (MDD-TECT) for x ≥ 0.297.

| cell (stage 2) | x | h | law α (no offset) | α\* (fitting seeds) |
|---|---|---|---|---|
| m50_r052 | 0.191 | 0.868 | 2.60 | 1.5 |
| m75_r052 | 0.221 | 0.628 | 1.88 | 2 |
| m100_r052 | 0.239 | 0.477 | 1.43 | 1 |
| m200_r052 | 0.360 | 0 | 0 | 0 |
| m100_r035 | 0.355 | 0 | 0 | 0 |
| m100_r080 | 0.155 | 1 | 3 | 3 |
| m75_r035 (held out) | 0.331 | 0 | 0 | test seeds, stage 2 |
| m150_r080 (held out) | 0.202 | 0.784 | 2.35 | test seeds, stage 2 |

**Reading.** The size effect on α is transport load per vehicle: trip length over op time, divided by the fleet.
The plain size term of `../rq3-alpha-size` missed its margin because it could not see the fleet. τ / a needs only the
layout, the op times and the fleet, all known before the shift. AGV utilization (L3) does as well and could be read
from a plant KPI, which matters for the online version. Stage 2 tests whether the policy keeps this when it plays the
law with its offset head, and whether the law transfers to the two held-out cells (criterion H).
