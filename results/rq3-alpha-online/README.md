# rq3-alpha-online: an online α law, h from the AGVs on duty in each regime (2026-10-29, user: "design the online α experiment (h from the on-duty fleet per regime, plus the spread law) with pre-declared criteria, and set it up in the lab")

Status: **done 10-30 04:45 (lab `rq3-alpha-online`, 1,000 / 1,000 tasks; paused 1 h 51 min on 10-29 at the user's
request). G1 PASS, selected law O1: today's knots read with the AGVs on duty, nothing refit** (Results). Stage 1 of 2.
Criteria pre-declared 10-29, before any result. Stage 2 (`../eval-alpha-online`) started 10-30 04:45.

## Why
- **`../eval-alpha-transport-size` (10-29) passed P1-P6.** The law α = min(4, max(0, 3 − 6σ) × h(x) × offset) with
  x = τ / a and a size gate on the pair heads is paper B's α policy across sizes and fleets.
  - Rule N named the next step: the online version, with h read from the fleet on duty in each regime, plus a Unity
    check.
  - Plan §3.1 (online α) had the same item.
- **Today's law reads a from the full fleet, so it plays one α per floor.** The fleet halves in short-fleet blocks
  (B2: 7 → 2 AGVs), so transport per vehicle doubles or more. The evidence that α should drop there:
  - `../rq3-alpha-size` (per-block attribution, fixed-α runs): short-fleet α\* 0.25-0.5 at 75-100 machines, against
    1.5-2 in the other blocks;
  - `../rq3-size-portfolio`'s exact map rule at 100 machines: MDD-TECT (α 0) while the fleet is short, −12.2%;
  - the v6 / v7 heads already lower α in short-fleet blocks (`../eval-alpha-offset`).
- **What per-block attribution cannot say:** what α to play in each regime when the two are chosen separately.
  Tardiness from a short block carries into the next block. So this stage plays every pair (α_full, α_short) on each
  instance and fits the online law on those surfaces.
- **Plus the spread law:** the law multiplies v7's spread factor by h. That product was never tested off σ 0
  (`../rq3-alpha-transport` "design choice"). Here the spread floors are held-out cells.

## Design (`run.py`, `launch.sh`, `analyze.py`)
- **Simulator:** the frozen v5 simulator (`../rq2-rebase-v5`, 6d13412a + sim.py), MDD-PTECT (quad), window tardiness,
  B2 window. As in `../rq3-alpha-transport`.
- **A pair (α_full, α_short):**
  - The simulator plays α_full while the whole fleet is on duty (normal and surge blocks) and α_short while fewer AGVs
    are (short-fleet blocks).
  - TwinConfig.ptime_alpha is set before every agent decision from the on-duty count at that moment (the engine's
    `_on_duty` rule). Stage 2's policy harness does the same.
  - The diagonal (a, a) is the earlier fixed-α run of the same instance (checked below).
  - Grid on both axes: α ∈ {0, 0.25, 0.5, 1, 2, 3, 4}, 49 pairs (stage 1's grid less 0.75 and 1.5).
- **Cells:**

  | role | cells | seeds |
  |---|---|---|
  | fit (13) | m15_b2; m20 / m30 / m50 / m75 / m100 / m150 / m200 r052; m50 / m100 r035; m50 / m100 / m200 r080 | fitting 2000-2039 |
  | held out (6) | m75_r035, m150_r080 (`../rq3-alpha-transport`'s held-out pairs); m15_s1, m50_s1, m100_s0.5, m100_s1 (speed spread, grid fleet, `../rq3-alpha-size`'s spread floors) | test 0-39 (statistics from fitting seeds) |

  - `../rq3-alpha-transport`'s fit cells less m200_r035. Its x is 0.53 at the full fleet and 1.07 at the short one,
    so every candidate law with x1 < 0.53 plays α 0 there. Its fixed-α curve falls monotonically to α 0, and it would
    cost a third of the run. `analyze.py` flags a selected law with x1 above 0.534.
  - About 80% of instances have at least one short-fleet block (a quarter of block time).
- **Floor statistics per cell** (fitting instances): τ = mean travel time to a candidate machine / mean op time
  (`../rq3-alpha-size` run.stats), a_full and a_short = AGVs on duty per machine in full / short blocks (B2: 7 and
  2 of 15; the grid: round(r M) and half that), σ the speed spread.

## Analysis (`analyze.py`, pre-declared 10-29)
- **Regret** of a cell at a pair: the % its pooled tardiness there exceeds that at its best grid pair. Off-grid pairs
  are interpolated bilinearly. **Room:** a cell's regret at its best single α (the diagonal). It is what any online
  law can gain over the best static α.
- **Candidate laws**, all α = min(4, S(σ) × h(x)), S(σ) = max(0, 3 − 6σ) (v7's factor, 3 on the homogeneous fit
  cells), h(x) = clip((x1 − x) / (x1 − x0), 0, 1), x = τ / a:

  | law | a | knots |
  |---|---|---|
  | S0 | full fleet in every block (today's law) | `../rq3-alpha-transport`'s (0.175, 0.297) |
  | S | full fleet in every block | refit here (the fair static reference) |
  | O1 | on duty (short blocks read their own x) | S0's, not refit |
  | O2 | on duty | one hinge, refit |
  | O3 | on duty | separate hinges for full and short blocks, refit (4 knots) |
  | M (comparator) | S0 in full blocks, α 0 (MDD-TECT) in short blocks | the map rule |

- **Fit:** grid search, minimizing the mean regret over the 13 fit cells (equal weight).
  - The full hinge has x0 ≥ B2's full-fleet x (0.099), so B2's full-fleet blocks keep v7's α 3.
  - Knot steps are 0.01 (S, O2) and 0.02 (O3).
  - Ties within 0.01 points go to the gentlest slope, then the median knot.
- **Selection:** leave-one-cell-out regret (refit without the cell, score the cell). Take the lowest of O1-O3, then
  the simplest within 0.5 points, in the order O1 < O2 < O3.
- **Gate G1 (go for stage 2):** the selected law's leave-one-cell-out regret is at least 1.0 point below S's.
  - `law_online.json` records G1, the selected law (as `online`), S0 (`static`) and M (`map`), each cell's statistics
    and pairs, and stage 2's law cells.
  - Stage 2's launcher refuses to run without G1 PASS.
- **Reported, not gating:**
  - each cell's best pair, best single α and room;
  - the per-regime split of the best pair (short-block tardiness only, exploratory);
  - O2 vs O3, i.e. whether one hinge in x serves both fleet states;
  - B2's regret under each law;
  - the held-out cells' interpolated regrets (stage 2 runs those laws exactly and judges transfer, criterion H').
- **Pipeline test (10-29, synthetic surfaces from the real cell statistics, quadratic bowls around a planted law):**
  - planted O2 (0.15, 0.45): recovered (0.149, 0.449), selected O2, G1 PASS (S 9.06% vs O2 0.49%);
  - planted O3 (0.18, 0.30 / 0.30, 0.60): recovered (0.179, 0.299 / 0.300, 0.580), selected O3. G1 FAIL there
    because that planted law leaves only 0.31 points of room.
- **Expectation, stated before the run (a guess, not a criterion):** the room is largest where the short block's x
  crosses the hinge.
  - B2's dry-run seed: the best pair (3, 0.5) is 21.6% better than α 3 throughout.
  - O1 sends B2's short blocks to α 0 (x 0.35 > 0.297), probably too low: on that seed (3, 0) is 6% behind (3, 0.5).
  - rq3-alpha-size's short-block α\* of 1.5-4 at 15-30 machines also argues against reusing the static knots.
  - So I expect O2 or O3 to be selected, and G1 to pass.

## Checks (10-29, scratchpad, before queueing)
- **The diagonal is the earlier fixed-α run, exactly:**
  - B2 s2000 at α 0 / 1 / 3 / 4 equals `../rq3-alpha-size` (205.410 / 190.473 / 197.234 / 188.912);
  - m100_r035 s0 at α 0 / 3 equals `../rq3-alpha-transport` (38.053 / 39.471);
  - m100_s1 s0 at α 0.5 / 1 equals `../rq3-alpha-size` (195.034 / 223.829);
  - m50_r080 s2000 at α 2 / 4 equals `../rq3-alpha-transport` (87.313 / 90.551).
- **The pair switches with the fleet:**
  - B2 s2000: 592 of 1,716 decisions fall in short blocks (2 AGVs). (3, 0) gives 163.7 and (0, 3) 203.8, against
    197.2 for (3, 3).
  - m50_r080 s2000: 2,315 of 6,070 decisions are short.
  - Instances without a short block (m100_r035 s0) play α_full throughout, so (1, 0) equals α 1.
- **Speed:** 49 pairs take 58 s on B2 and 78 s at 20 machines (dry run, 6 tasks at once).
- `analyze.py` ran through every section on the dry-run outputs (`--min-n 1`).
- **Edit after the start (10-29 ~17:55):** `run.py`'s `law_alpha_fn` crashed on an instance without a fleet schedule
  (found in stage 2's dry run) and was fixed in place (atomic replace). Only stage 2's law runs use it; the surface
  path the running tasks execute is unchanged.

## Cost
- 760 surface tasks (49 episodes each) and 240 statistics tasks.
- At `../rq3-alpha-transport`'s loaded rates: m200 ~54 / 40 min per seed (r052 / r080), m150 ~25, m100 ~12-16,
  smaller floors ~1-6. About 170 CPU-h, so ~12 h on 14 slots.
- The longest single task is an m200_r052 seed (~1 h).

## Run
`launch.sh [W]` runs `tasks.txt` (`run.py list`, longest first), W at a time; finished tasks are skipped.

## Outputs
- `out/<cell>/s<seed>.json`: tardiness per pair `"af|ash"` (total, per block, decisions / short decisions), the
  schedule, floor statistics (surf tasks); statistics only (stats tasks);
- `out/<cell>/law/s<seed>.json`: stage 2's law runs;
- `analysis.out`, `surfaces.csv`, `fit.json`, `law_online.json`.

## Results (10-30, `analysis.out`, `law_online.json`)

### Checks
- **The diagonal (a, a) equals the earlier fixed-α run of the same instance on 5,320 of 5,320 (cell, seed, α).** All
  19 cells, 40 seeds each.
- **Floor statistics** (fitting seeds, n 40 per cell), x = τ / a at the full / short fleet:

  | cell | x full | x short | | cell | x full | x short |
  |---|---|---|---|---|---|---|
  | m15_b2 | 0.099 | 0.346 | | m50_r035 | 0.276 | 0.553 |
  | m20_r052 | 0.108 | 0.217 | | m50_r080 | 0.124 | 0.249 |
  | m30_r052 | 0.141 | 0.282 | | m100_r035 | 0.355 | 0.690 |
  | m50_r052 | 0.191 | 0.383 | | m100_r080 | 0.155 | 0.311 |
  | m75_r052 | 0.221 | 0.430 | | m200_r080 | 0.234 | 0.467 |
  | m100_r052 | 0.239 | 0.478 | | m75_r035 (held out) | 0.331 | 0.662 |
  | m150_r052 | 0.310 | 0.620 | | m150_r080 (held out) | 0.202 | 0.403 |
  | m200_r052 | 0.360 | 0.719 | | spread cells (held out) | 0.089-0.227 | 0.313-0.454 |

  S0's hinge runs from x 0.175 (full α) to 0.297 (α 0).

### Surfaces: what an online α can gain
% vs α 3 throughout (3 | 3), pooled window tardiness, fitting seeds 2000-2039 (held-out cells: test seeds 0-39).
Room = regret of the best single α against the best pair, in points.

| cell | role | best pair | % | best single α | % | room |
|---|---|---|---|---|---|---|
| m15_b2 (B2) | fit | 4 \| 2 | −2.75 | 2 | −1.54 | 1.24 |
| m20_r052 | fit | 4 \| 2 | −2.87 | 4 | −2.32 | 0.56 |
| m30_r052 | fit | 4 \| 2 | −2.08 | 4 | −0.92 | 1.19 |
| m50_r052 | fit | 3 \| 0.5 | −1.88 | 2 | −0.10 | 1.81 |
| m75_r052 | fit | 2 \| 0 | −6.94 | 2 | −1.24 | **6.12** |
| m100_r052 | fit | 2 \| 0 | −10.40 | 1 | −2.12 | **9.24** |
| m150_r052 | fit | 0.5 \| 0 | −14.23 | 0 | −13.43 | 0.94 |
| m200_r052 | fit | 0 \| 0 | −15.80 | 0 | −15.80 | 0.00 |
| m50_r035 | fit | 3 \| 0 | −5.71 | 0.5 | −1.30 | **4.68** |
| m50_r080 | fit | 4 \| 2 | −0.32 | 4 | −0.30 | 0.02 |
| m100_r035 | fit | 0 \| 0 | −13.20 | 0 | −13.20 | 0.00 |
| m100_r080 | fit | 3 \| 0.5 | −0.76 | 3 | +0.00 | 0.77 |
| m200_r080 | fit | 3 \| 0 | −8.56 | 1 | −2.86 | **6.23** |
| m75_r035 | held out | 0.5 \| 0 | −10.15 | 0 | −9.28 | 0.96 |
| m150_r080 | held out | 2 \| 0 | −13.19 | 0.5 | −5.74 | **8.58** |
| m15_s1 | held out | 0.5 \| 0.5 | −17.22 | 0.5 | −17.22 | 0.00 |
| m50_s1 | held out | 0.5 \| 0 | −31.24 | 0.5 | −29.99 | 1.81 |
| m100_s0.5 | held out | 0.5 \| 0 | −24.89 | 0.25 | −23.37 | 2.03 |
| m100_s1 | held out | 0.25 \| 0 | −29.45 | 0 | −28.75 | 0.99 |

- **Mean room over the 13 fit cells: 2.52 points.** The five large values (4.7-9.2 points: m75 and m100 at r 0.52,
  m200_r080, m50_r035, the held-out m150_r080) all lie where the full fleet's x is below S0's upper knot and the short
  fleet's above it. There the best pair keeps a high α with the full fleet and plays α 0 (MDD-TECT) while it is short.
  B2, m50_r052 and m100_r080 cross the knot too, with 0.8-1.8 points.
- **Room is near 0 where one regime decides the cell:** α 0 throughout on the slow-transport floors (m200_r052,
  m100_r035), or a full fleet that never runs short of transport (m50_r080).
- **Per regime (exploratory, short-block tardiness at the best α_full):** the short block's best α is 2 at 15-30
  machines and 3 at m50_r080, 0.5 at m100_r080, and 0-0.25 everywhere else. This is the hinge again, read at the
  short fleet's x.
- B2's pooled best pair is 4 | 2, not the dry-run seed's 3 | 0.5 (that seed's 21.6% was one instance). B2's room
  (1.24 points) is near the fit cells' median (1.19).

### Laws
Mean regret over the 13 fit cells, %, in sample / leave-one-cell-out:

| law | knots | in sample | LOO |
|---|---|---|---|
| S0 (today's static law) | 0.175, 0.297 | 2.97 | 2.97 |
| S (static, refit) | 0.169, 0.299 | 2.98 | 3.32 |
| **O1 (on duty, S0's knots)** | 0.175, 0.297 | **1.19** | **1.19** |
| O2 (on duty, one hinge refit) | 0.189, 0.339 | 0.62 | 0.81 |
| O3 (full / short hinges refit) | 0.259, 0.319 / 0.020, 0.420 | 0.50 | 0.74 |
| M (S0 full, α 0 short) | 0.175, 0.297 | 2.24 | 2.24 |

Leave-one-out regret per cell (points):

| law | m15_b2 | m20 | m30 | m50 | m75 | m100 | m150 | m200 | m50_r035 | m50_r080 | m100_r035 | m100_r080 | m200_r080 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S0 | 2.82 | 2.95 | 2.13 | 1.45 | 6.44 | 9.22 | 0.94 | 0.00 | 4.69 | 0.32 | 0.00 | 0.77 | 6.90 |
| S | 2.82 | 2.95 | 2.13 | 1.51 | 6.96 | 9.22 | 4.48 | 0.00 | 4.82 | 0.32 | 0.00 | 0.77 | 7.15 |
| O1 | 2.21 | 1.66 | 2.37 | 1.24 | 0.58 | 1.91 | 0.94 | 0.00 | 1.54 | 1.14 | 0.00 | 1.07 | 0.80 |
| O2 | 2.21 | 2.69 | 1.86 | 0.98 | 0.10 | 0.41 | 0.77 | 0.00 | 0.64 | 0.66 | 0.00 | 0.05 | 0.14 |
| O3 | 1.51 | 1.96 | 1.33 | 0.43 | 0.43 | 0.26 | 1.31 | 0.00 | 0.80 | 1.11 | 0.00 | 0.45 | 0.00 |
| M | 2.21 | 8.95 | 4.23 | 1.24 | 0.58 | 1.91 | 0.94 | 0.00 | 1.54 | 5.68 | 0.00 | 1.07 | 0.80 |

(m20-m200 without a suffix are the r 0.52 cells.)

- **Selection (pre-declared): O1.** The lowest online LOO is O3's 0.74%, and O1's 1.19% lies within 0.5 points of it
  (0.45). The order O1 < O2 < O3 then picks O1.
- **G1: PASS.** O1's LOO 1.19% is 2.13 points below S's 3.32% (needs ≥ 1.0).
- **O1 fits nothing.** Its knots are `../rq3-alpha-transport`'s, so its in-sample and leave-one-out regrets are the
  same number. Here the gain comes from reading x at the on-duty fleet, not from a fit to these surfaces.
- **The static form is the limit, not its knots.** Refitting the static law (S) does not help (LOO 3.32% vs S0's
  2.97%). Reading the fleet per regime cuts the mean regret by 1.8 points with the same knots.
- **One hinge serves both fleet states** (O2 0.81% vs O3 0.74% LOO, reported, not gating).
- **The map rule M is O1 at most cells.** It fails where the short block still wants α > 0 (m20 8.95, m50_r080 5.68,
  m30 4.23 points). O1 keeps α there because the short fleet's x stays below the upper knot.
- **Where O1 costs:** m30 (+0.24 points vs S0), m50_r080 (+0.82) and m100_r080 (+0.30). At m100_r080 the short
  fleet's x (0.311) is just past the knot, so O1 plays 3 | 0 where the best pair is 3 | 0.5.
- **B2:** O1 plays 3 | 0 (short x 0.346 is past the knot), regret 2.21% against S0's 2.82%. The best pair is 4 | 2;
  O3 gets to 1.51% (its in-sample pair there is 3 | 0.55). So on B2 O1 sends the short blocks to α 0, lower than the surface wants. This is the
  guess stated before the run ("probably too low"), and it costs 0.7 points against O3.

### Held-out cells (interpolated on test-seed surfaces; stage 2 runs the laws exactly and judges H')

| cell | S0 pair | S0 regret | O1 pair | O1 regret |
|---|---|---|---|---|
| m75_r035 | 0 \| 0 | 0.96 | 0 \| 0 | 0.96 |
| m150_r080 | 2.35 \| 2.35 | 13.06 | 2.35 \| 0 | **0.23** |
| m15_s1 | 0.70 \| 0.70 | 1.21 | 0.70 \| 0 | 1.34 |
| m50_s1 | 0.65 \| 0.65 | 3.42 | 0.65 \| 0 | 1.20 |
| m100_s0.5 | 0.92 \| 0.92 | 6.95 | 0.92 \| 0 | 1.57 |
| m100_s1 | 0.46 \| 0.46 | 3.53 | 0.46 \| 0 | 1.56 |

- O1 cuts the regret on 4 of 6 held-out cells, by 2-13 points, ties on one, and costs 0.13 points on m15_s1.
- **The spread law × on-duty hinge** (the product `../rq3-alpha-transport` left untested off σ 0) improves on S0 on
  the 50-100-machine spread floors (by 2.0-5.4 points) and costs 0.13 points at 15 machines. Stage 2's exact law runs
  decide H'.

### Reading against the expectation
- **Expected:** O2 or O3 selected, G1 PASS, room largest where the short block's x crosses the hinge.
- **Found:** G1 passes, and the room sits where expected. But the simplest law is selected: today's knots, read with
  the AGVs on duty, reach within 0.45 points of the best refit.
- So for paper B the online law needs no new parameters. "Read the hinge at the fleet on duty" is the whole change.
- What O1 leaves on the table is mostly small floors' short blocks (B2, m20, m30), where it plays α 0 and the
  surfaces want about 2.

### Stage 2 (started 10-30 04:45)
- `law_online.json` changes the pair (static → O1) on six of the eight policy cells: B2, m15_s1, m50_r052, m75_r052,
  m100_r052, m100_r080.
  - m100_r035 and m200_r052 play 0 | 0 under both laws and are not rerun.
- 18 policy tasks and 520 law runs (13 cells × 40 seeds), 14 at a time. Lab `eval-alpha-online`.
