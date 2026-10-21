# rq3-alpha-size: auto-tuned α across floor size and machine heterogeneity (2026-10-21, user: paper B gate)

Status: **running locally since 10-21 11:20 (lab `rq3-alpha-size`, 1,520 tasks, 16 at a time; ETA about 15:20).**
Paper plan: `docs/Plans/PAPER_PLAN_2026-10-21.md` (paper B).

## Question
The best machine rule is PECT / PTECT, scored `wait + p + α·p²/p̄` (PTECT uses `max(travel, wait)`; α 0 = ECT / TECT
exactly). Two results suggest its best α is a property of the floor:
- `../rq3-alpha-curve` (15 machines): α* falls with machine-speed spread, roughly α* ≈ max(0, 3 − 6.5σ); a σ-tuned α
  gains 6-17% over α 3 on heterogeneous floors.
- `../rq3-size-catalog`: on homogeneous floors MDD-TECT (α 0) goes from 18.5% worse than MDD-PTECT (α 3) at 15
  machines to 19.0% better at 200. So α* also falls with floor size.

Can α be set from observable floor statistics, so one rule works across sizes and heterogeneity, without per-floor
tuning and without a learned switching policy?

## Design (`run.py`)
- **Cells:** floor ∈ {15, 20, 30, 50, 75, 100 machines} × speed spread s ∈ {0, 0.5, 1.0} = 18 fitting cells, plus
  2 held-out cells (30 and 75 machines at s 0.75) used only for testing.
  - Floors are `../rq3-size-grid`'s conditions (round(0.52 M) AGVs, B2 load × 0.9 above 15 machines; 15 = B2), through
    `../rq3-size-oracle/sizesim.py`.
  - Spread s > 0 adds a fixed per-machine speed factor log-uniform in [1/(1+s), 1+s], capacity-matched per machine type
    and with B2-tight due dates, exactly as `../rq2-hetero-screen` (generalized to any machines per type).
- **Configs per instance (18):** MDD × {PECT, PTECT} × α ∈ {0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4}, each held all
  episode. Window tardiness, total and per regime block (normal / surge / short fleet).
- **Seeds:** fitting 2000-2039 (choose α*, fit the formula), test 0-39 (report). Held-out cells: test seeds only.
- **Observable statistics per instance** (all known before or while running the floor):
  - σ: std of the log machine speed factors (from processing-time data);
  - travel: mean AGV travel time from a machine (or the input) to a candidate machine of each type, and its spread
    across candidates (floor geometry / AGV speed);
  - processing: mean op time and its spread across candidate machines;
  - AGVs on duty per machine.
- **Frozen code:** `../rq2-rebase-v5` (6d13412a + its sim.py), the simulator every size result used.

## Checks (done 10-21, one seed each, scratchpad)
- 15 machines with spread (s 0.5 and 1.0, seed 0): MDD-PTECT at α 3 / 0 / 1 and MDD-PECT at α 1.5 equal
  `../rq3-alpha-curve` exactly (8 / 8). The generalized spread code reproduces the 15-machine study.
- Spread 0: α 3 equals the grid's MDD-PTECT and α 0 the catalog's MDD-TECT at 50 machines (seed 0) and 100 machines
  (seed 1), exactly (4 / 4).
- Statistics move as expected: travel / processing time is 0.04 at 15 machines, 0.09 at 50 and 0.14 at 100; σ ≈ 0.28-0.37
  at s 1.0.
- First look (100 machines, s 1.0, seed 0): PTECT α 0 / 0.25 / 0.5 / 1 / 2 / 3 / 4 = 197 / 189 / 195 / 224 / 291 / 400 /
  486 k job-s: α* ≈ 0.25, 53% better than α 3.

## Analysis (pre-declared; `analyze.py`)
`analyze.py` implements the steps below and the gate. Pipeline test (10-21, synthetic data with a planted law
α* = max(0, 3.5 − 5σ − 20τ)): it recovers (3.5, 5, 20) exactly, selects M3 and passes G1-G3. Regret is a cell's
pooled tardiness at the formula's per-instance α (rounded to the α grid) over its pooled tardiness at its best single
α, averaged over cells with equal weight; it can be slightly negative because per-instance α can beat one α per cell.
1. **α curves:** per cell, pooled fitting-seed tardiness vs α for PTECT and PECT; α* per cell (grid argmin), and the
   set of α within 1% of the best.
2. **Candidate formulas** for α* (PTECT primary, PECT reported), fitted on the 18 cells' fitting seeds:
   - M0: one constant α for all cells (the best single tuned α);
   - M1: α = max(0, a − b·σ) (the 15-machine law, refitted);
   - M2: α = max(0, a − c·τ), τ = travel / processing time (size only);
   - M3: α = max(0, a − b·σ − c·τ).

   Fit criterion: pooled regret (tardiness at the formula's α, rounded to the nearest grid value, vs each cell's
   best α). Selection: leave-one-cell-out regret; the simplest model within 0.5 points of the best wins.
3. **Test (seeds 0-39) and held-out cells:** the selected formula sets α per instance from that instance's
   statistics. It is compared with α 3 (MDD-PTECT), α 0 (MDD-TECT), M0, and each cell's best α chosen on fitting
   seeds. Pooled %, seed-bootstrap CI, Wilcoxon, Holm across cells.
4. **Per regime:** α* per regime block (normal / surge / short fleet), for the online follow-up (lower α when the fleet
   is short).

## Gate (go / no-go for paper B's auto-tuned rule)
- **G1, size matters:** at s 0, α* falls with floor size by at least 1.0 between 15 and 100 machines.
- **G2, the formula works:** the selected formula includes a size term (M2 or M3) and beats M1 by at least 1 point of
  leave-one-cell-out regret. On test seeds it is within 2% of the per-cell best α, pooled over cells. It beats α 3
  significantly (Holm) in at least half of the cells and is never significantly worse than α 3.
- **G3, it transfers:** on the held-out cells (30 / 75 machines at s 0.75) it beats α 3 and is within 3% of the
  hindsight best α there.

**If G1-G3 pass:** the auto-tuned α is paper B's headline. Next come the online version (α per regime / from live
statistics) and a Unity check through the Python rule mode. **If G2 or G3 fail:** report the α curves as a design
guideline (α* by size and spread) without a deployable formula.

## Cost
1,520 tasks (18 cells × 80 seeds + 2 × 40), 18 episodes each, in the fast simulator. Measured: about 1 s per episode at 15
machines, 3 s at 50, and 7-16 s at 100 (seed-dependent). About 40-50 core-hours: ~3 h on 16 local workers, largest
floors first. Lab: `rq3-alpha-size` (local).
