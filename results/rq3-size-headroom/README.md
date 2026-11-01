# rq3-size-headroom: why does switching headroom shrink on larger floors? (2026-10-25, user: remaining work plan §2c)

Status: **done 10-26 (lab `rq3-size-headroom`, local, 20 workers; 460/460 outputs, all checks pass).** Test B
(transport) supported; test A (averaging) passes only on per-instance medians and not pooled. Answer: the headroom
shrinks because the floor becomes transport-bound and the v5 rules cannot follow; with MDD-TECT in the set the room is
−13 to −21% at every size and fleet. Dry run 10-25: the checks pass, and 6 descriptor files are kept as outputs (same
code).

## Why
The v5 switching ceiling falls with floor size: −9.4 / −10.3 / −3.3 / −1.0% vs MDD-PTECT at 15 / 50 / 100 / 200
machines (`../rq3-size-oracle`). Two parts of the answer are known (`docs/Plans/REMAINING_WORK_PLAN_2026-10-23.md` §2c):
- **Arithmetic:** the oracle saves about the same tardiness per instance (13.7-16.2 k job-s at 15-100 machines), but
  the floor's total grows much faster.
- **The action set:** every v5 machine rule (PECT, PTECT) carries the processing-time weight α g(p), which trades
  travel and queue for faster machines. MDD-TECT ranks machines on travel and queue only (PTECT's score with α 0 and no
  availability scaling). Adding it brings 100 machines back to −10.6% (`../rq3-size-portfolio`).

Two explanations are untested:
- **Averaging:** a floor with more machines has smoother aggregate dynamics, so fewer swings to exploit.
- **Transport scarcity:** the grid keeps 0.52 AGVs per machine at every size, but trips get longer on a bigger floor,
  so transport binds harder. Short-fleet blocks are where TECT pays at 100 machines.

This run varies the AGV ratio at three floor sizes to separate the two. It gives the thesis a mechanism, not just a
curve.

## Design (`run.py`, `launch.sh`, `analyze.py`)
- **Oracle:** rq2-rebase-v5's greedy switching oracle, unchanged (900 s slots, hold-to-end tails, frozen code 6d13412a
  and its `sim.py`), over two action sets:
  - `v5`: the 6 v5 pairs;
  - `v5tect`: v5 + MDD-TECT (as rq3-size-portfolio's m100 `v5tect`).
- **Floors:** the floor-size grid's m50 / m100 / m200 (B2 load × 0.9, arrival cap off, M bays; `../rq3-size-grid`).
- **AGVs per machine r ∈ {0.35, 0.52, 0.80}:** round(r M) AGVs on duty in full-fleet blocks, half that in short-fleet
  blocks. r 0.52 is the grid's own condition.

  | floor | r 0.35 | r 0.52 (grid) | r 0.80 |
  |---|---|---|---|
  | m50 | 18 (9) | 26 (13) | 40 (20) |
  | m100 | 35 (18) | 52 (26) | 80 (40) |
  | m200 | 70 (35) | 104 (52) | 160 (80) |

- **Only the fleet changes:** each instance's jobs, phases, regime blocks and machine speeds are the same at every r.
  `check` verifies this on seeds 0-2 per floor, and every descriptor file carries a hash of the instance less its AGV
  fields.
- **Seeds:** B2 test seeds 0-19, paired across ratios (same instance) and across the two action sets.
- **Reused:** at r 0.52 the v5 oracle exists for all three floors (`../rq3-size-oracle`, seeds 0-39) and the v5tect
  oracle for m100 (`../rq3-size-portfolio`). Those are not rerun.
- **Descriptors** (`desc`): MDD-PTECT and MDD-TECT held all episode, with the state at each slot's first decision:
  - backlog per machine (remaining plus queued plus inbound work), its spread over machines, and the regime;
  - WIP, jobs waiting for an AGV per on-duty AGV;
  - AGV utilization over the slot (busy AGV-seconds over on-duty AGV-seconds from the AGV schedule), machine
    utilization, trips and path per trip.
- **Checks** (launch and analysis): MDD-PTECT at r 0.52 equals the grid's (seeds 0-2 per floor). Analysis: held
  MDD-PTECT / MDD-TECT in the descriptor runs equal the oracles' slot-0 tails, reused files included, and the instance
  hash is equal across ratios.
- **Cost (dry run, one process each):**
  - one episode takes ~3.5 s at m50, ~8.5 s at m100 and ~30 s at m200;
  - an oracle replays 144 (v5) or 168 (v5tect) episodes;
  - 280 oracle tasks and 180 descriptor tasks, about 180-200 core-h, m200 three quarters of it.

## Analysis (pre-declared 10-25, before any oracle result)
Per cell: the v5 and v5tect ceilings vs MDD-PTECT (pooled, paired seed-bootstrap CI), what MDD-TECT adds over the v5
oracle, MDD-TECT held, the hindsight best fixed pair, saving per instance, switches per episode, per-slot tail spread
(how much the choice matters in a slot), and the descriptors.

**Test A, averaging.** At each ratio:
- (a) the backlog CV over slots falls from m50 to m200 (Mann-Whitney, one-sided);
- (b) the v5tect ceiling still shrinks from m50 to m200 (per-instance %, Mann-Whitney, one-sided).

Averaging is supported if (a) and (b) both hold at r 0.80, where transport is least scarce.

**Test B, transport scarcity.** On m100 and on m200:
- (c) the v5 ceiling at r 0.80 is larger than at r 0.52 (same instances, paired Wilcoxon);
- (d) what MDD-TECT adds over the v5 oracle is larger at r 0.35 than at r 0.80 (paired Wilcoxon).

Transport scarcity is supported if (c) and (d) hold on both floors.

Both tests can pass. The analysis then reports how much of the m50 → m100 / m200 drop in the v5 ceiling the larger
fleet recovers, and, within each floor, Spearman correlations of the per-instance v5 gain with AGV utilization,
backlog CV, the tail spread and switches.

**Predictions (plan §2c):**
- If averaging drives the shrinkage, load variability falls with size and the v5tect ceiling still falls with size.
- If transport scarcity drives it, more AGVs per machine restore the v5 ceiling, and fewer AGVs make TECT matter more.

**Dry-run hints (seed 0 only, not evidence):**
- MDD-PTECT at m200 is 731 k job-s at r 0.35 against 159 at r 0.80, with AGVs 96% busy at r 0.35. At m50 the ratio
  barely matters (58.1 vs 56.8).
- Path per trip grows with size at a fixed ratio: ~315 m at m50, ~450 m at m100, ~710 m at m200.

## Lab
`lab show rq3-size-headroom`. Launch `bash results/rq3-size-headroom/launch.sh 20`; done when 460 files exist under
`out/*/*/*/s*.json`; analysis `analyze.py` → `analysis.out`, `summary.csv`, `instances.csv`.

## Results (10-26)
460/460 outputs, 20 paired instances per cell, all checks pass. The run took 1 day 3.6 h wall time, about 10 h of it
paused (SIGSTOP 10-25 evening to 10-26 morning). The `seconds` field of oracles in flight during the pause includes
it. `analysis.out` has every CI and descriptor; `summary.csv` per cell, `instances.csv` per instance.

### Ceilings (pooled % vs MDD-PTECT held, < 0 better; every oracle ceiling excludes 0)
| floor | AGVs on duty, r 0.35 / 0.52 / 0.80 | v5 oracle | v5 + MDD-TECT oracle | MDD-TECT adds | MDD-TECT held | AGV util | path per trip (m) |
|---|---|---|---|---|---|---|---|
| m50 | 18 / 26 / 40 | −7.0 / −11.6 / −14.2 | −13.0 / −15.6 / −14.5 | −6.5 / −4.5 / −0.4 | +1.1 / +21.5 / +39.9 | 0.82 / 0.62 / 0.42 | 323 / 315 / 311 |
| m100 | 35 / 52 / 80 | −1.2 / −3.2 / −6.1 | −13.6 / −10.4 / −6.4 | −12.6 / −7.5 / −0.4 | −13.2 / −2.0 / +21.7 | 0.94 / 0.75 / 0.52 | 460 / 451 / 448 |
| m200 | 70 / 104 / 160 | −0.7 / −1.0 / −1.9 | −16.6 / −20.7 / −16.6 | −15.9 / −19.9 / −15.0 | −16.6 / −20.2 / −12.0 | 0.99 / 0.91 / 0.68 | 718 / 711 / 701 |

MDD-PTECT's tardiness (k job-s, mean) at r 0.35 / 0.52 / 0.80: m50 284 / 97 / 74, m100 2,141 / 609 / 258, m200
8,374 / 2,890 / 757. Machine utilization falls to 0.38 at m200 r 0.35, where the AGVs are 99% busy.

### Pre-declared tests
- **Test B, transport scarcity: supported** ((c) and (d) on both floors).
  - (c) More AGVs regrow the v5 ceiling. Per-instance medians at r 0.35 / 0.52 / 0.80: m100 −1.4 / −4.9 / −10.8%,
    m200 −1.0 / −1.1 / −2.6%. r 0.80 − r 0.52: −2.1 points at m100 (p 0.048), −1.3 at m200 (p 0.006).
  - (d) Fewer AGVs make MDD-TECT worth more. What it adds, median, r 0.35 vs r 0.80: m100 −11.9 vs −0.0% (p 5e-5),
    m200 −20.2 vs −6.3% (p 0.012).
  - Within each floor, instances with busier AGVs leave less v5 room. Spearman of the v5 gain with AGV utilization:
    +0.48 / +0.71 / +0.54 at m50 / m100 / m200; with jobs waiting per AGV: +0.58 / +0.79 / +0.57 (all p < 1e-4). The
    ratios are pooled here, so this partly restates the manipulation.
- **Test A, averaging: passes as declared at r 0.80, but not pooled.**
  - (a) The backlog CV over slots falls from m50 to m200 at r 0.35 (0.60 → 0.47, p 0.003) and r 0.80 (0.55 → 0.48,
    p 0.017), not at r 0.52 (0.53 → 0.51, p 0.32). The fall is small.
  - (b) The per-instance v5 + MDD-TECT ceiling shrinks at r 0.80 (median −19.6 → −11.0%, p 0.023), not at r 0.35 or
    0.52. Pooled it does not shrink at any ratio: m50 −13.0 / −15.6 / −14.5% against m200 −16.6 / −20.7 / −16.6%.
  - The m50 r 0.80 median is pulled by its low-tardiness instances (bottom half by MDD-PTECT −22%, top half −17%).
    The pooled figure weights by tardiness, as every ceiling in this study and in `../rq3-size-oracle` does.
  - **Reading:** averaging is at most a small effect. With MDD-TECT in the set, the room does not shrink with size.

### Fleet capacity versus trip length
- **A larger fleet recovers part of the v5 drop at 100 machines and little at 200.** From m50 r 0.52 (−11.6%), the
  v5 ceiling drops to −3.2% at m100, and r 0.80 brings it back to −6.1% (34% of the drop). At m200 it drops to −1.0%,
  and r 0.80 brings it to −1.9% (8%).
- **At 100 machines the fleet decides.** MDD-TECT held is −13% vs MDD-PTECT at r 0.35, −2% at 0.52 and +22% at 0.80;
  what it adds to the oracle falls from −12.6% to −0.4%.
- **At 200 machines the trip length decides.** At r 0.80 the AGVs are 68% busy, with 0.59 jobs waiting per AGV. Even
  so, MDD-TECT held is −12%, and adding it to the oracle gains −15%. Trips are ~700 m against ~315 m at 50 machines,
  and a larger fleet barely shortens them.
- **At 50 machines MDD-TECT held never wins** (+1 to +40%), yet it adds −6.5 / −4.5% to the oracle at r 0.35 / 0.52.
  It pays only in some slots there (its slot share is 30 / 23 / 14%).

### Answer to §2c
On larger floors the bottleneck moves from the machines to transport. Every v5 machine rule weights processing time
(α g(p)), which sends jobs to faster machines at the cost of travel and queue. Holding the job stream fixed:
1. More AGVs per machine regrow the v5 ceiling, and fewer make MDD-TECT, which ranks on travel and queue alone, worth
   more (test B).
2. Fleet capacity is not all of it. At 200 machines trips stay ~700 m with any fleet, and MDD-TECT pays 12-20% at
   every ratio.
3. Averaging is small. Load variability falls only slightly with size, and the pooled ceiling with MDD-TECT in the set
   is −13 to −21% at every size and fleet.

**The shrinking headroom is a limit of the v5 action set on transport-bound floors, not of large floors.** The room
is there, one pair away. This is the large-floor failure of `../rq3-transfer-map`: the room under 2.5% at ≥ 75
machines is room within v5. Two consequences for the plan:
- **The α action (§2d) can reach it:** PTECT with α 0 and no availability scaling is TECT's score.
- **The 100-machine fine-tune (§2b) keeps the v5 pairs,** so its ceiling there is the v5 oracle's −3.2% (r 0.52),
  not the −10.4% with MDD-TECT.

### Caveats
- **Greedy oracle, hindsight per slot:** the ceilings are what switching could gain, not what a policy gets (the
  final policy captures about 0.38 of the room on most floors, `../rq3-transfer-map`).
- **Only the fleet is matched within a floor, not across floors.** Each size follows the grid's own floor (B2 load
  × 0.9, cap off, M bays), so sizes also differ in layout and trip length by design. Machine utilization is 0.54-0.61
  in every cell except the starved m200 r 0.35 (0.38).
- **20 seeds per cell** (the grid used 40 at r 0.52). The r 0.52 v5 ceilings here (−11.6 / −3.2 / −1.0%) match
  `../rq3-size-oracle` (−10.3 / −3.3 / −1.0% on 40 seeds).
- **The Spearman figures pool the three ratios,** so they mix the manipulation with instance variation.
