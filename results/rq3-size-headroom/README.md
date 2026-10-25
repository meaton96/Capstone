# rq3-size-headroom: why does switching headroom shrink on larger floors? (2026-10-25, user: remaining work plan §2c)

Status: **queued 10-25 (lab `rq3-size-headroom`, local, 20 workers; ~180-200 core-h, ETA about 10-26 morning).**
Dry run 10-25: the checks pass, and 6 descriptor files are kept as outputs (same code).

## Why
The v5 switching ceiling falls with floor size: −9.4 / −10.3 / −3.3 / −1.0% vs MDD-PTECT at 15 / 50 / 100 / 200
machines (`../rq3-size-oracle`). Two parts of the answer are known (`docs/Plans/REMAINING_WORK_PLAN_2026-10-23.md` §2c):
- **Arithmetic:** the oracle saves about the same tardiness per instance (13.7-16.2 k job-s at 15-100 machines), but
  the floor's total grows much faster.
- **The action set:** v5 has no travel-aware rule. Adding MDD-TECT brings 100 machines back to −10.6%
  (`../rq3-size-portfolio`).

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

## Results
Pending.
