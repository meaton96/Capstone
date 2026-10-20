# rq3-size-grid: floor-size cross-evaluation grid (2026-10-20, user: floor-size study phase 3)

Status: **reference half (fixed pairs + final-policy family on the new floors) running locally; arms half waits on
`train-size-m50` / `-m100` / `-mix`.** Umbrella: `../rq3-size-train/README.md`.

## Question
Does a policy carry over to floor sizes it was not trained on, and how far? Hypotheses (from the umbrella README):
1. performance falls with the size ratio |log(test / train)|, not with absolute size;
2. mixed-size training stays flat across 15-100 machines;
3. it degrades less than single-size policies beyond 100.

## Design
- **Twin**, test seeds 0-39, B2 window / slots / reward (`../rq3-gen-v5/run_task.py`, frozen v5 code), c ~ U[1.75, 2.5].
  Measure: window tardiness, pooled over test seeds, vs MDD-PTECT (seed-bootstrap 95% CI, Wilcoxon) and vs the floor's
  best v5 pair.
- **Test floors** (same per-machine scaling as `../rq3-size-v5`):

  | floor | machines | AGVs on duty (short blocks) | floor export | runs here |
  |---|---|---|---|---|
  | m15 | 15 | 7 (2), B2 itself | floor7 | arms only (pdr, T15 from rq3-gen-v5 base / eval-v5-cmix test) |
  | m20 | 20 | 10 (5) | `../dev-size-floors/floors/m20` | all |
  | m30 | 30 | 16 (8) | `../dev-size-floors/floors/m30` | all |
  | m50 | 50 | 26 (13) | `../agv-ratio-scale/floors/m50` | arms only (pdr, T15 from rq3-size-v5) |
  | m75 | 75 | 39 (20) | `../dev-size-floors/floors/m75` | all |
  | m100 | 100 | 52 (26) | `../agv-ratio-scale/floors/m100` | arms only (pdr, T15 from rq3-size-v5) |
  | m150 | 150 | 78 (39) | `../dev-size-floors/floors/m150` | all |
  | m200 | 200 | 104 (52) | `../dev-size-floors/floors/m200` | all |

  Above 15 machines: round(0.52 M) AGVs (the fleet wrapper keeps agvCount = bays), B2 load × 0.9, arrival cap off.
  The new m50 / m100 exports are identical to the agv-ratio-scale ones (`../dev-size-floors`).
- **Arms** (5 training seeds each; every policy group also runs MDD-PTECT as the run-to-run check):

  | arm | group | trained on | checkpoints |
  |---|---|---|---|
  | T15 | `cmix` | 15 machines (the final-policy family) | `train-v5-cmix_s0-4` |
  | T50 | `size50` | 50 | `train-size-m50_s0-4` |
  | T100 | `size100` | 100 | `train-size-m100_s0-4` |
  | Tmix | `sizemix` | 15 + 50 + 100 | `train-size-mix_s0-4` |

  Plus `pdr`: the 6 v5 pairs + MDD-TECT per floor.
- **Chunks:** 10 test seeds per task up to m100, 5 on m150 / m200. One m200 episode takes 5-13 minutes locally (peak memory 0.74 GB)
  (rule or policy alike; the random warm-up length varies by seed), and an m100 10-seed chunk took ~2.6 h on the
  cluster, so 10-seed m200 chunks of 7 policies would come close to the cluster's 10 h limit.
- **1000 machines** is left out for now: `../dev-size-1000-probe` measured ~6 h and 7.7 GB per episode (a full cell
  would be ~6,000 core-hours). A reduced cell is a decision for the user.

## Parts and lab experiments
- `rq3-size-grid-ref`: `pdr` + `cmix` on m20 / m30 / m75 / m150 / m200, 56 tasks, local (`launch_local.sh ref 16`;
  the local machine was idle while the cluster queue held the training jobs).
- `rq3-size-grid`: `size50` / `size100` / `sizemix` on all 8 floors, 120 tasks, cluster (`submit.sh arms`), after
  training.

## Analysis (`analyze.py` -> `analysis.out`, `summary.csv`, `grid.csv`)
Per floor: the pairs vs MDD-PTECT and the best pair; the hindsight best pair; each arm's family mean of 5 vs MDD-PTECT
and vs the best pair, with the spread over training seeds (each seed's pooled %, one-sample t-test of the 5 against
0: the training-seed-level test); Tmix vs each single-size arm, paired per test seed. Then the arm x floor matrix and the
transfer curve (single-size arms: gap vs |log10(test / train)|, Spearman). Run-to-run check: MDD-PTECT in every policy
group equals the pdr group's on the same floor and seeds.

Already in place before the run (from `../rq3-size-v5`, reproduced here): T15 family −1.69% at 15, −1.29% at 50,
+2.60% at 100 machines.

## Results
Pending.
