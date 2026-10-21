# rq3-size-grid: floor-size cross-evaluation grid (2026-10-20, user: floor-size study phase 3)

Status: **reference half done 10-20 14:05 (local, 8 h, 56 / 56 tasks); arms half (`rq3-size-grid`) waits on
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
  cluster, which is ~1.6x slower per core than the local machine. A job's time limit cannot be raised after
  submission (the partition itself allows 20 days), so the big floors use short chunks and generous limits.
- **1000 machines:** `../dev-size-1000-probe` measured ~6 h and 7.7 GB per episode (a full cell would be ~6,000
  core-hours). The user chose a reduced column (10-20): `../rq3-size-m1000` (5 test seeds, MDD-PTECT, SRT-PTECT, the
  final policy and one seed of each new arm).

## Parts and lab experiments
- `rq3-size-grid-ref`: `pdr` + `cmix` on m20 / m30 / m75 / m150 / m200, 56 tasks, local (`launch_local.sh ref 16`;
  the local machine was idle while the cluster queue held the training jobs).
- `rq3-size-grid`: `size50` / `size100` / `sizemix` on the floors up to 100 machines (m15-m100), 72 tasks, cluster
  (`submit.sh arms`), after training. Narrowed 10-21 after the advisor meeting (50-100 machines is the realistic range;
  m150 / m200 keep their reference half). The 1000-machine arms (`rq3-size-m1000` future) were cancelled for the same
  reason.

## Analysis (`analyze.py` -> `analysis.out`, `summary.csv`, `grid.csv`)
Per floor: the pairs vs MDD-PTECT and the best pair; the hindsight best pair; each arm's family mean of 5 vs MDD-PTECT
and vs the best pair, with the spread over training seeds (each seed's pooled %, one-sample t-test of the 5 against
0: the training-seed-level test); Tmix vs each single-size arm, paired per test seed. Then the arm x floor matrix and the
transfer curve (single-size arms: gap vs |log10(test / train)|, Spearman). Run-to-run check: MDD-PTECT in every policy
group equals the pdr group's on the same floor and seeds.

Already in place before the run (from `../rq3-size-v5`, reproduced here): T15 family −1.69% at 15, −1.29% at 50,
+2.60% at 100 machines.

## Results: reference half (10-20)
Twin, test seeds 0-39, window tardiness; % vs MDD-PTECT, pooled over test seeds, < 0 better. Run-to-run check:
MDD-PTECT in the policy groups equals the pdr group's on 40 / 40 seeds on every floor. `analysis.out`, `summary.csv`,
`grid.csv`.

### Rule landscape by floor size

| floor | MDD-PTECT mean (k job-s) | per machine | best pair (vs MDD-PTECT) | worst pair | hindsight best pair |
|---|---|---|---|---|---|
| m15 (B2) | 145.5 | 9.7 | MDD-PECT −0.5% | ATC-PECT +13.1% | −4.5% |
| m20 | 76.4 | 3.8 | ATC-PECT −3.8% | SRT-PTECT +9.8% | −8.1% |
| m30 | 92.8 | 3.1 | ATC-PTECT −5.2% | SRT-PECT +8.5% | −9.6% |
| m50 | 158.2 | 3.2 | MDD-PTECT | SRT-PECT +10.5% | −4.6% |
| m75 | 302.8 | 4.0 | MDD-PTECT | ATC-PECT +22.2% | −2.3% |
| m100 | 456.0 | 4.6 | MDD-PTECT | ATC-PECT +31.5% | −0.7% |
| m150 | 1,634.9 | 10.9 | MDD-PTECT | ATC-PECT +55.9% | −0.26% |
| m200 | 2,848.9 | 14.2 | MDD-PTECT | ATC-PECT +71.1% | −0.17% |

- The room for any rule choice shrinks with size. The per-instance best of the 6 pairs in hindsight gains 8-10% over
  MDD-PTECT at 20-30 machines, 0.7% at 100 and under 0.3% at 150-200.
- ATC goes from the best pair (20-30 machines) to by far the worst (+36-71% at 150-200). SRT-PTECT stays within ~2%
  of MDD-PTECT on the big floors.
- Tardiness per machine is flat from 20 to 100 machines (3-5 k job-s) and rises 2-3x at 150-200: the large floors run
  in a more congested regime at the same 0.52 AGVs per machine (likely longer AGV trips on a bigger floor; not
  measured here).
- m15 is B2 itself (full load, 7 AGVs = 0.47 per machine); m20 and up use B2 load × 0.9 and 0.52 AGVs per machine.
  So the m15 → m20 step changes load as well as size.

### T15: the policies trained on 15 machines (train-v5-cmix)

| floor | size ratio vs 15 | family mean of 5 (95% CI), W/L | training seeds s0 / s1 / s2 / s3 / s4 | final policy s3 (95% CI), W/L |
|---|---|---|---|---|
| m15 | 1 | **−1.69** [−2.86, −0.54], 28/12 | −0.3 / −2.3 / −1.0 / −1.7 / −3.2 | −1.71 [−3.92, +0.44], 24/16 |
| m20 | 1.3 | **−5.28** [−9.05, −1.99], 34/6 | −2.3 / −3.8 / −7.7 / −7.1 / −5.4 | **−7.11** [−12.27, −2.55], 33/7 |
| m30 | 2 | **−4.59** [−7.05, −2.96], 36/4 | −2.8 / −3.4 / −5.7 / −5.7 / −5.3 | **−5.69** [−8.60, −3.60], 35/5 |
| m50 | 3.3 | −1.29 [−2.55, +0.03], 26/14 | −0.0 / −0.9 / −2.6 / −2.1 / −0.9 | −2.12 [−4.56, +0.11], 29/11 |
| m75 | 5 | +1.24 [−0.24, +2.77], 22/18 | +2.8 / +2.0 / −0.5 / −0.1 / +2.0 | −0.09 [−1.60, +1.82], 22/18 |
| m100 | 6.7 | **+2.60** [+1.27, +3.88], 16/24 | +3.8 / +3.4 / +1.0 / +0.4 / +4.4 | +0.44 [−0.64, +1.70], 21/19 |
| m150 | 10 | **+6.51** [+5.68, +7.45], 3/37 | +8.1 / +9.9 / +1.3 / +1.5 / +11.7 | **+1.48** [+0.73, +2.50], 8/28 |
| m200 | 13.3 | **+9.54** [+8.80, +10.44], 0/40 | +11.2 / +13.8 / +2.9 / +2.7 / +17.1 | **+2.68** [+1.88, +3.66], 3/34 |

Bold: CI excludes 0. Training-seed-level t-test of the 5 seeds against 0: p 0.028 (m15), 0.006 (m20), 0.002 (m30),
0.049 (m50), 0.13 (m75), 0.03 (m100), 0.039 (m150), 0.031 (m200).

- **Transfer falls with the size ratio.** Family gap vs |log10(test / 15)|: Spearman ρ +0.93 (p 0.0009, 8 floors;
  ρ +0.88 vs the floor's best pair). Up to 2x the training size (20-30 machines) the policies beat MDD-PTECT by
  5-7% and roughly match the floor's own best pair (family −1.5% vs ATC-PECT at m20, +0.6% vs ATC-PTECT at m30).
  At 3x (m50) the edge is −1.3%, at 5x (m75) it is gone, and from 6.7x (m100) on they are worse than MDD-PTECT,
  increasingly so (+6.5% at 10x, +9.5% at 13x, losing on 37-40 of 40 instances).
- **The training seeds split in two on large floors.** s2 and s3 (the final policy) degrade gently (+1.3-2.9% at
  150-200); s0, s1 and s4 lose 8-17%. On small floors all five gain. So the size limit depends on the training seed,
  not only on the recipe; with 5 seeds this cannot be pinned to anything measurable yet.
- **The final policy (s3)** is significantly better than MDD-PTECT up to 30 machines, ties it at 50-100 (CIs
  include 0) and is 1.5-2.7% worse at 150-200.
- **Reading:** above ~75 machines MDD-PTECT is close to the per-instance best rule, so a rule-selecting policy can
  only lose, and any choice of ATC or PECT costs a lot there. The losses are what a 15-machine policy does when it
  keeps choosing rules that pay off on small floors. This is consistent with `../rq3-size-remap` (input scaling is
  not the cause). Whether training on 50 / 100 / mixed sizes fixes it is the arms half.

### Thesis reading (so far)
The proposal's "any floor size" claim does not hold for the 15-machine-trained policy. It holds within about 3x of
the training size (15-50 machines: better than or equal to MDD-PTECT, family and final policy), and fails beyond
~5x. The arms half tests whether a policy trained on larger or mixed floors moves or widens that window.
