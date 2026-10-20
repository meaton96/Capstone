# rq3-size-m1000: reduced 1000-machine column of the floor-size grid (2026-10-20, user: "yes")

Status: **reference half (`rq3-size-m1000-ref`) queued on the cluster 10-20 14:19 (array 21932264); arms half
(`rq3-size-m1000`) waits on `rq3-size-grid`.** Part of the floor-size study (`../rq3-size-train/README.md`); grid up to
200 machines: `../rq3-size-grid/README.md`.

## Why
In the grid the policies trained on 15 machines lose more the bigger the floor (family vs MDD-PTECT +2.6% at 100,
+6.5% at 150, +9.5% at 200; final policy s3 +0.4%, +1.5%, +2.7%). The user asked (10-19) whether a policy trained on
100 machines can go to 1000. This column shows how the 4 training arms behave at 1000 machines. With 5 test seeds it
can show whether a policy does anything different from MDD-PTECT there and in which direction, not rank the arms.

## Design
- **Floor:** `../dev-size-floors/floors/m1000` (layout D, 1000 bays, 186 × 232.5 m). The grid's per-machine scaling:
  200 machines per type, 520 AGVs on duty (short blocks 260), B2 load × 0.9, arrival cap off, c ~ U[1.75, 2.5]. Twin,
  B2 window / slots / reward, frozen v5 code, through the grid's harness (`../rq3-size-grid/task.py`).
- **Test seeds 0-4**, one episode per task.

  | group | policy |
  |---|---|
  | `mdd` | MDD-PTECT (reference; the best pair at 50-200 machines) |
  | `srt` | SRT-PTECT (the runner-up at 150-200 machines, within ~2% of MDD-PTECT) |
  | `t15` | the final policy (train-v5-cmix s3) |
  | `t50` / `t100` / `tmix` | one seed of train-size-m50 / -m100 / -mix |

- **Seed pick for the new arms** (`pick.py`, before any of them runs at 1000): the seed with the lowest tardiness vs
  MDD-PTECT at 200 machines in `../rq3-size-grid`. The same rule gives s3 for the 15-machine arm (+2.68% vs s2 +2.9%),
  which is also the final policy.
- **Cost:** `../dev-size-1000-probe` took 5.9 h and 7.7 GB per episode locally (seed 0). The cluster's nodes are ~1.6x
  slower than the local Ryzen 9 7950X (ten m100 MDD-PTECT episodes, seeds 0-9: 92 s locally vs 149 s on the
  cluster), and episode time varies ±50% by seed (seed 0 was the fastest of those ten). So expect ~10-16 h per task,
  ~30 tasks ≈ 300-450 core-hours (above the ~180 quoted before the speed check). Limits: 2 days, 20 GB per task.
  The reference half runs with `--nice=100` so the queued training jobs stay ahead.
- **Reproduction check:** seed 0 of MDD-PTECT and the final policy repeats the local probe on the cluster.

## Analysis (`analyze.py` -> `analysis.out`, `summary.csv`)
Per policy vs MDD-PTECT over the 5 seeds: pooled %, each seed's %, wins / losses / identical, exact sign test over the
non-identical seeds (no CI with 5 seeds), and the same pick's pooled % at 200 machines in the grid for the trend.

## Notes
- At 1000 machines the floor is far more congested: MDD-PTECT's window tardiness on seed 0 is 53,336 k job-s, about
  53 k per machine, against 14 k at 200 machines and 3-5 k at 20-100. The column tests the policies in that regime,
  which the size-only reading has to keep in mind.
- In the probe (seed 0) the final policy's tardiness equals MDD-PTECT's exactly.
- The task file corrects run_task's episode count for `--pdr none` (it counted it as one pair; that gave the probe's
  rc=1).

## Results
Pending.
