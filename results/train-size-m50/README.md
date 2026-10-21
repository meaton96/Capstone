# train-size-m50

Training arm of the floor-size study (sizes: 50). Design, code and checks: `../rq3-size-train/README.md`. Results go to the cross-evaluation grid (`rq3-size-grid`).

## Training (done 10-21 00:59, lab `train-size-m50`, cluster jobs 21929092-96)
All 5 seeds trained the full 100k agent steps (48 updates, 3,840 episodes) and saved `checkpoint.pt`
(`results/train-size-m50_s<k>/`, also on the cluster for the grid). Most of the 29 h wall time was queue wait.

| seed | training time | critic explained variance (last update) |
|---|---|---|
| s0 | 7.3 h | +0.63 |
| s1 | 10.3 h | +0.70 |
| s2 | 10.0 h | +0.36 |
| s3 | 10.6 h | +0.59 |
| s4 | 9.5 h | +0.60 |

Episode returns in the logs are not comparable across seeds (each update draws different instances); the evaluation is
`rq3-size-grid` (arms half), which waits on `train-size-m100` and `train-size-mix`.

Yardstick (`../rq3-size-oracle`): the v5 switching oracle at 50 machines is −10.3% vs MDD-PTECT, the same room as at 15
machines (−9.4%); the best observable v5 regime map is only −0.6% exact there (`../rq3-size-portfolio`).
