# train-size-m100

Training arm of the floor-size study (sizes: 100). Design, code and checks: `../rq3-size-train/README.md`. Results go to the cross-evaluation grid (`rq3-size-grid`).

## Training (done 10-23 07:17, lab `train-size-m100`, cluster jobs 21929100-105)
All 5 seeds trained the full 100k agent steps (48 updates, 3,840 episodes) and saved `checkpoint.pt`
(`results/train-size-m100_s<k>/`, also on the cluster for the grid). Most of the 3.5-day wall time was queue wait; the jobs
started 10-21 to 10-22.

| seed | training time | critic explained variance (last update) |
|---|---|---|
| s0 | 18.6 h | +0.64 |
| s1 | 20.6 h | +0.72 |
| s2 | 19.5 h | +0.74 |
| s3 | 19.1 h | +0.69 |
| s4 | 20.2 h | +0.77 |

The 24 h job limit left 3.4-5.4 h of margin. Episode returns in the logs are not comparable across seeds. The evaluation
is `rq3-size-grid` (arms half, floors up to 100 machines), which waits on `train-size-mix`.

Yardsticks at 100 machines:
- v5 switching ceiling −3.3% vs MDD-PTECT (`../rq3-size-oracle`);
- the observable rule "short fleet → MDD-TECT" −12.2% (`../rq3-size-portfolio`);
- this arm chooses among v5 pairs only, so TECT is outside its action set.

**Evaluated 10-24 in `rq3-size-grid` (arms half):** never significantly worse than MDD-PTECT from 15 to 100 machines;
−3.7 to −4.4% at 20-50. The best single policy for 15-100 machines in the study.
