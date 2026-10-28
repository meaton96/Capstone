# train-size-mix

Training arm of the floor-size study (sizes: 15,50,100). Design, code and checks: `../rq3-size-train/README.md`. Results go to the cross-evaluation grid (`rq3-size-grid`).

## Training (done 10-24 01:48, lab `train-size-mix`, cluster jobs 21929106-08, 21929112, 21929114)
- **Steps:** all 5 seeds trained the full 100k agent steps and saved `checkpoint.pt` (`results/train-size-mix_s<k>/`, also
  on the cluster for the grid). With 12 envs that is 65 updates and 4,152 episodes, mixed over the 15 / 50 / 100-machine
  floors.
- **Wall time:** the jobs ran one after another from 10-23 morning to 10-24 01:48; most of the 4.3-day lab wall time was
  queue wait.

| seed | training time | critic explained variance (last update) |
|---|---|---|
| s0 | 15.8 h | +0.84 |
| s1 | 14.8 h | +0.77 |
| s2 | 15.2 h | +0.65 |
| s3 | 15.7 h | +0.75 |
| s4 | 17.0 h | +0.58 |

- **Job limit:** the 24 h limit left 7-9 h of margin.
- **Returns:** episode returns in the logs are not comparable across seeds (the floor sizes differ by episode).
- **Evaluation:** `rq3-size-grid`'s arms half (Slurm 21959231, 72 tasks, started 10-24 01:49), together with
  `train-size-m50` / `-m100`.

**Evaluated 10-24 in `rq3-size-grid` (arms half):** flat near MDD-PTECT (−0.65 to +0.06%), mainly because s0 and s2
collapsed to MDD-PTECT (identical on every test instance); s3 is −0.8 to −6.0% on every floor.
