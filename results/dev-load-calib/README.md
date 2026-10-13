# dev-load-calib: which load keeps a 6 h episode stationary, and when does transport bind in the twin? (2026-10-04)

**Why.** rq2-twin-blocks' 6 h episodes showed the backlog growing all episode (tardiness per hour ~3x first to last
hour): the load levels were tuned for 1.5 h windows. Fixed rules only (MDD-TECT, SRT-TECT), twin, failures off,
c ~ U[1.75, 2.5], 21,600 s window after a random warm-up, seeds 0-19. `run.py` -> `run.out`, `runs.json`;
`blocks.py` -> `blocks.out`.

**One regime per episode** (tardiness per hour, 1000 job-s, hours 1-6, MDD-TECT):

| utilization outside lulls | AGVs | hours 1 -> 6 | late % | AGV busy |
|---|---|---|---|---|
| 0.7-1.4 (old default) | 7 | 10.9 -> 48.6 (grows all 6 h) | 50 | 0.41 |
| **0.6-1.2** | 7 | 5.5 -> 15.5 by h3, then 14-16 (**levels off**) | 39 | 0.37 |
| 0.55-1.1 | 7 | 4.7 -> 19.8 at h5 -> 11.9 (noisy) | 33 | 0.36 |
| 0.5-1.0 | 7 | 1.1 -> ~10 | 27 | 0.34 |
| 0.6-1.2 | 3 | like 7 AGVs (transport not binding) | 42 | 0.75 |
| 0.6-1.2 | 2 | 10.4 -> 61.6 (**transport binds, grows**) | 57 | 0.91 |

**Regime-block mixes** (blocks of 5,400 s; normal = 0.6-1.2, surge = 1.0-1.6, short = normal load with 2 AGVs on duty):
B0 normal only: levels off (h6/h3 0.73); B1 normal:surge 2:1: creeps up (1.24-1.31); **B2 normal:surge:short 2:1:1:
levels off at ~31 (1.01-1.11)**; B3 normal:short 3:1: levels off (0.83).

**Decisions taken from it:** normal load = utilization 0.6-1.2; the training regime is B2 (rq2-twin-fleet,
train-due-twin); in the twin transport binds only at 2 AGVs (no zone blocking), Unity at ~4, so Unity fleet blocks use
3 AGVs (rq2-oracle-due-long).
