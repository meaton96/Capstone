# twin-compute: training throughput and compute cost, Unity vs the event-based twin, CPU vs GPU (2026-10-04/05)

**Question.** For the thesis's Unity-vs-twin comparison: how fast does PPO train on each backend, where does the time
go, and how much compute does a training run take (cluster wall time, core-hours, GPU-hours, cloud cost)?

**Setup.** One short PPO run per config on RIT's `sporc` partition (jobs 21801284-21801292), 10 updates each, the
production PPO settings (rollout 128, minibatch 64, 4 epochs), the same network (obs v3, 15 machines x 256 job rows),
reward and training distribution (tardiness, c ~ U[1.75, 2.5], load base / utilhi, machine failures off since the twin
does not model them, 5,400 s windows). Unity: player `~/capstone/linux_server_v3` (= the 10-03 22:24 v3 build); twin:
DES-1k (kinematic) on the 7-AGV floor export, envs spread over worker processes (`--twin-workers`). GPU jobs use
`~/capstone/.venv-gpu` (torch 2.1.1+cu121) on one A100. `env/train.py` writes `timing.json`: seconds in env stepping,
inference (incl. truncation and GAE bootstraps) and PPO updates; steps/s excludes startup. Launcher
`linux_server/slurm/submit_bench.sh` + `bench.sbatch`; analysis `analyze.py` (`summary.csv`).

| config | node | steps/s | env / infer / update | startup | core-h per 1M steps | GPU-h per 1M | days per 3M steps |
|---|---|---|---|---|---|---|---|
| Unity CPU, 6 envs, 18 cores (production) | skl | 6.9 | 87 / 2 / 11% | 125 s | 730 | - | 5.1 |
| Unity CPU, 12 envs, 36 cores | spr | 10.5 | 93 / 1 / 6% | 174 s | 954 | - | 3.3 |
| Unity CPU, 24 envs, 36 cores | spr | 10.7 | 91 / 1 / 7% | 232 s | 931 | - | 3.2 |
| Unity + A100, 12 envs, 24 cores | skl | 7.9 | 98 / 1 / 1% | 246 s | 847 | 35 | 4.4 |
| Unity + A100, 24 envs, 36 cores | skl | 9.7 | 99 / 0 / 1% | 265 s | 1,032 | 29 | 3.6 |
| twin CPU, 16 envs, 36 cores | spr | 150 | 12 / 34 / 54% | 2 s | 67 | - | 0.23 |
| twin CPU, 32 envs, 36 cores | spr | 122 | 8 / 13 / 78% | 2 s | 82 | - | 0.28 |
| **twin + A100, 16 envs, 24 cores** | skl | **380** | 44 / 18 / 38% | 10 s | **18** | **0.7** | **0.09 (~2 h)** |
| twin + A100, 32 envs, 36 cores | skl | 470 | 45 / 13 / 42% | 23 s | 21 | 0.6 | 0.07 (~1.8 h) |

All 9 configs done 10-05 15:25 (the two 36-core GPU configs waited ~22 h on queue priority).
Local 4070 Ti (train-due-twin, 10-05): ~310 steps/s per run with 3 runs sharing the GPU and a Unity sweep on the CPU.

**Findings.**
1. **Unity training is simulator-bound.** 87-98% of the loop is env stepping; a GPU does not help (the A100 run was
   slower, on an older Skylake node), and 24 players are no faster than 12: all envs step in lockstep, so each step
   waits for the slowest player, and the players compete for cores. The production setup (6.9 steps/s) means about
   5 days per 3M steps, as the cluster runs showed.
2. **The twin is network-bound on CPU** (update 54-78%), so it gains most from a GPU: 380 steps/s, **36x the best
   Unity config** and about **50x fewer core-hours** per step (18 core-h + 0.7 GPU-h vs about 930 core-h per 1M).
   A 3M-step run: about 2 h instead of 3-5 days; 32 envs reach 470 steps/s (GPU still not saturated).
3. **Cost per 3M-step run** at illustrative cloud prices of $0.04 per core-hour and $1.50 per GPU-hour (check current
   list prices before citing): Unity about $110, twin + GPU about $5. The ratio does not depend much on the prices.
4. **Startup:** Unity players need 2-4 min to launch and connect (plus the warm-up of every episode), the twin seconds.

**Caveats.** 10 updates per run (steady-state rate, startup excluded); node generations differ (skl = Skylake,
spr = Sapphire Rapids; the fastest twin run was on the older node, so it is if anything understated); minibatch 64 is
not tuned for GPUs (larger minibatches would favour the GPU further); the twin omits machine failures and zone
blocking, so equal steps are not equal realism (that is twin-transfer's question, RQ1).

**For the thesis.** Train and iterate on the twin (hours), and use Unity for evaluation and a final reference run:
the compute paragraph of the twin-transfer comparison, outcome A/B/C in `FUTURE_EXPERIMENTS.md`.
