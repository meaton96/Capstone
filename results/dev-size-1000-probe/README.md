# dev-size-1000-probe

Can the twin run a 1000-machine floor (520 AGVs on duty), and how long does one B2-structured episode take, for MDD-PTECT and for the final policy (cmix s3)? `probe.py`, `launch.sh` (8 h cap each); `timing.json`, `probe_*.out`, `probe_*.time` (peak memory). Floor: `../dev-size-floors/floors/m1000`. Part of `../rq3-size-train/README.md`.

## Results (done 10-20 05:36)
| run | wall time, one episode | peak memory | window tardiness (job-s) | jobs exited | decisions |
|---|---|---|---|---|---|
| MDD-PTECT | 21,424 s (5.95 h) | 7.7 GB | 53,335,636 | 15,015 | 71,359 |
| final policy (cmix s3) | 21,039 s (5.84 h) | 7.7 GB | **53,335,636** (identical) | 15,015 | 71,359 |

- **The twin runs a 1000-machine floor:** 1,000 machines, 520 AGVs on duty, observation caps 1,000 machine rows /
  17,152 job rows. It takes about 6 h and 7.7 GB per episode on one core.
- **The final policy's episode is identical to MDD-PTECT's**, with the same tardiness, jobs and decision count. At
  1000 machines the policy makes no choice that changes the outcome from MDD-PTECT's (its picks either were MDD-PTECT
  or did not matter).
- **probe.py exit code 1 is a harness bug, not a failure.** rq3-gen-v5's `n_policies` counts `--pdr none` as one pair
  and so expected 2 episodes. The episode completed and `episodes.csv` is complete.
- **For the grid:** a full 40-seed, 26-policy cell at 1000 machines would be about 6,000 core-hours. Feasible only
  reduced, e.g. 5 seeds × (MDD-PTECT, the best 2-3 pairs, each arm's pick), about 200 core-hours.
- **Caveat:** 0.52 AGVs per machine may under-supply transport on a 186 × 233 m floor (trips grow with the floor; the
  twin's knee law has AGVs per machine rising as M^0.3). So "1000 machines" here also means "transport-heavier".
