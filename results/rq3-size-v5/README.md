# rq3-size-v5: floor size in the twin (50 and 100 machines) for the v5 rules and the final policy (2026-10-17, user)

Status: **queued** (lab `rq3-size-v5`, cluster, 8 twin tasks).

## Why
RQ3's conditions include floor size, which had no current result. The 10-14 claims doc said "the twin can't run it",
but that holds only for tiled 105-machine floors. Plain 50- and 100-machine floors load in the twin
(agv-ratio-twin-v5, rq4-mfail-ccal). The user (10-17) asked to set this up with the final policy, the PPO trained on
mixed due dates (cmix pick s3), and to report all policies' numbers.

## Method (`task.py`, `submit.sh`, `pull.sh`, `analyze.py`)
- **Harness:** a thin wrapper around `../rq3-gen-v5/run_task.py` (same frozen code, B2 window / slots / reward and
  fleet wrapper). It adds two conditions:
  - **m50:** 50 machines (10 per type), layout D, floor `../agv-ratio-scale/floors/m50` (50 bays). This is
    dev-hetero-tune's validated m50 family: 26 AGVs on duty (short-fleet blocks 13), the B2 load mix × 0.9, arrival
    cap off, due dates c ~ U[1.75, 2.5].
  - **m100:** 100 machines (20 per type), floor `../agv-ratio-scale/floors/m100`, the same per-machine scaling:
    52 AGVs (short 26).
  - **m15** is B2 itself (rq3-gen-v5 base + eval-v5-cmix test base).
- **Groups:**
  - pdr: the 6 v5 pairs + MDD-TECT;
  - ppo: the B2-trained s0-s4;
  - cmix: s0-s4;
  - loose: s0-s4.

  Every policy group also runs MDD-PTECT as a run-to-run check. Every policy was trained on the 15-machine floor
  only, so this measures transfer.
- **Analysis:** per floor, the best pair and the pair ranking vs 15 machines (Kendall τ), and each pick / family mean
  vs MDD-PTECT and vs the floor's best pair.
- **Checks (local smoke, seed 0):**
  - m50 / cmix: 6 episodes in 100 s (cmix s3 −2.9%, s0 +4.6% vs MDD-PTECT).
  - m100 / cmix: 6 episodes in 297 s.
  - No observation-cap warnings; the network's set encoders take 50 / 100 machine rows.

## Results
Pending.
