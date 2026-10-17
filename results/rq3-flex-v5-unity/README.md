# rq3-flex-v5-unity: machine flexibility in Unity for the v5 rules and the final policy (2026-10-17, user)

Status: **smoke queued** (lab `dev-flex-unity-smoke`, then `rq3-flex-v5-unity`, cluster, 120 Unity tasks).

## Why
RQ3's conditions include machine flexibility. The twin refuses it, because Unity draws each machine's secondary
capabilities from the scenario seed, and the only flexibility result (`rq3-flex`, 10-02) is outdated: flow time,
old rules, pre-audit build. The user (10-17) asked to set it up with the final policy, the PPO trained on mixed due
dates (cmix pick s3), and to report all policies' numbers.

## Method (`task.sh`, `array.sbatch`, `submit.sh`, `analyze.py`)
- **Instances:** Unity, B2 test seeds 0-19 with `--machine-flexibility p --secondary-time-multiplier 1.25`. Each
  machine can also run each other type with probability p, and a secondary type runs 25% slower. Levels:
  `flex-p0.3`, `flex-p0.5`.
- **Settings:** the same code, player and settings as `../eval-v5-cmix-unity` (6d13412a snapshot, `linux_server/`
  5e6e0f87 without `-pyrules`, random warm-up, 21,600 s window, 900 s slots, deterministic).
- **Groups:**
  - fa: SRT-PECT, SRT-PTECT, MDD-PECT, MDD-PTECT;
  - fb: ATC-PECT, ATC-PTECT, B2-trained s4;
  - fc: MDD-PTECT, loose s2, cmix s3.

  fc's MDD-PTECT repeats fa's: a check that the flexible floor is identical per seed.
- **Reference:** no flexibility = B2 in Unity on the same seeds (rq4-agvfail-v5-rules level none, eval-v5-cmix-unity
  base).
- **Analysis:** per level, the best pair and the ranking vs no flexibility (Kendall τ); the picks vs MDD-PTECT and vs
  the best pair; and the cost of flexibility per policy.
- **Smoke first:** one task (flex-p0.5 / fc / s0) on the cluster. The full run waits until it is checked.

## Results
Pending.
