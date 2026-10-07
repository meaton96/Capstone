# rq2-oracle-due-long: Unity switching headroom at the calibrated load (2026-10-04 .. 10-06)

Unity H15 greedy switching oracle, tardiness objective, at the calibrated load (utilization 0.6-1.2 outside lulls,
dev-load-calib), c ~ U[1.75, 2.5], random warm-up, machine failures as generated. Player `linux_server_v4`, frozen
Python `env_snapshot/` (10-04 19:15), seeds 0-7 per setting (`run.sh`, per-setting `analysis.out`). It ran 22.9 h
locally, slowed by parallel twin training.

| setting | window | slots | regime | best fixed pair on average | oracle gain vs it (mean / median) | vs each seed's own best pair (mean / median) |
|---|---|---|---|---|---|---|
| long-u | 6 h | 1,800 s (12) | one regime, normal load | ATC-ECT | 10.8% / 9.7% (8/8 seeds > 2%) | 8.5% / 8.8% |
| blocks-u | 6 h | 1,800 s (12) | blocks of 1.5 h: normal 2 : surge 1 : short fleet 1 (3 AGVs on duty) | MDD-TECT | 12.8% / 8.8% (8/8) | 6.8% / 5.6% |
| short-u | 1.5 h | 900 s (6) | one regime, normal load | SRT-TECT | 22.0% / 13.8% (7/8) | 14.5% / 6.1% |

(short-u's mean is inflated by seeds with almost no tardiness: seeds 2, 4, 5, 6 have 0.6-3.3 thousand job-s, so small
absolute changes are large percentages; read the median.)

## Findings

1. **The headroom found on the twin is there in Unity too.**
   - blocks-u (the Unity counterpart of the twin's B2 training regime): +12.8% mean / +8.8% median over the best pair
     on average, with 30-min slots.
   - Twin B2: +10.8% mean over MDD-TECT, with 15-min slots.
   - The oracle uses several job families in Unity too: MDD 40%, ATC 31%, SRT 28% of slots on blocks-u.
2. **The best fixed pair depends on the regime and the simulator.**

   | setting | best pair on average | where MDD-TECT ranks |
   |---|---|---|
   | long-u | ATC-ECT | 5th, 11.3% from the per-seed best |
   | blocks-u | MDD-TECT, but MDD-ECT, ATC-ECT and ATC-TECT are within 1.1 points of it | 1st |
   | short-u | SRT-TECT by tardiness sum; MDD-ECT best by mean gap | |
   | twin B2 | MDD-TECT, clearly | 1st |

   An action prior on a single pair (train-due-twin-slot-prior-kl uses MDD-TECT) is therefore a choice tied to the
   regime: right for B2-like training on the twin, wrong for long-u. Any Unity training with a prior should pick the
   pair from that regime's fixed-pair results, not reuse MDD-TECT.
3. **Each instance has its own best pair here too.**
   - Per-seed best pairs on long-u: ATC-TECT 3, ATC-ECT 3, EDD-ECT 1, MDD-ECT 1.
   - Choosing the right fixed pair per instance in hindsight is worth 2.8-7.7% (gap of the best-on-average pair to the
     per-seed best). Switching within the episode adds a further 6.8-8.5% on top.
   - Compare with base-eplen (cluster, running) on whether the per-instance preference fades with longer episodes.
4. **SRWT machine rules are never competitive** (34-65% from the per-seed best), as on the twin.
5. **Time in system is not traded off:** the tardiness-oracle schedules are also 1.1-1.7% better on time in system
   than each seed's best fixed pair (long-u, blocks-u).

## For the open decision (train-tardiness: regime and backend)

- Both 6 h settings show about 9-13% headroom in Unity, so either regime is worth training on.
- blocks-u matches the twin's B2 regime, except that the short fleet has 3 AGVs on duty in Unity vs 2 on the twin. It
  is the natural choice if the policy is pre-trained on the twin (twin-transfer).
- Caveats:
  - The twin results since 10-05 suggest RL struggles to capture this headroom (slot policies at parity with
    MDD-TECT; the action prior collapsed to MDD-TECT).
  - dev-oracle-bc (running) tests whether the network can imitate the oracle at all.
  - rq2-oracle-slotlen (cluster) tests how much headroom is lost with 30-min instead of 15-min slots. The Unity runs
    here used 30-min slots.
