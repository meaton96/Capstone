# rq2-oracle-due: Unity switching oracle on the tardiness objective (2026-10-03)

**Question.** The twin screen (`../rq2-twin-due/README.md`) found that due dates at c = 2 with due-date job rules give
a switching oracle a 6.5% median gain over the best fixed pair. Does that hold in Unity, with machine failures on?
Step 3 of `docs/Plans/PDR_RULE_SET_PLAN_2026-10-02.md` §9.1.

**Answer: yes, narrowly.** Oracle against the best-on-average fixed pair, MDD-TECT (tardiness integral over the
window):

| measure | Unity (8 seeds) | go bar |
|---|---|---|
| median gain | 3.43% | >= 3% |
| mean gain | 5.69% | |
| seeds above 2% | 5 / 8 (62%) | >= 60% |
| switching alone (vs each seed's best pair), median | 1.43% | |
| vs SRT-TECT (best flow rule), median | 3.94% | |

- **MDD-TECT is the best fixed pair** (2.7% from the per-seed best on average, 3 of 8 wins); SRT-TECT (2 wins) and
  ATC-ECT (2) follow; SRWT pairs are 22-36% behind.
- **The oracle uses MDD 42%, SRT 38%, ATC 15%** of the slots; EDD 4%, SPT 2% (both drop candidates for the head).
- **Flow time does not suffer:** the oracle schedules have 0.37% less time in system than each seed's best flow pair.
- **8 seeds is thin:** seed 0 alone gives +20%, three seeds give under 2%. The twin's 40-seed halves differed 2.5% vs
  7.6% in median, so the Unity median is uncertain by a few points either way.
- **Twin cross-check (failure-free seeds 0, 2, 4):** fixed-pair tardiness within 1.9-6.2% of the twin, rank correlation
  0.66-0.96, best pair the same on 1 of 3; Unity's switching gain is larger than the twin's on all three (11.8 / 5.6 /
  1.1% vs 1.6 / 2.2 / 0.0%).

**Setup.** `run.sh`: `env/switch_oracle.py` per seed, randomized generator (machine failures on seeds 1, 3, 5, 6, 7),
`--params '{"due_date_allowance": 2.0}'`, random warm-up, 5,400 s window, 6 x 900 s stages, pairs H15 (job SRT / SPT /
MDD / EDD / ATC x machine ECT / TECT / SRWT), `env/config/rewards/tardiness.json`, player `linux_server_dev/` (10-03
01:58, `BUILD_NOTES.md`). 8 workers, 10-03 02:34-05:48, 720 episodes. `analyze.py` -> `analysis.out`, `per_seed.csv`.
In the per-seed lines of `run.out`, "time in system" is switch_oracle's label for the reward quantity, here tardiness.
