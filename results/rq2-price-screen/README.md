# rq2-price-screen (2026-10-02)

**Question:** on the 15-machine floor, does TECT's travel price λ (score max(travel, queue) + p + λ × travel) change
the best fixed machine rule? And does the best λ change with how scarce AGVs are? This screen comes before making
priced TECT its own machine rule.

**Answer:** a little. The price helps only where AGVs bind, and by about 1%.

| regime | best rule on average | its gain over plain TECT | per-seed best variant vs best-on-average rule |
|---|---|---|---|
| base-w (7 AGVs) | ECT (λ makes TECT 0.2-0.4% worse) | 0.24% | 0.69% |
| agv4 | TECT λ 2 | 0.96% | 0.50% |
| agv3 | TECT λ 1 (λ 4: +1.2% worse) | 0.36% | 0.44% |
| amax1 | TECT λ 2 | 1.38% | 0.37% |

- **Best λ shifts with the fleet:** 0 with 7 AGVs, 1-2 when AGVs bind, and λ 4 is too much on the 15-machine floor.
  This is the same direction as the linked floor (rq2-linked-price).
- **Per seed** the price moves time in system by −3.8% to +2.2% against plain TECT. Even picking the best variant
  per seed beats the best-on-average rule by only 0.4-0.7%.
- **Jobs exited** barely change (107-110).
- **Implication:** adding priced TECT to the machine head gives the oracle a rule that is sometimes better. But per-seed
  choice among ECT / TECT / priced TECT is worth under 1% over the best fixed rule, so this lever alone is unlikely to
  create the several-percent switching headroom training needs. It is worth adding only together with a regime where
  AGV supply changes within the episode (AGV breakdowns, bursts) or on the linked floor (λ 1: 0.8-2.4%).

**Setup:**
- **Regimes:** base-w, agv4, agv3 and amax1 from rq2-oracle-screen, each with its overrides.
- **Episodes:** randomized generator, random warm-up, 5,400 s agent window, layout D, job rule SRT, seeds 0-4.
- **Rules:** SRT-ECT, and SRT-TECT at λ 0 / 1 / 2 / 4, with one `evaluate.py` per (regime, variant, seed).
- **Player:** `linux_server/` (10-02 16:13 build), 80 jobs (100 episodes), 0 failed, 18:49-19:17.
- **Check:** all 16 SRT-ECT / SRT-TECT λ 0 episodes that overlap the oracle screen (seeds 0-1) equal its stage-1
  fixed returns exactly.
- **Duplicate seeds:** amax1 equals base-w on seeds 0 and 2, because the arrival cap does not bind there (seen before
  in the oracle screen).

**Files:** `run.sh`, `analyze.py` (prints the tables), `summary.csv`, `episodes_all.csv`, and per-job folders
`<regime>/<variant>/s<seed>/`.
