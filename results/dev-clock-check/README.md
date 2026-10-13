# dev-clock-check: does one Unity player stay exact past 2^17 s on the SimTime build? (2026-10-03)

base-due-c2 seed 0 (21 pairs, ~188,800 simulated s) in ONE player with `--allow-clock-drift`, player
`linux_server_due/`. Every pair shared with `rq2-oracle-due/s0` stage 1 (run in players under 120,000 s) equals it to
<= 3.6e-15, including the six pairs that ran entirely past 2^17 s (MDD-ECT ... ATC-SRWT). **The SimTime fix (17f1a9e0)
holds**; evaluate.py's PLAYER_SIM_BUDGET_S, `--allow-clock-drift` and switch_oracle.py's per-stage player splitting
were removed (ec417a6b). Players built before 3d22f9c7 still drift past 131,072 s.
