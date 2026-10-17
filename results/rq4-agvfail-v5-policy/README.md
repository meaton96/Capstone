# rq4-agvfail-v5-policy: the v5 policies under AGV breakdowns (Unity, 2026-10-14)

Status: **done** 2026-10-15 00:51 (cluster array 21894124, 280 / 280). Results and reading:
`../rq4-agvfail-v5-rules/README.md` ("Results (policy half)").

The policy half of the RQ4 v5 rerun. Setup, groups, checks and the analysis are shared with the rules half and
described in `../rq4-agvfail-v5-rules/README.md`. Its outputs are `runs/<level>/{ppoa,ppob,bc}/s<seed>/`. Policies:
- PPO s0-s4 (train-v5-nokl, train-v5-final) and BC v5 s0-s2 at l3000 / l1500;
- PPO s3-s4 also without breakdowns (s0-s2 and BC at none are in `../eval-transfer-unity-v5`).
