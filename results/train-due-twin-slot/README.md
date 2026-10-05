# train-due-twin-slot: PPO on the twin with slot actions and a 3 h horizon (2026-10-05, running)

Same regime as train-due-twin; the agent picks a rule pair per 900 s slot, held for every decision in it
(`env/env_wrappers/slot_env.py`, `--slot-seconds 900`), discount horizon 10,800 s, 3 seeds x 100k agent steps
(~4,200 six-hour episodes each), rollout 32. Why: `../eval-due-twin/README.md`. Wrapper checks (10-05): MDD-TECT held
per slot equals the fixed run exactly; replaying an oracle schedule reproduces the oracle exactly (B2 seeds 0 and 3).
Evaluation: `../eval-due-twin-slot/` (queued in the lab). Results: added here when done.
