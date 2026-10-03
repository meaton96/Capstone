# dev-due-check: Unity's due-date rules and tardiness reward (2026-10-03)

- **Parity (passed):** `python -m des_twin.parity --agvs 1 --max-decisions 400` on `due2_s0.json` (randomized seed 0,
  c = 2, failures off) for MDD_ECT, ATC_TECT, EDD_SRWT: 400 / 400 decisions with identical observations and masks, and
  the player's exported jobs (with due dates) equal the Python resolution. Run with `--out <player>/Results` and moved
  here, because since eb61a6c2 `-decisionlogdir` is overwritten under ML-Agents (separate fix).
- **Tardiness smoke (`tard_smoke/`):** evaluate.py, c = 2, SRT-TECT / MDD-TECT / ATC-ECT / EDD-ECT x seeds 0-1: return
  = -window_tardiness / 1000 exactly; flow-rule trajectories unchanged by due dates. Seed 0 against the twin: time in
  system 0.3-3.3% apart, tardiness 0.6-15.5% (the allowance amplifies flow differences), best pair ATC-ECT in Unity
  vs SRT-TECT in the twin. `tard_smoke_guard.out`: first attempt, stopped by the reward's no-due-date guard on the
  player's unseeded start-up episode (guard now skips seed -1).
