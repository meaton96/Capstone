# 12 head-rule baselines via evaluate.py (2026-09-27)

These are the fixed-rule baselines for the two-head action space (`docs/features/DECISION_POINTS.md` §1). They were
run through `evaluate.py`, the same path `rnd02` checkpoints will be scored on.

## Setup

- **Player:** `linux_server_dev` (two-branch build, 2026-09-26 17:20).
- **Instances:** `--scenario-generator randomized --episode-duration-seconds 5400`, held-out seeds 0-19. Seeds 0-8
  are the `rnd_load_s0-8` instances.
- **Runs:** `run.sh` (4 workers × 5 seeds) and `run_part3.sh`. Seeds 15-19 split in two because queuing all 60
  scenarios at once exceeded gRPC's 4 MB cap (`part3/` is that failed attempt; ignore it).
- **Merged:** `episodes_all.csv` (240 episodes) and `summary_all.csv`. Gaps are to the best of the 12 rules on
  each seed.

## Results

- 240/240 episodes, 0 deadlocks, 0 timeouts; every episode truncated at 5400 s.
- **Best on average:** SPT-ECT on total flow (+8.1% mean gap). On mean flow per exited job, SRT-TECT (+2.8%) and
  SRT-ECT (+3.1%).
- **No rule dominates.** Total-flow wins spread over 9 of the 12 rules (max 4 each), and worst/best per seed
  averages +25.8%.
- **FIFO is last on every metric** and never wins.

## Caveats

- **Only the first 5400 s from t = 0.** evaluate.py has no random warm-up, so these episodes cover only the start
  of each scenario. Training (`--random-warmup`) starts at random phase offsets.
- **Both flow metrics are censored.** They count only jobs that exited inside the window, and jobs exited varies
  44-52 by rule. Read total flow together with jobs exited.
- **The ranking differs from the batch-runner sweep** (`gen_rules0926`, full episodes, seeds 0-8): per-seed
  Spearman rank correlation 0.06-0.87. ECT/TECT still lead and FIFO trails, but the order inside the top group is
  regime-dependent.
