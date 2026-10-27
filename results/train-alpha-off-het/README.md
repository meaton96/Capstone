# train-alpha-off-het: the spread law sets α, the head picks an offset, training arm on spread floors (2026-10-25, user: "the law sets α, the head picks an offset")

Status: **trained 10-26** (lab `train-alpha-off-het`, local, 18:55-22:02, 3.0-3.1 h per seed). All 5 seeds reached
48 updates (step 98,304). Evaluated 10-26 (`../eval-alpha-offset`): −16.40% vs MDD-PTECT at s 1, +0.70 vs the final policy on B2
(n.s.); H1' and H2' PASS. Evaluation: `../eval-alpha-offset` (a lab future waiting on this arm and `../train-alpha-off-b2`).

## Question
Does law + offset keep `../train-alpha-het`'s gain on spread floors (−5 to −13% vs the final policy at s 0.5-1) without
its +4.1% cost on B2? The law detects the spread per floor; the head only adjusts α around it (e.g. lower in short-fleet
blocks, as the v6 arms did).

## Setup
- **Code:** `code/` = `../dev-alpha-offset/code` (v7 action schema: job {SRT, MDD, ATC} × machine {PECT, PTECT} ×
  offset {×1, ×0.5, ×0.25, ×1.5, ×2}; α = min(4, max(0, 3 − 6σ) × offset), σ = std of the floor's log machine speed
  factors; checks O1-O4 passed, see its README) plus `c_range` (1.25 3.5).
- **Recipe:** the final policy's (train-v5-cmix), exactly as `../train-alpha-het`:
  - B2 floor7, 7 AGVs (2 in short-fleet blocks), B2 load mix, c ~ U[1.25, 3.5] in training;
  - 16 envs, shared-baseline groups of 4, rollout 128, batch 256;
  - 900 s slots, 5,400 s discount horizon, entropy 0.01 → 0.001;
  - 100k agent steps (48 updates), no prior, no KL.
- **Speed spread:** 0,1 (s ~ U[0, 1] per instance, `code/hetero.py`, unchanged from `../train-alpha-het`: factors
  log-uniform in [1/(1+s), 1+s], capacity-matched per type, B2-tight due dates). The same training instances as
  `../train-alpha-het` for the same seed.
- **Launch:** `launch.sh` → `../scripts/alpha_offset_train.sh train-alpha-off-het "0 1 2 3 4" 2 0,1`. Seeds 0-4 run in
  parallel locally, one process per seed with 2 twin workers and a 12 GB address-space cap. The launcher resumes from
  the latest checkpoint and skips seeds with `DONE_s<seed>`. It only trains.
- **Outputs:** `../train-alpha-off-het_s<seed>/` (checkpoints, TensorBoard with per-head entropies, `hetero.json`) and
  `train_s<seed>.out`.
