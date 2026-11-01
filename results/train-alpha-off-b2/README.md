# train-alpha-off-b2: the spread law sets α, the head picks an offset, training arm on B2 (2026-10-25, user: "the law sets α, the head picks an offset")

Status: **trained 10-26** (lab `train-alpha-off-b2`, local, 19:08-22:10, about 3 h per seed). All 5 seeds reached 48
updates (step 98,304). Evaluated 10-26 (`../eval-alpha-offset`): −2.55% on B2 (−0.87 vs the final policy, p 0.048) and −16.42% at
s 1 without spread floors in training. Evaluation: `../eval-alpha-offset` (a lab future waiting on this arm and `../train-alpha-off-het`).

## Question
On the training floor itself, does the offset head only add to the final policy's action space? On B2 σ = 0, so the law
is α 3 and offset ×1 is exactly the final policy's action; the other offsets reach α 0.75 / 1.5 / 4. Expect ≈ the final
policy (B2's α curve is flat within about 1% between 1.5 and 4), perhaps a lower α in short-fleet blocks
(`../train-alpha-b2`: 1.8 there vs 2.4).

## Setup
- **Code:** `code/` = `../dev-alpha-offset/code` (v7 action schema; checks O1-O4 passed) plus `c_range` (1.25 3.5).
- **Recipe:** the final policy's, exactly as `../train-alpha-b2` (see `../train-alpha-off-het/README.md`).
- **Speed spread:** 0,0: the plain B2 generator through `code/hetero.py` (checked identical in `../dev-alpha-action`);
  σ 0 on every instance.
- **Launch:** `launch.sh` → `../scripts/alpha_offset_train.sh train-alpha-off-b2 "0 1 2 3 4" 2 0,0`. Seeds 0-4 in parallel
  locally, one process per seed with 2 twin workers and a 12 GB address-space cap; resumes; train only.
- **Outputs:** `../train-alpha-off-b2_s<seed>/` and `train_s<seed>.out`.
