# dev-slot-horizon: a shorter discount horizon in slot mode, and both objectives logged (2026-10-07, handoff fix 2)

## Why

- Audit M3: training maximizes a discounted return (horizon 10,800 s, about 0.92 per 900 s slot) with a bootstrap past
  the window, while evaluation and the oracle score the undiscounted 6 h window. With 1-2% headroom, that mismatch is
  as large as the signal.
- The credit trace (`docs/experiments/review_1007/CREDIT_ASSIGNMENT_TRACE_1007.md` 4b) measured the signal-to-noise of
  one slot's choice: 0.048 at a 3,000 s horizon against 0.027 at 10,800 s, about 3x fewer samples for the same
  confidence. A longer horizon doubles the effect but multiplies the noise by 5.

## What changed

- No new training code: `--discount-horizon-s 5400` (the upper end of the handoff's 3,000-5,400 s range).
- `env/evaluate.py` now reports each episode's discounted return (`discounted_return`, the training gamma, no
  bootstrap) next to the scored window tardiness. It takes the gamma from the checkpoints (`--discount-horizon-s`
  overrides it; 0 = off), and a slot env discounts within its slots with the same gamma.

## Run

dev-prior-offset with `--discount-horizon-s 5400` (seeds 0-1, 100k steps, offset prior, KL 0.05). Compare with
dev-prior-offset (10,800 s), which differs only in the horizon.

## Result (10-08 04:57, cluster CPU; `analysis.out`)
- **Held-out B2 seeds 0-39:**
  - s0 plays MDD-TECT in 100% of slots (= MDD-TECT, 0.000%).
  - s1 plays it in 96.7% (MDD-SRWT 2.7%) and is +0.4% (4 wins, 5 losses).
- **The critic improved:** explained variance is 0.84-0.85 at the 5,400 s horizon, against 0.75 at 10,800 s. That is
  the expected variance cut of the shorter horizon.
- **The policy didn't leave the prior:** mean p(ATC) 0.02-0.04, Spearman with ATC-ECT's gap −0.31 / −0.33.
- **Reading:** the shorter horizon makes the value estimate better but does not overcome the MDD-TECT prior's KL pull.
  Same outcome as dev-prior-offset and dev-big-batch.
