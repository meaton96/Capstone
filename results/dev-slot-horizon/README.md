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

## Result

(pending: `analysis.out`)
