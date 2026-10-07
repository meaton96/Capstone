# dev-oracle-bc-ev: behavior cloning on expected-value oracle labels (2026-10-07, handoff fix 7)

## Why

dev-oracle-bc cloned the switching oracle's choice on the instance's one realized future. The credit trace showed that
a slot's hindsight-best choice on one future predicts its sign on other futures only about half the time. So those
labels are mostly noise, and cloning them overstates the realizable headroom. The handoff asks for:
- labels from an expected-value oracle (the mean over several redrawn futures per slot);
- better still, regret labels that switch away from the default only when the predicted gain beats its uncertainty
  (Wei et al. 2026, arXiv:2605.23957).

## What changed

- `collect_ev.py` (new):
  - dev-oracle-bc's H15 greedy oracle, but at each slot the 15 pairs' hold-to-end tails are also scored on M redrawn
    futures (splice at the slot's first decision).
  - The schedule follows the best mean over the realized and the redrawn futures, and the obs is captured along it.
  - Stores `tails_fut` (24, M, 15) next to the realized `tails`.
  - Each spliced run is checked to reach the slot start in the same state.
- `../dev-oracle-bc/bc_train.py --labels`:
  - `oracle` (default, unchanged): the realized future;
  - `ev`: the mean over futures;
  - `ev-safe`: a pair other than `--default-pair` (MDD-TECT) counts by the lower confidence bound of its paired gain,
    mean - z SE (`--safe-z`, default 1). Exact ties go to the default.

**Local check** (seed 0, 3 slots, 2 futures, `check/`):
- slot-0 realized tails equal collect.py's exactly, the replay is exact, and there are 0 state mismatches;
- the EV choice matched the realized-future best on 1 of 3 slots (the EV schedule picked MDD-TECT where the single-future
  oracle picked SRT-ECT).

## Runs (lab)

- `dev-oracle-bc-ev-collect` (cluster, future): 4 futures, B2 0-39 + 2000-2159, about 75 CPU-min per seed (about
  250 CPU-h). It waits for rq2-oracle-steady and rq2-realizable: `env/` must be synced to the cluster first, and not
  while jobs run from the synced code.
- `dev-oracle-bc-ev` (local, after it): `run.sh`. Runs bc_train with `ev` and `ev-safe`, then rolls out the cloned
  networks on B2 0-39 with the three references. Compare with dev-oracle-bc (-2.2 to -2.6% vs MDD-TECT).

## Result

(pending)
