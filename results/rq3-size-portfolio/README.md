# rq3-size-portfolio: exact checks of the size rule sweep's proxies (2026-10-20, user: step 3)

Status: **3a (regime maps) running locally since 10-20 22:45 (`rq3-size-portfolio-map`, 160 tasks); 3b (portfolio
oracles, `rq3-size-portfolio`, 120 tasks, ~60 core-hours) queued after `rq3-size-catalog-m200`.** Steps 1-2:
`../rq3-size-oracle`, `../rq3-size-catalog`.

## Why
The catalog screen found, by proxy, that at 100 machines the rule that matters is outside the v5 set: the regime map
{short fleet → MDD-TECT, otherwise MDD-PTECT} scored −3.5% vs MDD-PTECT on test seeds, as large as the whole v5
switching oracle there (−3.3%). Proxies piece together runs that held one pair all episode, so they ignore what a
switch carries over. This step measures the same policies exactly, and asks how far a size-specific portfolio raises
the switching ceiling.

## Method (`run.py`, `analyze.py`, `launch_map.sh`, `launch_oracle.sh`)
- **Simulator:** the oracle's frozen simulator on the floor-size grid's conditions (`../rq3-size-oracle/sizesim.py`;
  frozen v5 code and sim.py of rq2-rebase-v5). B2 test seeds 0-39.
- **3a, regime maps:** per floor, the map rq3-size-catalog fitted over all 84 pairs on fitting seeds 2000-2039, applied
  at each 900 s slot's first decision by the regime at that time (as rq2-rebase-v5's map policies):

  | floor | normal | surge | short fleet | proxy (test) |
  |---|---|---|---|---|
  | m15 | ATC-PTECT | MDD-PTECT | MDD-PECT | −0.53% |
  | m30 | ATC-PECT | ATC-PECT | MDD-PECT | −6.11% |
  | m50 | ATC-PTECT | MDD-PTECT | MDD-PECT | −2.33% |
  | m100 | MDD-PTECT | MDD-PTECT | **MDD-TECT** | −3.53% |

  At 15-50 machines the best map over all 84 pairs uses only v5 pairs; at 100 machines it needs MDD-TECT.
- **3b, portfolio oracles:** rq2-rebase-v5's greedy oracle (900 s slots, hold-to-end tails) over
  - m100 `v5tect`: the 6 v5 pairs + MDD-TECT (what adding one rule to the action space would allow);
  - m100 `block6`: the catalog's greedy per-block set, MDD-PTECT, MDD-TECT, ATC-PTECT, FIFO-PTECT, SLACK-TECT,
    MOD-PTECT (block proxy −11.6%);
  - m50 `block6`: MDD-PTECT, ATC-PTECT, CR-TECT, MDD-PECT, ATC-PECT, SLACK-TECT (block proxy −11.2%).

  Each is compared with the v5 oracle on the same seeds (−10.3% at m50, −3.3% at m100).
- **Frozen choices:** `run.py` asserts its maps and 6-pair sets equal rq3-size-catalog's `<cond>/portfolios.json`.
- **Checks:** at m15 the map equals rq2-rebase-v5's exact value for the same map on seeds 0-2 (to 1e-6; rerun at
  launch). The 3b oracles' MDD-PTECT (slot-0 tail) is compared with the catalog's on all seeds. One m100 map run
  (seed 1) holds MDD-PTECT until the short-fleet block starts, then MDD-TECT: 1,185 vs 1,204 k job-s for MDD-PTECT
  held all episode.
- **Cost:** 3a ~6 core-minutes at 4 workers' pace (m100 ~25 s per episode); 3b ~60 core-hours (m100 7-pair ~42 min per
  instance, 6-pair ~36 min; m50 ~11 min).

## Results
Pending.
