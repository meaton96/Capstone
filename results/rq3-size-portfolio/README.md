# rq3-size-portfolio: exact checks of the size rule sweep's proxies (2026-10-20, user: step 3)

Status: **3a (regime maps) done 10-20 22:50 (`rq3-size-portfolio-map`, local, 5 min, 160 / 160); 3b (portfolio
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
### 3a: the fitted regime maps, exact (10-20)
B2 test seeds 0-39, vs MDD-PTECT held all episode, pooled; < 0 better. `analysis.out`, `summary.csv`.

| floor | map (normal / surge / short fleet) | exact | proxy | v5 switching oracle | map vs v5 oracle (paired) |
|---|---|---|---|---|---|
| m15 | ATC-PTECT / MDD-PTECT / MDD-PECT | −0.96% [−3.35, +1.20], 22/17, p 0.30 | −0.53% | −9.42% | +9.3%, 5/35 |
| m30 | ATC-PECT / ATC-PECT / MDD-PECT | **−5.01%** [−7.91, −2.31], 31/9, p 0.0006 | −6.11% | (not run) | |
| m50 | ATC-PTECT / MDD-PTECT / MDD-PECT | −0.55% [−3.39, +2.11], 27/13, p 0.16 | −2.33% | −10.27% | +10.8%, 0/40 |
| m100 | MDD-PTECT / MDD-PTECT / **MDD-TECT** | **−12.21%** [−16.27, −8.08], 28/6, p 4e-5 | −3.53% | −3.29% | **−9.2%**, 19/21, p 0.14 |

- **At 100 machines a one-line rule beats the v5 switching oracle's whole ceiling almost fourfold.** Use MDD-TECT
  while the fleet is short, MDD-PTECT otherwise: −12.2% vs MDD-PTECT, and −10.5% vs MDD-TECT held all episode.
  - **Not driven by a few instances:** the median instance gains 5.6%. On the 6 seeds with no short-fleet block in
    the window the rule equals MDD-PTECT; of the 34 where it acts, it wins 28. Without the 1 / 3 / 5 heaviest seeds the
    pooled gain stays −12.7 / −11.5 / −11.2%. Single instances gain up to 25% (seed 13: 1,749 vs 2,319 k job-s).
  - **Against the v5 oracle** it wins on the heavy instances (seed 13: 1,749 vs the oracle's 2,312) and loses
    narrowly on light ones where it does nothing (19 / 21; pooled −9.2%, p 0.14). The v5 oracle cannot choose TECT.
  - **The proxy understated it fourfold** (−3.5%): it credits each pair only with tardiness accrued inside its own
    block, from a run that held that pair all episode. Handling the short-fleet block with TECT also cuts the lateness
    that spills into the blocks after it, which the exact run counts.
  - **Observable:** the rule needs only the number of AGVs on duty (the short-fleet regime halves it), applied at the
    next 900 s slot.
- **At 30 machines the v5 map is a solid observable rule** (−5.0%, p 0.0006: ATC in normal / surge, MDD-PECT when the
  fleet is short), about as good as the 15-machine-trained policy family there (−4.6%, `../rq3-size-grid`).
- **At 15 and 50 machines the maps gain little and are not significant** (−1.0%, −0.6%), consistent with
  rq2-rebase-v5's finding that observable maps over v5 capture almost nothing of the oracle at 15 machines.
- **Reading for the thesis:**
  - The best observable rule changes with floor size, and at 100 machines the decisive rule (TECT) lies outside the
    rule set the policies were trained to choose from. That set was picked on the 15-machine floor, where MDD-TECT
    is 18.5% worse than MDD-PTECT.
  - At 100 machines, the policies (the 15-machine family +2.6%; `train-size-m100` still training on v5 actions) are
    far behind a fixed rule the floor's size suggests. The size limit is partly an action-space limit, not only a
    transfer limit.
  - Why TECT on big floors and not small ones: short-fleet blocks put 26 AGVs on 100 machines, against 2 on 15, so
    AGVs per machine are not scarcer on the big floor. The difference is distance: trips are much longer on a
    100-machine layout, so choosing the machine the AGV reaches soonest (TECT) pays, while processing-time weighting
    (PECT / PTECT) does not. This is a hypothesis; the 200-machine catalog (running) is the next check.
- **3b** (queued) measures the switching ceiling once MDD-TECT is in the portfolio (`v5tect`) and with the catalog's
  6-pair sets.
