# rq3-size-catalog: the 84-pair rule catalog at 30, 50 and 100 machines (2026-10-20, user)

Status: **queued in the lab after `rq3-size-oracle` (local, 240 tasks, 16 at a time).** Step 2 of the size rule sweep
(step 1: `../rq3-size-oracle`). Step 3, the exact oracle for a size-specific portfolio, is decided after this.

## Question
Do other rules become strong, or complementary, as the floor grows? The v5 set ({SRT, MDD, ATC} × {PECT, PTECT})
was picked on the 15-machine floor for strength, and on that floor the full catalog had no regime specialists
(`../rq2-portfolio-screen`). On larger floors the v5 ranking changes (`../rq3-size-grid`: ATC best at 20-30, worst at
150-200), so a different rule set might hold more switching headroom there.

## Method (`run.py`, `analyze.py`, `launch.sh`)
- **Screen:** `../rq2-portfolio-screen` unchanged: 12 job rules (SPT LPT SRT LRT FIFO PTWINQ EDD SLACK CR MDD MOD ATC)
  × 7 machine rules (SMPT SRWT MMUR ECT TECT PECT PTECT, quad α 3) = 84 pairs, each held all episode, window
  tardiness split by 1.5 h regime block (the blocks sum to the total; checked on m30 seed 0 to 2e-12).
- **Floors:** m30, m50, m100 through `../rq3-size-oracle/sizesim.py` (the grid's conditions; frozen v5 code). Each task
  checks its MDD-PTECT against the grid's. 15 machines = the original screen's runs.
- **Seeds:** test 0-39 (reporting), fitting 2000-2039 (portfolio selection; the 15-machine screen's fitting seeds
  2000-2039 are used for comparability, not all 160).
- **Cost:** ~2.6 s per m30 episode under load (84 pairs on m30 seed 0 took 221 s), ~4.5 s at m50, ~15 s at m100:
  ~40 core-hours.

## Analysis (`analyze.py`, rq2-portfolio-screen's own proxy and greedy functions)
Per floor, vs MDD-PTECT, pooled:
- best fixed pairs on test seeds; where the v5 pairs rank; the best pair outside v5;
- best pair per regime (normal / surge / short fleet) and the most frequent per-block winners;
- proxies for the full 84-pair pool and for v5: per-instance best and per-block best (hindsight), one pair per regime
  (observable), fitting | test;
- greedy portfolios (K ≤ 6, fitting seeds): "block" (per-block best) and "map" (regime map). The block portfolio is the
  candidate for the step-3 exact oracle.

Caveat (as in the original screen): each block's value comes from a run where that pair was held all episode, so
carry-over between pairs is ignored. The proxies rank portfolios; the exact oracle (step 3) measures one.

15-machine reference (this analysis, fitting seeds 2000-2039): test proxies full pool block −11.19% / inst −5.42% /
map −0.53%; v5 block −7.91% / inst −4.51% / map −0.53% (the original screen's values).

## Results
Pending.
