# rq3-size-catalog: the 84-pair rule catalog at 30, 50, 100 and 200 machines (2026-10-20, user)

Status: **30 / 50 / 100 machines done 10-20 19:10 (local, 2 h 41 min, 240 / 240, every task's MDD-PTECT = the grid's). 200 machines
(`rq3-size-catalog-m200`: `launch.sh 16 m200`, 80 tasks, ~93 core-hours) added by the user 10-20, runs last, after
`rq3-size-oracle-m200`.** Step 2 of the size rule sweep
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

## Results: 15-100 machines (10-20)
Test seeds 0-39, % vs MDD-PTECT pooled, < 0 better. Hindsight proxies (inst = best pair per instance, block = best pair
per 1.5 h regime block) are bounds, not policies; map = one pair per regime (normal / surge / short fleet), chosen on
fitting seeds, is observable. `analysis.out`, `summary.csv`, `<cond>/portfolios.json`.

### Best fixed pairs

| floor | 1st | 2nd | 3rd | v5's best | best outside v5 |
|---|---|---|---|---|---|
| m15 | MDD-PECT −0.5 | MDD-PTECT 0 | SRT-PECT +4.8 | (all top 4 are v5) | CR-PTECT +12.4 (#6) |
| m30 | ATC-PTECT −5.2 | ATC-PECT −4.4 | MDD-PECT −1.8 | ATC-PTECT #1 | MOD-PECT +3.2 (#5) |
| m50 | MDD-PTECT 0 | MDD-PECT +2.0 | ATC-PTECT +2.6 | MDD-PTECT #1 | EDD-PTECT +6.5 (#5) |
| m100 | **MDD-TECT −1.9** | MDD-PTECT 0 | SRT-TECT +3.8 | MDD-PTECT #2 | **MDD-TECT −1.9 (#1)** |

### Best pair per regime

| floor | normal | surge | short fleet |
|---|---|---|---|
| m15 | MDD-PECT −0.9 | MDD-PTECT 0 | MDD-PECT −0.4 |
| m30 | ATC-PTECT −7.2 | ATC-PTECT −8.4 | MDD-PECT −1.7 |
| m50 | ATC-PECT −5.6 | ATC-PTECT −0.8 | MDD-PTECT 0 |
| m100 | MDD-PTECT 0 | MDD-TECT −0.1 | **MDD-TECT −11.5** |

### Portfolio proxies (test seeds)

| floor | full pool: map / block / inst | v5: map / block / inst | greedy block K = 6 (fitting) | its test block |
|---|---|---|---|---|
| m15 | −0.5 / −11.2 / −5.4 | −0.5 / −7.9 / −4.5 | MDD-PTECT, ATC-PECT, MOD-PECT, SRT-PTECT, FIFO-PECT, PTWINQ-PTECT | −7.3 |
| m30 | −6.1 / −15.5 / −10.0 | −6.1 / −13.2 / −9.6 | MDD-PTECT, ATC-PECT, ATC-PTECT, MDD-PECT, MOD-PTECT, MOD-PECT | −13.9 |
| m50 | −2.3 / −14.9 / −8.2 | −2.3 / −8.9 / −4.6 | MDD-PTECT, ATC-PTECT, CR-TECT, MDD-PECT, ATC-PECT, SLACK-TECT | −11.2 |
| m100 | **−3.5** / −13.0 / −8.5 | **0.0** / −2.7 / −0.7 | MDD-PTECT, MDD-TECT, ATC-PTECT, FIFO-PTECT, SLACK-TECT, MOD-PTECT | −11.6 |

### Reading
- **Up to 50 machines the v5 set holds the useful rules.** At 30 and 50 machines the best observable regime map uses
  only v5 pairs (−6.1% / −2.3%: ATC in normal / surge blocks, MDD in short-fleet blocks), and the full catalog adds
  hindsight complementarity (block −15.5 vs −13.2 at 30; −14.9 vs −8.9 at 50) but no better observable choice.
- **At 100 machines v5 runs out and the TECT machine rule takes over.** The v5 set's hindsight bounds collapse (block
  −2.7%, per instance −0.7%, regime map 0.0%), while the full catalog's stay as large as on small floors (block
  −13.0%, per instance −8.5%). The best fixed pair is MDD-TECT (−1.9%), outside v5, and in short-fleet blocks it is
  11.5% better than MDD-PTECT. The per-block winners are mostly TECT pairs (MDD-TECT 23 of 200, EDD-TECT 19,
  SLACK-TECT 12, SRT-TECT 11).
- **Why TECT:** short-fleet blocks put 26 AGVs on 100 machines, so transport binds. TECT ranks machines by earliest
  completion including travel, without PECT / PTECT's processing-time weighting. When AGVs are scarce, reaching a
  machine sooner matters more than the machine's speed. v5's machine rules (PECT / PTECT) were picked on the
  15-machine floor, where TECT is far behind (MDD-TECT +18.5% on test seeds there, rq2-portfolio-screen).
- **An observable size-aware rule:** at 100 machines the regime map {short fleet → MDD-TECT, otherwise MDD-PTECT}
  scores −3.5% on test seeds (fitting −2.2%). That is as large as the v5 switching oracle's whole ceiling there
  (−3.3%, `../rq3-size-oracle`), from a rule that only needs the fleet size. It is a proxy (each block's value comes from
  a run that held the pair all episode); an exact run of that policy is cheap and is the first thing step 3 should
  measure.
- **For the training arms:** `train-size-m100` (and the mixed arm) choose among v5 pairs only, so the rule that matters
  most at 100 machines is not in their action space. Their ceiling is the v5 oracle's −3.3%, not what the floor
  allows.
- **The user's hypothesis holds:** the rules that give switching headroom change with floor size. ATC carries it at
  30-50 machines (within v5), TECT at 100 (outside v5).
