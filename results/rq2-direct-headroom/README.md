# rq2-direct-headroom: does a direct job×machine action space have more headroom than rule pairs?

Status: done 2026-10-07 22:32 (local, 160/160). 32-future recheck (rq2-direct-headroom-f32, cluster): 70/80 on
10-08 03:38; the 10 direct seeds that hit the 12 h limit were resubmitted with 24 h.

## Why
The fallback plan, if the PDR agent cannot be made to work, is a GNN that chooses job and machine directly (L2D,
Song 2022 style). Those are hard to train, so first check whether the bigger action space has anything to learn on our
floor.

Rule selection is capped by its rule set (Link et al. 2026, arXiv:2604.24117). Direct choice is not, but a better
non-rule choice has to exist. Background: `docs/experiments/review_1007/FINDINGS_1007.md`.

## Method (`run.py`)
- **Setting:** B2 twin setting (`../rq2-oracle-steady/sim.py`), tardiness objective, test seeds 0-39.
- **Base policy:** MDD-TECT, the best fixed pair on B2.
- **One-step rollout (Bertsekas):** for every agent decision in the first 1,800 s of the 6 h window, in order:
  1. try each option;
  2. play the rest of the episode with the base;
  3. keep the option with the lowest window tardiness;
  4. move to the next decision, replaying the kept choices.
- **Options:**
  - **rule:** the distinct (job, machine) outcomes of the 15 H15 pairs at that decision.
  - **direct:** every legal (job, machine), capped at 60: all rule outcomes plus a seeded sample of the rest.
- **Scoring:**
  - **hindsight:** on the instance's real future. Clairvoyant.
  - **expected8:** mean over 8 futures in which the arrivals after the decision time are replaced by other B2
    instances' arrivals. Only current-state knowledge, so a policy could achieve this in principle.
  - The rollout's choices are always finally scored on the real instance.
- **Forcing a choice:** `des_twin.engine.rank_jobs` / `select_machine` are wrapped to return a preset job and
  machine. The twin is deterministic; forcing the base's own choice reproduces the base exactly (asserted at startup).
- **Output:** `B2/<rule|direct>-<hindsight|expected8>/s<seed>.json` (per-decision rows included). `analyze.py`
  produces the table in `analysis.out`.

## Smoke test (seed 0, first 300 s only, 4 futures; output in the 10-07 session scratch, not kept)

| cell | gain over MDD-TECT | decisions | deviations from base | outside rule set |
|---|---|---|---|---|
| rule, hindsight | +9.8% | 42 | 1 | 0 |
| direct, hindsight | +12.5% | 40 | 3 | 3 |
| rule, expected4 | +6.5% | 42 | 3 | 0 |
| direct, expected4 | +5.6% | 34 | 13 | 12 |

- A single deviation in the first 5 minutes moving window tardiness by about 10% is the chaos the credit trace found.
  One seed means nothing; the 40-seed means and paired differences are the result.
- Timing at 300 s: rule-expected4 took 68 s, direct-expected4 332 s. The full run is about 70 CPU-h.

## How to read the result
- **direct − rule (paired, expected8)** is the extra headroom a GNN could exploit without foresight.
  - Clearly positive (e.g. ≥ 2 percentage points with a CI excluding 0): a direct-action learner has room.
  - About 0: it would hit the same wall as rule selection.
- **hindsight − expected8** within each variant is the share of the gain that needs foresight. It bears on the
  look-ahead question (rq3-lookahead).
- **Caveats:**
  - A one-step rollout over a 30-min stretch is a lower bound on what a full policy could do.
  - The 60-option cap binds at high WIP (the `capped` column).

## Results (B2 seeds 0-39; `analysis.out`)
| cell | gain over MDD-TECT, pooled | mean per seed ± 95% | decisions rolled | deviations from base | outside the rule set |
|---|---|---|---|---|---|
| rule, hindsight | +10.6% | +11.1 ± 2.0 | 128 | 4.0 | 0 |
| direct, hindsight | +13.1% | +13.6 ± 2.4 | 128 | 6.5 | 5.2 |
| rule, expected8 | 0.0% | −0.5 ± 3.0 | 127 | 15.6 | 0 |
| direct, expected8 | −1.8% | −2.3 ± 2.9 | 123 | 35.3 | 28.8 |

Paired direct − rule: **+2.5 ± 1.1 points in hindsight**; **−1.8 ± 2.6 points with 8 redrawn futures**.

## Reading
- **With the realized future**, 30 minutes of one-step choices alone nearly reach the full-window switching oracle
  (−10.8%). Direct job×machine choice adds 2.5 points over rule outcomes, mostly by leaving the rule set (5 of 6.5
  deviations are choices no H15 pair makes).
- **Knowing only the present** (scored over redrawn futures), neither action space gains anything. The direct
  rollout deviates far more often (35 vs 16 times) and ends slightly worse. With 8 futures it chases noise in a larger
  option set.
- **For the GNN question:** on B2, a bigger action space buys nothing a non-clairvoyant controller can use, at least
  greedily. A trained GNN could still find multi-step strategies a one-step rollout cannot, so this is evidence, not
  proof. The 32-future recheck tests whether the rollout's estimate is the limit.

## 32-future recheck (rq2-direct-headroom-f32; 70/80 tasks, 10-08)
| cell | pooled gain over MDD-TECT | mean per seed ± 95% | seeds |
|---|---|---|---|
| rule, expected32 | +1.1% | +1.2 ± 2.1 | 40 |
| direct, expected32 | −2.3% | −2.0 ± 3.2 | 30 (10 timed out, resubmitted) |

Paired direct − rule: −3.0 ± 3.2 points (30 seeds).

- A less noisy estimate gives the rule rollout a small gain (+1.1%).
- The direct rollout is still worse: more options, judged on a noisy estimate, cost more than the extra freedom buys.
- The conclusion stands: no extra usable headroom for a direct action space on B2, at least greedily.
