# rq2-direct-headroom: does a direct job×machine action space have more headroom than rule pairs?

Status: running (lab, local, 16 workers, started 2026-10-07). Results are filled in when it finishes.

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

## Results
(pending)
