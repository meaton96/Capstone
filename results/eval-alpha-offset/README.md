# eval-alpha-offset: the law + offset arms on floors with machine-speed spread (2026-10-25, user: "the law sets α, the head picks an offset")

Status: **done 10-26 (lab `eval-alpha-offset`, 10 tasks, 14 min): H1' PASS, H2' PASS.** Law + offset keeps the het
arm's spread-floor gain (−13.9 points vs the final policy at s 1) and costs nothing on B2. The B2-trained offset arm
does as well on spread floors and is the best of the arms on B2 (−2.55%). Criteria were pre-declared 10-25, before
any training.

## Why
`../eval-alpha-spread` found that the v6 α head (`../train-alpha-het`) reaches the spread law on spread floors (H1 PASS)
but plays one compromise α of 0.7-1.4 everywhere. So it does not read the spread (H2 FAIL) and costs +4.1% vs the final
policy on B2 (H3 FAIL). The fix tested here (`../dev-alpha-offset`, option (c) there) sets α per floor from the spread
law, α_law = max(0, 3 − 6σ), and lets the third head pick an offset on it: ×{1, 0.5, 0.25, 1.5, 2}, α capped at 4.

On B2 (σ 0) the law is α 3, so offset ×1 is exactly the final policy's action. Two questions:
- Does the het-trained offset arm keep the v6 het arm's gain on spread floors?
- Does it lose the cost on B2?

## Design
- **The same cells, seeds and arguments as `../eval-alpha-spread`,** so its outputs are the references and are not
  rerun:
  - 15-machine B2 floors with machine-speed spread s ∈ {0, 0.25, 0.5, 0.75, 1} (s 0 is B2), through `hetero.py`
    (unchanged; check O2 in `../dev-alpha-offset`);
  - test seeds 0-39 (training seeds start at 10,000);
  - B2 due dates (c 1.75-2.5) and `../rq3-gen-v5/run_task.py`'s arguments: twin, tardiness reward, random warm-up,
    21,600 s window, 900 s slots, 5,400 s discount horizon.
- **New groups (`task.py`):**

  | group | what | code |
  |---|---|---|
  | `off-het` | train-alpha-off-het s0-s4 + MDD-PTECT (α 3) + MDD-PTECT@x1 (the law alone, continuous α), decision log | `../train-alpha-off-het/code` (v7) |
  | `off-b2` | train-alpha-off-b2 s0-s4 + the same two rules, decision log | `../train-alpha-off-b2/code` (v7) |

- **Reused from `../eval-alpha-spread`:** the final policy (train-v5-cmix s0-s4), train-alpha-het / -b2 (v6), and the
  fixed-rule grid (MDD × {PECT, PTECT} × α 0-4, the 30 v6 combinations, each instance's σ).
- **Tasks:** 10 (`tasks.txt`: 5 cells × 2 groups), 10 at a time (`launch.sh`). One process per task with a 12 GB
  address-space cap.
- **Cost:** at eval-alpha-spread's rate of 4.5-6 s per episode, 280 episodes per task, so about 25-30 min wall.

## Analysis (`analyze.py`, pre-declared 10-25)
Window tardiness, % vs MDD-PTECT (α 3), pooled over instances, with a 95% instance-bootstrap CI, seeds better / worse
and a Wilcoxon test. The statistics are imported from `../eval-alpha-spread/analyze.py` (`compare()`), so both
evaluations are scored the same way.
- **Checks:**
  - each new group's MDD-PTECT equals eval-alpha-spread's het-group MDD-PTECT bit for bit (same instances);
  - the law alone is the same in both groups and equals MDD-PTECT at s 0 (law α 3);
  - `speed_sigma` equals the grid's σ, and `alpha_law` = max(0, 3 − 6σ).
- **Per cell, references:**
  - the law alone (v7: MDD-PTECT at α = max(0, 3 − 6σ), continuous);
  - eval-alpha-spread's refit law (max(0, 1.8 − 4σ) on the α grid);
  - the hindsight best fixed MDD-PTECT α;
  - the per-instance best v6 combination.
- **Per cell, policy stacks:** final, b2, het, off-b2 and off-het, each as the mean over its 5 training seeds per
  instance, with each seed's own % also shown.
- **Paired:** off-het vs final, vs het and vs the law alone, per cell; off-b2 vs final, per cell.
- **Offsets chosen:** per slot, by cell and by regime block (normal / surge / short fleet):
  - offset shares and the mean log2 offset;
  - the mean played α (law × offset, capped at 4) against the law;
  - machine-head and job-head shares.

**Criteria (verdict):**
- **H1', off-het keeps the het arm's gain on spread floors:** at s 0.5, 0.75 and 1 it beats the final policy
  significantly (Wilcoxon, Holm over the 3 cells). Pooled over those cells' 120 instances, it is at most 1.0% worse than
  train-alpha-het.
- **H2', off-het costs nothing on B2:** at s 0 it is within +1% of the final policy and not significantly worse. A
  positive difference with Wilcoxon p < 0.05 fails. This is eval-alpha-spread's H3 rule.

**Secondary (reported, not part of the verdict):**
- off-b2 under H2''s rule at s 0;
- **H3', the head adds to the law:** off-het vs the law alone, pooled over all 5 cells (better, Wilcoxon p < 0.05);
- **H4', the regime adjustment:** off-het's mean log2 offset is lower in short-fleet blocks than in normal blocks,
  pooled over cells and per training seed.

**For reference** (eval-alpha-spread, test seeds, % vs MDD-PTECT at s 0 / 0.25 / 0.5 / 0.75 / 1):

| row | s 0 | s 0.25 | s 0.5 | s 0.75 | s 1 |
|---|---|---|---|---|---|
| final policy | −1.69 | −1.52 | −2.40 | −1.72 | −2.89 |
| train-alpha-het | +2.38 | −1.43 | −7.69 | −12.78 | −15.92 |

- The law 3 − 6σ on fitting seeds (`../dev-alpha-offset/law_choice.out`): 0.00 / −0.84 / −4.54 / −10.05 / −12.69.
- What the result would mean:
  - if H1' and H2' both pass, law + offset is the deployable α policy for paper B;
  - if H2' passes and H1' fails, the head's offsets do not recover what het's free α found beyond the law;
  - if H2' fails, the head leaves the law's α 3 on B2, as v6 did.

## Dry run (10-25, scratchpad; cells s 0 and 1, seeds 0-1)
- **Setup:** both groups ran with `../dev-alpha-offset`'s smoke checkpoint standing in for the arms, 4 tasks in 37 s.
- **Checks:** all passed: MDD-PTECT = eval-alpha-spread het 8 / 8; the law the same in both groups and = MDD-PTECT at
  s 0 8 / 8; σ and α_law 32 / 32.
- **Analysis:** it printed every section; the criteria sections print n/a where cells are missing.
- **The numbers are meaningless:** the checkpoint is a 3-update smoke.
- **Cosmetic note:** rows equal to MDD-PTECT show up as "0/2 worse". The reference is the grid's float64 MDD-PTECT@3,
  while evaluate.py's MDD-PTECT is float32 (differences about 1e-8, also in eval-alpha-spread).

## Outputs
- `out/<cell>/<group>/` (evaluate.py: `episodes.csv` with `speed_sigma` / `alpha_law`, `decisions.csv`, `eval.out`,
  `task.json`);
- `out/<cell>/DONE_*`, `logs/`;
- `analysis.out`, `summary.csv`, `offset_choices.csv`.

## Results (10-26)
`analysis.out`, `summary.csv`, `offset_choices.csv`.

**Checks (all pass):**
- both new groups' MDD-PTECT equals eval-alpha-spread's bit for bit (400 / 400);
- the law alone is the same in both groups and equals MDD-PTECT at s 0 (280 / 280);
- σ and α_law are as defined (2,800 / 2,800 rows).

### Per cell (window tardiness, % vs MDD-PTECT at α 3, family of 5, test seeds 0-39)
| row | s 0 (B2) | s 0.25 | s 0.5 | s 0.75 | s 1 |
|---|---|---|---|---|---|
| final policy (v5) | −1.69 | −1.52 | −2.40 | −1.72 | −2.89 |
| train-alpha-het (v6, free α) | +2.38 | −1.43 | −7.69 | −12.78 | −15.92 |
| train-alpha-b2 (v6) | −0.60 | −2.02 | −3.49 | −6.23 | −7.16 |
| **train-alpha-off-het** (v7) | −1.01 | −1.65 | −7.37 | −12.33 | −16.40 |
| **train-alpha-off-b2** (v7) | **−2.55** | −1.17 | −6.01 | −11.22 | −16.42 |
| law alone, α = max(0, 3 − 6σ) | 0.00 | −0.48 | −6.58 | −11.97 | −16.09 |
| best fixed α (hindsight) | 0.00 (α 3) | −1.41 (1.5) | −7.80 (1) | −12.20 (1) | −17.22 (0.5) |
| per-instance best v6 combination (hindsight) | −6.82 | −8.52 | −13.85 | −17.90 | −21.89 |

Every v7 row at s 0.5-1 excludes 0. Training-seed ranges at s 1: off-het −17.7 to −14.7, off-b2 −17.5 to −13.8.
At s 0, off-b2's seeds are −3.4 to −1.8.

### Criteria (pre-declared)
- **H1', keeps the het arm's gain: PASS.**
  - off-het vs the final policy: −5.09 / −10.80 / −13.91 points at s 0.5 / 0.75 / 1 (p 6e-8 / 2e-11 / 9e-12,
    significant after Holm).
  - Pooled over those 120 instances it is +0.08% [−0.47, +0.66] vs train-alpha-het (bound +1.0%).
- **H2', costs nothing on B2: PASS.** At s 0, off-het vs the final policy is +0.70 [−1.05, +2.14] (p 0.54). The v6 het
  arm was +4.1 there.
- **Secondary:**
  - **off-b2 on B2 (H2' rule): better than the final policy,** −0.87 [−1.97, +0.08] (p 0.048).
  - **H3', the head adds to the law: yes, by a little.** off-het vs the law alone, all 5 cells pooled: −0.73
    [−1.66, +0.10] (p 0.005). Per cell: −1.01 / −1.17 / −0.85 / −0.41 / −0.38.
  - **H4', lower offsets when the fleet is short: pooled yes, 4 of 5 training seeds.** Mean log2 offset −0.73 in
    short-fleet blocks vs −0.55 in normal blocks. Per seed, short − normal is +0.19 / −0.06 / −0.13 / −0.60 / −0.31.

### What the head chose (per slot)
- **off-het mostly halves the law.** ×0.5 is chosen in 54-61% of slots, and the mean log2 offset is −0.5 (about ×0.7).
  The played α is 2.11 against the law's 3 at s 0, and 0.60 against 0.72 at s 1. Lowering α on B2 costs it nothing,
  but it does not gain either (−1.01%, n.s.).
- **off-b2 stays near the law.** ×1 and ×1.5 are chosen in 30-36% and 27-33% of slots, and the mean log2 offset is
  +0.1. The played α is 3.11 at s 0 and 0.89 at s 1.
- **Both adjust α by regime, in the same direction.** It is lower in short-fleet blocks and higher in surges. For
  off-b2 at s 0: −0.12 / +0.09 / +0.26 log2 (short / normal / surge). This is the direction `../rq3-size-headroom`
  predicts: when transport is scarce, the travel-and-queue rule (α 0) pays.
- **Job head:** off-het plays MDD in 51-57% of slots and ATC in 36-43%; off-b2 is closer to even (MDD 42-51%, ATC
  46-56%). SRT is rare (2-7%).

### Reading
- **The law carries the spread gain, and the head carries the B2 edge.**
  - At s 1 the law alone is −16.1%, and the arms add only 0.3 points.
  - At s 0 the law is MDD-PTECT itself, and the edge is the head's: −1.0% (off-het) and −2.55% (off-b2).
- **One policy now serves both kinds of floor.** off-b2 is trained only on B2, yet it matches the het arms on spread
  floors (−16.4% at s 1) and has the largest B2 edge of any arm here. Against the per-instance best v6 combination,
  it captures 0.37 of the room on B2 (2.55 / 6.82; the final policy 0.25) and 0.75 at s 1.
- **The v6 failure is fixed.**
  - v6's free α head found one compromise α and paid +4.1% on B2.
  - With the law in the action, the head only has to learn an offset. Its baseline (×1) is already right on every
    floor.
- **For paper B:** law + offset is the deployable α policy (the README's pre-declared reading for H1' and H2'
  passing). The open choice is which arm to carry forward:
  - **off-b2:** better on B2 and on par at high spread; it needs no spread floors in training.
  - **off-het:** 1.1-1.4 points better at s 0.5-0.75 (−7.37 / −12.33 vs −6.01 / −11.22; point estimates, not tested).

### Caveats
- **The law was fit on these floors.** `3 − 6σ` comes from fitting seeds of the same 15-machine spread family
  (`../dev-alpha-offset/law_choice.out`). off-b2 never trained on spread floors, but its spread performance borrows
  the law's knowledge of them. Transfer to other floor families (size, layout) is untested; `../rq3-size-headroom`
  suggests large floors want low α too.
- **off-b2's B2 gain is marginal:** p 0.048, with a CI touching 0 (+0.08). Its 5 training seeds agree (all −1.8 to
  −3.4).
- **Hindsight references** (best fixed α, best v6 combination) are per-instance or per-cell oracles, not policies.

