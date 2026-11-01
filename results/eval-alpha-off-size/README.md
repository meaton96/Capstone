# eval-alpha-off-size: the law + offset arms on the floor-size grid (2026-10-26, user: "evaluation of train-alpha-off-b2 and train-alpha-off-het on the size grid (50/100/200 machines), with pre-declared criteria")

Status: **done 10-27 (6 h 56 min, 39 / 39 tasks; criteria pre-declared 10-26, before any result).** S1: off-het
PASS, off-b2 FAIL. S2: FAIL for both. S3: off-b2 PASS, off-het FAIL. **D: carry forward off-het.** Lab
`eval-alpha-off-size` (local, 20 workers).

## Why
- **`../eval-alpha-offset` (10-26):** the law + offset α policy (v7) passed H1' and H2' on 15-machine floors with
  machine-speed spread. The law α = max(0, 3 − 6σ) carries the spread gain, and the head carries the B2 edge.
  - off-b2, trained on B2 only, is the best arm on B2 (−2.55% vs MDD-PTECT).
  - off-het is slightly better at medium spread.
- **Floor size is the axis where the final policy fails.**
  - `../rq3-size-grid`: T15 is +2.60% at 100 machines and +9.54% at 200 vs MDD-PTECT.
  - `../rq3-size-headroom`: the room on large floors lies at low α. MDD-TECT, which is PTECT's score at α 0, is −19%
    at 200 machines, while v5's own room is −0.2%.
- **The size-grid floors are homogeneous (σ 0),** so the law gives α 3 = MDD-PTECT everywhere. Any move of α is the
  head's: offsets ×0.5 / ×0.25 give α 1.5 / 0.75, and ×1.5 / ×2 give 4 (the cap). α 0 is out of reach.
- **Questions:**
  - Does the 15-machine-trained head avoid T15's loss on large floors?
  - Does it lower α where that pays?
  - Which arm should paper B carry forward (plan §9 item 7)?

## Design (`task.py`, `launch.sh`, `analyze.py`)
- **Floors:** the grid's m50 / m100 / m200 (`../rq3-size-grid` conditions, as `../rq3-size-v5`): M bays, round(0.52 M)
  AGVs on duty (short blocks half: 26 (13) / 52 (26) / 104 (52)), B2 load × 0.9, arrival cap off, c ~ U[1.75, 2.5].
- **Harness:** `../rq3-gen-v5/run_task.py`'s B2 arguments (twin, tardiness reward, random warm-up, 21,600 s window,
  900 s slots, 5,400 s discount horizon) and its fleet wrapper.
  - evaluate.py is the arms' own v7 snapshot (`../train-alpha-off-*/code/env`, identical for both), run in process
    with a 12 GB address-space cap.
  - `task.py` checks that `evaluate` and `scenarios` import from that snapshot.
- **Groups** (test seeds 0-39; each also runs MDD-PTECT as the run-to-run check against the grid's pdr group, bit for
  bit):

  | group | policies | decision log |
  |---|---|---|
  | `off-b2` | train-alpha-off-b2 s0-s4 | yes (offset per slot) |
  | `off-het` | train-alpha-off-het s0-s4 | yes |
  | `alpha` | MDD-PTECT@1.5, MDD-PTECT@0.75 (the fixed α the offsets reach) | no |

- **Chunks** as the grid: m50 one 40-seed task per group, m100 10-seed, m200 5-seed. That is 39 tasks, longest first
  (`tasks.txt`).
- **Each task writes `blocks.json`** (the regime blocks per seed) for the per-regime offset analysis.
- **References, not rerun** (same floors, seeds and frozen generator; % vs MDD-PTECT, test seeds 0-39):

  | floor | MDD-PTECT (k job-s) | T15 (final policy) | T100 | MDD-TECT | best v5 pair | hindsight best v5 pair |
  |---|---|---|---|---|---|---|
  | m15 (B2) | 145.5 | −1.69 | −1.02 | — | MDD-PECT −0.5 | −4.5 |
  | m50 | 158.2 | −1.29 | −3.70 | +29.9 | MDD-PTECT | −4.57 |
  | m100 | 456.0 | +2.60 | −0.08 | −1.91 | MDD-PTECT | −0.72 |
  | m200 | 2,848.9 | +9.54 | — | −19.03 | MDD-PTECT | −0.17 |

  - Sources: pdr (v5 pairs + MDD-TECT) and T15 from `../rq3-size-v5` (m50, m100) and `../rq3-size-grid` (m200); T100
    from `../rq3-size-grid` (m50, m100; not run at m200).
  - B2: `../eval-alpha-offset` s 0, where off-b2 is −2.55% and off-het −1.01%.

## Analysis (`analyze.py`, pre-declared 10-26 before any result)
- **Metric:** window tardiness, % vs MDD-PTECT pooled over test seeds 0-39, with a 95% instance-bootstrap CI, test
  seeds better / worse, and a Wilcoxon test. The statistics are imported from `../eval-alpha-spread/analyze.py`, as in
  `../eval-alpha-offset`.
- **Family:** the per-instance mean over the 5 training seeds. Each seed's own pooled % is shown with a one-sample
  t-test of the 5.
- **Checks:**
  - every group's MDD-PTECT equals the grid pdr group's, seed for seed;
  - every policy has all 40 seeds.
- **Per floor:** MDD-PTECT@1.5 / @0.75; the best fixed reachable α (in sample) and the per-instance best of α 3 / 1.5 /
  0.75 (hindsight); MDD-TECT; the best and hindsight-best v5 pair; T15, T100 and the two arms.
- **Paired:**
  - each arm vs T15 and vs T100 (family vs family, per test seed);
  - off-het vs off-b2;
  - capture of the reachable fixed-α room: (arm − MDD-PTECT) / (per-instance best α − MDD-PTECT), pooled.
- **Offsets chosen:** shares, mean log2 offset and mean played α (3 × offset, capped at 4), per floor and by regime
  (normal / surge / short fleet). B2's (m15, `../eval-alpha-offset` s 0) comes first, and each training seed is also
  shown alone.

**Criteria (verdict), per arm:**
- **S1, no loss with size:**
  - at m100 and m200 the family beats T15 (paired, Wilcoxon, Holm over the 2 floors);
  - on none of m50 / m100 / m200 is it significantly worse than MDD-PTECT (pooled > 0 with Wilcoxon p < 0.05 fails).
- **S2, gain at 200 machines:** the family is better than MDD-PTECT at m200 (pooled < 0, Wilcoxon p < 0.05).
- **S3, the head lowers α with size:** the mean log2 offset over policy slots at m200 is below B2's (m15), pooled,
  and for at least 4 of the 5 training seeds.

**D, which arm to carry forward (plan §9 item 7):**
- off-b2 by default: it is the best arm on B2 and needs no spread floors in training.
- off-het replaces it if it is better than off-b2 on at least 2 of the 3 size floors (paired family vs family, Wilcoxon,
  Holm over the 3) and significantly worse on none.

**Secondary (reported):** the arms vs T100 at m50 / m100, the capture of the reachable α room, and the offsets by
regime.

**What the outcomes would mean:**
- **S1 and S2 pass:** one α policy, trained on 15 machines, covers spread and size. That is paper B's transfer claim,
  without training at the largest size.
- **S1 passes, S2 fails:** the head avoids T15's loss but cannot reach the large-floor room. The reachable α stops at
  0.75, while the room lies at α 0. The fix would be a size term in the law, or an offset that reaches 0.
- **S1 fails:** the head's 15-machine choices cost on large floors, as T15's do. Training at size (T100) stays the
  answer.

## Cost and dry run (10-26, scratchpad)
- **m50, seeds 0-1, one off-het checkpoint plus MDD-PTECT and MDD-PTECT@0.75:**
  - MDD-PTECT equals the grid's pdr group bit for bit on both seeds (57,743 / 269,404 job-s);
  - σ 0 and α_law 3 are recorded on every episode;
  - the decision log has the offsets. The checkpoint chose ×0.5 (α 1.5) in all 48 slots.
  - An m50 episode takes about 18-25 s.
- **m200, seed 0, one off-b2 checkpoint plus MDD-PTECT:**
  - MDD-PTECT equals the grid's bit for bit (158,078 job-s);
  - 200 machine rows in the observation; each episode takes ~320 s at 0.7 GB resident;
  - offsets logged per slot. The numbers mean nothing yet: one seed, one checkpoint.
- **Analysis:** it ran on the dry-run outputs (`--min-n 1`) and printed every section; criteria print n/a where floors
  are missing.
- **Estimate:**
  - per-episode times: ~320 s at m200 (dry run), about 100 s at m100 (`../rq3-size-grid`), ~20 s at m50;
  - 600 episodes per floor make about 55 core-h at m200, 17 at m100 and 4 at m50;
  - that is about 75 core-h in all, 4-5 h on 20 workers. An m200 arm task (30 episodes) is ~2.7 h.

## Run
`launch.sh [W]` runs `tasks.txt` W at a time (default 20), with logs in `logs/`. A finished task
(`out/<floor>/<group>_s<a>-<b>/DONE`) is skipped, so relaunching resumes. Lab: done when 39 `DONE` files exist;
analysis `analyze.py` → `analysis.out`, `summary.csv`, `offset_choices.csv`.

## Outputs
- `out/<floor>/<group>_s<a>-<b>/`: `episodes.csv` (with `speed_sigma` / `alpha_law`), `decisions.csv` (arms),
  `blocks.json`, `eval.out`, `cond.json`, `scenarios/`, `DONE`;
- `analysis.out`, `summary.csv`, `offset_choices.csv`;
- `diag_rules.out` (exploratory rule-choice breakdown, 10-27).

## Results (10-27, `analysis.out`, `summary.csv`, `offset_choices.csv`)
**Checks pass.** Every group's MDD-PTECT equals the grid's pdr group on all 40 seeds at every floor, and every policy
has all 40 seeds. A 200-machine episode took ~660 s with 20 tasks running (the 320 s of the dry run was one task
alone), so the run took 6 h 56 min instead of 4-5 h.

**Per floor** (window tardiness, % vs MDD-PTECT pooled over test seeds 0-39; family = mean of the 5 training seeds,
Wilcoxon p; B2 from `../eval-alpha-offset` s 0):

| | m15 (B2) | m50 | m100 | m200 |
|---|---|---|---|---|
| MDD-PTECT@1.5 (offset ×0.5) | | +0.16 | −2.65 (p 0.0002) | −6.01 (38/2) |
| MDD-PTECT@0.75 (offset ×0.25) | | +5.11 | −4.28 (p 0.006) | −10.88 (38/2) |
| per-instance best of α 3 / 1.5 / 0.75 (hindsight) | | −3.77 | −5.53 | −10.94 |
| MDD-TECT (α 0, out of the head's reach) | | +29.9 | −1.91 | −19.03 |
| hindsight best v5 pair | −4.5 | −4.57 | −0.72 | −0.17 |
| T15 (final policy) | −1.69 | −1.29 | +2.60 | +9.54 |
| T100 | −1.02 | −3.70 | −0.08 | |
| **off-b2** | **−2.55** | **−2.30** (p 0.0005) | **+0.66** (p 0.68) | **+3.99** (0/40) |
| **off-het** | **−1.01** | **−0.90** (p 0.05) | **−1.07** (p 0.09) | **−0.27** (p 0.24) |

- **Training seeds spread widely at m200:** off-b2 +14.9 / −5.4 / +2.5 / +3.6 / +4.5, off-het +7.9 / +1.0 / +4.0 /
  −9.5 / −4.9.
- **Paired (family vs family, per test seed):**
  - vs T15: off-b2 −1.88 at m100 (p 0.011) and −5.06 at m200 (37/3); off-het −3.58 (p 0.001) and −8.96 (40/0).
  - vs T100: off-b2 +1.45 at m50 (p 0.045) and +0.74 at m100 (p 0.023); off-het +2.91 (p 0.062) and −0.99
    (p 0.13).
  - off-het vs off-b2: +1.44 at m50 (p 0.22), −1.72 at m100 (p 0.0008), −4.10 at m200 (39/1).
- **Capture of the reachable fixed-α room** (arm vs the per-instance best of α 3 / 1.5 / 0.75): off-b2 +0.61 / −0.12 /
  −0.36 at m50 / m100 / m200, off-het +0.24 / +0.19 / +0.03.
- **Offsets chosen** (mean log2 offset, played α; m15 / m50 / m100 / m200):
  - off-b2: +0.08 / −0.15 / −0.22 / −0.25, α 3.11 / 2.79 / 2.70 / 2.67;
  - off-het: −0.60 / −0.86 / −0.90 / −0.81, α 2.11 / 1.78 / 1.74 / 1.85.
  - ×0.25 (α 0.75), the best fixed α at m100 and m200, gets only 17% (off-b2) and 16% (off-het) of the m200 slots.

**Criteria:**
- **S1, no loss with size:** off-b2 **FAIL**. It beats T15 at m100 and m200, but is +3.99% vs MDD-PTECT at m200 (worse
  on 40 / 40 seeds). off-het **PASS**: it beats T15 at both and is never significantly worse than MDD-PTECT.
- **S2, gain at 200 machines:** **FAIL** for both (off-het −0.27%, p 0.24; off-b2 +3.99%).
- **S3, the head lowers α with size:** off-b2 **PASS** (−0.25 at m200 vs +0.08 on B2; lower for 4 of 5 seeds, s3 is
  the exception). off-het **FAIL**: lower pooled (−0.81 vs −0.60) but for only 3 of 5 seeds (s0 and s1 rise again at
  m200; at m50 and m100 all 5 are lower than on B2).
- **D:** off-het is better than off-b2 at m100 and m200 after Holm, and not worse at m50, so **carry forward off-het**
  (plan §9 item 7).

**Why the arms miss the room (exploratory, after the verdict; `diag_rules.py` → `diag_rules.out`):**
- **The pre-declared reading of S1 pass / S2 fail does not hold here.** It blamed the reach: offsets stop at α 0.75
  while the room lies at α 0. But the room within reach is large. Fixed α 0.75 is −10.9% at m200 and −4.3% at m100,
  and α 1.5 is −6.0% at m200.
- **The head lowers α, but not far enough.** Mean played α at m200 is 1.85 (off-het) and 2.67 (off-b2).
- **The other two heads keep 15-machine choices that cost on large floors.**
  - The v5 pairs at m200: MDD-PECT +17.7%, ATC-PTECT +42.6%, ATC-PECT +71.1% (SRT-PTECT +1.9%).
  - Across the 10 arm policies at m200, the share of slots with the PECT machine rule tracks the result (Spearman ρ
    +0.68, p 0.03). The four policies that pick PECT in under 5% of slots average −4.7%, the other six +6.2%.
  - The best of them, off-het s3 (no PECT, mean α 1.23), is −9.5%, about 87% of the reachable room.
  - Neither the ATC share (ρ +0.12) nor the mean α in PTECT slots (ρ +0.25) explains the spread on its own.
- **The fixed-α rows extend `../rq3-alpha-size` to 200 machines.** On these homogeneous floors α* is 3 at 50 machines,
  about 0.75-1 at 100 (rq3-alpha-size's fitting seeds: 1, −2.1%), and at most 0.75 at 200. rq3-alpha-size's size
  term failed its gate over 15-100 machines; the 200-machine point is where it would pay.

**What it means:**
- **For paper B's transfer claim:** law + offset (off-het) trained on 15 machines no longer loses on large floors.
  - It is never significantly worse than MDD-PTECT from 50 to 200 machines.
  - It beats the final policy by 3.6 points at 100 machines and 9.0 at 200.
  - It does not gain at 200 machines, where a fixed α 0.75 is −10.9%.
  - At 50 and 100 machines it is no better than T100 (+2.9 / −1.0 points, n.s.).
- **The arm choice (rule D) trades B2 for size.** off-b2 is better on B2 (−2.55% vs −1.01%), and off-het is better on
  the large floors.
- **Candidate next steps** (not run, not decided):
  - **(a) α head only:** evaluate off-het with the job and machine heads fixed to MDD-PTECT and only the offset free.
    No training. It tests whether the offset head alone captures the reachable room (−6 to −11% at m200).
  - **(b) A size or transport term in the law:** α set from a travel / processing descriptor, so the law, not the
    head, lowers α on large floors. It is tested on the grid with the same criteria.

**Caveats:**
- The PECT diagnosis is post hoc, over 10 policies.
- T100 was not run at m200.
- MDD-TECT is not exactly PTECT at α 0: it has no availability scaling.
- The family mean hides seeds that do capture the room (off-het s3 −9.5%, off-b2 s1 −5.4% at m200), and seeds that
  lose badly (off-b2 s0 +14.9%).
- The law was fit on 15-machine floors, so on σ-0 floors it is α 3 by construction. These floors test the head, not
  the law.
