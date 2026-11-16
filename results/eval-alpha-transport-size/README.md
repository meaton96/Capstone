# eval-alpha-transport-size: the α law with a transport term and a size gate on the pair heads (2026-10-28, user: "design the size / transport term in the law, with the pair heads restrained above ~50 machines, with pre-declared criteria, and set it up in the lab")

Status: **done 10-29 08:06 (lab, local, 9 h 28 min; 61 policy tasks + 320 simulator law runs). P1-P6 PASS, H FAIL.**
The transport law + gate is carried forward as paper B's α policy across sizes and fleets, as a fit to these
floors: of the two held-out size / fleet pairs, one transfers and one does not. Stage 2 of 2. Criteria pre-declared
10-28, before any result. The launcher ran because stage 1's gate F1 passed (`../rq3-alpha-transport/law.json`).

## Why
- `../eval-alpha-only-size` (rule N: both) found two limits on large floors.
  - The pair heads cost 2-6 points above 50 machines.
  - The offset head plays about the same α at every size, while the best fixed α falls to ≤ 0.75 at 200 machines.
- Stage 1 (`../rq3-alpha-transport`) fits a size / transport term h(x) for the law on fixed-α curves.
- This stage puts the fitted law into the carried-forward policy (train-alpha-off-het, v7), with a size gate on the
  pair heads, and tests it on the size grid and on two fleets.

## Design (`task.py`, `launch.sh`, `analyze.py`)
- **The law:** α = min(4, max(0, 3 − 6σ) × h(x) × offset), h from `law.json`; x is the cell's floor statistic from
  its fitting instances. The floors are homogeneous, so the law plays 3 h(x) × offset.
  - `task.py` replaces `env_wrappers.twin_env.alpha_law` with v7's law times the cell's h (rounded to 3 decimals, as
    `law.json`). It also patches `evaluate.alpha_law`, which only labels the episode rows.
  - The frozen v7 code is not edited, and the policy's inputs do not contain the law.
  - On B2, h = 1 by construction (x0 ≥ B2's x) and the gate is off, so the policy is v7 off-het exactly. B2 is not
    rerun.
- **The gate:** above 50 machines the job and machine heads are fixed to MDD-PTECT and only the offset head chooses
  (`../eval-alpha-only-size`'s AlphaOnlyPolicy). At 50 machines or fewer the full policy plays.
- **Cells** (homogeneous; r052 = the grid, r035 / r080 = round(0.35 / 0.80 M) AGVs on duty, as
  `../rq3-size-headroom`): m50_r052, m75_r052, m100_r052, m200_r052, m100_r035, m100_r080. Test seeds 0-39.
- **Groups** (train-alpha-off-het s0-s4; family = per-instance mean of the 5):

  | group | cells | what |
  |---|---|---|
  | `lg` | all six | transport law + gate (the candidate), plus MDD-PTECT as the run-to-run check |
  | `lf` | the five above 50 machines | transport law, full policy (no gate) |
  | `v7` | m75_r052, m100_r035, m100_r080 | v7 law, full policy (the other cells have it in `../eval-alpha-off-size`) |

- **Rules, in the simulator** (equal to this harness, checked below):
  - the α grid on test seeds (`../rq3-alpha-size` for m50-m100 r052, `../rq3-alpha-transport` for the rest);
  - **the law alone**, MDD-PTECT at the cell's law α, on the six cells and the two held-out cells (m75_r035,
    m150_r080) (`../rq3-alpha-transport/run.py ... law`, in this stage's launch).
- **Earlier references on the same seeds:** v7 off-het full and alpha-only at m50 / m100 / m200 r052
  (`../eval-alpha-off-size`, `../eval-alpha-only-size`); the final policy T15 (`../rq3-size-v5`, `../rq3-size-grid`).
- **Harness:** `../eval-alpha-off-size`'s (the arm's own v7 snapshot in process, the fleet wrapper, a 12 GB
  address-space cap, decision log). Chunks of 5 seeds at m200, 10 at 75-100 machines, 40 at m50.

## Checks (10-28, scratchpad, with stage 1's synthetic pipeline-test law as a stand-in)
- **Gated `lg`, m100_r035, seeds 0-1, offset forced to ×1:** equals the simulator at the law's α (0.779) exactly
  (36.318 / 6517.277). Every decision was MDD-PTECT@x1. MDD-PTECT equals `../rq3-size-headroom`'s (39.471 /
  7003.075). The episode rows carry alpha_law 0.779.
- **`lf`, m100_r035 seed 0:** plays the full policy (MDD-PTECT@x0.5 23 of 24 slots, one ATC) under the patched law.
- **`lg`, m50_r052 seed 0, h = 1:** equals `../eval-alpha-off-size`'s off-het s0 and MDD-PTECT exactly (50.082 /
  57.743).
- `analyze.py` ran on these outputs (`--min-n 1`) through every section.

## Analysis (`analyze.py`, pre-declared 10-28 before any result)
- **Metric:** window tardiness, % vs MDD-PTECT pooled over test seeds 0-39, 95% instance-bootstrap CI, better / worse,
  Wilcoxon. Paired comparisons use the family; each training seed's pooled % and a one-sample t-test are reported.
- **Per cell:**
  - the α grid, the cell's α*_fit (best grid α on fitting seeds) and the hindsight best single α on test seeds;
  - the law alone, v7 full / alpha-only, `lf`, `lg`, and T15.
- **Played:** offset shares, mean played α, PECT share and non-MDD share per cell and group.
- **Criteria (family of 5):**
  - **P1, the term helps at 200 machines:** `lg` better than v7 alpha-only at m200_r052 (p < 0.05).
  - **P2, reaches the fixed-α room:** at m100_r052 and m200_r052, `lg` is within +1 point of MDD-PTECT at α*_fit and
    not significantly worse.
  - **P3, no loss:** `lg` is not significantly worse than MDD-PTECT on any of the six cells, nor than v7 off-het full
    at m50_r052.
  - **P4, the gate:** `lg` is better than `lf` at m200_r052 (p < 0.05), and not significantly worse than `lf` at
    m75_r052, m100_r052, m100_r035 or m100_r080.
  - **P5, a guard on the offset head:** `lg` is not significantly worse than the law alone on any cell above 50
    machines.
  - **P6, fleets:** `lg` is better than v7 off-het full at m100_r035 (p < 0.05) and not significantly worse at
    m100_r080.
  - **H, transfer of the law (the law alone, held-out cells m75_r035 and m150_r080):** within +2% of the cell's best
    single α on its test seeds. Where α 3 is at least 2% worse than that best, the law must also beat α 3 (p < 0.05).
- **N, the next step:**
  - P1, P2 and P3 PASS → the transport law + gate becomes paper B's α policy across sizes and fleets, carried forward
    in place of v7 off-het. Next: the online version (h from the on-duty fleet per regime) and a Unity check.
  - P1 PASS, P2 FAIL → if the law alone reaches α*_fit (within +1 point at m100 and m200 r052), the offset head
    costs on large floors, so the next step is the law alone above the gate. Otherwise the law's form stops short,
    so the next step is a refit with another variable or form.
  - P1 FAIL → the term does not reach the policy. Report the α curves, and v7 + gate (`../eval-alpha-only-size`) as
    the large-floor result.
  - P4 FAIL → once α is right, the pair heads stop costing; drop the gate.
  - H FAIL → the law is a fit to these floors, not a rule that transfers to new size / fleet pairs. The paper must
    say so.
- **Expectation, stated before the run (a guess, not a criterion):**
  - The curves already known suggest α* near 0-0.5 at 200 machines (MDD-TECT −19%) and a flat optimum near 1 at 100.
    So the law should send α low at m200, and P1 should pass.
  - The het head halves the law α on average (×0.5 in ~60% of slots), so `lg` plays below the law. At m200 that may
    not hurt (TECT is best). At m100, where the curve turns up below α 0.75, it might, which would show up in P2 / P5.
  - At m100_r080 the law should stay near 3, and the gate may cost there (P4 / P6).

## Cost
- Policy tasks: m200 ~660 s per episode under a 20-way load (16 chunks, 25-30 episodes each, ~5 h); m100 ~105-180 s;
  m75 ~70 s; m50 ~40 s. About 160 CPU-h, so ~8-9 h on 20 slots.
- Law runs: 320 simulator episodes, ~2-3 CPU-h, at the end of the queue.

## Run
`launch.sh [W]` checks F1, then runs `tasks.txt` (`eval ...` = `task.py`, `law ...` = `../rq3-alpha-transport/run.py
... law`), W at a time. Finished tasks are skipped.

## Outputs
- `out/<cell>/<group>_s<a>-<b>/`: `episodes.csv`, `decisions.csv`, `eval.out`, `cond.json` (with the law), `blocks.json`,
  `DONE`;
- the law runs in `../rq3-alpha-transport/out/<cell>/law/`;
- `analysis.out`, `summary.csv`, `played.csv`.

## Results (10-29, `analysis.out`, `summary.csv`, `played.csv`)

### Checks
- Every cell and group has 5 training seeds × 40 test seeds. The α label on every episode row equals the cell's law α.
- MDD-PTECT run inside the harness equals the simulator at α 3 on 40 / 40 seeds in every cell.
- Gated runs: 0 of 4,800 decisions per cell differ from MDD-PTECT on the job and machine heads.
- At m100_r080 the law is 3 (h = 1), so `lf` must equal `v7` there. It does, per training seed to the decimal.
- The law α per cell is stage 1's (`law.json`): m50_r052 2.603 (h 0.87), m75_r052 1.883, m100_r052 1.431, m200_r052 0,
  m100_r035 0, m100_r080 3, held out m75_r035 0 and m150_r080 2.351.

### Per cell (window tardiness, % vs MDD-PTECT at α 3, test seeds 0-39; policies are the family of 5)

| cell | law α (α*_fit, hindsight best) | best single α | law alone | v7 full | v7 alpha-only | `lf` | `lg` | final T15 |
|---|---|---|---|---|---|---|---|---|
| m50_r052 | 2.60 (1.5, 2) | −1.67 | −0.98 | −0.90 | −0.46 | (= `lg`) | +0.21 | −1.29 |
| m75_r052 | 1.88 (2, 2) | −1.61 | −0.46 | −0.75 | | −0.71 | **−2.30** | +1.24 |
| m100_r052 | 1.43 (1, 0.75) | −4.28 | −2.90 | −1.07 | −3.21 | −1.59 | **−4.25** | +2.60 |
| m200_r052 | 0 (0, 0) | −19.03 | −19.03 | −0.27 | −4.74 | −10.81 | **−19.03** | +9.54 |
| m100_r035 | 0 (0, 0) | −14.30 | −14.30 | −0.50 | | −8.57 | **−14.30** | |
| m100_r080 | 3 (3, 2) | −0.01 | 0 | −0.22 | | −0.22 | −0.14 | |
| m75_r035 (held out) | 0 (–, 0) | −9.28 | **−9.28** | | | | | |
| m150_r080 (held out) | 2.35 (–, 0.5) | −5.74 | −1.72 | | | | | |

`lf` at m50 is not run, since the gate is off at 50 machines and `lg` is the full policy there. CIs, better / worse
counts and per-seed numbers are in `analysis.out`.

### Criteria

| | test | result | |
|---|---|---|---|
| P1 | m200_r052 `lg` vs v7 alpha-only | −15.01% [−16.67, −13.04], 34/6, p 4.5e-8 | PASS |
| P2 | `lg` vs MDD-PTECT at α*_fit | m100 −0.61% [−1.62, +0.26] (p 0.25); m200 0.00 (`lg` is that rule) | PASS |
| P3 | `lg` vs MDD-PTECT, six cells; vs v7 full at m50 | never worse (+0.21 at m50 to −19.03 at m200); vs v7 full at m50 +1.12% [+0.24, +2.02], p 0.064 | PASS (m50 borderline) |
| P4 | gate: `lg` vs `lf` | m200 −9.21% (40/0, p 1.8e-12); m100_r035 −6.27%; m100_r052 −2.71%; m75 −1.60% (p 0.12); m100_r080 +0.07% | PASS |
| P5 | `lg` vs the law alone above 50 machines | m75 −1.85% (p 0.014); m100 −1.39% (p 0.28); equal at m200, m100_r035; m100_r080 −0.14% | PASS |
| P6 | fleets: `lg` vs v7 full | m100_r035 −13.86% (p 4.3e-9); m100_r080 +0.07% (p 0.47) | PASS |
| H | law alone vs the best single α, held-out cells | m75_r035 0.00 (the law picks α 0, the best), beats α 3 by 9.28%; m150_r080 **+4.27% [+0.93, +7.78]** (limit +2), beats α 3 by 1.72% (p 0.003) | **FAIL** |

**N:** P1, P2 and P3 pass, so the transport law + gate becomes paper B's α policy across sizes and fleets, carried
forward in place of v7 off-het. The declared next steps are the online version (h from the on-duty fleet per regime)
and a Unity check. H fails, so the paper must present the law as a fit to these floors, not as a rule shown to
transfer to new size / fleet pairs.

### Reading
- **Where the law sends α to 0, the gated policy is a fixed rule.**
  - At m200_r052 and m100_r035, α = 3 h × offset is 0 for every offset and the pair heads are fixed. So `lg` is
    MDD-TECT, and all five training seeds give the same −19.03 / −14.30.
  - P1, P2 at m200 and the first half of P6 are therefore the law choosing MDD-TECT; the learned part adds nothing on
    those floors. The room is the one `../rq3-size-headroom` found (v5 + MDD-TECT −13..−21%). What this stage adds is
    a rule that finds it from a floor statistic, and a policy that keeps it.
- **Where the law's α is between 0 and 3 above the gate, the offset head adds to the law.**
  - It beats the law alone at m75 (−1.85, p 0.014) and at m100 (−1.39, n.s.).
  - `lg` matches or passes the hindsight best single α: m75 −2.30 vs −1.61, m100 −4.25 vs −4.28.
  - The head plays about 0.6 × the law (mean played α 1.15 at m75 and 0.86 at m100, against 1.88 and 1.43). At m100
    that lands on the best α (0.75-1), below where the law alone sits.
- **The pair heads cost even once α is right, so the gate stays (P4).**
  - `lf` loses 9.2 points to `lg` at m200 and 6.3 at m100_r035.
  - The loss is uneven across training seeds. At m200, s1, s3 and s4 come within 1-2 points of MDD-TECT (−17.9,
    −18.8, −16.7), while s0 and s2 lose most of the room (+2.0, −2.7).
- **At 50 machines the transport term does not help the full policy.**
  - `lg` (the full policy at law α 2.60) is +0.21 vs MDD-PTECT, against −0.90 for v7 full; the final policy T15
    (−1.29) is best there.
  - Every training seed is worse under the new law than under v7's (by 0.2-2.1 points). The loss passes the declared
    test (p 0.064) but its bootstrap CI excludes 0, so it is a small real cost, not noise. Why lowering α from 3 to
    2.6 hurts a policy whose best fixed α is 1.5 was not tested.
- **The law transfers on the short-fleet side, not on the large-floor, large-fleet side (H).**
  - m75_r035 has x 0.33; the law sends α to 0, which is the best single α.
  - m150_r080 has x 0.20, close to m50_r052's 0.19. The law reads light transport (α 2.35) while the best α is 0.5.
    It still beats α 3, but takes 1.7 of the 5.7 points of room.
  - Stage 1 misses on the same side: m150_r052 (LOO 3.5%) and m200_r080 (law 1.56 vs α*_fit 1.0). Travel per AGV
    leaves out something that grows with floor size at a large fleet. What that is (congestion with 120 AGVs, longer
    trips than the per-machine statistic shows) was not tested.
- **Against the final policy:** T15 is worse than MDD-PTECT above 50 machines (+2.60 at m100, +9.54 at m200), while
  `lg` is never worse than MDD-PTECT on any cell. So paper B's α policy, not T15, is the one to quote above 50
  machines.
