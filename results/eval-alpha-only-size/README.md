# eval-alpha-only-size: does the offset head alone take the large-floor room? (2026-10-28, user: "yes setup the alpha head only check")

Status: **done 10-28 (lab `eval-alpha-only-size`, 30 / 30 tasks, 5 h 23 min).** Criteria pre-declared 10-28, before
any result. **Result (ao-het): O1 PASS, O2 PASS, O3 PASS, O4 FAIL, B PASS, so rule N says both.** With the pair fixed
to MDD-PTECT, the offset head takes −4.7% at 200 machines, where the full arm took −0.3%. The pair heads cost 4.5
points there, but α 0.75 is still 6.9 points better. The offset head does not lower α with size. Next: a size /
transport term in the law, with the pair heads checked again.

## Why
- **`../eval-alpha-off-size` (10-27):** on the size grid the law + offset arms never lose to MDD-PTECT (off-het, S1
  PASS), but neither gains at 200 machines (off-het −0.27%, S2 FAIL), although fixed α 0.75 is −10.9% there and α 1.5
  −6.0%.
- **Its exploratory mechanism** (`diag_rules.out`, post hoc) points two ways:
  - the offset head lowers α only partway (mean played α 1.85 for off-het, 2.67 for off-b2 at 200 machines);
  - the machine head keeps choosing PECT, which costs on large floors (MDD-PECT +17.7% at 200 machines). Over the 10
    arm policies the PECT share tracks the result (ρ +0.68, p 0.032), the α in PTECT slots does not (ρ +0.25).
- **This check separates the two.** Fix the job and machine heads to MDD-PTECT and let only the offset head choose. If
  that takes the room, the loss came from the pair heads; if not, the offset head (or the law it multiplies) is the
  limit.
- It decides between the two next steps the 10-27 write-up named: (a) restrain the pair heads on large floors, or (b) a
  size / transport term in the law.

## Design (`task.py`, `launch.sh`, `analyze.py`)
- **Alpha-only policy:** each checkpoint of `train-alpha-off-het` / `-b2` s0-s4, with the job head fixed to MDD and the
  machine head to PTECT. The offset head's argmax is played, so each slot is MDD-PTECT@x⟨offset⟩, i.e. α 3 / 1.5 /
  0.75 / 4 / 4 at σ 0.
  - The actor is factorized (one Categorical per branch given the state, `models/actor_critic.py`), so fixing heads 0
    and 1 leaves the offset head's distribution unchanged. It sees the states its own MDD-PTECT play produces (closed
    loop).
  - `task.py` replaces `evaluate.CheckpointPolicy` with a subclass in its own process before `build_policies` runs.
    The frozen v7 code is not edited. Policy names get `~alphaonly`.
- **Floors:** B2 (m15 = `../rq3-gen-v5` base) and the grid's m50 / m100 / m200 (`../rq3-size-grid` conditions, as
  `../eval-alpha-off-size`): M bays, round(0.52 M) AGVs on duty (short blocks half), B2 load × 0.9, arrival cap off,
  c ~ U[1.75, 2.5].
- **Harness:** `../eval-alpha-off-size/task.py`'s (the arms' own v7 snapshot run in process, the fleet wrapper, a 12 GB
  address-space cap, decision log), test seeds 0-39, the grid's chunks.
- **Groups:**

  | group | floors | policies | why |
  |---|---|---|---|
  | `ao-het` | m15-m200 | train-alpha-off-het s0-s4 alpha-only, MDD-PTECT (m15 also MDD-PTECT@1.5 / @0.75) | the carried-forward arm (rule D, 10-27); MDD-PTECT is the run-to-run check |
  | `ao-b2` | m15-m200 | train-alpha-off-b2 s0-s4 alpha-only | the arm whose head did lower α with size (S3 PASS) but lost 4.0% at 200 |
  | `off-het`, `off-b2` | m15 | the full arms, MDD-PTECT | harness check against `../eval-alpha-offset`'s B2 runs, and the m15 full-arm reference |

  30 tasks, longest first (`tasks.txt`).
- **References, not rerun:**
  - m50-m200 full arms and fixed α: `../eval-alpha-off-size/out` (off-het / off-b2 / alpha groups);
  - MDD-PTECT: the grid's pdr runs (`../rq3-size-v5`, `../rq3-size-grid`) and `../rq3-gen-v5/base/pdr` on B2;
  - final policy T15: `../rq3-size-v5`, `../rq3-size-grid`, and `../eval-v5-cmix/test/base/cmix` on B2.

## Checks (10-28, before queueing; scratchpad, B2 seeds 0-1, train-alpha-off-het s0)
- **Wrapper plumbing:** with the offset head forced to index 0 (×1) the alpha-only policy equals MDD-PTECT exactly
  (95,842.008 / 29,594.482 job-s). Forced to index 2 (×0.25) it equals MDD-PTECT@0.75 exactly (113,659.836 /
  29,271.109). So (MDD, PTECT, ×1) is v5's α 3 and the offsets map as designed.
- **Unforced:** every alpha-only decision played MDD / PTECT (48 / 48), with the network's offsets (47 × ×0.5, 1 ×
  ×1.5). It differs from the full arm on seed 0 (95,617.6 vs 94,702.2) and equals it on seed 1, where the full arm
  played MDD-PTECT throughout.
- **Harness:** the full arm here equals `../eval-alpha-offset`'s B2 run of the same checkpoint (94,702.211 /
  32,341.408), and MDD-PTECT equals `../rq3-gen-v5/base/pdr`.
- **Analysis:** `analyze.py` ran on the dry-run outputs, and on a stand-in m200 group (a copy of an off-het chunk). It
  flagged the stand-in's non-MDD / PTECT decisions (309 / 600) and computed every criterion.

## Analysis (`analyze.py`, pre-declared 10-28 before any result)
- **Metric:** window tardiness, % vs MDD-PTECT pooled over test seeds 0-39, 95% instance-bootstrap CI, test seeds
  better / worse, Wilcoxon. Family = per-instance mean of the 5 training seeds; each seed's own pooled % and a
  one-sample t-test of the 5.
- **Per floor:** MDD-PTECT@1.5 / @0.75 and their per-instance best with α 3, T15, both full arms and both alpha-only
  arms.
- **Paired:** each alpha-only arm vs its full arm, vs MDD-PTECT@0.75 and vs T15; ao-het vs ao-b2; capture of the
  reachable fixed-α room, (arm − MDD-PTECT) / (per-instance best α − MDD-PTECT), for full and alpha-only arms.
- **Offsets:** per floor, alpha-only vs full: offset shares, mean log2 offset, mean played α, by regime; per training
  seed.
- **Criteria, per arm (ao-het decides N):**
  - **O1, gain at 200 machines:** the alpha-only family is better than MDD-PTECT at m200 (pooled < 0, Wilcoxon
    p < 0.05). This is the S2 both full arms failed.
  - **O2, the pair heads cost at 200 machines:** the alpha-only family is better than its full arm at m200 (paired,
    p < 0.05).
  - **O3, no loss with size:** the alpha-only family is not significantly worse than MDD-PTECT on m50, m100 or m200.
  - **O4, reaches the static room at 200 machines:** the alpha-only family is within +1% of MDD-PTECT@0.75 at m200 and
    not significantly worse.
  - **B, B2 cost (a guard, not a goal):** on B2 the alpha-only family is within +1 point of its full arm and not
    significantly worse. A FAIL means the pair heads carry the B2 edge, so an alpha-only policy is not a standalone
    replacement.
  - **N, next step (ao-het):**
    - O1 and O4 PASS → the offset head picks α well once the pair is fixed; the large-floor loss is the pair heads'.
      Next: restrain the pair heads on large floors (e.g. alpha-only above a size, or the pair heads trained there).
    - O1 PASS, O4 FAIL → both: the pair heads cost part of the room and the offset head stops short. Next: the law's
      size / transport term, with the pair heads checked again.
    - O1 FAIL → the offset head does not take the room even with the pair fixed. Next: a size / transport term in the
      law.
- **Expectation, stated before the run:** off-het played mean α 1.85 at m200 with its own pairs. If its offsets stay
  near that with MDD-PTECT fixed, the result should sit between α 3 (0%) and α 1.5 (−6.0%); O1 likely PASS, O4 likely
  FAIL. The closed loop can change the offsets, so this is a guess, not a criterion.

## Cost
- Episodes per process (eval-alpha-off-size, 20-way load): m200 ~660 s, m100 ~105 s, m50 ~40 s, m15 ~4 s.
- m200: 16 tasks of 25-30 episodes ≈ 4.6-5.5 h each, all running at once; m100 / m50 / m15 fill the other 4 slots.
  About 100 CPU-h in all, so ~5.5-6 h wall on 20 slots.

## Run
`launch.sh [W]` runs `tasks.txt` W at a time (logs in `logs/`). Finished tasks (`out/<floor>/<group>_s<a>-<b>/DONE`)
are skipped, so relaunching resumes.

## Outputs
- `out/<floor>/<group>_s<a>-<b>/`: `episodes.csv`, `decisions.csv`, `eval.out`, `cond.json`, `blocks.json`, `DONE`;
- `analysis.out`, `summary.csv`, `offset_choices.csv`.

## Results (10-28; `analysis.out`, `summary.csv`, `offset_choices.csv`)

### Checks
- Every alpha-only policy ran 40 / 40 test seeds on every floor. Not one of its 4,800 decisions per floor and arm
  was anything other than MDD / PTECT.
- The run-to-run MDD-PTECT in the ao-het groups equals the grid's pdr reference on 40 / 40 seeds on every floor.
- The m15 full arms equal `../eval-alpha-offset`'s B2 runs on 200 / 200 seed-runs for both arms.

### Per floor, % vs MDD-PTECT (window tardiness, test seeds 0-39, family = mean of 5)

| floor | α 1.5 | α 0.75 | best α (hindsight) | T15 | off-het | off-b2 | **ao-het** | ao-b2 |
|---|---|---|---|---|---|---|---|---|
| m15 (B2) | +1.02 | +5.62 | −2.10 | −1.69 | −1.01 | **−2.55** | −0.33 (p 0.84) | −0.57 (p 0.72) |
| m50 | +0.16 | +5.11 | −3.77 | −1.29 | −0.90 | **−2.30** | −0.46 (p 0.34) | −0.53 (p 0.63) |
| m100 | −2.65 | **−4.28** | −5.53 | +2.60 | −1.07 | +0.66 | −3.21 [−4.59, −1.88], 33/7 | −1.83 |
| m200 | −6.01 | **−10.88** | −10.94 | +9.54 | −0.27 | +3.99 | −4.74 [−5.71, −3.98], 39/1 | −2.39 |

- ao-het's training seeds at m200 are −2.9, −0.3, −4.8, −9.4, and −6.3 (t-test p 0.037). At m100 they are −2.5,
  −1.2, −3.2, −4.6, and −4.5 (p 0.0078).
- ao-het is the best learned policy run on these floors at 100 and 200 machines. It is 5.7 and 13.0 points better
  than T15 there (paired, 40 / 0 seeds at m200). At m100 it also beats the floor-trained arms of `../rq3-size-grid`
  (T100 −0.08, T50 +1.48, Tmix +0.06) and the hindsight best v5 pair (−0.72). The lower α that v5 cannot reach is
  why. It is still 6.9 points behind fixed α 0.75 at m200 (4 / 36 seeds). At m100 it is 1.1 points behind α 0.75,
  which is not significant (p 0.48).
- Capture of the per-instance fixed-α room at m200: ao-het 0.43, ao-b2 0.22, off-het 0.03, off-b2 −0.36. At m100:
  0.58 / 0.33 / 0.19 / −0.12.

### Pre-declared criteria

| criterion | ao-het | ao-b2 |
|---|---|---|
| O1 gain at m200 | −4.74 [−5.70, −4.02], p 1e-11, **PASS** | −2.39 [−2.76, −1.99], p 4e-8, **PASS** |
| O2 better than its full arm at m200 | −4.48 [−5.11, −3.97], 39/1, **PASS** | −6.13 [−7.12, −5.42], 40/0, **PASS** |
| O3 not worse on m50 / m100 / m200 | −0.46 / −3.21 / −4.74, **PASS** | −0.53 / −1.83 / −2.39, **PASS** |
| O4 within +1% of α 0.75 at m200 | +6.89 [+6.07, +7.73], 4/36, **FAIL** | +9.53 [+8.60, +10.74], **FAIL** |
| B on B2, within +1 point of its full arm | +0.69 [−0.32, +1.92], p 0.19, **PASS** | +2.03 [+0.84, +3.42], p 0.0015, **FAIL** |

**N (ao-het, O1 PASS, O4 FAIL): both.** The pair heads cost part of the room, and the offset head stops short. The
next step is a size / transport term in the law, with the pair heads checked again. The pre-run expectation (between
α 3 and α 1.5, O1 PASS, O4 FAIL) held.

### Where the pair heads help and where they cost (alpha-only − full arm, paired)

| floor | het | b2 |
|---|---|---|
| m15 | +0.69 (p 0.19) | +2.03 (p 0.0015) |
| m50 | +0.44 (p 0.23) | +1.82 (p 0.016) |
| m100 | −2.16 (p 0.0004) | −2.48 (p 0.004) |
| m200 | −4.48 (p 9e-12) | −6.13 (p 2e-12) |

- The pair heads earn their keep below about 50 machines and cost above it. The cost grows with size. For off-b2 the
  B2 edge (−2.55 vs −0.57 alpha-only) sits mostly in the pair heads. That is B's FAIL, so ao-b2 does not replace
  off-b2 on B2.
- For het the pair heads are a wash on m15 / m50 (n.s. either way), so ao-het is never significantly worse than
  off-het and is better from 100 machines.

### The offset head (section 3 of `analysis.out`)
- **It plays the same offsets whether the pair is fixed or not.** The per-seed mean played α changes by at most 0.07
  between the full and alpha-only runs (ao-het m200 1.84 vs 1.85). The closed loop under MDD-PTECT does not move it.
  So the 4.5-point gap between the arms at m200 is the pair heads alone, not a different α.
- **It does not lower α with size.** ao-het plays mean α 2.13 / 1.78 / 1.73 / 1.84 on m15 / m50 / m100 / m200, and
  ao-b2 3.12 / 2.80 / 2.71 / 2.67. Meanwhile the fixed-α optimum moves from α 3 (B2) to α 0.75 (m200). The head's
  ×0.25 share at m200 is only 0.16 (het) / 0.18 (b2). The regimes barely differ (het m200 normal 1.74, short 1.96,
  surge 1.93).
- **The level of α explains most of each policy's result (exploratory, post hoc).**
  - Over the 10 alpha-only policies, the per-seed mean played α against the per-seed pooled % gives Spearman ρ +0.80
    (p 0.005) at m200 and +0.84 (p 0.002) at m100.
  - Against the fixed-α curve interpolated in log α at each policy's mean α (policies with mean α ≤ 3), the median
    residual is −0.1 points at m200 and −1.2 at m100. So the head's slot-by-slot timing adds little beyond its
    average level at 200 machines and about a point at 100.
  - One outlier: b2 s1 at m200 is −6.4 against −1.9 predicted.
  - Mean α under a mix of offsets is a crude summary (convexity), so this reading is a hint, not a test.

### Reading
- **Both of the 10-27 mechanisms are real, and they add up.** At m200, fixing the pair to MDD-PTECT moves off-het from
  −0.3% to −4.7%. The pair heads' PECT play was the larger part of the 10-27 shortfall. Lowering α the rest of the way
  (−4.7% to −10.9%) needs an α the head never chooses, because it was trained on 15-machine floors, where α near 3 is
  right. The law gives 3 at σ 0 whatever the size.
- **What follows for the design.**
  - The law needs a term that brings α down on large, transport-bound floors (`../rq3-size-headroom`: the large-floor
    room comes from transport, i.e. fleet and trip length, not size as such). Then the offset head has to do
    nothing new.
  - The pair heads need restraining above about 50 machines. Options: alpha-only above a size, or the pair heads
    trained on large floors. Otherwise any lower α they are given is played against PECT.
- **Standing results.**
  - On B2, off-b2 full stays the best v7 policy (−2.55%).
  - From 100 machines up, ao-het is the best learned policy (m100 −3.2%, m200 −4.7%), but it is no better than a
    fixed α 0.75 that a size-aware law would pick.
  - None of this changes the final policy (T15).
- **Caveats.**
  - The test floors are homogeneous (σ 0), so the law's σ term is idle here and the size term is untested.
  - The pair-head cost was measured only against MDD-PTECT, not against the pair a large-floor-trained head would
    choose.
  - One set of 40 test seeds per floor.
