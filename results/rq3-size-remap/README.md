# rq3-size-remap: why the final policy does not transfer to 100 machines (no-training diagnostic) (2026-10-19, user)

Status: **done** 2026-10-19 17:53 (cluster array 21926204, 3 h 5 min, 20 / 20). `analysis.out`, `summary.csv`.

## Why
In `../rq3-size-v5`, the final policy (cmix, trained on 15 machines only) keeps its edge on 50 machines (pick −2.1%)
but not on 100: the pick is +0.4% and the 5-seed family +2.6% vs MDD-PTECT. The user's original proposal claims the
policy transfers to any floor size, and the user (10-19) asked to look at this more.

The architecture is built to be size-agnostic:
- the action is a rule pair;
- machines and jobs go through set encoders with mean / max pooling.

But three of the 18 global scalars carry the floor size, and they were constant or narrow during training:

| scalar | definition | at 15 machines (training) | at 100 machines |
|---|---|---|---|
| s[12] | floor size, n / 100 | always 0.15 | 1.0 (never seen) |
| s[1] | active jobs, squash(WIP, 30) | about 0.4-0.65 | about 0.9 (near saturation) |
| s[15] | options at a routing decision, squash(eligible machines, 5) | 3 per type → 0.375 | 20 per type → 0.8 |

(The 64 × 64 floor grid also gets denser on large floors. It is not touched here.)

Hypothesis: the transfer gap comes from these inputs leaving the training range, not from the encoders. If so,
rewriting them into the 15-machine range at test time should restore most of the transfer, with no training.

## Method (`task.py`, `submit.sh`, `pull.sh`, `analyze.py`)
- **Remap:** `task.py` wraps `TwinSchedulingEnv._obs` (frozen v5 code) and rewrites only these scalars before the
  policy sees them.
  - The counts behind each squash are recovered exactly (inverse squash, rounded to the integer they are), then scaled
    by 15 / n.
  - Remaps: `size` (s[12] := 0.15), `wip` (s[1] := squash(WIP × 15/n, 30)), `opt` (s[15] := squash(options × 15/n,
    5) on routing decisions only; a dispatch decision's options are one machine's queue), and `all`.
- **Floors / seeds / policies:** rq3-size-v5's m50 (26 AGVs) and m100 (52 AGVs), test seeds 0-39, the cmix family
  s0-s4 (final pick s3) + MDD-PTECT. m100 runs in 10-seed chunks. 4 remaps × (1 + 4) = 20 tasks.
- **Analysis:** per floor and remap, the pick and the family mean vs MDD-PTECT and vs no remap (= rq3-size-v5),
  paired over seeds.

## Checks (local, 10-19)
- **Identity at 15 machines:** `--remap all --cond base` reproduces eval-v5-cmix's B2 test runs exactly (12 / 12
  episodes, seeds 0-1, max |diff| 0).
- **The remap is active at 50 machines** (seed 0):
  - MDD-PTECT is unchanged (57,742.97);
  - cmix s1 / s4 / s0 change (55,918.7 → 53,433.0, 55,948.6 → 51,751.1, 60,417.1 → 60,388.0); s2 / s3 are unchanged
    on this seed.
- **Unit checks** at n = 50: size → 0.15; WIP 120 → 36; routing options 10 → 3; dispatch options untouched.

## Results
**Checks.** MDD-PTECT equals rq3-size-v5's run on 40 / 40 seeds in every floor × remap.

Final policy (cmix) vs MDD-PTECT, test seeds 0-39; pooled % [95% CI], Wilcoxon p. The "vs none" column is paired
against the same policy without the remap.

| floor | remap | cmix s3 | cmix mean of 5 | mean of 5 vs none |
|---|---|---|---|---|
| m50 | none | −2.1 [−4.6, +0.2], p 0.02 | −1.3 [−2.6, +0.1], p 0.06 | – |
| m50 | size | −2.4, p 0.005 | −1.5, p 0.01 | −0.2 (n.s.) |
| m50 | wip | −3.1, p 0.03 | −1.9, p 0.03 | −0.6 (n.s.) |
| m50 | opt | −3.0, p 0.005 | −1.7, p 0.03 | −0.4 (p 0.08) |
| m50 | **all** | **−4.2** [−6.4, −1.9], p 0.001 | **−1.9** [−3.4, −0.4], p 0.008 | **−0.6** [−1.3, −0.1], p 0.015 |
| m100 | none | +0.4 (n.s.) | **+2.6** [+1.3, +3.9], p 0.006 | – |
| m100 | size | +0.2 | +2.3, p 0.003 | −0.3 (n.s.) |
| m100 | wip | +1.2 | +2.8, p 0.01 | +0.2 (n.s.) |
| m100 | opt | +0.5 | +2.6, p 0.01 | −0.0 |
| m100 | **all** | +1.3 | **+2.8** [+1.3, +4.3], p 0.01 | +0.2 (n.s.) |

## Reading
1. **The hypothesis is rejected for 100 machines.** Rewriting all three size-dependent inputs into the training range
   leaves the family +2.3 to +2.8% behind MDD-PTECT, the same as without the remap. The out-of-range global scalars
   are not what stops the transfer to 100 machines.
2. **At 50 machines they matter a little.** All three remaps together improve the family by a further 0.6 points
   (p 0.015; pick −4.2% vs −2.1%). Partial input normalization helps where the policy already transfers.
3. **So the 100-machine gap is a change in the floor's dynamics, not in the inputs' scale.** On 100 machines the rule
   landscape is different:
   - the ATC pairs are +23-31% behind MDD-PTECT (+11-13% at 15 machines);
   - the hindsight best pair is only −0.7%.

   A policy trained on 15 machines picks pairs that suit 15-machine states. The same-looking state calls for a
   different pair on a big floor.
4. **For the floor-size study:** test-time input fixes will not close this. Training on the larger floors, or on a
   mix of sizes, is what can. It supports the user's 15 / 50 / 100 / mixed training design, and it argues against
   spending a training arm on the size-normalized-input variant.

