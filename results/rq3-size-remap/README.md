# rq3-size-remap: why the final policy does not transfer to 100 machines (no-training diagnostic) (2026-10-19, user)

Status: **queued** (lab `rq3-size-remap`, cluster, 20 twin tasks).

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
Pending.
