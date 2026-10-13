# dev-big-batch: 2,048 slot steps per update (2026-10-07, handoff fix 6a)

## Why

The slot runs update on 512 slot steps (16 envs x 32), about 21 episodes. With an SNR of 0.03-0.05 per slot, a few
thousand per update is the right order (credit trace section 6, item 5): more samples per gradient, not more epochs.

## What changed

Flags only: `--rollout-length 128 --batch-size 256`, so 16 x 128 = 2,048 slot steps per update, still 8 minibatches x 4
epochs. Over the same 100k steps there are 49 updates instead of 195.

## Run

dev-prior-offset with these flags, seeds 0-1. Compare with dev-prior-offset.

## Result (10-08 04:11, cluster CPU; `analysis.out`)
- **Held-out B2 seeds 0-39:** both seeds play MDD-TECT in 100% of slots, exactly MDD-TECT (0.000%). It is the same as
  dev-prior-offset and train-due-twin-slot-prior-kl.
- **Learning curves** match the prior-kl reference within noise (s0 −182 → −197, s1 −201 → −180 by fifth).
  Explained variance is 0.75, the same as the reference.
- **The M1 fix is visible:** output weight norm 1.53 (offset prior) vs 0.55-0.71 (scaled-init prior).
- **The policy still never leaves MDD-TECT.** Mean p(ATC) is 0.02-0.05. It is correlated with the right instances
  (Spearman −0.34 to −0.36 with ATC-ECT's gap), but far from flipping the argmax.
- **Reading:** 4× the samples per update did not move the deterministic policy. With the MDD-TECT prior at p = 0.8 and
  KL 0.05, the per-slot signal (SNR 0.03-0.05) is too weak to overcome the KL pull. That holds with the fixed init
  (dev-prior-offset) and with bigger batches.
