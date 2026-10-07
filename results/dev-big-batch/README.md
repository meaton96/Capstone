# dev-big-batch: 2,048 slot steps per update (2026-10-07, handoff fix 6a)

## Why

The slot runs update on 512 slot steps (16 envs x 32), about 21 episodes. With an SNR of 0.03-0.05 per slot, a few
thousand per update is the right order (credit trace section 6, item 5): more samples per gradient, not more epochs.

## What changed

Flags only: `--rollout-length 128 --batch-size 256`, so 16 x 128 = 2,048 slot steps per update, still 8 minibatches x 4
epochs. Over the same 100k steps there are 49 updates instead of 195.

## Run

dev-prior-offset with these flags, seeds 0-1. Compare with dev-prior-offset.

## Result

(pending: `analysis.out`)
