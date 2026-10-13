# base-eplen: does each instance's preferred rule fade with episode length? (2026-10-06/07, RIT cluster)

**Question.** The user's advisor suggested episodes of 10,000s of jobs instead of the current ~300 (6 h window). So far
each 6 h instance seemed to have "its own" best rule, and dev-congestion-signal found that the local preference
follows each 1.5 h block's realized arrivals: sampling noise that longer windows and longer regime blocks should
average out. Does it?

**Setup.**
- All 15 H15 fixed pairs on the twin (tardiness), seeds 0-39.
- Agent windows 6 / 24 / 96 / 200 h, which is about 270 / 1,100 / 4,500 / 9,400 jobs out.
- Mix B2 (the training regime: normal 2 : surge 1 : short fleet 1) with regime blocks of 1.5 h and of 6 h.
- Mix B0 (normal profile only, the calibrated load that levels off) at 1.5 h, added as a stability control after a
  24 h test showed B2's tardiness per 6 h chunk rising.
- The generator is rq2-twin-fleet's with horizon = window + 2.5 h. 6 h / 1.5 h / B2 reproduces rq2-twin-fleet's fixed
  pairs exactly on all 40 seeds.
- 480 cluster tasks (`run.py`, `submit.sh`, `slurm/twin_array.sbatch`), about 5 h. Three tasks died at start-up with
  empty logs, and one more in rq2-oracle-slotlen. Since then the array script logs a line before the virtualenv
  activation; the four were resubmitted and completed.

## Results (`analyze.py`, `analysis.out`)

**Whole episodes.**
- *hindsight gain:* each instance's own best fixed pair vs the best pair on average
- *own best = overall best:* share of instances whose own best pair is the best pair on average
- *rank rho:* agreement of an instance's ranking of the 15 pairs with the average ranking
- *MDD-TECT tardiness per job:* in seconds

| mix, block | window | jobs out | best pair on average | own best = overall best | hindsight gain (mean / median) | rank rho | MDD-TECT tardiness per job |
|---|---|---|---|---|---|---|---|
| B2, 1.5 h | 6 h | 271 | MDD-TECT | 20% | 7.9% / 4.8% | 0.81 | 604 |
| | 24 h | 1,129 | MDD-TECT | 28% | 5.4% / 2.8% | 0.85 | 1,039 |
| | 96 h | 4,541 | **ATC-TECT** | 53% | 1.5% / 0.0% | 0.92 | 1,303 |
| | 200 h | 9,419 | **ATC-TECT** | 75% | 0.6% / 0.0% | 0.94 | 1,326 |
| B2, 6 h | 6 h | 265 | MDD-TECT | 28% | 9.7% / 5.9% | 0.77 | 663 |
| | 24 h | 1,110 | MDD-TECT | 28% | 6.6% / 2.5% | 0.82 | 1,235 |
| | 96 h | 4,486 | MDD-TECT | 38% | 3.9% / 1.4% | 0.88 | 1,740 |
| | 200 h | 9,319 | MDD-TECT | 45% | 2.6% / 0.2% | 0.89 | 1,807 |
| B0, 1.5 h | 6 h | 262 | ATC-TECT | 40% | 4.5% / 2.6% | 0.83 | 350 |
| | 24 h | 1,045 | ATC-TECT | 53% | 2.1% / 0.0% | 0.91 | 416 |
| | 96 h | 4,199 | ATC-ECT | 55% | 1.5% / 0.0% | 0.98 | 483 |
| | 200 h | 8,736 | ATC-TECT | 53% | 0.9% / 0.0% | 0.99 | 470 |

**6 h chunks inside the long episodes** (same statistics per chunk, on a warm floor):

| mix, block | window | chunks | chunk hindsight gain (mean / median) | chunk's best pair = its episode's best |
|---|---|---|---|---|
| B2, 1.5 h | 24 / 96 / 200 h | 160 / 640 / 1,320 | 11.1 / 8.2 / 8.2% (median 6.0-7.5%) | 39 / 29 / 27% |
| B2, 6 h | 24 / 96 / 200 h | 160 / 640 / 1,320 | 13.7 / 18.1 / 18.8% (median 8.5-12.8%) | 39 / 27 / 23% |
| B0, 1.5 h | 24 / 96 / 200 h | 160 / 640 / 1,320 | 6.2 / 7.3 / 6.9% (median 3.6-4.9%) | 43 / 35 / 35% |

**Stability**, MDD-TECT tardiness per 6 h chunk over a 200 h episode:
- B2, 1.5 h blocks: 174 -> 342 -> 350, then 320-470
- B2, 6 h blocks: 190 -> 390 -> 600, then 370-610
- B0: 97-147 throughout

## Reading

1. **The advisor's point holds for the episode score.** With longer windows, the per-instance preference fades:
   - hindsight gain falls from 7.9% at 6 h to 0.6% at 200 h (B2, 1.5 h blocks)
   - ranking agreement rises from 0.81 to 0.94, and to 0.99 on B0
   - most instances end up with the same best pair

   "Each episode has its own best rule" was largely a small-sample effect of about 300-job episodes.
2. **The local preference does not fade.** Inside long episodes, each 6 h chunk still has its own best pair, worth 8%
   in hindsight on B2 (19% with 6 h blocks), and only a quarter of the chunks agree with their episode's best. Long
   episodes average this variation away from the score, but it stays there for a switching policy to exploit.
   Longer regime pieces (6 h blocks) make the chunk-to-chunk differences larger and more persistent: the "longer
   pieces" idea gives a policy bigger and steadier targets.
3. **6 h episodes measure a floor that is still warming up.** On B2, tardiness per 6 h chunk roughly doubles over the
   first day, then levels off at a high but stable level: about 350-400 with 1.5 h blocks, 550 with 6 h blocks. Over
   200 h it does not keep growing. B0 stays near 100-150. Tardiness per job at 6 h (604) is less than half the steady
   state (1,326). Every result so far (oracles, RL training and evaluation) was measured on this transient.
4. **Which pair is best depends on the episode length.**
   - B2 with 1.5 h blocks: MDD-TECT is best on average at 6 and 24 h, ATC-TECT at 96 and 200 h (best for 75% of
     instances at 200 h).
   - B2 with 6 h blocks: MDD-TECT stays best at every length, with growing agreement.
   - B0: ATC-TECT (once ATC-ECT) is best at every length.

   So the MDD-TECT action prior (train-due-twin-slot-prior-kl) and the "MDD-TECT is the strongest fixed rule" result
   both hold for the short, warming-up episodes and for long regime blocks. They do not hold for B2 with 1.5 h blocks
   at steady state.

## Implications for training and evaluation (proposed, not yet decided)

- **Evaluate** on long windows (>= 96 h, or 24 h after a day-long warm-up), scored per 6 h chunk at steady state.
  Per-episode noise then no longer dominates the comparison with fixed rules.
- **Train** with long or continuing episodes: the discount horizon (3 h) and truncation bootstrapping already handle
  length. This also removes the warm-up share (about 15% of each 6 h episode) and the end-of-window effects behind
  the "MDD-TECT in the last hours" habit of the 400k slot policy. A 200 h twin episode costs about 70 s per fixed rule
  on one cluster CPU.
- **Pick the prior's default pair at steady state.** For B2 with 1.5 h blocks that is ATC-TECT, not MDD-TECT.
- **Regime blocks:** 6 h blocks give larger, more persistent differences between chunks, a better-posed switching
  problem than 1.5 h blocks. rq2-oracle-slotlen shows 1 h slots keep 82% of the switching headroom.
