# eval-v5-cmix-unity: do the due-date-trained PPO policies keep their twin gains in Unity? (2026-10-15)

Status: **done**, B2 seeds 0-39 (lab `eval-v5-cmix-unity`):
- first pass, seeds 0-19: 2026-10-16 01:26 (cluster array 21899614, 1 h 46 min, 79 / 79, plus the smoke task, job
  21898765);
- extension, seeds 20-39: 2026-10-16 15:00 (array 21901847, 80 / 80; submitted 11:58, queued behind
  rq2-oracle-bc-future-data4k).

Outputs: `analysis.out` / `summary.csv` (seeds 0-39, the main result), `analysis_s0-19.out` / `summary_s0-19.csv` (the
first pass, reproduced byte for byte by `analyze.py --seeds 0-19`), `analysis_s20-39.out` / `summary_s20-39.csv` (the
new half alone).

## Why
eval-v5-cmix (twin only, 200 held-out B2 seeds) gave the thesis' first positive RL result:
- PPO trained at loose due dates (c 2.5-3.5) is −5.3% total tardiness vs MDD-PTECT at loose c.
- The mixed-c policy (c 1.25-3.5) is −2.1% at base c.

Every Unity check so far (eval-transfer-unity-v5 at base; rq3-gen-v5-unity at load-1.2, due-tight and lay-g) used
only the base-trained PPO s4. Neither of the new policies, and not the loose condition, has run in Unity. On
rq3-gen-v5-unity, MDD-PTECT's tardiness moved by +4% between Unity and the twin under some conditions. A twin gain of
5% needs this check before the thesis reports it as a result on the physical floor.

## Method (`task.sh`, `array.sbatch`, `submit.sh`, `analyze.py`; `conds.json` built from `../rq3-gen-v5/run_task.py`)
- **Conditions:** the same definitions as `../eval-v5-cmix/cmix_task.py`:
  - `due-loose`: c 2.5-3.5;
  - `base`: c 1.75-2.5 (B2).
- **Seeds:** B2 test seeds 0-39, the twin runs' instances (eval-v5-cmix's test seeds). 0-19 ran first; 20-39 were
  added 10-16 to test base c with more power.
- **Policies:**
  - the 6 v5 pairs;
  - the loose-trained pick, train-v5-loose s2;
  - the mixed-c pick, train-v5-cmix s3;
  - the base-trained pick, train-v5-final s4 (the final policy).

  The picks were made on the fitting seeds 2000-2159 in eval-v5-cmix.
- **Groups (4 tasks per seed, 160 in all, 3-4 episodes each):**

  | group | policies | conditions |
  |---|---|---|
  | pa | SRT-PECT, SRT-PTECT, MDD-PECT, MDD-PTECT | due-loose |
  | pb | ATC-PECT, ATC-PTECT, base s4 | due-loose |
  | pc | MDD-PTECT, loose s2, cmix s3 | due-loose, base |

- **Reuse at base:** the 6 pairs and s4 come from `../eval-transfer-unity-v5` and the no-breakdown rows of
  `../rq4-agvfail-v5-rules` / `-policy`.
  - The code, player and settings are the same, and their instance hashes and MDD-PTECT rows are identical across
    those sources on seeds 0-39 (one instance hash and one config hash per seed on 40 / 40; MDD-PTECT
    eval-transfer-unity-v5 = rq4-agvfail-v5-rules on 40 / 40, checked 10-16).
  - pc's MDD-PTECT checks the reuse seed for seed. At due-loose it repeats pa's as a run-to-run check.
- **Same setup as `../rq3-gen-v5-unity`:**
  - code 6d13412a (`../rq2-rebase-v5/code/env`, the checkpoints' code);
  - player `linux_server/` 5e6e0f87 (glibc-patched cluster copy, no `-pyrules`);
  - random warm-up, 21,600 s window, 900 s slots, deterministic;
  - discount horizon 10,800 s, as in the reused rows. It only changes the discounted-return column.

  The frozen evaluate.py runs through `../rq4-agvfail-v5-rules/eval_wrap.py`, which raises Unity's timeout_wait to
  1,800 s and changes nothing else.
- **Twin side:** `../rq3-gen-v5/<cond>/{pdr, ppo}` and `../eval-v5-cmix/test/<cond>/{loose, cmix}`, on the same seeds
  0-19.
- **Analysis (`analyze.py` → `analysis.out`, `summary.csv`; `--seeds A-B` → `analysis_sA-B.out`, `summary_sA-B.csv`):**
  per condition, on the seeds complete in both simulators:
  - each pick vs MDD-PTECT and vs the simulator's best v5 pair, in Unity and in the twin: pooled window tardiness,
    seed-bootstrap 95% CI, seeds better / worse, Wilcoxon, k job-s saved per instance;
  - Unity minus twin of each pooled %, with a paired seed-bootstrap CI: does the gain transfer?
  - the 6 pairs' ranking, Unity vs twin (Kendall τ, best pair in each);
  - MDD-PTECT Unity vs twin;
  - the hindsight best v5 pair.
- **Power caveat:** the twin's base-condition gains are not significant on these seeds, neither on 0-19 (loose s2
  −1.7%, cmix s3 −1.4% vs MDD-PTECT) nor on 0-39 (−1.2 / −1.7%). At base this check can show direction and size, not
  significance. At due-loose the twin gains are −6.9% (loose s2) and −6.7% (cmix s3) on 0-39, both with CIs below 0.
- **Extension (10-16):** `submit.sh full 0 39` (the seed range is an argument, default 0 19; it lists only unfinished
  tasks and refuses to rewrite `tasks.txt` while a `cmix-unity` array is still in the queue). Same code, player,
  checkpoints and settings as the first pass.
- **Smoke:** `submit.sh smoke` runs base / pc / s0 on the cluster (lab `dev-cmix-unity-smoke`). Its output counts
  toward the full run. **Passed 10-15:**
  - MDD-PTECT s0 94,388.418 job-s, identical to the reused eval-transfer-unity-v5 row, with the same instance hash
    (6711812f52868218).
  - Both picks ran in Unity (3 episodes in 3,558 s on skl-a-42, about 20 min per episode): loose s2 86.6 k and
    cmix s3 86.0 k, against 89.2 k and 92.6 k in the twin.

## Results (B2 seeds 0-39, Unity next to the twin on the same seeds)

**Checks pass on 40 / 40 seeds:**
- pc's MDD-PTECT equals the reused base rows and pa's at due-loose, with the same instance hashes.
- Each due-loose seed has one instance hash across pa / pb / pc.
- In the twin, MDD-PTECT in the loose / cmix groups equals rq3-gen-v5's.

**Each pick vs MDD-PTECT** (pooled window tardiness, seed-bootstrap 95% CI, seeds better / worse, Wilcoxon p,
k job-s saved per instance). The last column is Unity minus twin of the pooled %, seeds resampled jointly:

| condition | policy | Unity | twin | Unity − twin |
|---|---|---|---|---|
| due-loose | loose s2 | **−6.2** [−10.7, −2.3], 27/13, p 0.008, 4.4 k | **−6.9** [−12.0, −2.6], 33/7, p 0.0002, 4.8 k | +0.7 pp [−3.8, +5.7] |
| due-loose | cmix s3 | −2.8 [−5.8, −0.3], 26/14, p 0.051, 2.0 k | **−6.7** [−12.2, −2.4], 31/9, p 0.0003, 4.7 k | +3.9 pp [−0.6, +9.1] |
| due-loose | base s4 | +3.6 [−2.1, +8.5], 21/19, p 0.68, −2.6 k | +0.2 [−6.4, +5.5], 26/14, p 0.2, −0.2 k | +3.4 pp [−0.0, +7.5] |
| due-loose | hindsight best v5 pair | −6.6 [−10.3, −3.9], 29/0, 4.6 k | −8.5 [−13.9, −4.9], 35/0, 5.9 k | +1.9 pp [−0.9, +5.5] |
| base | loose s2 | −0.4 [−3.6, +3.0], 21/19, p 1.0, 0.6 k | −1.2 [−3.5, +1.3], 26/14, p 0.29, 1.7 k | +0.8 pp [−1.5, +3.4] |
| base | cmix s3 | +0.5 [−2.5, +3.6], 24/16, p 0.65, −0.7 k | −1.7 [−3.9, +0.4], 24/16, p 0.18, 2.5 k | +2.2 pp [−0.4, +5.0] |
| base | base s4 | +0.3 [−2.6, +3.2], 22/18, p 0.73, −0.4 k | +0.2 [−2.2, +2.4], 16/24, p 0.38, −0.2 k | +0.1 pp [−2.6, +3.1] |
| base | hindsight best v5 pair | −3.8 [−5.2, −2.6], 31/0, 5.6 k | −4.5 [−6.5, −2.9], 32/0, 6.6 k | +0.7 pp [−1.5, +3.1] |

The twin columns equal eval-v5-cmix's test-seed rows for the picks (loose s2 −1.2 / −6.9, cmix s3 −1.7 / −6.7, base
s4 +0.2 / +0.2 at base / due-loose).

**Against each simulator's best v5 pair:**
- At due-loose the best pair is MDD-PTECT in both simulators, so the rows above apply unchanged.
- At base it is MDD-PECT in both, effectively tied with MDD-PTECT (−0.2% Unity, −0.5% twin). Against it: Unity
  loose s2 −0.3, cmix s3 +0.7, base s4 +0.5; twin −0.7, −1.3, +0.6. None is significant.

**The two halves** (pooled % vs MDD-PTECT, Unity / twin, and the Unity − twin gap in pp; bold = gap CI excludes 0):

| condition | policy | seeds 0-19 (first pass) | seeds 20-39 | seeds 0-39 |
|---|---|---|---|---|
| due-loose | loose s2 | −8.2 / −7.9 (−0.2) | −4.5 / −6.0 (+1.6) | −6.2 / −6.9 (+0.7) |
| due-loose | cmix s3 | −4.9 / −6.4 (+1.5) | −0.9 / −7.0 (+6.0) | −2.8 / −6.7 (+3.9) |
| due-loose | base s4 | −0.0 / −0.2 (+0.2) | +7.1 / +0.6 (**+6.5** [+3.8, +10.7]) | +3.6 / +0.2 (+3.4) |
| base | loose s2 | +0.0 / −1.7 (+1.7) | −0.8 / −0.7 (−0.2) | −0.4 / −1.2 (+0.8) |
| base | cmix s3 | +3.6 / −1.4 (**+4.9** [+0.7, +9.4]) | −2.3 / −2.0 (−0.2) | +0.5 / −1.7 (+2.2) |
| base | base s4 | +2.9 / +0.1 (+2.8) | −2.1 / +0.3 (−2.3) | +0.3 / +0.2 (+0.1) |

On seeds 20-39 alone the loose s2 gain at due-loose is −4.5% [−10.6, +0.8] in Unity (14/6, p 0.12) and −6.0%
[−12.7, +0.1] in the twin (17/3, p 0.012).

**Rules, Unity vs twin (seeds 0-39):**

| | due-loose | base |
|---|---|---|
| best v5 pair, Unity / twin | MDD-PTECT / MDD-PTECT | MDD-PECT / MDD-PECT |
| pair ranking Kendall τ (6 pairs) | +1.00 (p 0.003) | +1.00 (p 0.003) |
| MDD-PTECT, Unity vs twin | +1.1% [−2.7, +4.5] (70.3 vs 69.6 k) | +1.3% [−0.9, +3.2] (147.4 vs 145.5 k) |

On seeds 20-39 alone MDD-PTECT is +3.3% [+0.9, +5.4] in Unity at base.

Pairs vs MDD-PTECT (Unity | twin):

| condition | SRT-PECT | SRT-PTECT | MDD-PECT | ATC-PECT | ATC-PTECT |
|---|---|---|---|---|---|
| due-loose | +22.0 \| +21.5 | +25.1 \| +23.2 | +1.5 \| +0.8 | +10.4 \| +7.4 | +9.0 \| +7.3 |
| base | +4.3 \| +4.8 | +6.0 \| +5.7 | −0.2 \| −0.5 | +14.1 \| +13.1 | +12.7 \| +11.3 |

**Per seed** (`analysis.out`, the "per seed" lines):
- Each pick's per-seed effect vs MDD-PTECT correlates between Unity and the twin at r 0.34 / 0.16 / 0.81 at due-loose
  and 0.69 / 0.49 / 0.41 at base (loose s2 / cmix s3 / base s4).
- At base the mixed-c pick's excess is still on the first pass's heavy instances: seeds 5, 2 and 10 carry 108 k of the
  129 k by which Unity exceeds the twin. Without them it is −1.0% in Unity vs −1.4% in the twin.

## First pass: B2 seeds 0-19 (10-16 01:26; `analysis_s0-19.out`, `summary_s0-19.csv`)

**Checks pass:**
- pc's MDD-PTECT equals the reused base rows on 20 / 20 seeds and pa's at due-loose on 20 / 20, with the same
  instance hashes.
- Each due-loose seed has one instance hash across pa / pb / pc.
- In the twin, MDD-PTECT in the loose / cmix groups equals rq3-gen-v5's on 20 / 20.

**Each pick vs MDD-PTECT** (pooled window tardiness, seed-bootstrap 95% CI, seeds better / worse, Wilcoxon p,
k job-s saved per instance). The last column is Unity minus twin of the pooled %, seeds resampled jointly:

| condition | policy | Unity | twin | Unity − twin |
|---|---|---|---|---|
| due-loose | loose s2 | **−8.2** [−15.5, −2.4], 13/7, p 0.04, 5.5 k | **−7.9** [−16.3, −2.8], 16/4, p 0.006, 5.2 k | −0.2 pp [−7.5, +8.0] |
| due-loose | cmix s3 | **−4.9** [−9.7, −1.1], 15/5, p 0.02, 3.3 k | **−6.4** [−14.7, −1.6], 15/5, p 0.01, 4.2 k | +1.5 pp [−3.9, +9.0] |
| due-loose | base s4 | −0.0 [−8.5, +4.8], 11/9, p 0.93, 0.0 k | −0.2 [−10.6, +7.0], 13/7, p 0.37, 0.1 k | +0.2 pp [−6.2, +7.8] |
| due-loose | hindsight best v5 pair | −8.0 [−13.2, −4.8], 16/0, 5.4 k | −9.1 [−19.3, −3.6], 18/0, 6.0 k | +1.2 pp [−3.6, +8.8] |
| base | loose s2 | +0.0 [−4.8, +4.5], 12/8, p 0.96, −0.0 k | −1.7 [−5.3, +1.7], 15/5, p 0.12, 2.4 k | +1.7 pp [−1.8, +6.1] |
| base | cmix s3 | +3.6 [−0.6, +7.2], 10/10, p 0.52, −5.0 k | −1.4 [−5.0, +1.2], 11/9, p 0.5, 1.9 k | **+4.9 pp** [+0.7, +9.4] |
| base | base s4 | +2.9 [−1.4, +7.6], 10/10, p 0.7, −4.0 k | +0.1 [−3.1, +2.3], 9/11, p 0.6, −0.1 k | +2.8 pp [−2.0, +7.9] |
| base | hindsight best v5 pair | −3.4 [−5.9, −2.0], 16/0, 4.8 k | −6.3 [−9.1, −3.9], 19/0, 8.9 k | +2.9 pp [−0.3, +5.8] |

**Against each simulator's best v5 pair:**
- At due-loose the best pair is MDD-PTECT in both simulators, so the rows above apply unchanged.
- At base the best pair is MDD-PTECT in Unity but MDD-PECT in the twin, which is −3.3% vs MDD-PTECT on these seeds.
  Against it, the twin rows become loose s2 +1.7, cmix s3 +2.1 and base s4 +3.6 (all n.s.). The Unity rows are as
  above.

**Rules, Unity vs twin:**

| | due-loose | base |
|---|---|---|
| best v5 pair, Unity / twin | MDD-PTECT / MDD-PTECT | MDD-PTECT / MDD-PECT |
| pair ranking Kendall τ (6 pairs) | +0.73 (p 0.056) | +0.73 (p 0.056) |
| MDD-PTECT, Unity vs twin | +3.0% [−3.7, +8.7] (67.7 vs 65.8 k) | −0.9% [−4.6, +2.2] (140.0 vs 141.2 k) |

Pairs vs MDD-PTECT at due-loose (Unity | twin):

| SRT-PECT | SRT-PTECT | MDD-PECT | ATC-PECT | ATC-PTECT |
|---|---|---|---|---|
| +22.8 \| +26.6 | +23.3 \| +26.4 | +1.4 \| +2.2 | +7.1 \| +5.5 | +6.5 \| +7.9 |

Base is in `../rq3-gen-v5-unity`.

**Per seed** (`analysis_s0-19.out`, the "per seed" lines):
- Each pick's per-seed effect vs MDD-PTECT is only weakly correlated between Unity and the twin: r 0.17 / 0.14 / 0.62
  at due-loose and 0.60 / 0.23 / 0.15 at base (loose s2 / cmix s3 / base s4).
- The base misses sit on a few heavy instances.
  - cmix s3: seeds 5, 2 and 10 carry 108 k of the 138 k by which Unity exceeds the twin. Without them it is +0.9% in
    Unity vs −0.5% in the twin.
  - base s4: seeds 2, 5 and 13 carry +102 k. The other 17 seeds net −24 k, for a total excess of +78 k.
  - Seed 5 is the largest single swing: cmix s3 is −30.8 k in the twin and +16.7 k in Unity.

**First-pass reading (superseded by seeds 0-39):** the loose-c gain transferred (−8.2 / −4.9% in Unity for loose s2
/ cmix s3 against −7.9 / −6.4% in the twin), and the base-c gain did not show (cmix s3 +3.6% in Unity against −1.4%
in the twin, a gap of +4.9 pp [+0.7, +9.4] carried by seeds 5, 2 and 10). Seeds 20-39 did not repeat the base-c gap
(−0.2 pp) and shrank the mixed-c pick's Unity gain at loose c.

## Reading (seeds 0-39)
1. **The loose-trained policy's gain at loose due dates holds in Unity.**
   - The loose s2 pick is −6.2% [−10.7, −2.3] vs MDD-PTECT in Unity (27 of 40 seeds better, p 0.008) against −6.9% in
     the twin on the same seeds. The gap is +0.7 pp [−3.8, +5.7].
   - It saves 4.4 k job-s per instance in Unity, 95% of what the hindsight best pair saves there (4.6 k).
   - The first pass's −8.2% was on the high side: the new half alone gives −4.5% (n.s. on 20 seeds).
   - The base-trained s4 does not gain at loose c in either simulator (+3.6 / +0.2%), so the gain comes from the
     training distribution.
2. **The mixed-c policy's loose-c gain only partly transfers.**
   - cmix s3 is −2.8% [−5.8, −0.3] in Unity (p 0.051) against −6.7% in the twin, a gap of +3.9 pp [−0.6, +9.1].
   - On seeds 20-39 alone it is −0.9% in Unity against −7.0% in the twin. The first pass (−4.9%) overstated it.
3. **At base c, Unity is now inconclusive rather than negative.**
   - Neither simulator shows a significant gain on these 40 seeds: loose s2 −0.4% / −1.2% and cmix s3 +0.5% / −1.7%
     (Unity / twin), all n.s.
   - The first pass's +4.9 pp gap for the mixed-c pick did not repeat on seeds 20-39 (−0.2 pp: −2.3% in Unity, −2.0%
     in the twin). On 40 seeds the gap is +2.2 pp [−0.4, +5.0], not significant. Most of it is still seeds 5, 2 and
     10; without them Unity and twin agree (−1.0 / −1.4%).
   - The Unity CI for the mixed-c pick (−2.5 to +3.6) contains both the twin's −1.7% and 0. 40 seeds cannot tell a
     1-2% gain from none. From the CI width, detecting a 1.7% effect in Unity (80% power) would take roughly 250
     seeds of the pick.
4. **Unity costs the learned policies a little, consistently but not significantly.**
   - All six 40-seed gaps (3 picks x 2 conditions) are at or above 0: +0.1 to +3.9 pp, mean +1.8 pp. None has a CI
     clear of 0 against MDD-PTECT; the closest is base s4 at loose c, +3.4 pp [−0.0, +7.5] (and [+0.03, +7.6] against
     the best pair, the same rule there, in another bootstrap draw).
   - A transfer cost of 1-2 pp matters for a 2% claim, not for a 6% one.
5. **The rules rank identically in both simulators.**
   - τ +1.00 at both c on 40 seeds (p 0.003). The best pair is MDD-PTECT at loose c and MDD-PECT at base in both
     simulators; MDD-PECT and MDD-PTECT are within 0.5% of each other at base.
   - MDD-PTECT's level differs by +1.1% (loose) and +1.3% (base), both n.s.
6. **20 seeds are too few for gains of 2-5%.**
   - Per-seed effects correlate only 0.16-0.81 between the simulators.
   - The two halves disagree: 2 of the 12 half-sample gaps have CIs clear of 0, one in each half, in different cells,
     and neither repeats in the other half.
7. **For the thesis:**
   - Report the loose-c result as holding on the physical floor: the loose-trained policy is −6.2% vs the best fixed
     pair in Unity on B2 test seeds 0-39 (27 / 40 better), next to −6.9% in the twin on the same seeds.
   - Report the mixed-c policy's loose-c gain as partly transferring: −2.8% in Unity (p 0.051) against −6.7% in the
     twin.
   - Keep the base-c gain (mixed-c family −2.1%, 200 seeds) a twin result. In Unity on 40 seeds the picks are at
     parity with MDD-PTECT (−0.4 / +0.5%). That is consistent with a small gain and with none. Drop the first pass's
     "4.9 pp worse in Unity": it did not replicate.
