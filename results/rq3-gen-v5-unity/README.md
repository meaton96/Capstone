# rq3-gen-v5-unity: Unity subset of the v5 RQ3 sweep (2026-10-14)

Status: **done** 2026-10-15 01:27 (cluster array 21894272, 3 h 29 min, 120 / 120; lab `rq3-gen-v5-unity`).
`analysis.out`.

## Why
Demo 1 slide 17 asks for the RQ3 sweep with the final policy "plus a small Unity subset": does the twin's RQ3
answer hold on the physical floor?

## Method (`task.sh`, `array.sbatch`, `submit.sh`, `analyze.py`)
- **Conditions** (chosen before the twin results; `conds.json` is built from `../rq3-gen-v5/run_task.py`):

  | condition | why |
  |---|---|
  | `load-1.2` | the busiest floor, so the most AGV traffic |
  | `due-tight` | the best pair changed in the H15 twin sweep |
  | `lay-g` | the deeper layout, where the H15 policies turned positive |

  `base` comes from `../eval-transfer-unity-v5` plus the no-breakdown rows of `../rq4-agvfail-v5-*`.
- **Seeds:** B2 0-19 (the twin sweep's instances).
- **Policies:** the 6 v5 pairs + the final PPO policy (`../eval-v5-final/selected.txt`). Groups pa = SRT/MDD pairs,
  pb = ATC pairs + PPO.
- **Same setup as `../eval-transfer-unity-v5`:** code 6d13412a, player `linux_server/` 5e6e0f87 (glibc-patched
  cluster copy, no `-pyrules`), random warm-up, 21,600 s window, 900 s slots.
- **Analysis:** per condition, the pair ranking in Unity and the twin (Kendall tau, same best pair or not), MDD-PTECT
  Unity against twin, and the final policy against the best pair and against MDD-PTECT in each simulator, on the
  same seeds.

## Results (B2 seeds 0-19, Unity next to the twin on the same seeds; final policy PPO s4)

| condition | best v5 pair, Unity / twin | pair ranking τ (6 pairs) | MDD-PTECT, Unity vs twin | PPO s4 vs best pair, Unity / twin | PPO s4 vs MDD-PTECT, Unity / twin | hindsight best pair, Unity / twin |
|---|---|---|---|---|---|---|
| base | MDD-PTECT / MDD-PECT | +0.73 | −0.9% (n.s.) | +2.9 / +3.6 (n.s.) | +2.9 / +0.1 | −3.4 / −3.1 |
| load-1.2 | MDD-PECT / MDD-PTECT | +0.60 | **+4.3%** [+1.4, +7.7] | −0.6 / +0.2 | −2.3 [−5.1, +0.3] / +0.2 | −3.0 / −2.6 |
| due-tight | **SRT-PECT** / MDD-PTECT | +0.33 | **+3.9%** [+1.3, +6.5] | −0.2 / +0.1 | −2.9 [−5.4, −0.5], p 0.14 / +0.1 | −3.0 / −3.7 |
| lay-g | MDD-PTECT / MDD-PECT | +0.60 | +0.7% | −1.7 / +0.4 | −1.7 / −0.8 | −4.3 / −4.9 |

**Pairs vs MDD-PTECT** (Unity | twin):

| pair | base | load-1.2 | due-tight | lay-g |
|---|---|---|---|---|
| MDD-PECT | +0.3 \| −3.3 | −1.7 \| +0.1 | −1.4 \| +0.2 | +0.1 \| −1.2 |
| SRT-PECT | +5.9 \| +5.0 | +2.2 \| +4.3 | **−2.7** \| +0.7 | +5.4 \| +1.7 |
| SRT-PTECT | +6.0 \| +4.7 | +1.7 \| +4.9 | **−2.6** \| +0.7 | +4.0 \| +2.1 |
| ATC-PECT | +15.9 \| +10.8 | +9.3 \| +11.4 | +8.1 \| +10.9 | +11.6 \| +12.0 |
| ATC-PTECT | +11.7 \| +10.0 | +11.2 \| +11.1 | +7.7 \| +10.0 | +11.9 \| +10.8 |

## Reading
1. **The twin's RQ3 answer holds in Unity.** In no condition does the final policy beat the best fixed pair: in Unity
   it is −1.7 to +2.9%, in the twin −0.6 to +3.6%, all n.s.
   - The hindsight headroom is the same in both simulators (−3 to −5%). No observable policy takes it in either.
2. **The rule families rank the same; the near-ties swap.**
   - The MDD pairs with a processing-time machine rule are best or within 2% everywhere.
   - ATC is 8-16% behind in both simulators.
   - The top pair changes between simulators only among pairs within about 3% of each other, hence τ 0.33-0.73 over
     6 pairs.
   - **Tight due dates favour SRT in Unity** (SRT-PECT −2.7% vs MDD-PTECT): the twin's 40-seed sweep had found the
     same (rq3-gen-v5 due-tight: SRT-PECT best).
3. **Unity is about 4% worse than the twin on the busier conditions** (load ×1.2, tight due dates), and equal at base
   and on layout G. That is the AGV congestion the free-flow twin lacks, small at 7 AGVs, as in twin-gap-reg.
4. **PPO s4 vs MDD-PTECT is −2.3% (load-1.2) and −2.9% (due-tight) in Unity but +0.1-0.2% in the twin.** Against
   each condition's best pair it is still at parity (−0.6 / −0.2%). In those conditions the reference rule itself
   loses ground in Unity, and the policy's switching between near-tied pairs absorbs part of that. It is not a
   policy edge, and it needs more than 20 seeds before it is reported as one.
5. **For the thesis (RQ3):**
   - The conditions-sweep answer (no condition where the policy beats the best pair; the conditions shift which rule
     is best, mainly the job half) is confirmed on the physical floor for the three conditions most likely to
     differ.
   - The twin is a valid stand-in for this question at 7 AGVs.
