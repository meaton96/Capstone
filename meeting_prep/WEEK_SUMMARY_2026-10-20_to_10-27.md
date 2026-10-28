# Week summary: 2026-10-20 to 10-27

% = tardiness vs MDD-PTECT on B2-family test seeds; negative is better. Details in `results/<name>/README.md`.

## TL;DR
- The thesis is now organized as papers, one per chapter. Writing restarted in `docs/_Thesis_New` (Ch. 1 and 2.1-2.5 done).
- Floor-size transfer is the main thread. A policy's edge follows the room a floor leaves. On large floors that room
  is in low α (travel-aware rules), which the v5 rule set lacks. Fine-tuning the weights does not fix it.
- The α policy (spread law + learned offset) works on heterogeneous floors and never loses on large ones, but it does
  not yet capture the 200-machine room.
- One run is still going: the 100-machine fine-tune on the cluster (ETA 10-30).

## Thesis organization
- **Outline by papers (10-23)**, `docs/Thesis/Outline_Papers` (35 pp): 1 Intro · 2 Problem setting and platform ·
  3 Physical floor (paper D) · 4 One parameter instead of a policy (B) · 5 Learning to switch rules (A) · 6 How far the
  edge lasts (T) · 7 3D plant for a DES twin (C) · 8 Ethics · 9 Conclusion. Background and Related Work now go in each
  chapter's own intro.
- **Writing (10-26 on)** in `docs/_Thesis_New`: Ch. 1 is done except 1.2 Contributions, which waits on results.
  Ch. 2.1-2.5 are done (2.5 twin on 10-27). Next: 2.6 benchmark, Ch. 3, Ch. 5.
- **Rules from the 10-21 review:** results appear only in results sections, history is told once, and the 160-page
  draft gets cut hard.

## Decisions this week
| date | decision |
|---|---|
| 10-21 (advisor) | Finish in spring (May 2027; papers by April). 50-100 machines is the realistic range, so the 1000-machine study is dropped. Plan the work as papers A / B / C (+ optional D). |
| 10-21 | Training under noise closed (fails every confirmatory cell). |
| 10-23 | α becomes part of the action (not a 7th rule pair). |
| 10-25 | α-head fix: the spread law sets α per floor and the head picks an offset. Fine-tune targets are 100 machines + spread 1.0 (not 200). Anchored PPO closed; future-table code merged. Mixed-size training dropped; the claim becomes "train at the largest deployment size". |
| 10-27 | off-het carried forward as the α policy (pre-declared rule D). |

## Timeline (remaining-work plan, 10-23)
| when | experiments | writing |
|---|---|---|
| now - mid Nov | finish transfer (T); online α | paper A |
| mid Nov - Dec | α-action policy; Unity check of the tuned rule | paper B outline |
| Dec - Feb | DT loop: Unity → twin sync, B2 gate first | paper B |
| Feb - Mar | DT factorial; GNN only if time | paper C |
| Mar - Apr | none | thesis assembly |

## Finished experiments
| experiment | result |
|---|---|
| `rq3-size-grid` (ref 10-20, arms 10-24) | 15-machine policy: −5.3% at 20 machines, −1.3% at 50, +2.6% at 100, +9.5% at 200 (holds within ~3x its training size). Transfer runs downward: the 100-machine-trained policy (T100) is best on 15-100 and never worse. Mixed-size training had 2 of 5 seeds collapse to MDD-PTECT. |
| `rq3-size-oracle` / `-catalog` / `-portfolio` (10-20/21) | v5 switching ceiling −9.4 / −10.3 / −3.3 / −1.0% at 15 / 50 / 100 / 200 machines. The best machine rule flips with size: MDD-TECT −19.0% at 200. The rule "short fleet → MDD-TECT" gives −12.2% at 100. |
| `rq3-alpha-size` (10-21) | Spread law α ≈ max(0, 1.8 − 4σ) holds at 15-100 machines. The size term fails its gate, so there is no static size-aware α. |
| `eval-v5-future` (10-21) | PPO with the per-job future table: −2.6% (from scratch) / −3.0% (BC init), the best learned policies. Only ~0.4 pts of that comes from reading the forecast. |
| `eval-v5-uncert-dr` (10-21) | Training under noise never beats the noise-free policy. Closed. |
| `rq2-replan` (10-22) | Planner gain saturates at 5-min re-plans (−11.2%). Event triggers never beat periodic re-planning. |
| `eval-v5-slot` (10-22) | PPO decision interval: 10 min is best (−3.0%), 3-30 min is a plateau, below 3 min is unstable (1 min +6.2%). Answers the advisor's "why 5 / 15 min". |
| `base-late-share` (10-22) | 51.8% of B2 jobs are late under MDD-PTECT ("about half late" holds). |
| `eval-alpha-spread` (10-23) | α-as-action v6 reaches the spread law on spread floors (−16% at s 1) but settles on one α everywhere: +4.1% on B2. Fail. |
| `rq3-transfer-map` (10-25) | Across 42 conditions the edge tracks the room a condition leaves (ρ +0.78), not the distance from training. The policy captures ~0.38 of that room and loses its edge only at ≥ 75 machines or under noise. |
| `eval-alpha-offset` (10-26) | Law + offset (v7) passes both checks: −16.4% at s 1, no cost on B2. Trained on B2 only it is the best arm on B2 (−2.55%). |
| `rq3-size-headroom` (10-26) | Headroom shrinks on big floors because of transport (fleet size, ~700 m trips), not averaging. The v5 + MDD-TECT ceiling is −13 to −21% at every size and fleet. |
| `eval-ft` het arm (10-26) | 24 fine-tune updates gain only −0.47 pts (n.s.), 18% of the way to the target. The limit is the rule set, not the weights. |
| `eval-alpha-off-size` (10-27) | off-het is never worse than MDD-PTECT at 50-200 machines and beats the final policy by 3.6 / 9.0 pts at 100 / 200. It gains nothing at 200 (−0.27%) even though fixed α 0.75 gives −10.9% there. |

## Running
- `train-ft-m100`: fine-tunes the final policy on 100 machines (cluster, 5 seeds). 1 of 5 done, the rest are queued
  on Slurm; ETA 10-30.
- `eval-ft-m100` is waiting on it.

## Open questions for the meeting
1. **200-machine room:** test the α head alone (job / machine fixed to MDD-PTECT), or add a size / transport term to
   the law?
2. **Which policy the thesis calls "final":** v5 mixed-due-date (paper A data) or law + offset?
3. **When to start the Unity → twin sync:** it takes 5-7 weeks, the longest lead time in the plan.
4. **Static benchmark instances (Ch. 2.6) and machine failures with the final policy (Ch. 6.3):** run them, or say why not?
5. **Optional GP-evolved rule baseline for paper B:** about 2-3 days.
