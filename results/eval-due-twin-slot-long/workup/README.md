# Bad-instance workup of the 400k slot policies (2026-10-06)

Question: is there a property of the generated scenario that the instances where the slot policies lose to MDD-TECT
("bad") have and the ones where they win ("good") do not? If so, the generator could be steered.

**Answer: no.** No scenario feature separates bad from good instances, on 40 or on 200 instances, and no classifier
predicts a loss better than chance. What does explain a loss is an outcome of the instance, not a regime knob: the
policy plays ATC/EDD-TECT and loses where ATC is the wrong family on that instance (MDD-TECT is near-optimal there).
Which family is right is partly predictable from transport congestion, which is a state the policy could observe.

## Data

- **Discovery, seeds 0-39:** `../episodes.csv` (all 5 checkpoints) + slot choices from
  `../../eval-due-twin-slot-long-actions/` (same run with `--decision-log`, tardiness identical to the original:
  max |diff| 0) + fixed pairs and oracle from `rq2-twin-fleet` B2.
- **Replication, seeds 1000-1199:** `../../eval-due-twin-slot-long-ext/` (s0, s3, MDD-TECT, ATC-ECT, decision logs).
  These are new instances of the same B2 regime, outside held-out 0-39 and training (>= 10,000).
- **Label:** mean gap of s0 and s3 to MDD-TECT, window tardiness (Spearman 0.94 with the s0-s3 mean on 0-39). Bad /
  good = top / bottom quartile of the label. On 1000-1199 the two checkpoints lose on 86 / 200 instances (mean gap
  -2.2%, median -0.9%), the same picture as on 0-39 (15 / 40).
- **Features** (`workup.py`, 46 per instance, inside the 6 h agent window):
  - regime shares and where they fall (thirds of the window), regime changes, whether a 2-AGV block precedes the window
  - nominal and realized load, per-slot peaks and variability, segment kinds
  - fluid backlogs per machine type and for transport (arriving work or moves vs capacity, from t = 0)
  - due-date tightness, op means and long-op share

## Results

`workup.out`; figures `timeline_0_39.png`, `timeline_ext.png`, `features_bad_vs_good.png`, `gap_scatter.png`,
`outcome_features_0_39.png`; per-instance tables `features_0_39.csv`, `features_ext.csv`.

1. **No scenario feature separates bad from good.** On 0-39 the strongest correlates are transport peaks / backlog
   (rho ~0.3), tighter due dates (rho -0.27) and more jobs. None survive Benjamini-Hochberg over the 46 features
   (q >= 0.72). On 1000-1199 every feature has |rho| <= 0.21, with q >= 0.13. The top features in the two sets differ.
   No 0-39 hint replicates at q < 0.1. The bad / middle / good distributions overlap almost completely
   (`features_bad_vs_good.png`), and the regime timelines show no pattern (`timeline_*.png`).
2. **Scenario features do not predict a loss.** Predicting gap > 0 from the 46 scenario features:
   - L2 logistic regression: AUC 0.40 (leave-one-out on 0-39), 0.53 (trained on 0-39, tested on 1000-1199), 0.54
     (5-fold on 1000-1199)
   - random forest: AUC 0.35, 0.56 and 0.56

   These are at chance. There is no "bad region" of the generator's space to remove.
3. **What does explain a loss: how ATC does vs MDD-TECT on the instance.** Spearman of the gap with ATC-ECT's gap to
   MDD-TECT: 0.58 (0-39) and 0.61 (1000-1199). In the bad quartile ATC-ECT is 15-18% worse than MDD-TECT; in the good
   quartile it is about 10% better. The fixed pairs on 0-39 show the same thing: the best pair is ATC-TECT on 8 of the
   10 good instances. In the 10 bad ones it is MDD-TECT 4 times and SRT-ECT 3 times, never ATC. The oracle holds
   MDD-TECT through the bad instances (s4, s37, s36 in `timeline_0_39.png`).
4. **What the policies do.** At 400k the seeds learned different strategies (slot shares, 0-39):
   - s0: ATC-TECT 39%, MDD-TECT 34%, EDD-TECT 27%
   - s3: ATC-TECT 65%, SRT-TECT 28%
   - s1: a mix of ATC/EDD/MDD/SRT
   - s2: MDD-ECT / ATC-ECT
   - s4: SRT-TECT 62%

   s0 opens with ATC/EDD-TECT and moves to MDD-TECT for the last 1-2 h on almost every instance, whatever the regime.
   This looks like a time-in-episode habit, not a reaction to the floor state. On bad instances the policies use
   MDD-TECT more (median share of slots 21% vs 6% on 1000-1199) and ATC less (48% vs 67%). So they react somewhat in the right
   direction, but late or not enough.
5. **ATC vs MDD-TECT is partly predictable from congestion.** On 1000-1199, ATC-ECT's gap to MDD-TECT correlates with
   the transport backlog (max rho 0.41, mean 0.38), transport load (0.38), machine backlog (0.29) and less normal-regime
   time (-0.33); all q < 0.001. Under transport congestion (2-AGV blocks, high load) MDD-TECT beats ATC. The policy gap
   correlates only weakly with these (0.15). The policy already picks up part of the signal but does not act on it
   enough.

## Implications

- **Steering the generator away from the bad instances is not possible:** no scenario property marks them (results 1-2).
  Even with one, it would be the wrong fix. The deployed floor does not avoid the hard instances, and training only on
  instances where ATC works would make the policy worse at the ones where it does not.
- The losses come from the choice between the ATC and MDD families, and the right choice depends on the instance's
  congestion. Options, roughly in order of cost:
  - **Action prior / residual on MDD-TECT** (start from the best fixed pair and make deviations pay for themselves).
    This keeps the instances where MDD-TECT is right from becoming losses.
  - **Check the observation carries congestion:** AGV queue / transport backlog, machine backlog, fleet size now. If
    the policy cannot see them, it can only learn a time-of-day habit (result 4).
  - **Oversample congested / short-fleet training instances** (the opposite of steering away), so ATC-vs-MDD mistakes
    cost more during training.
- Seed-to-seed differences are large (result 4), so any change needs 5+ seeds to judge.

Follow-up (10-06): a transport-backlog scalar adds no information beyond obs v3 on the twin; see `../../dev-congestion-signal/README.md`.
