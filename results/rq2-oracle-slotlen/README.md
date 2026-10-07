# rq2-oracle-slotlen: switching headroom vs decision-slot length (2026-10-06, RIT cluster)

Question (user, 10-06): would a longer decision interval make a difference? Longer slots give fewer decisions with
larger effects each (easier to learn from), but slower reactions.

**Setup.**
- The H15 greedy switching oracle of rq2-twin-fleet: twin, B2 regime, tardiness, 6 h agent window, hold-to-end
  tails, seeds 0-39.
- Slot lengths: 1,800 / 3,600 / 5,400 / 10,800 s here; 900 s from rq2-twin-fleet.
- Every slot length's fixed pairs equal rq2-twin-fleet's exactly (seed 26 checked locally before the run).
- 160 cluster tasks (`run.py`, `submit.sh`, `slurm/twin_array.sbatch`), about 3.5 h. One task (10,800 s, seed 39)
  died at start-up on its node with empty logs; it was resubmitted and completed.

## Result (`analyze.py`, `analysis.out`)

| slot | decisions / 6 h | oracle vs MDD-TECT (mean of sums) | share of the 15-min gain | oracle vs each instance's own best fixed pair (mean of sums / median) | oracle switches per episode |
|---|---|---|---|---|---|
| 15 min | 24 | 10.8% | 100% | 6.3% / 7.9% | 6.0 |
| 30 min | 12 | 9.9% | 91% | 5.3% / 5.2% | 4.0 |
| 1 h | 6 | 8.9% | 82% | 4.2% / 3.8% | 2.4 |
| 1.5 h | 4 | 8.2% | 76% | 3.5% / 2.6% | 1.5 |
| 3 h | 2 | 6.9% | 64% | 2.2% / 0.4% | 0.6 |

Picking each instance's best fixed pair in hindsight (no switching at all) is worth 4.8% over MDD-TECT.

## Reading

- **The oracle's 10.8% at 15-min slots has two parts:**
  - About 4.8 points come from knowing which fixed pair suits the instance.
  - About 6 points come from switching within the episode.
  - The switching part shrinks steadily with slot length (6.3 -> 5.3 -> 4.2 -> 3.5 -> 2.2 points), but slowly at
    first: 1 h slots keep 82% of the total gain with a quarter of the decisions.
- **For training:** 30-min or 1-h slots cost 1-2 points of headroom but give each decision 2-4 times the effect.
  Per-slot effects are about 1% at 15 min (dev-congestion-signal). This is a cheap knob to try against the weak
  learning signal found in train-due-twin-slot-prior-kl. The Unity oracle (rq2-oracle-due-long) already used 30-min
  slots for the 6 h settings and found 8.8-12.8%.
- **Instance choice is a large part of the headroom.** About half of what the oracle gains over MDD-TECT comes from
  telling which kind of instance it is in. This is the part the policy finds hard: the state predicts the right family
  with AUC <= 0.70 (dev-congestion-signal), and the preference follows each block's realized arrivals.
  base-eplen tests whether that per-instance preference fades with longer episodes.
