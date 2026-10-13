# train-due-twin-slot: PPO on the twin with slot actions and a 3 h horizon (2026-10-05)

Same regime as train-due-twin; the agent picks a rule pair per 900 s slot, held for every decision in it
(`env/env_wrappers/slot_env.py`, `--slot-seconds 900`), discount horizon 10,800 s, 3 seeds x 100k agent steps
(~4,200 six-hour episodes each), rollout 32. Why: `../eval-due-twin/README.md`. Wrapper checks (10-05): MDD-TECT held
per slot equals the fixed run exactly; replaying an oracle schedule reproduces the oracle exactly (B2 seeds 0 and 3).
Training: 1.7 h per seed (~16 agent steps/s); curves rose 15-17% on seeds 1-2 (-218 -> -180, -206 -> -176), ~3% on
seed 0, still rising at the end.

## Held-out result (eval-due-twin-slot, 10-05 19:46; twin, B2 seeds 0-39, deterministic)

| policy | mean tardiness vs MDD-TECT | median gap | seeds better than MDD-TECT | worst seed | Wilcoxon p | oracle gap closed (median) |
|---|---|---|---|---|---|---|
| seed 0 | +3.00% | +0.90% | 17 / 40 | +25.8% | 0.023 (worse) | -0.07 |
| seed 1 | -0.42% | **-3.04%** | **27 / 40** | +29.5% | 0.22 | 0.16 |
| seed 2 | +0.27% | -1.75% | 26 / 40 | +19.8% | 0.43 | 0.15 |
| untrained | +52.8% (median) | | 0 / 40 | | | |

(Oracle: about 11% below MDD-TECT on these seeds; `../eval-due-twin-slot/analysis.out`.)

**Reading.** Slot actions + the 3 h horizon turned an 11-17% loss (per-decision run) into **parity with the best fixed
pair**: seeds 1-2 beat MDD-TECT on about two thirds of the instances and by 1.8-3.0% in the median, but a few bad
instances (up to +30%) cancel the mean, and neither difference is significant; seed 0 is significantly worse. The
first RL result here that matches the strongest fixed rule. Next: train longer (curves still rising), look at the bad
instances (which blocks: surge / short fleet?), more training seeds, then Unity evaluation.

## Where it loses: bad-instance analysis (10-05; `../eval-due-twin-slot/bad_instances.py` -> `bad_instances.out`,
## slot choices `../eval-due-twin-slot/bad_instances_actions.out`)

- **The policy loses where MDD-TECT is already near-optimal.** Worst 8 seeds vs the other 32: policy gap +11 / +10%
  vs **-7 / -5%** (s1 / s2); MDD-TECT's gap to the seed's own best pair 1.3% vs 12.6%; oracle gain 8% vs 17%.
  Spearman with the policy gap over 40 seeds: MDD-TECT's gap to own best -0.74, oracle gain -0.63, jobs in window
  +0.43, MDD-TECT tardiness +0.40, short-fleet share +0.38; surge share, c and warm-up ~0.
- **What it does** (s1, deterministic, 7 seeds): ATC-ECT in 67% of slots, MDD-ECT 15%, SRT-ECT 10%, TECT almost
  never; 4.4 switches per 24-slot episode. It wins where the oracle uses ATC (seeds 1, 9: -27%, -7%) and loses where
  the oracle holds MDD-TECT (seeds 37, 34, 36: +30%, +13%, +8%).
- **It is not just ATC-ECT:** vs the fixed ATC-ECT pair the policy is 12% better on mean tardiness (s1 -12.3%,
  s2 -11.7%; better on 34 / 30 of 40 seeds; Wilcoxon p < 0.001). It learned ATC-ECT as a default plus useful
  switching, which brings it to MDD-TECT's level (ATC-ECT alone is 6.5% worse in the median); it has not learned to
  recognize the instances (heavier, more short-fleet time) where MDD-TECT should be held, and it barely uses TECT.
- **Next:** train-due-twin-slot-long (400k steps, 5 seeds, running) tests whether more training fixes it. If not:
  start from the best fixed pair (an action prior / bias toward MDD-TECT, so the policy learns deviations from it),
  or more training instances of the heavy / short-fleet kind.
