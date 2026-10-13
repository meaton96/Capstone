# train-due-twin-slot-long: slot policies to 400k agent steps (2026-10-05/06)

Same setup as `../train-due-twin-slot/README.md` (B2 regime, 900 s slots, 10,800 s discount horizon, tardiness
reward), trained to 400k agent steps instead of 100k: s0-2 resumed from the train-due-twin-slot checkpoints (100k),
s3-4 started fresh. Five runs in parallel on the local machine, about 15-16 agent steps/s each (5.5 h for the resumed
seeds, 6.9 h for the fresh ones). Launcher `run_local.sh`, curve summary `analyze.py`.

Training return by fifth of training (first -> last): s0 -191 -> -180, s1 -190 -> -196, s2 -185 -> -181,
s3 -200 -> -184, s4 -206 -> -186. The resumed seeds barely moved after 100k. The fresh seeds rose for the first
fifth and then flattened. Episodes draw different load regimes, so these curves are noisy.

## Held-out result (eval-due-twin-slot-long, 10-06 06:05; twin, B2 seeds 0-39, deterministic)

`../eval-due-twin-slot-long/analysis.out`. Fixed pairs reproduce the B2 numbers (max |diff| 1.45e-05). Best fixed
pair MDD-TECT: mean tardiness 172.39. Oracle: 153.78 (about 11% below MDD-TECT).

| policy | mean tardiness vs MDD-TECT | median gap | seeds better | worst seed | Wilcoxon p | oracle gap closed (median) | 100k checkpoint (median gap) |
|---|---|---|---|---|---|---|---|
| s0 (resumed) | **-2.44%** | -2.57% | 24 / 40 | +11.5% | **0.046 (better)** | 0.22 | +0.90% |
| s1 (resumed) | +0.57% | -1.10% | 24 / 40 | +14.5% | 0.98 | 0.11 | -3.04% |
| s2 (resumed) | +1.15% | -0.20% | 22 / 40 | +22.7% | 0.60 | 0.02 | -1.75% |
| s3 (fresh) | -0.93% | **-4.17%** | 24 / 40 | +18.8% | 0.20 | 0.23 | |
| s4 (fresh) | +2.34% | +2.93% | 18 / 40 | +34.8% | 0.16 | -0.18 | |
| s3 untrained | +0.15% | 0.00% | 6 / 40 | +9.8% | 0.42 | 0.00 | |

(Mean tardiness from `-return`; paired over the 40 seeds. The untrained s3 network happens to act almost like
MDD-TECT under the deterministic argmax. The untrained s0 network in eval-due-twin-slot was +52.8%.)

**Reading.** Training four times as long did not move the slot policies past parity with MDD-TECT. Over the 5 seeds
the median gap ranges from -4.2% to +2.9% and the mean gap from -2.4% to +2.3%. s0 is the only seed significantly
better than MDD-TECT, and only just (p = 0.046, mean -2.4%). Going from 100k to 400k made s0 better, but s1 and s2
worse, so seed-to-seed variation is larger than the effect of the extra training. The worst instances shrank for the
resumed seeds (+26-30% at 100k, +12-23% now), but every seed still loses on 15-22 of 40 instances. The median oracle
gap closed is at most 0.23, so the 11% oracle headroom is still mostly unused.

**Implication.** More steps of the same PPO setup is not the lever; the bad-instance analysis in
`../train-due-twin-slot/README.md` still applies (the policy does not recognize instances where MDD-TECT should be
held). Candidates from there: an action prior / bias toward MDD-TECT so the policy learns deviations from it, or more
heavy / short-fleet training instances.

## Where it loses: bad-instance analysis (10-06; `../eval-due-twin-slot-long/bad_instances.py` -> `bad_instances.out`)

Same script as at 100k, over all 5 seeds.

- **Same instances as at 100k.** 7 of the 8 worst seeds (mean gap over the 5 policies) were also in the 100k worst 8
  (7, 20, 26, 33, 34, 36, 37); Spearman of the per-instance mean gaps, 400k vs 100k, 0.67. Extra training did not fix
  the losing instances.
- **Same cause.** On the worst 8, MDD-TECT is within 1.6% of the instance's best fixed pair (12.5% on the other 32),
  and the oracle gains 9% (17% elsewhere). Spearman with the mean policy gap: MDD-TECT's gap to own best -0.58, oracle
  gain -0.46. Four of the worst 8 have MDD-TECT as the best fixed pair. The regime mix (surge / short-fleet share, c,
  op mean, warm-up) correlates weakly (|rho| <= 0.26), so the instances that should be held on MDD-TECT are not
  visibly marked by the regime.
- **Consistent across seeds.** s0-s3 agree on which instances are hard (each seed's gap vs the others' mean,
  rho 0.50-0.59); s4 is the outlier (rho 0.14, loses on 22 / 40). Lost to MDD-TECT: s0 15, s1 16, s2 18, s3 16, s4 22
  of 40.
- **Implication:** the policy's problem is systematic (it loses where MDD-TECT is already
  near-optimal), not noise that more steps average out. This points to the action-prior option: start from MDD-TECT
  and make deviations pay for themselves.

Thorough workup with 200 replication instances: `../eval-due-twin-slot-long/workup/README.md` (no scenario property separates bad from good; losses follow ATC vs MDD-TECT on the instance).
