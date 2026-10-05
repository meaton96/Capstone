# eval-due-twin: twin-trained PPO vs fixed pairs and the oracle (held-out, twin; 2026-10-05)

**Setup.** `run.sh`: train-due-twin's 3 final checkpoints (+ the untrained init) on held-out seeds 0-39 of the
training regime (rq2-twin-fleet's B2: 6 h windows, regime blocks normal / surge / 2-AGV, c ~ U[1.75, 2.5], failures
off), deterministic (argmax) actions, `evaluate.py --twin`. Fixed pairs and the switching oracle per seed come from
`../rq2-twin-fleet/tasks/B2_s*.json` (MDD-TECT / SRT-TECT rerun here: equal to 1.5e-5). `analyze.py` -> `analysis.out`.

| policy | mean tardiness | median gap vs MDD-TECT | median gap vs seed's best pair | seeds beating MDD-TECT |
|---|---|---|---|---|
| oracle (B2) | 153.8 | about -11% | | |
| MDD-TECT (best fixed on average) | 172.4 | 0 | | |
| trained s0 | 190.8 | +17.2% | +20.9% | 1 / 40 |
| trained s1 | 193.3 | +11.4% | +19.4% | 6 / 40 |
| trained s2 | 200.7 | +13.9% | +22.2% | 7 / 40 |
| untrained init | 247.8 | +44.5% | +55.5% | 0 / 40 |

**Finding: the policy learned (init +44.5% -> +11-17%) but is worse than one fixed pair.** Training curves
(train-due-twin) are flat within instance noise; explained variance 0.99-1.00, entropy 0.9 -> 0.27-0.49.

**Diagnosis (`../dev-due-twin-actions`, s1, seeds 0-5, decision log):** the deterministic policy spreads its job-rule
choices almost evenly (ATC 22, SPT 21, MDD 19, SRT 19, EDD 19%) and machine rules 34 / 33 / 32%, while confident per
decision (mean top probability 0.92). It switches rule decision by decision on small state differences, which gives an
incoherent priority order (MDD now, SPT next); a fixed rule wins by being consistent. One decision's choice barely
moves tardiness hours later, so per-decision credit assignment is too weak to learn "keep MDD-TECT".

**Proposed fixes (twin, ~2.7 h per run):** (1) slot actions: the agent picks a rule pair per 900 s slot (24 choices per
6 h episode, like the oracle and periodic rule switching in the literature), a wrapper holds the action within the
slot; (2) a longer discount horizon (~10,800 s) since tardiness consequences play out over hours (rq2-twin-warm).
