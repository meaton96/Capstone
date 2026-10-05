# train-due-twin: PPO on the event-based twin, per-decision actions (2026-10-05)

3 seeds x 3M decisions on the local RTX 4070 Ti (`run_local.sh`; the cluster could not start it before 10-06 09:20),
~315 steps/s per run, 2.6-2.7 h each. Regime B2 (rq2-twin-fleet): 6 h windows, regime blocks normal / surge / 2-AGV
short-fleet, c ~ U[1.75, 2.5], failures off; tardiness reward, obs v3, H15 heads, PopArt, 3,000 s discount horizon.
Learning curves (`analyze.py`): flat within instance noise; explained variance 0.99-1.00; entropy 0.9 -> 0.27-0.49.
**Held-out result and diagnosis: `../eval-due-twin/README.md`** (11-17% worse than MDD-TECT; rule mixing). Superseded
by train-due-twin-slot.
