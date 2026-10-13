# rq2-oracle-due-mix: Unity oracle on the training mix (c x load), held-out seeds 20-39 (2026-10-04)

**Setup.** `run.sh`: H15 oracle (6 x 900 s), tardiness, random warm-up, 5,400 s window, machine failures as generated,
7 AGVs, per episode c ~ U[1.75, 2.5] and load profile base / utilhi (50:50; `RandomizedParams.due_date_allowance_range`,
`load_mix`), seeds 20-39 (held out from the twin screen that chose the mix). Player `linux_server_dev/` (obs v3); a
first launch on `linux_server_due/` failed every seed on the sensor size (`run_failed_due_player.out`). Seed 32 timed
out at Unity startup under load (rerun queued in the lab). `../rq2-oracle-due/analyze.py` -> `analysis.out`.

**Result (19 seeds):** **10.50% median over the best-on-average pair** (MDD-ECT; mean 15.47%), **14 / 19 seeds above
2%** (74%); switching alone 3.04% median (mean 10.07%); 16.10% median over SRT-TECT. Oracle job rules MDD 42%, SRT
32%, ATC 17%, EDD 6%, SPT 4%; time in system unchanged (-0.02%).
- The mix raises the gain over a single fixed pair a lot (3.4% at fixed c = 2): with c and load varying, no pair is
  good everywhere (best pair MDD-ECT is 13.9% from the per-seed best on average; per-seed winners MDD-ECT 6, ATC-TECT 3).
- Switching alone (3.0%) is above c = 2's 1.8%: the mix also helps within an episode.
