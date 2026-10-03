# rq2-twin-due: does a due-date objective give rule switching headroom? (twin screen, 2026-10-03)

**Question.** On time in system, a switching oracle beats the best fixed rule pair by about 1% (Unity
rq2-switch-oracle, oracle screen), too little to train for. Does a tardiness objective with TWK due dates, with or
without due-date job rules, give a policy several percent? Decides `docs/Plans/PDR_RULE_SET_PLAN_2026-10-02.md` §9.

**Answer: yes on the twin, at one allowance and only with due-date rules in the job head.** Training-like windows
(random warm-up), 40 instances, due dates at c = 2 (about half the jobs late). Oracle against the strongest fixed pair
of all 30, MDD-TECT:

| action set | oracle gain, mean | median [90% CI] | seeds above 2% |
|---|---|---|---|
| time in system, current 12 pairs (vs SRT-TECT) | 1.0% | 0.5% [0.3, 0.7] | 25% |
| tardiness, current 12 pairs | 2.6% | 1.1% | 48% |
| tardiness, all 30 pairs (current + EDD, SLACK, CR, MDD, MOD, ATC) | 10.8% | 6.7% [5.3, 8.2] | 75% |
| tardiness, candidate head H15: job {SRT, SPT, MDD, EDD, ATC} x machine {ECT, TECT, SRWT} | 9.4% | 6.5% [4.9, 7.6] | 75% |
| tardiness, H10: same job rules x {ECT, TECT} | 7.6% | 5.3% [2.5, 6.3] | 65% |

(CI: seed bootstrap of the median, best pair held fixed; the time-in-system row is against its own best pair.)

1. **The due-date rules make the headroom, not the objective alone.** Switching the current 12 pairs on tardiness
   gains a 1.1% median over MDD-TECT. The strongest fixed rule changes with the objective: MDD-TECT has 4.4% less mean
   tardiness than SRT-TECT and wins 34 of 40 instances against it. Its time in system is 1.2% above the
   per-instance best fixed pair, against 0.7% for SRT-TECT.
2. **The candidate head keeps nearly all of it.** H15 reaches 6.5% of the full catalog's 6.7% median. Its oracle uses
   MDD 35%, SRT 34%, ATC 25% of the 900 s slots and EDD only 4% (EDD is the drop candidate). Dropping SRWT (H10) costs
   about a point.
3. **Choice depends on load, with flow rules in lulls and due-date rules under load.** Lull slots: SPT 28%, SRT 26%,
   EDD 15%. Overload slots: MDD 31%, SRT 31%, ATC 22%. This is the direction the literature reports (due-date rules
   help below saturation, Holthaus & Rajendran 1997), but the split was different in the 20-seed pre-change set
   (SRT held overload there), so treat it as indicative.
4. **It costs no flow time.** The tardiness-oracle schedules have 0.7% *less* time in system than the best fixed
   flow pair.

## Caveats (read before citing a number)

- **Same absolute gain, smaller base.** Per 90-minute window the oracle saves about 1,300-1,400 job-seconds of
  tardiness, against 1,250 job-seconds of time in system on the flow objective. The percentage is about ten times
  larger because tardiness (25,000 job-s per window at c = 2) is the part of flow time above the allowance
  (132,000 job-s). Tardiness is a standard objective in its own right (Luo 2020, Hou et al. 2024 report it), but
  report the absolute figure next to the percentage.
- **Only c of about 2 works.** c = 1.5 (75% late) roughly halves it: H15 median 3.6%, 62.5% of seeds above 2%.
  At c >= 3 tardiness is sparse (4 of 40 windows have none at c = 3, 15 of 40 at c = 4, 32 of 40 at c = 6): large
  percentages on near-zero values, and a poor reward signal.
- **Instances differ a lot.** Seeds 0-19 vs 20-39: H15 median 2.5% vs 7.6%, share above 2% 55% vs 95%. The first
  20-seed run sat on the go threshold for this reason; 40 seeds is the estimate to use.
- **Twin, not Unity.** No machine failures (the twin does not model them; the training mix has them in 2/3 of
  episodes) and no zone blocking. On time in system the twin's fixed pairs land within 0.5-1.7% of pre-change Unity
  runs and pick the same best pair on 4 of 5 instances; its oracle gain was up to 0.8 points higher than Unity's on 2
  of 6 instances. A Unity oracle check with failures is required before training.
- **Due-date rules are twin-only.** `env/des_twin/rules.py` (committed in 3d22f9c7); not in `DispatchingEngine.cs`.
- **Greedy oracle, 6 x 900 s,** perfect foresight within each stage: an achievable schedule, not a bound in
  either direction for a learned policy.

## AGV-selection change during the first launch

The first launch (10-02 23:48) ran the main grid on the twin's old AGV selection (any idle AGV before any returning
one). Between launches, `eb61a6c2` (10-03 00:36 on disk, 00:47 committed) changed both Unity's `AGVPool` and the twin
to pick the nearest of idle and returning AGVs, so the 80 c = 1.5 tasks of 00:40 used the new one. Every run changes
under the switch (time in system about 1.5% per run, tardiness 4-6%), so that set mixed versions.
- `tasks_oldsel/` (360 tasks, old selection, seeds 0-19) is kept; it is the set like-for-like with the pre-change
  Unity runs. At c = 2 it gave the 30-pair oracle 7.6% mean, 3.7% median, 60% of seeds above 2%
  (`analysis_oldsel/`).
- `tasks_newsel_unstamped/` (the 80 c = 1.5 tasks) equals the rerun's c = 1.5 files exactly.
- `tasks/` is the full rerun on the current twin (code stamp `1d5d59fd5c2c` at 3d22f9c7 in every file; the runner
  refuses to add tasks from other code), extended to seeds 20-39 because the 20-seed result sat on the threshold.

## Setup

- **Question:** the switching oracle finds about 1% on time in system with the current 12 pairs. Does a due-date
  objective (tardiness), with or without due-date job rules, give a policy the several percent a retrain needs?
  Plan: `docs/Plans/PDR_RULE_SET_PLAN_2026-10-02.md` §9 (the user asked for this twin test 10-02 23:30).
- **Simulator:** the event-based twin `env/des_twin` (DES-1k: Unity's AGV motion, routes, docks and handshakes, no
  zone reservations), kinematic transport, layout D, 7 AGVs, onTransport, floor exported from Unity
  (`linux_server_des/Results/G1/rnd_load_s0/D/agv7_SPT_ECT_s0/des_floor.json`). An episode takes 0.3-0.5 s.
- **Instances:** randomized generator (rnd_load defaults) with machine failures off, seeds 0-19 (the held-out seeds;
  the jobs are identical to the default instances, only the failure block is dropped). 5,400 s agent window.
  - `warm`: random warm-up, i.e. the training distribution (= the oracle screen's mf-off regime).
  - `t0`: from t = 0 (= rq2-switch-oracle's setting).
- **Due dates:** total work content (Blackstone et al. 1982), d_i = r_i + c x TWK_i, TWK_i = the job's remaining work
  at arrival (fastest eligible machine per op), operation due dates split by work for MOD
  (`des_twin.scenario.assign_due_dates`). Allowance c in {1.5, 2, 3, 4, 6}, calibrated as Sels et al. 2012 do by the
  share of jobs late under a flow rule: under SRT-ECT on warm windows about 77 / 46 / 15 / 5 / 0% (flow / TWK median
  1.9).
- **Due-date job rules (twin only, `des_twin/rules.py`):** EDD; SLACK (d - t - w); CR ((d - t) / w); MDD
  (max(d, t + w), Baker & Bertrand 1982); MOD (max(d_op, t + p), Baker & Kanet 1983); ATC ((1/p) exp(-max(0, d - t
  - w) / (k p_mean)), k = 2, Vepsalainen & Morton 1987, without their waiting-time look-ahead). w = remaining work.
- **Objectives** (window integrals over [first agent decision, end of window], units of 1000 job-seconds):
  - tis: time in system of every job (the WIP integral, what the flow_time reward sums);
  - tard: tardiness of every job, finished or not (the late-WIP integral: open jobs past due, integrated). This is
    what a tardiness reward would sum.
- **Oracle:** greedy, as `env/switch_oracle.py`: stage 1 runs every pair for the whole window; stage k fixes the
  pairs chosen for segments 1..k-1 and tries every pair from segment k on; 6 x 900 s. Every schedule is a real run.
- **Pair sets:** P12 = the current heads (job SPT / SRT / PTWINQ / FIFO x machine ECT / TECT / SRWT); P30 = those job
  rules plus the six due-date rules, x the same 3 machine rules; H15 = job SRT / SPT / MDD / EDD / ATC x ECT / TECT /
  SRWT; H10 = the same job rules x ECT / TECT.
- **Tasks:** per setting and seed: tis-P12 once; tard-P12 and tard-P30 at each c; H15 / H10 at c 1.5 and 2 (warm).
- **Tasks run:** warm seeds 0-39 and t0 seeds 0-19: tis-P12; tard-P12 and tard-P30 at every c; warm H15 / H10 at c 1.5
  and 2. 820 tasks, 10-03 01:00-01:33, 24 workers, no Unity.

## Files

- `run.py`: the runner (`--settings`, `--seeds`, `--extra PAIRSET:C`); `analyze.py [tasks_dir]`: all tables.
- `tasks/<setting>_s<seed>_<objective>_<pairset>_c<c>.json`: one oracle each (schedule, every candidate run, both
  objectives, code stamp).
- `analysis/`: `decision.csv` (the table above, with intervals), `headroom.csv` (every setting / set / c),
  `fixed_gaps.csv` (fixed pairs on tardiness), `calibration.csv` (vs the pre-change Unity runs), `analyze.out`.
- Logs: `run.out`, `run_heads.out`, `run_s20-39.out`, `run_heads_s20-39.out`; first launch `run_oldsel.out`,
  `run_newsel_unstamped_c1.5.out`.

## References for the method

Blackstone, Phillips & Hogg 1982 (TWK due dates; IJPR 20(1)); Baker 1984 (sequencing rules and due-date assignment in
a job shop; Management Science 30(9)); Baker & Bertrand 1982 (MDD; J. Operations Management 3); Baker & Kanet 1983
(MOD; J. Operations Management 4(1)); Vepsalainen & Morton 1987 (ATC; Management Science 33(8)); Sels et al. 2012
(calibrating c by the share late); Rajendran & Holthaus 1999 and Holthaus & Rajendran 1997 (TWK, c 3-8).
