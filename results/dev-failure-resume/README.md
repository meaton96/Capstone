# dev-failure-resume: preempt-resume for operations interrupted by a machine failure (2026-10-28)

Status: **PASS** (lab `dev-failure-resume`, 10-28 16:34-22:10). Landed in the main checkout 10-28 evening (applied
from worktree branch `failure-resume`, uncommitted on main df03817d); player installed in `linux_server/`
(Simulation.dll `22891261...`, `linux_server/BUILD_NOTES.md`) and synced, glibc-patched, to the cluster with `env/`.
Reruns: `rq4-mfail-ccal-resume`, `agv-study-fail-resume` (`docs/Plans/FAILURE_RESUME_RERUNS_2026-10-28.md`).

## Why

The user (10-28): resume the interrupted operation instead of restarting it. Until now a machine failure sent the job
in process back to routing and its operation started again from zero, possibly on another machine, and the elapsed
work was lost. The thesis listed this as a limitation (`Future_Work.tex`, `Platform.tex` §domain-machines,
`SCENARIO_GENERATORS.md` §10 item 5). The standard DES treatment of a breakdown is preempt-resume: the part stays on
the machine, processing is suspended for the repair and the remaining time is completed afterwards (the default
"preempt" failure rule in Arena / Simio; Pinedo's preemptive-resume breakdowns).

## What changed

Model (both simulators, default since 2026-10-28):
- The job in process stays on the failed machine, in state Processing, with its remaining time frozen. It is not
  re-routed. After the repair it finishes the rest; the machine takes no new job until then.
- Unity stops the processing countdown from the tick the failure fires in to the repair harvest, so the finish (and
  almost-done) tick moves back by exactly R - F + 1 ticks (F, R = failure and repair harvest ticks).
- The work done before the failure is busy time (utilization); `lost_work` is 0. The held job's pre-dispatched AGV,
  if any, is released by failure step 5 as before, and the almost-done flag is re-armed so it fires again after the
  repair.
- Everything else is unchanged: queued jobs are re-routed off the failed machine, pickups cancelled, transits
  redirected / held, deferral, calendar-time aging, as-good-as-new repairs.
- `-failurerestart` (player) / `TwinConfig.failure_restart` (twin) restore the old model, to reproduce results made
  before 10-28.

Code:
| file | change |
|---|---|
| `Machines/PhysicalMachine.cs` | `AcknowledgeFailure(keepActiveJob)`: keeps the job and `remainingTime`, re-arms almost-done; `AcknowledgeRepairComplete` resumes the progress bar |
| `FailureCoordinator.cs` | `RestartOnFailure` (`-failurerestart`); step 1 holds the job; busy time closed at the failure and reopened at the repair; `repair_done` rows carry the resumed job |
| `HeadlessBatchRunner.cs`, `Types/EpisodeRecord.cs` | the new flag in the known flags; record comment |
| `env/des_twin/engine.py` | `TwinConfig.failure_restart`; `MachineRT.susp` (stopped tick, finish, almost-done) set in `_fail` and shifted in `_repair_done`; `remaining_proc` frozen while down; op rows get `paused` seconds; `repair_done` fail rows carry the resumed job |
| `env/des_twin/docking.py` | `dock(..., failure_restart=)` |
| `env/des_twin/rules.py` | comment: the RECT / RTECT risk term `p / 2 + R` models the old restart |
| `env/tests/test_twin_failures.py` | invariants test in both modes; new exact test (finish moves back by R - F + 1); utilization / lost work in both modes |

## Evidence so far

- **Twin tests:** `test_twin_failures.py` 44 passed (new: `test_resume_moves_the_finish_back_by_the_stopped_ticks` at
  3 failure positions, and both modes of the invariants and utilization tests). Rest of `env/tests`: 399 passed.
- **Old model reproduced exactly:** the new engine with `failure_restart=True` against main's engine (df03817d) on
  12 heavy-failure runs (seeds 0-2 x SPT_ECT / MDD_TECT / SRT_SRWT / MDD_PTECT, Weibull λ 1,500 s): summary,
  decisions, op rows, fail rows, fail log, AGV trace and job rows all identical.
- **Failures off:** no code path changes (the empty-trace test passes; `remaining_proc` and op rows only differ when a
  trace is set).
- **Size of the change** on those 12 heavy-failure runs: resume cuts mean flow time 5-38% (seed 0: −35%, seed 1:
  −5..−19%, seed 2: −36..−38%) with the same jobs finished; restart had lost 7,000-37,000 s of work per run.
- **Player:** `CapstoneBuild-resume/linux_server_dev`, "[LinuxServerBuild] Succeeded (0 errors)", 0 "is missing";
  Simulation.dll md5 `22891261e82b72d3bb3dc0fbc3764e38` (current `linux_server/` is `5e6e0f87`). Built from a copy of
  the worktree's Capstone/ (`build.sh`). No C# changed on main since the current player (`git diff 71e34bdb HEAD --
  Capstone/Assets` is empty), so this build = the current player + this change.

## Unity parity: PASS (`cells.sh`, `analyze.sh` → `analysis.out`, `parity_exact.csv`; lab `dev-failure-resume`)

On dev-csharp-failures' scenarios (layout D, release, lane parking, injected `failureSchedule`), 38 runs in `runs/`
(`build.sh` and `cells.sh` still point at the worktree, where they ran):
- **24/24 resume cells exact** (1 AGV, 8 schedules x SPT_ECT / MDD_TECT / PTWINQ_SRWT): the twin matches Unity event
  for event (`failparity.py`: every event, decision and failure row, 37-85 failures per run; summary metrics within
  float noise, |tardiness| ≤ 0.03%).
- **3/3 `-failurerestart` cells exact** against the twin's `failure_restart`, and all 10 output files of each are
  identical to the pyrules player's run of the same cell (`runs/base/`).
- **Regression, 7 AGVs, Unity's native samplers:** fail_s0 / fail_s1 with `-failurerestart` and failure-free nofail_s0
  on the new player: all 10 output files identical to the pyrules player.
- **The change at 7 AGVs** (native samplers, restart → resume): fail_s0 SPT_ECT makespan 7,438 → 6,322 s, mean flow
  2,599 → 2,539 s, mean tardiness 845 → 775 s, 40 jobs both; fail_s1 MDD_TECT mean flow 1,009 → 1,022 s, tardiness
  0 → 2 s (one seed; resume changes which machine is free when, so a single run can go either way).
