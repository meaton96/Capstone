# dev-fleet-check: Unity fleet schedule (agvSchedule) (2026-10-04)

Build `linux_server_v4/` 10-04 18:32 (`BUILD_NOTES.md`): scenario "agvSchedule" [{"start", "agvCount"}]; AGVs above the
active count take no new task (AGVPool.IsOffDuty -> AGVController.IsOutOfService), finish their task and park; global
scalar 13 = AGVs on duty per machine.
- `sched1of2`: twin parity with 2 AGVs / 1 on duty, 300 decisions, PASSED (decisions and times identical, so the
  off-duty AGV never moves). It found one twin mismatch, fixed: event flag 2 now excludes off-duty AGVs as
  AGVPool.GetIdleAGV does.
- `noschedule_agv1`: parity at 1 AGV PASSED (no regression).
- `repro_*`: without a schedule v4 equals linux_server_dev on 3 pairs x seed 3 (every episodes.csv column and the
  config hash).
