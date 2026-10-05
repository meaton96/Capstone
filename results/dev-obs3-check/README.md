# dev-obs3-check: observation schema v3 / action schema v4 build (2026-10-03)

Build `linux_server_dev/` 10-03 22:24 (`BUILD_NOTES.md`): job columns 17-20 (has due date, signed-squashed slack and
time to due, late flag), global scalars 16-17 (share late / negative slack), job head SRT, SPT, MDD, EDD, ATC.
- Smoke: 600 s, 2 pairs, scripts linked, evaluate.py accepts the 17,928-float sensor.
- Twin vs Unity observation parity (`parity_*`, 1 AGV, due2_s0, 400 decisions): MDD-ECT, ATC-TECT, SRT-SRWT PASSED
  (masks identical; every column within 1e-5 except the two by-design ones); the new columns are non-zero over a full
  episode (up to ~0.96 in magnitude). Explanation of the observation: `docs/experiments/FINDINGS_2026-10-04_05.md`.
