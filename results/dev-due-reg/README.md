# dev-due-reg: the due-date build reproduces the previous build without due dates (2026-10-03)

**Passed.** `linux_server_dev/` (due-date port, 10-03 01:58) against `linux_server/` (3d22f9c7, 00:47).
- **Batch (C# only):** rnd_load_s0 / s1 (machine failures) / s2 x SRT_TECT, SPT_ECT, FIFO_ECT, PTWINQ_SRWT, D, 7 AGVs,
  releasePrevious, lane, full drain: 12 / 12 cells equal on every results.csv and job_completions.csv field (instance
  hashes included) except the timestamp and the new tardiness columns. `compare_batch.py`; raw cells in each player's
  `Results/dev-due-reg/`.
- **Python path:** evaluate.py, SRT-TECT / SPT-ECT / FIFO-ECT / PTWINQ-SRWT x seeds 0-1, random warm-up, 5,400 s: old
  `env/` (`git archive HEAD`, run with `python -S` so the editable install cannot redirect imports) + old player vs new
  `env/` + new player, 8 / 8 episodes equal on every shared column (`py_old/`, `py_new/`).
- The `*brokenbuild*.out` logs are from a first 01:51 build that copied the Editor's Library and linked no scripts;
  it was replaced (`linux_server_dev/BUILD_NOTES.md`).
