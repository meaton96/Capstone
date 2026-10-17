"""
@file future_table.py
@brief Future job table (2026-10-15, rq2-oracle-bc-future): the next arriving jobs as a per-job table, one row per job,
       like the live job table. Twin only: the twin knows the scenario's future, the player does not.

@details des_twin.lookahead summarizes the future in 15 numbers (counts, loads and slack per 30 / 60 / 120 min window).
eval-v5-lookahead-fcast (10-15) found that a PPO policy given those features does not use them. This table keeps the
schedule itself: what arrives, when, with which operations and due date, so a network can learn its own summary.

Rows: the next @p rows jobs arriving in [now, now + horizon), in arrival order, read the way des_twin.lookahead reads
them: through `twin.unc.plan_future_jobs(now, arrived)` when the twin has an uncertainty layer (the published plan,
not the world's future; results/eval-v5-lookahead-fcast's PublishedFuture adapter feeds perturbed schedules through
the same path), else the twin's pending arrivals. A job only needs `arrival`, `ops` (per operation {machine id:
time}), `types` and `due` (None without due dates), so the uncertainty layer's light plan jobs work too.

Columns (FUTURE_JOB_FEATURES = 20, float32, squashes as des_twin.observation's; work = an operation's mean time over
its eligible machines, min work = its shortest time, the live table's remaining-work convention):
  0  present (1 real row, 0 padding)
  1  arrival offset from now / horizon (clamped to [0, 1])
  2  squash(arrival offset, 300 s)
  3  squash(rank of the job among the future jobs, 32) (the mean over present rows gives the count in the table;
     independent of the row cap, so the first rows of a larger table equal a smaller table)
  4  squash(cumulative work of the jobs up to this one / machines, 1,800 s) (the max over rows gives the offered load)
  5  squash(operations, 3)
  6  squash(work, 1,200 s)
  7  squash(min work, 1,200 s)
  8  squash(first operation's shortest time, 300 s)
  9  squash(first operation's eligible machines, 5)
  10-14  squash(work of the job's operations of each machine type, 600 s), types in observation.TYPE_INDEX order
  15 has a due date
  16 signed_squash(due - arrival - min work, 1,200 s): slack at arrival
  17 signed_squash(due - now, 1,200 s)
  18 signed_squash((due - arrival - min work) / min work, 2): slack relative to the work
  19 squash(mean eligible machines per operation, 5)
"""
import numpy as np

from .observation import AGE_SCALE, COUNT_SCALE, OPS_SCALE, TIME_SCALE, TYPE_INDEX, signed_squash, squash

FUTURE_JOB_FEATURES = 20
FUTURE_PRESENT_COL = 0
FUTURE_ROWS = 64
FUTURE_HORIZON = 7200.0
CUM_WORK_SCALE = 1800.0
TYPE_WORK_SCALE = 600.0
SLACK_RATIO_SCALE = 2.0
RANK_SCALE = 32.0


def future_jobs(twin, horizon: float = FUTURE_HORIZON):
    """@brief The jobs arriving in [now, now + horizon), in arrival order: the published plan when the twin has an
    uncertainty layer (twin.unc.plan_future_jobs, as des_twin.lookahead), else the twin's pending arrivals."""
    now = twin.now
    if getattr(twin, "unc", None) is not None:
        pending = twin.unc.plan_future_jobs(now, {j.id for j in twin.order})
    else:
        pending = twin.pending[twin.pending_i:]
    out = []
    for job in pending:
        if job.arrival >= now + horizon:
            break
        if job.arrival >= now:
            out.append(job)
    return out


def future_job_table(twin, rows: int = FUTURE_ROWS, horizon: float = FUTURE_HORIZON) -> np.ndarray:
    """@brief The (rows, FUTURE_JOB_FEATURES) table at the twin's current time (see the module docstring)."""
    t = np.zeros((int(rows), FUTURE_JOB_FEATURES), dtype=np.float32)
    now = float(twin.now)
    n_machines = max(len(twin.floor.machines), 1)
    cum = 0.0
    for i, job in enumerate(future_jobs(twin, horizon)[:rows]):
        r = t[i]
        offset = float(job.arrival) - now
        means = [float(np.mean([float(v) for v in op.values()])) for op in job.ops]
        mins = [float(min(op.values())) for op in job.ops]
        work, min_work = sum(means), sum(mins)
        cum += work
        r[0] = 1.0
        r[1] = min(max(offset / horizon, 0.0), 1.0)
        r[2] = squash(offset, TIME_SCALE)
        r[3] = squash(i, RANK_SCALE)
        r[4] = squash(cum / n_machines, CUM_WORK_SCALE)
        r[5] = squash(len(job.ops), OPS_SCALE)
        r[6] = squash(work, AGE_SCALE)
        r[7] = squash(min_work, AGE_SCALE)
        if job.ops:
            r[8] = squash(mins[0], TIME_SCALE)
            r[9] = squash(len(job.ops[0]), COUNT_SCALE)
            r[19] = squash(float(np.mean([len(op) for op in job.ops])), COUNT_SCALE)
        for w, typ in zip(means, job.types):
            k = TYPE_INDEX.get(typ)
            if k is not None:
                r[10 + k] += w
        for k in range(len(TYPE_INDEX)):
            r[10 + k] = squash(r[10 + k], TYPE_WORK_SCALE)
        if job.due is not None:
            due = float(job.due)
            r[15] = 1.0
            r[16] = signed_squash(due - float(job.arrival) - min_work, AGE_SCALE)
            r[17] = signed_squash(due - now, AGE_SCALE)
            if min_work > 0:
                r[18] = signed_squash((due - float(job.arrival) - min_work) / min_work, SLACK_RATIO_SCALE)
    return t
