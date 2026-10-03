"""Dispatching rules, ported from DispatchingEngine.cs.

A rule is "<JOB>_<MACHINE>": the job half ranks jobs (a machine's queue, or the pool of jobs waiting to be
routed) and the machine half picks the machine a routed job goes to. Scores are float32 and ties go to the
first candidate, as in the C# ArgMin/ArgMax, so equal-score cases resolve the same way.

Due-date job rules (EDD, SLACK, CR, MDD, MOD, ATC) exist only here, for the rq2-twin-due screen (2026-10-02); they
are not in DispatchingEngine.cs yet. They need due dates on the jobs (scenario.assign_due_dates) and raise without.
"""
import math

import numpy as np

f32 = np.float32
JOB_RULES = ("SPT", "LPT", "SRT", "LRT", "FIFO", "PTWINQ") + ("EDD", "SLACK", "CR", "MDD", "MOD", "ATC")
DUE_DATE_RULES = ("EDD", "SLACK", "CR", "MDD", "MOD", "ATC")
ATC_K = 2.0    # ATC look-ahead parameter k (Vepsalainen & Morton 1987 report k of about 1.5-3 working well)
MACHINE_RULES = ("SMPT", "SRWT", "MMUR", "ECT", "TECT")


def parse_rule(name):
    parts = name.upper().split("_")
    if len(parts) != 2 or parts[0] not in JOB_RULES or parts[1] not in MACHINE_RULES:
        raise ValueError(f"unsupported rule {name!r} (expected JOB_MACHINE, e.g. SPT_ECT; Random is not modelled)")
    return parts[0], parts[1]


def _argmin(ids, score):
    best, best_s = ids[0], f32(np.finfo(np.float32).max)
    for i in ids:
        s = score(i)
        if s < best_s:
            best, best_s = i, s
    return best


def _argmax(ids, score):
    best, best_s = ids[0], f32(np.finfo(np.float32).min)
    for i in ids:
        s = score(i)
        if s > best_s:
            best, best_s = i, s
    return best


def rank_jobs(job_rule, ids, sim, proc_time):
    """DispatchingEngine.RankJobs. proc_time(job_id) -> float32."""
    if len(ids) == 1:
        return ids[0]
    if job_rule == "SPT":
        return _argmin(ids, proc_time)
    if job_rule == "LPT":
        return _argmax(ids, proc_time)
    if job_rule == "SRT":
        return _argmin(ids, lambda j: sim.remaining_work(j))
    if job_rule == "LRT":
        return _argmax(ids, lambda j: sim.remaining_work(j))
    if job_rule == "FIFO":
        # time in the current queue: Job.since is when the job entered its state (JobData.StateEntryTime)
        return _argmax(ids, lambda j: f32(sim.now - sim.jobs[j].since))
    if job_rule == "PTWINQ":
        loads = sim.all_machine_loads()
        return _argmin(ids, lambda j: f32(proc_time(j) + sim.work_in_next_queue(j, loads)))
    if job_rule in DUE_DATE_RULES:
        return _rank_due(job_rule, ids, sim, proc_time)
    raise ValueError(job_rule)


def _due(sim, j):
    due = sim.jobs[j].due
    if due is None:
        raise ValueError(f"job {j} has no due date: due-date rules need scenario.assign_due_dates")
    return due


def _rank_due(job_rule, ids, sim, proc_time):
    """Due-date job rules. d = job due date, w = remaining work (SRT's quantity), t = now, p = proc_time (this
    machine at dispatch, the fastest eligible machine at routing, as SPT). All lower-first except ATC.
      EDD    d                                   (earliest due date)
      SLACK  d - t - w                           (minimum slack)
      CR     (d - t) / w                         (critical ratio; a late job is negative and goes first)
      MDD    max(d, t + w)                       (modified due date, Baker & Bertrand: EDD while jobs can
                                                  still finish on time, SRT once they cannot)
      MOD    max(d_op, t + p)                    (modified operation due date, Baker & Kanet; d_op from
                                                  assign_due_dates, the current operation's share of the allowance)
      ATC    (1/p) exp(-max(0, d - t - w) / (k p_mean)), highest first (apparent tardiness cost, Vepsalainen &
             Morton; slack without their waiting-time look-ahead, p_mean over the candidates, k = ATC_K)"""
    t = sim.now
    if job_rule == "EDD":
        return _argmin(ids, lambda j: _due(sim, j))
    if job_rule == "SLACK":
        return _argmin(ids, lambda j: _due(sim, j) - t - float(sim.remaining_work(j)))
    if job_rule == "CR":
        return _argmin(ids, lambda j: (_due(sim, j) - t) / max(float(sim.remaining_work(j)), 1e-6))
    if job_rule == "MDD":
        return _argmin(ids, lambda j: max(_due(sim, j), t + float(sim.remaining_work(j))))
    if job_rule == "MOD":
        def mod(j):
            job = sim.jobs[j]
            _due(sim, j)
            return max(job.op_due[job.cur_op], t + float(proc_time(j)))
        return _argmin(ids, mod)
    if job_rule == "ATC":
        p_mean = sum(float(proc_time(j)) for j in ids) / len(ids)
        scale = ATC_K * max(p_mean, 1e-6)

        def atc(j):
            p = max(float(proc_time(j)), 1e-6)
            slack = _due(sim, j) - t - float(sim.remaining_work(j))
            return math.exp(-max(0.0, slack) / scale) / p
        best, best_s = ids[0], -math.inf
        for j in ids:
            s = atc(j)
            if s > best_s:
                best, best_s = j, s
        return best
    raise ValueError(job_rule)


def select_machine(machine_rule, cands, job_times, loads, utils, travel):
    """DispatchingEngine.SelectMachine over parallel candidate lists (float32)."""
    if len(cands) == 1:
        return cands[0]
    if machine_rule == "SMPT":
        score = job_times
    elif machine_rule == "SRWT":
        score = loads
    elif machine_rule == "MMUR":
        score = utils
    elif machine_rule == "ECT":
        score = [f32(q + p) for q, p in zip(loads, job_times)]
    elif machine_rule == "TECT":
        score = [f32(max(tr, q) + p) for tr, q, p in zip(travel, loads, job_times)]
    else:
        raise ValueError(machine_rule)
    b = 0
    for i in range(1, len(score)):
        if score[i] < score[b]:
            b = i
    return cands[b]
