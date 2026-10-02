"""Dispatching rules, ported from DispatchingEngine.cs.

A rule is "<JOB>_<MACHINE>": the job half ranks jobs (a machine's queue, or the pool of jobs waiting to be
routed) and the machine half picks the machine a routed job goes to. Scores are float32 and ties go to the
first candidate, as in the C# ArgMin/ArgMax, so equal-score cases resolve the same way.
"""
import numpy as np

f32 = np.float32
JOB_RULES = ("SPT", "LPT", "SRT", "LRT", "FIFO", "PTWINQ")
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
