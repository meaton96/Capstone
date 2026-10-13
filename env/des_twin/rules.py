"""Dispatching rules, ported from DispatchingEngine.cs.

A rule is "<JOB>_<MACHINE>": the job half ranks jobs (a machine's queue, or the pool of jobs waiting to be
routed) and the machine half picks the machine a routed job goes to. Scores are float32 and ties go to the
first candidate, as in the C# ArgMin/ArgMax, so equal-score cases resolve the same way.

Due-date job rules: EDD, MDD and ATC are ported to DispatchingEngine.cs (2026-10-03, same formulas, double scores);
SLACK, CR and MOD exist only here (rq2-twin-due screen). Due dates come from the scenario's per-job "dueDate"
(resolve_jobs) or scenario.assign_due_dates. A job without one counts as due at +infinity, as JobData.DueDate in C#:
the rules rank it last.
"""
import math

import numpy as np

f32 = np.float32
JOB_RULES = ("SPT", "LPT", "SRT", "LRT", "FIFO", "PTWINQ") + ("EDD", "SLACK", "CR", "MDD", "MOD", "ATC")
DUE_DATE_RULES = ("EDD", "SLACK", "CR", "MDD", "MOD", "ATC")
ATC_K = 2.0    # ATC look-ahead parameter k (Vepsalainen & Morton 1987 report k of about 1.5-3 working well)
MACHINE_RULES = ("SMPT", "SRWT", "MMUR", "ECT", "TECT")
# Reliability-aware machine halves (2026-10-09, plan §3; twin only, not in the RL action heads or DispatchingEngine.cs).
# p = processing time on the candidate, q = its load (machine_load), tr = TECT's travel estimate, A = availability,
# MTBF / age / Weibull (k, lam) at the twin's information level (TwinConfig.reliability: static / online / oracle),
# kappa = TwinConfig.reliability_kappa, R = TwinConfig.reroute_cost:
#   RECT    q / A + p + kappa P (p / 2 + R),  P = 1 - exp(-p / MTBF): the chance the operation is interrupted (a
#           restart loses about p / 2 and the job is re-routed); q / A: the queue drains at the long-run availability
#   RTECT   max(tr, q / A) + p + kappa P (p / 2 + R)
#   RECTA   RECT with P the Weibull conditional failure probability over the job's exposure window given the age a:
#           1 - exp(-((a + q + p) / lam)^k + ((a + q) / lam)^k)
#   RTECTA  RTECT with RECTA's P
#   HAECT   hard-avoid control: ECT over the candidates with A >= the floor median (all candidates if none qualify)
RELIABILITY_RULES = ("RECT", "RTECT", "RECTA", "RTECTA", "HAECT")
# Processing-time-weighted machine halves (2026-10-10, dev-pect-rule; ported to DispatchingEngine.cs PtimeScores and
# made the RL machine head, action schema v5, in dev-pect-unity the same day: keep the two in step; defaults quad,
# alpha 3 = TwinConfig = Unity PtimeRuleParams = config.PTIME_RULE_PARAMS). dev-b2-pect-check / dev-hetero-tune found that RECT / RTECT with no reliability information
# and no failures beat ECT / TECT by 15-25%: their risk term is a convex penalty on the candidate's processing time p.
# These are that weighting as a rule of its own, with the shape g chosen by TwinConfig.ptime_shape:
#   PECT    q / A + p + alpha g(p)
#   PTECT   max(tr, q / A) + p + alpha g(p)
# q, p, tr as ECT / TECT; alpha = TwinConfig.ptime_alpha; A = TwinConfig.ptime_avail (default 1: the plain queue; the
# reliability rules' blind form uses the floor availability, so a value < 1 reproduces them exactly). Shapes, over the
# routed job's eligible candidates (p_min the fastest, p_mean their mean):
#   lin     g = p                                   (score = wait + (1 + alpha) p; p - p_min ranks identically)
#   quad    g = p^2 / p_mean                        (scale-free: alpha is dimensionless)
#   ratio   g = p (p / p_min - 1)                   (scale-free, zero on the fastest candidate)
#   exp     g = (1 - exp(-p / tau)) (p / 2 + R)     (the reliability rules' risk term at a common MTBF tau and
#                                                    re-route cost R: TwinConfig.ptime_tau / ptime_reroute, seconds)
PTIME_RULES = ("PECT", "PTECT")
PTIME_SHAPES = ("lin", "quad", "ratio", "exp")


def parse_rule(name):
    parts = name.upper().split("_")
    if len(parts) != 2 or parts[0] not in JOB_RULES or parts[1] not in MACHINE_RULES + RELIABILITY_RULES + PTIME_RULES:
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
    """Job j's due date; +infinity when it has none (JobData.DueDate)."""
    due = sim.jobs[j].due
    return math.inf if due is None else due


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
            op_due = math.inf if job.op_due is None else job.op_due[job.cur_op]
            return max(op_due, t + float(proc_time(j)))
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


def _interrupt_prob(rule, p, q, term):
    a, mtbf, _, k, lam, age = term
    if not math.isfinite(mtbf):
        return 0.0
    if rule in ("RECTA", "RTECTA"):
        age = age or 0.0
        return 1.0 - math.exp(-(((age + q + p) / lam) ** k - ((age + q) / lam) ** k))
    return 1.0 - math.exp(-p / mtbf)


def reliability_scores(machine_rule, job_times, loads, travel, rel):
    """Scores (lower first) of a reliability-aware machine rule; rel = Twin.reliability_terms(cands)."""
    kappa, reroute = rel["kappa"], rel["reroute"]
    out = []
    for p, q, tr, term in zip(job_times, loads, travel, rel["terms"]):
        p, q, tr = float(p), float(q), float(tr)
        a = max(term[0], 1e-6)
        if machine_rule == "HAECT":
            out.append(f32(q + p) if term[0] >= rel["median_A"] else math.inf)
            continue
        wait = q / a
        if machine_rule in ("RTECT", "RTECTA"):
            wait = max(tr, wait)
        risk = _interrupt_prob(machine_rule, p, q, term) * (p / 2.0 + reroute)
        out.append(f32(wait + p + kappa * risk))
    if machine_rule == "HAECT" and all(math.isinf(s) for s in out):
        out = [f32(float(q) + float(p)) for p, q in zip(job_times, loads)]
    return out


def ptime_scores(machine_rule, job_times, loads, travel, pt):
    """Scores (lower first) of a processing-time-weighted machine rule (PTIME_RULES); pt = Twin.ptime_params():
    {"alpha", "shape", "tau", "reroute", "avail"}. Float64 arithmetic, float32 score, as the reliability rules."""
    alpha, shape, avail = pt["alpha"], pt["shape"], pt["avail"]
    if shape not in PTIME_SHAPES:
        raise ValueError(f"processing-time shape must be one of {PTIME_SHAPES}, not {shape!r}")
    ps = [float(p) for p in job_times]
    p_min = max(min(ps), 1e-6)
    p_mean = max(sum(ps) / len(ps), 1e-6)
    tau, reroute = pt["tau"], pt["reroute"]
    out = []
    for p, q, tr in zip(ps, loads, travel):
        q, tr = float(q), float(tr)
        wait = q / avail
        if machine_rule == "PTECT":
            wait = max(tr, wait)
        if shape == "lin":
            g = p
        elif shape == "quad":
            g = p * p / p_mean
        elif shape == "ratio":
            g = p * (p / p_min - 1.0)
        else:
            g = (1.0 - math.exp(-p / tau)) * (p / 2.0 + reroute)
        out.append(f32(wait + p + alpha * g))
    return out


def select_machine(machine_rule, cands, job_times, loads, utils, travel, rel=None, pt=None):
    """DispatchingEngine.SelectMachine over parallel candidate lists (float32). The reliability-aware halves
    (RELIABILITY_RULES, twin only) also need rel = Twin.reliability_terms(cands); the processing-time-weighted
    halves (PTIME_RULES; also in Unity) pt = Twin.ptime_params()."""
    if len(cands) == 1:
        return cands[0]
    if machine_rule in PTIME_RULES:
        if pt is None:
            raise ValueError(f"machine rule {machine_rule} needs its parameters (pt=Twin.ptime_params())")
        score = ptime_scores(machine_rule, job_times, loads, travel, pt)
    elif machine_rule in RELIABILITY_RULES:
        if rel is None:
            raise ValueError(f"machine rule {machine_rule} needs reliability terms (rel=Twin.reliability_terms)")
        score = reliability_scores(machine_rule, job_times, loads, travel, rel)
    elif machine_rule == "SMPT":
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
