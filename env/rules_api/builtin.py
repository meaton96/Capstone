"""Reference ports of the built-in dispatching rules to the rules API (rules_api, 2026-10-11).

Each function reproduces its counterpart in env/des_twin/rules.py (rank_jobs / _rank_due / select_machine /
ptime_scores) and Capstone/.../DispatchingEngine.cs (RankJobs / SelectMachine / PtimeScores) decision for decision:
the same quantities, the same float32 / float64 arithmetic, the same tie-breaking (registry.pick_job / pick_machine).
tests/test_rules_api.py and results/dev-pyrules check it on whole twin episodes; edit them only together with those.

Job rules (lower first unless marked max):
  SPT p | LPT p (max) | SRT w | LRT w (max) | FIFO time in queue (max) | PTWINQ p + WINQ
  EDD d | SLACK d - t - w | CR (d - t) / w | MDD max(d, t + w) | MOD max(d_op, t + p) | ATC (1/p) exp(-max(0, d-t-w) /
  (k p_mean)) (max)        p, w, d, d_op, t: JobView.p, remaining_work, due, op_due, ctx.now
Machine rules (lower first):
  SMPT p | SRWT q | MMUR u | ECT q + p | TECT max(tr, q) + p (+ lambda tr) | PECT q/A + p + alpha g(p) |
  PTECT max(tr, q/A) + p + alpha g(p)       p, q, u, tr: MachineView.ptime, load, util, travel; lambda: ctx.travel_price
SLACK, CR and MOD exist only in the twin (Unity has no SLACK / CR rule; it has no operation due dates, so MOD sees +inf).
The twin's reliability halves (RECT, RTECT, RECTA, RTECTA, HAECT) are not ported: they need the failure model's
per-machine terms, which are not part of the view.
"""
import math

import numpy as np

from .registry import rule

f32 = np.float32
F32_MAX = float(np.finfo(np.float32).max)
ATC_K = 2.0            # = des_twin.rules.ATC_K = DispatchingEngine.AtcK
PTIME_SHAPES = ("lin", "quad", "ratio", "exp")


# ── job rules ────────────────────────────────────────────────────────────────────────────────────────────────────────

@rule("job", "SPT")
def spt(job, ctx):
    """Shortest processing time."""
    return job.p


@rule("job", "LPT", maximize=True)
def lpt(job, ctx):
    """Longest processing time."""
    return job.p


@rule("job", "SRT")
def srt(job, ctx):
    """Shortest remaining work (sum of the remaining operations' fastest times)."""
    return job.remaining_work


@rule("job", "LRT", maximize=True)
def lrt(job, ctx):
    """Longest remaining work."""
    return job.remaining_work


@rule("job", "FIFO", maximize=True)
def fifo(job, ctx):
    """Longest in its current queue."""
    return job.time_in_queue


@rule("job", "PTWINQ")
def ptwinq(job, ctx):
    """Processing time plus the least load among the next operation's machines."""
    return f32(job.p + job.winq)


@rule("job", "EDD")
def edd(job, ctx):
    """Earliest due date (+inf without one)."""
    return job.due


@rule("job", "SLACK")
def slack(job, ctx):
    """Minimum slack d - t - w (twin only)."""
    return job.due - ctx.now - float(job.remaining_work)


@rule("job", "CR")
def cr(job, ctx):
    """Critical ratio (d - t) / w (twin only)."""
    return (job.due - ctx.now) / max(float(job.remaining_work), 1e-6)


@rule("job", "MDD")
def mdd(job, ctx):
    """Modified due date max(d, t + w)."""
    return max(job.due, ctx.now + float(job.remaining_work))


@rule("job", "MOD")
def mod(job, ctx):
    """Modified operation due date max(d_op, t + p) (operation due dates: twin only)."""
    return max(job.op_due, ctx.now + float(job.p))


@rule("job", "ATC", maximize=True)
def atc(job, ctx):
    """Apparent tardiness cost (1/p) exp(-max(0, d - t - w) / (k p_mean)), k = 2."""
    scale = ATC_K * max(ctx.p_mean, 1e-6)
    p = max(float(job.p), 1e-6)
    slack_ = job.due - ctx.now - float(job.remaining_work)
    return math.exp(-max(0.0, slack_) / scale) / p


# ── machine rules ────────────────────────────────────────────────────────────────────────────────────────────────────

@rule("machine", "SMPT")
def smpt(m, job, ctx):
    """Shortest processing time on the machine."""
    return m.ptime


@rule("machine", "SRWT")
def srwt(m, job, ctx):
    """Least queued work (the load)."""
    return m.load


@rule("machine", "MMUR")
def mmur(m, job, ctx):
    """Least utilized so far."""
    return m.util


@rule("machine", "ECT")
def ect(m, job, ctx):
    """Earliest completion: load + processing time."""
    return f32(m.load + m.ptime)


@rule("machine", "TECT")
def tect(m, job, ctx):
    """Travel-aware earliest completion: max(travel, load) + processing time (+ travel price x travel)."""
    s = f32(max(m.travel, m.load) + m.ptime)
    if ctx.travel_price > 0.0 and float(m.travel) < F32_MAX:
        s = f32(s + f32(f32(ctx.travel_price) * m.travel))
    return s


def ptime_score(m, job, ctx, travel):
    """PECT (travel False) / PTECT (True) score of candidate m, as des_twin.rules.ptime_scores: float64 arithmetic
    over the float32 inputs, a float32 score; p_min / p_mean over the job's candidates (cached per decision)."""
    pt = ctx.pt
    key = ("ptime_stats", job.id)
    stats = ctx.cache.get(key)
    if stats is None:
        ps = [float(c.ptime) for c in job.machines]
        stats = ctx.cache[key] = (max(min(ps), 1e-6), max(sum(ps) / len(ps), 1e-6))
    p_min, p_mean = stats
    shape = pt["shape"]
    p, q = float(m.ptime), float(m.load)
    wait = q / pt["avail"]
    if travel:
        wait = max(float(m.travel), wait)
    if shape == "lin":
        g = p
    elif shape == "quad":
        g = p * p / p_mean
    elif shape == "ratio":
        g = p * (p / p_min - 1.0)
    elif shape == "exp":
        g = (1.0 - math.exp(-p / pt["tau"])) * (p / 2.0 + pt["reroute"])
    else:
        raise ValueError(f"processing-time shape must be one of {PTIME_SHAPES}, not {shape!r}")
    return f32(wait + p + pt["alpha"] * g)


@rule("machine", "PECT")
def pect(m, job, ctx):
    """Processing-time-weighted ECT: load / A + p + alpha g(p)."""
    return ptime_score(m, job, ctx, False)


@rule("machine", "PTECT")
def ptect(m, job, ctx):
    """Processing-time-weighted TECT: max(travel, load / A) + p + alpha g(p)."""
    return ptime_score(m, job, ctx, True)
