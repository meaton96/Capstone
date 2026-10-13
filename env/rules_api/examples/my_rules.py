"""Example rule mod for the rules API (dev-pyrules, 2026-10-11): two rules that exist in neither simulator's code.

Load it with evaluate.py --py-rules env/rules_api/examples/my_rules.py --pdr OPSLACK-PECT (or PSLACK-LECT, ...), or
in Python with rules_api.load_rules("env/rules_api/examples/my_rules.py") and rules_api.resolve("OPSLACK-LECT").

Job rule OPSLACK: slack per remaining operation, (d - t - w) / ops_left, lowest first (Kanet & Hayya's S/OPN), a due-date
rule with neither a C# nor a twin implementation. A job without a due date has +inf slack and goes last; among such
jobs the shortest remaining work goes first (so the rule degrades to SRT without due dates).

Machine rule LECT: least expected completion with a load-dependent travel blend: max(travel, load) + p + 0.5 p^2 /
p_mean + 0.1 * fleet-shortage x travel, where the shortage term charges travel more when few AGVs are on duty relative
to the WIP. It shows how a rule reads the decision context (ctx.fleet, ctx.wip) and caches a per-decision statistic.
"""
import math

from rules_api import rule


@rule("job", "OPSLACK")
def opslack(job, ctx):
    """Slack per remaining operation; SRT among jobs without a due date."""
    w = float(job.remaining_work)
    if math.isinf(job.due):
        return 1e30 + w          # after every job with a due date, shortest remaining work first
    return (job.due - ctx.now - w) / max(job.ops_left, 1)


@rule("machine", "LECT")
def lect(m, job, ctx):
    """max(travel, load) + p + 0.5 p^2 / p_mean + 0.1 x (WIP per on-duty AGV / 10) x travel."""
    key = ("lect_pmean", job.id)
    p_mean = ctx.cache.get(key)
    if p_mean is None:
        p_mean = ctx.cache[key] = max(sum(float(c.ptime) for c in job.machines) / len(job.machines), 1e-6)
    p, q, tr = float(m.ptime), float(m.load), float(m.travel)
    shortage = (ctx.wip / max(ctx.fleet, 1)) / 10.0
    return max(tr, q) + p + 0.5 * p * p / p_mean + 0.1 * shortage * tr
