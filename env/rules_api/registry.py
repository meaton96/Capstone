"""Rule registry, pick functions and rule pairs (rules_api, 2026-10-11).

A rule is a scoring function registered under a name:

    @rule("job", "SLACK2")
    def slack2(job, ctx):                    # job: JobView, ctx: DecisionView
        return (job.due - ctx.now - float(job.remaining_work)) / max(job.ops_left, 1)

    @rule("machine", "PECT2")
    def pect2(m, job, ctx):                  # m: MachineView (a candidate of `job`)
        return float(m.load) + float(m.ptime) + 2.0 * float(m.ptime) ** 2 / ...

The lowest score is picked (maximize=True: the highest). Ties go to the first candidate in the simulator's order,
exactly as the built-in rules (C# ArgMin / ArgMax / ArgMinIdx, des_twin.rules._argmin / select_machine):
  - job rules scan the candidates keeping the best with a strict comparison, starting from float32 max (min) or
    float32 lowest (max), so a candidate whose score is +inf (min) is never preferred over the first;
  - machine rules start from the first candidate and replace it only on a strictly better score.
A decision with one candidate never calls the rule (the simulators short-circuit it too). A NaN score is an error.

A pair "JOB-MACHINE" (RulePair) combines a job rule and a machine rule; resolve("MYJOB-PECT") looks both halves up.
Names are case-insensitive and stored upper-case; '-' separates the halves, so a rule name may not contain '-'.
The built-in rules (rules_api.builtin) are registered on import; user modules are loaded with load_rules(path or
module name), which imports them so their @rule decorators run.
"""
import importlib
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

F32_MAX = float(np.finfo(np.float32).max)

_JOB, _MACHINE = {}, {}
KINDS = {"job": _JOB, "machine": _MACHINE}


class Rule:
    __slots__ = ("kind", "name", "fn", "maximize", "module", "doc")

    def __init__(self, kind, name, fn, maximize, module, doc):
        self.kind, self.name, self.fn, self.maximize, self.module, self.doc = kind, name, fn, maximize, module, doc

    def __repr__(self):
        return f"Rule({self.kind}:{self.name}{' max' if self.maximize else ''} from {self.module})"


def _norm(name):
    name = str(name).strip().upper()
    if not name or "-" in name or "_" in name or " " in name:
        raise ValueError(f"rule name {name!r}: must be non-empty, without '-', '_' or spaces (pairs are JOB-MACHINE)")
    return name


def rule(kind, name, maximize=False, replace=False):
    """Decorator registering a scoring function as a job or machine rule. maximize: the highest score wins.
    replace: allow re-registering an existing name (otherwise a clash is an error, so a mod cannot silently shadow a
    built-in rule)."""
    if kind not in KINDS:
        raise ValueError(f"rule kind must be 'job' or 'machine', not {kind!r}")
    key = _norm(name)

    def deco(fn):
        table = KINDS[kind]
        old = table.get(key)
        if old is not None and not replace and old.fn is not fn:
            if getattr(old.fn, "__qualname__", None) == getattr(fn, "__qualname__", None) and old.module == fn.__module__:
                pass          # the same module imported twice (e.g. as a file and as a module): keep the new one
            else:
                raise ValueError(f"{kind} rule {key} is already registered ({old}); pass replace=True to override")
        table[key] = Rule(kind, key, fn, bool(maximize), fn.__module__, (fn.__doc__ or "").strip())
        return fn
    return deco


def get(kind, name):
    key = str(name).strip().upper()
    table = KINDS[kind]
    if key not in table:
        raise KeyError(f"unknown {kind} rule {name!r}; registered: {', '.join(sorted(table))}")
    return table[key]


def names(kind):
    return sorted(KINDS[kind])


def _check(score, rule_, cand):
    if score != score:
        raise ValueError(f"{rule_.kind} rule {rule_.name} scored candidate {cand} NaN")
    return score


def pick_job(rule_, ctx):
    """The job a job rule picks among ctx.jobs (JobView list); returns the job id."""
    jobs = ctx.jobs
    if not jobs:
        raise ValueError("no candidate jobs")
    if len(jobs) == 1:
        return jobs[0].id
    fn = rule_.fn
    best = jobs[0].id
    if rule_.maximize:
        best_s = -F32_MAX
        for j in jobs:
            s = _check(fn(j, ctx), rule_, j.id)
            if s > best_s:
                best, best_s = j.id, s
    else:
        best_s = F32_MAX
        for j in jobs:
            s = _check(fn(j, ctx), rule_, j.id)
            if s < best_s:
                best, best_s = j.id, s
    return best


def pick_machine(rule_, job, ctx):
    """The machine a machine rule picks among job.machines (MachineView list); returns the machine id."""
    ms = job.machines
    if not ms:
        raise ValueError(f"job {job.id} has no candidate machine")
    if len(ms) == 1:
        return ms[0].id
    fn = rule_.fn
    b, best_s = 0, _check(fn(ms[0], job, ctx), rule_, ms[0].id)
    for i in range(1, len(ms)):
        s = _check(fn(ms[i], job, ctx), rule_, ms[i].id)
        if (s > best_s) if rule_.maximize else (s < best_s):
            b, best_s = i, s
    return ms[b].id


class RulePair:
    """A job rule + a machine rule, used as one dispatching rule "JOB-MACHINE" in either simulator.

    pick(view) -> (job id, machine id or -1): at routing the job rule picks from the pool and the machine rule picks
    among that job's candidates; at dispatch only the job rule runs (machine = -1)."""
    is_py_rule = True     # marker the simulators test for (no import of this module needed there)

    def __init__(self, job_rule, machine_rule):
        self.job = job_rule if isinstance(job_rule, Rule) else get("job", job_rule)
        self.machine = machine_rule if isinstance(machine_rule, Rule) else get("machine", machine_rule)
        self.name = f"{self.job.name}-{self.machine.name}"

    def pick_job(self, ctx):
        return pick_job(self.job, ctx)

    def pick_machine(self, job, ctx):
        return pick_machine(self.machine, job, ctx)

    def pick(self, ctx):
        jid = self.pick_job(ctx)
        if ctx.kind == "dispatch":
            return jid, -1
        return jid, self.pick_machine(ctx.job(jid), ctx)

    def __repr__(self):
        return f"RulePair({self.name})"

    def __eq__(self, other):
        return isinstance(other, RulePair) and other.job is self.job and other.machine is self.machine

    def __hash__(self):
        return hash(self.name)


def resolve(name):
    """RulePair for "JOB-MACHINE" (or "JOB_MACHINE" when neither half contains '_'), looked up in the registry."""
    text = str(name).strip().upper()
    if text.startswith("PY:"):
        text = text[3:]
    parts = text.split("-") if "-" in text else text.split("_")
    if len(parts) != 2:
        raise ValueError(f"python rule pair {name!r}: expected JOB-MACHINE")
    return RulePair(parts[0], parts[1])


def load_rules(spec):
    """Import rule modules so their @rule decorators register. spec: a module name (e.g. "rules_api.examples.my_rules"),
    a .py file path, "builtin" (the built-in ports only; always loaded), or a comma list of these. Returns the loaded
    modules."""
    from . import builtin  # noqa: F401  (registers the reference rules)
    mods = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part or part.lower() == "builtin":
            continue
        path = Path(part)
        if part.endswith(".py") or path.exists():
            path = path.resolve()
            if not path.is_file():
                raise FileNotFoundError(f"rule module {part!r} not found")
            mod_name = f"pyrules_{path.stem}_{abs(hash(str(path))) & 0xffffff:x}"
            if mod_name in sys.modules:
                mods.append(sys.modules[mod_name])
                continue
            s = importlib.util.spec_from_file_location(mod_name, path)
            mod = importlib.util.module_from_spec(s)
            sys.modules[mod_name] = mod
            s.loader.exec_module(mod)
        else:
            mod = importlib.import_module(part)
        mods.append(mod)
    return mods


def isfinite(x):
    return math.isfinite(float(x))
