"""Dispatching rules written in Python, for both simulators (dev-pyrules, 2026-10-11).

Write a rule as a scoring function over a neutral decision view (rules_api.views) and register it by name:

    from rules_api import rule

    @rule("job", "SLACK2")                       # lower score = picked first (maximize=True: higher)
    def slack2(job, ctx):
        return (job.due - ctx.now - float(job.remaining_work)) / max(job.ops_left, 1)

    @rule("machine", "QECT")
    def qect(m, job, ctx):
        return float(m.load) + 1.5 * float(m.ptime)

A pair "SLACK2-QECT" (or "SLACK2-PECT": built-in halves are registered too) then runs as a dispatching rule:
  - twin:  TwinConfig(py_rule=rules_api.resolve("SLACK2-QECT"), ...) as the fixed rule, or gen.send(pair) per
           decision in agent mode; TwinSchedulingEnv.step(pair)
  - Unity: a player launched with -pyrules sends the view each decision and applies the pick Python sends back;
           UnitySchedulingEnv.step(pair) does both (env_wrappers.unity_env)
  - evaluate.py: --py-rules path/to/my_rules.py --pdr SLACK2-QECT (twin or Unity); "py:NAME" in --pdr is always a
           Python rule, so built-in C# / twin rules and Python rules can be compared in one run.
Every built-in rule has a reference port in rules_api.builtin (SPT LPT SRT LRT FIFO PTWINQ EDD SLACK CR MDD MOD ATC;
SMPT SRWT MMUR ECT TECT PECT PTECT) that reproduces the simulators' own rules decision for decision. An example mod is
rules_api/examples/my_rules.py; results/dev-pyrules/README.md has the checks and limits.
"""
from .registry import RulePair, get, load_rules, names, pick_job, pick_machine, resolve, rule
from .views import DecisionView, JobView, MachineView
from . import builtin  # noqa: F401  (registers the reference rules)

__all__ = ["rule", "RulePair", "resolve", "load_rules", "names", "get", "pick_job", "pick_machine",
           "DecisionView", "JobView", "MachineView"]
