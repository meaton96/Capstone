"""
@file test_rules_api.py
@brief Tests for the Python rule API (env/rules_api, dev-pyrules 2026-10-11): the registry and pick semantics (tie-breaking
       as the built-in rules), the Unity view parser, and twin parity: every built-in pair re-implemented with the API
       makes the twin's own decisions, decision for decision, in fixed and agent mode (the full B2 check is
       results/dev-pyrules/twin_parity.py).
"""
import json
import math
import os

import numpy as np
import pytest

import rules_api
from rules_api import DecisionView, JobView, MachineView, RulePair, resolve, rule
from rules_api.registry import KINDS, pick_job, pick_machine
from rules_api.unity_adapter import parse_view
from des_twin import Floor, TwinConfig
from des_twin.engine import Twin
from des_twin.scenario import resolve_jobs

f32 = np.float32
FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "des_twin")
FLOOR = os.path.join(FIX, "convoy_agv3", "des_floor.json")
JOBS = ["SPT", "LPT", "SRT", "LRT", "FIFO", "PTWINQ", "EDD", "SLACK", "CR", "MDD", "MOD", "ATC"]
MACHINES = ["SMPT", "SRWT", "MMUR", "ECT", "TECT", "PECT", "PTECT"]
PT = {"alpha": 3.0, "shape": "quad", "tau": 5544.0, "reroute": 0.0, "avail": 1.0}


def _job(jid, p=100.0, due=None, rw=None, since=0.0, now=10.0, machines=None, winq=0.0, op=0, ops=3):
    return JobView(jid, op, ops, 0.0, since, now, due, None, f32(p), remaining_work=f32(p if rw is None else rw),
                   winq=f32(winq), machines=machines)


def _m(mid, p, q=0.0, u=0.0, tr=0.0):
    return MachineView(mid, f32(p), f32(q), f32(u), f32(tr))


def _ctx(jobs, kind="routing", now=10.0):
    return DecisionView(kind, now, 0, jobs, -1 if kind == "routing" else 7, pt=PT)


# ── registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

def test_builtin_rules_registered():
    assert set(JOBS) <= set(rules_api.names("job"))
    assert set(MACHINES) <= set(rules_api.names("machine"))
    assert resolve("mdd_ptect").name == "MDD-PTECT" == resolve("py:MDD-PTECT").name


def test_register_and_clash():
    @rule("job", "TESTJOBX")
    def testjob(job, ctx):
        return -float(job.p)
    try:
        assert rules_api.get("job", "testjobx").fn is testjob
        with pytest.raises(ValueError):
            @rule("job", "TESTJOBX")
            def other(job, ctx):
                return 0.0
        with pytest.raises(ValueError):
            rule("job", "BAD-NAME")
        with pytest.raises(KeyError):
            resolve("NOSUCHRULE-ECT")
    finally:
        KINDS["job"].pop("TESTJOBX", None)


def test_load_example_mod():
    rules_api.load_rules(os.path.join(os.path.dirname(rules_api.__file__), "examples", "my_rules.py"))
    pair = resolve("OPSLACK-LECT")
    assert pair.job.name == "OPSLACK" and pair.machine.name == "LECT"
    # OPSLACK: slack per remaining op, lowest first; jobs without a due date go last
    jobs = [_job(1, due=None, rw=10), _job(2, due=1000.0, rw=100, ops=3), _job(3, due=400.0, rw=100, ops=1)]
    assert pick_job(pair.job, _ctx(jobs)) == 3


# ── pick semantics ───────────────────────────────────────────────────────────────────────────────────────────────────

def test_ties_go_to_first_and_max_rules():
    ctx = _ctx([_job(5, p=50), _job(6, p=20), _job(7, p=20), _job(8, p=50)])
    assert pick_job(rules_api.get("job", "SPT"), ctx) == 6
    assert pick_job(rules_api.get("job", "LPT"), ctx) == 5
    jb = _job(1, machines=(_m(10, 30, 5), _m(11, 20, 15), _m(12, 25, 10)))
    assert pick_machine(rules_api.get("machine", "ECT"), jb, _ctx([jb])) == 10       # 35 = 35 = 35: first
    assert pick_machine(rules_api.get("machine", "SMPT"), jb, _ctx([jb])) == 11


def test_infinite_scores_keep_first_and_nan_raises():
    ctx = _ctx([_job(1), _job(2), _job(3)])
    assert pick_job(rules_api.get("job", "EDD"), ctx) == 1           # no due dates: all +inf -> first, as C# / twin

    @rule("job", "TESTNAN")
    def nan_rule(job, ctx):
        return math.nan
    try:
        with pytest.raises(ValueError):
            pick_job(rules_api.get("job", "TESTNAN"), ctx)
    finally:
        KINDS["job"].pop("TESTNAN", None)


def test_single_candidate_not_scored():
    @rule("job", "TESTBOOM")
    def boom(job, ctx):
        raise AssertionError("scored a single candidate")
    try:
        assert pick_job(rules_api.get("job", "TESTBOOM"), _ctx([_job(4)])) == 4
    finally:
        KINDS["job"].pop("TESTBOOM", None)


def test_pair_pick_dispatch_has_no_machine():
    pair = resolve("SPT-ECT")
    assert pair.pick(_ctx([_job(1, p=9), _job(2, p=3)], kind="dispatch")) == (2, -1)
    j1 = _job(1, p=9, machines=(_m(4, 9, 100), _m(5, 12, 0)))
    assert pair.pick(_ctx([j1])) == (1, 5)


# ── Unity view ───────────────────────────────────────────────────────────────────────────────────────────────────────

def test_parse_unity_view():
    raw = json.dumps({
        "v": 1, "kind": "routing", "decision": 12, "t": 1234.5, "machine": -1, "wip": 9, "fleet": 2, "nAgvs": 7,
        "travelPrice": 0.0, "pt": {"alpha": 3.0, "shape": "quad", "tau": 5544.0, "reroute": 0.0, "avail": 1.0},
        "jobs": [{"id": 3, "op": 1, "ops": 4, "arrival": 100.0, "since": 1200.25, "due": None, "p": 61.2000008,
                  "rw": 300.5, "winq": 0.0, "m": [[2, 61.2000008, 10.0, 0.5, 3.25], [5, 70.0, 0.0, 0.25, 9.5]]},
                 {"id": 8, "op": 0, "ops": 2, "arrival": 1100.0, "since": 1100.0, "due": 5000.0, "p": 40.0,
                  "rw": 90.0, "winq": 12.5, "m": [[1, 40.0, 0.0, 0.0, 1.0]]}]})
    v = parse_view(raw)
    assert (v.kind, v.decision, v.now, v.wip, v.fleet, v.n_agvs) == ("routing", 12, 1234.5, 9, 2, 7)
    j = v.jobs[0]
    assert j.due == math.inf and j.op_due == math.inf and j.ops_left == 3
    assert j.p == f32(61.2000008) and isinstance(j.p, np.float32)
    assert j.time_in_queue == f32(1234.5 - 1200.25)
    assert [m.id for m in j.machines] == [2, 5] and j.machines[0].travel == f32(3.25)
    assert v.jobs[1].due == 5000.0
    assert resolve("FIFO-SRWT").pick(v) == (8, 1)        # job 8 waited 134.5 s, job 3 34.25 s
    with pytest.raises(ValueError):
        parse_view(dict(json.loads(raw), v=99))


# ── twin parity ──────────────────────────────────────────────────────────────────────────────────────────────────────

def _floor():
    with open(FLOOR) as f:
        return Floor(json.load(f))


def _jobs(floor, seed=0, hours=3.0):
    from scenarios.randomized import RandomizedParams, randomized_scenario
    p = RandomizedParams(agv_count=3, failure_probability=0.0, horizon_seconds=hours * 3600.0,
                         due_date_allowance_range=(1.75, 2.5), op_machine_spread=0.3)
    return resolve_jobs(randomized_scenario(seed, p), floor)


def _fingerprint(tw):
    return (list(tw.decision_rows), [(j.id, j.exit_time) for j in tw.jobs.values()], tw.now, tw.makespan)


@pytest.mark.parametrize("job_rule", JOBS)
def test_twin_fixed_mode_parity_every_job_rule(job_rule):
    floor, jobs = _floor(), None
    jobs = _jobs(floor, seed=JOBS.index(job_rule) % 3)
    for machine_rule in (MACHINES[JOBS.index(job_rule) % 7], "PTECT"):
        ref = Twin(floor, jobs, TwinConfig(rule=f"{job_rule}_{machine_rule}", agv_count=3)).run()
        py = Twin(floor, jobs, TwinConfig(rule=None, py_rule=resolve(f"{job_rule}-{machine_rule}"), agv_count=3)).run()
        assert _fingerprint(py) == _fingerprint(ref), f"{job_rule}-{machine_rule}"
        assert py.summary()["rule"] == f"py:{job_rule}-{machine_rule}"


@pytest.mark.parametrize("machine_rule", MACHINES)
def test_twin_agent_mode_parity_every_machine_rule(machine_rule):
    """Agent mode after a warm-up: the built-in halves sent as strings vs the RulePair sent per decision."""
    floor = _floor()
    jobs = _jobs(floor, seed=1)
    out = []
    for py in (False, True):
        tw = Twin(floor, jobs, TwinConfig(rule="SPT_ECT", agv_count=3, warmup_seconds=1800.0,
                                          episode_duration_seconds=5400.0, ptime_shape="ratio", ptime_alpha=2.0))
        job_rule = JOBS[MACHINES.index(machine_rule) + 3]
        action = resolve(f"{job_rule}-{machine_rule}") if py else (job_rule, machine_rule)
        gen, n = tw.agent_decisions(), 0
        try:
            next(gen)
            while True:
                gen.send(action)
                n += 1
        except StopIteration:
            pass
        assert n > 50
        out.append(_fingerprint(tw))
    assert out[0] == out[1]


def test_twin_mixed_actions_and_example_mod_runs():
    """Python and built-in pairs can alternate per decision; a rule that exists only as a mod runs end to end."""
    rules_api.load_rules(os.path.join(os.path.dirname(rules_api.__file__), "examples", "my_rules.py"))
    floor = _floor()
    jobs = _jobs(floor, seed=2)
    tw = Twin(floor, jobs, TwinConfig(rule="MDD_PTECT", agv_count=3, warmup_seconds=600.0))
    gen, k = tw.agent_decisions(), 0
    pairs = [resolve("OPSLACK-LECT"), ("MDD", "PTECT"), resolve("OPSLACK-PECT")]
    try:
        next(gen)
        while True:
            gen.send(pairs[k % 3])
            k += 1
    except StopIteration:
        pass
    assert k > 100 and tw.summary()["jobs_completed"] > 0


def test_twin_py_rule_rejects_illegal_pick():
    @rule("job", "TESTBADPICK")
    def bad(job, ctx):
        return 0.0
    floor = _floor()
    jobs = _jobs(floor, seed=0, hours=1.0)

    class Liar(RulePair):
        def pick_job(self, ctx):
            return -12345
    try:
        tw = Twin(floor, jobs, TwinConfig(rule=None, py_rule=Liar("TESTBADPICK", "ECT"), agv_count=3))
        with pytest.raises(ValueError):
            tw.run()
    finally:
        KINDS["job"].pop("TESTBADPICK", None)
