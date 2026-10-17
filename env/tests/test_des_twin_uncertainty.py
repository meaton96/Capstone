"""Twin uncertainty layer (2026-10-11, dev-twin-uncertainty): des_twin.uncertainty (configs, keyed CRN draws, the
world derived from the plan, conditional redraw), the engine hooks (actual durations, pre-dispatch on the estimate,
world arrivals), des_twin.observed (the degraded view: delays, dropouts, local truth, I1), the planner v2 (fork + belief
transform) and the plumbing. The full bit-exactness check against 95af8606 and the planner regression against the
stored rq2-oracle-perturbed / rq2-head-to-head results are results/dev-twin-uncertainty/check_exact.py and
check_planner.py; here the cheap versions on the 15-machine fixture floor."""
import json
import math
import os

import numpy as np
import pytest

from des_twin import Floor, TwinConfig
from des_twin import branch
from des_twin import uncertainty as U
from des_twin.engine import EXITED, NEEDS_ROUTING, PROCESSING, QUEUED, Twin
from des_twin.observation import ObservationBuilder
from des_twin.planner import belief_transform, plan_episode, window_tardiness
from des_twin.scenario import episode_settings, resolve_jobs
from des_twin.uncertainty import UncertaintyConfig, UncertaintyTrace

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "des_twin")
FLOOR = os.path.join(FIX, "convoy_agv3", "des_floor.json")
fork_only = pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")

_CACHE = {}


def _floor(frame=False):
    key = ("floor", frame)
    if key not in _CACHE:
        with open(FLOOR) as f:
            data = json.load(f)
        if frame:
            from tests.test_twin_env import with_frame
            data = with_frame(data)
        _CACHE[key] = Floor(data)
    return _CACHE[key]


def _scenario(seed, horizon=3 * 3600.0, warmup=0.0, cap=0.0):
    key = ("sc", seed, horizon, warmup, cap)
    if key not in _CACHE:
        from scenarios.randomized import RandomizedParams, randomized_scenario
        p = RandomizedParams(agv_count=3, failure_probability=0.0, horizon_seconds=horizon,
                             due_date_allowance_range=(1.75, 2.5))
        sc = randomized_scenario(seed, p)
        if warmup or cap:
            sc["stochastic"] = dict(sc.get("stochastic") or {}, warmupSeconds=warmup, episodeDurationSeconds=cap)
        _CACHE[key] = sc
    return _CACHE[key]


def _plan(seed, **kw):
    return resolve_jobs(_scenario(seed, **kw), _floor())


def _pool(seed):
    return _plan(seed + 70000)


def _trace(seed, spec, plan=None, **kw):
    plan = plan or _plan(seed)
    cfg = UncertaintyConfig.from_spec(spec)
    pool = _pool(seed) if (cfg.active("E4") or cfg.active("E4n")) else None
    return UncertaintyTrace.build(plan, _floor(), cfg, seed, rush_pool=pool, **kw)


def _twin(seed, spec=None, rule="MDD_PECT", frame=False, warmup=0.0, cap=0.0, **cfg_kw):
    plan = _plan(seed)
    trace = None if spec is None else _trace(seed, spec, plan)
    jobs = plan if trace is None else trace.world_jobs
    return Twin(_floor(frame), jobs, TwinConfig(rule=rule, agv_count=3, uncertainty=trace, warmup_seconds=warmup,
                                                episode_duration_seconds=cap, **cfg_kw))


def _play(tw, pairs, slot=900.0):
    """Agent mode with a cycling pair per slot; returns (decision count, window tardiness)."""
    gen = tw.agent_decisions()
    t0, n = None, 0
    try:
        dec = gen.send(None)
        while True:
            if dec.kind == "pause":
                dec = gen.send(None)
                continue
            if t0 is None:
                t0 = tw.now
            pair = pairs[int((tw.now - t0) // slot) % len(pairs)]
            dec = gen.send(tuple(pair.split("-")))
            n += 1
    except StopIteration:
        pass
    return n, window_tardiness(tw, t0 or 0.0)


def _fp(tw):
    return (tw.summary(), tw.metrics(), list(tw.decision_rows), tw.op_rows(), list(tw.trace), tw.job_rows())


# ── Configuration ────────────────────────────────────────────────────────────────────────────────────────────

def test_spec_parsing_and_roundtrip():
    c = UncertaintyConfig.from_spec("E2d=realistic,O1=severe,info=I1,p1_lead=900")
    assert c.params("E2d")["sigma_d"] == 0.08 and c.params("O1")["mean"] == 300.0
    assert c.info == "I1" and c.p1_lead == 900.0 and c.active("E2d") and not c.active("E1")
    assert UncertaintyConfig.from_spec("realistic").levels == {t: "realistic" for t in U.MIX_TYPES}
    assert UncertaintyConfig.from_spec("none").is_zero() and UncertaintyConfig.zero().is_zero()
    assert not UncertaintyConfig.from_spec("O3=mild").is_zero()
    assert UncertaintyConfig.from_dict(c.to_dict()) == c
    assert UncertaintyConfig(levels={"E1": {"sigma_op": 0.2}}).params("E1") == {"sigma_job": 0.0, "sigma_op": 0.2}
    with pytest.raises(ValueError):
        UncertaintyConfig(levels={"E4": "mild", "E4n": "mild"})
    with pytest.raises(ValueError):
        UncertaintyConfig.from_spec("E9=mild")
    with pytest.raises(ValueError):
        UncertaintyConfig(levels={"E1": "huge"}).params("E1")
    rng = np.random.default_rng(0)
    mixed = UncertaintyConfig.mix(rng)
    assert set(mixed.levels) == set(U.MIX_TYPES) and all(v in U.LEVELS for v in mixed.levels.values())


def test_trace_json_roundtrip_and_injection_schedule():
    tr = _trace(1, "E1=realistic,E2d=severe")
    data = json.loads(json.dumps(tr.to_json(schedule=True)))
    back = UncertaintyTrace.from_json(data, _plan(1), _floor())
    jd = tr.world_jobs["jobs"][3]
    assert back.factor(jd["id"], 0, jd["ops"][0]["eligible"][0][0]) == tr.factor(jd["id"], 0, jd["ops"][0]["eligible"][0][0])
    assert back.bias_at(5, 7000.0) == tr.bias_at(5, 7000.0)
    sched = data["machineBiasSchedule"]
    assert sched["dt"] == U.BIAS_DT and len(sched["machines"]) == len(_floor().machines)
    assert data["actualFactors"][str(jd["id"])][0][0] == tr.factor(jd["id"], 0, jd["ops"][0]["eligible"][0][0])


# ── Keyed draws, CRN, nesting, distributions ─────────────────────────────────────────────────────────────────

def test_keyed_draws_do_not_depend_on_call_order():
    tr = _trace(2, "E1=realistic,O1=realistic,O3=realistic")
    keys = [(j, o, m) for j in range(20) for o in range(3) for m in range(5)]
    a = [tr.factor(*k) for k in keys]
    b = [tr.factor(*k) for k in reversed(keys)][::-1]
    assert a == b
    assert tr.report_delay(3, 1, QUEUED, 0) == tr.report_delay(3, 1, QUEUED, 0) != tr.report_delay(3, 1, QUEUED, 1)
    other = _trace(2, "E1=realistic", draw=1)
    assert other.factor(*keys[0]) != a[0]                 # another world of the same plan


def test_e1_factors_are_mean_one_with_the_level_cv():
    tr = _trace(3, "E1=severe")
    f = np.array([tr.factor(j, o, m) for j in range(400) for o in range(3) for m in range(3)])
    sj, so = 0.15, 0.20
    cv = math.sqrt(math.exp(sj * sj + so * so) - 1.0)
    # the job/op factor is shared across machines, so the effective sample is ~ 1200 job-ops: allow 4 SE of that
    se = cv / math.sqrt(1200)
    assert abs(f.mean() - 1.0) < 4 * se
    assert abs(f.std() / f.mean() - cv) < 0.03


def test_levels_scale_the_same_standardized_draws():
    """CRN across levels: the sign of each op's error is the same at every level, and event sets are nested."""
    plan = _plan(4)
    tr = {lv: _trace(4, f"E1={lv},E3={lv},E4={lv},O1={lv}", plan) for lv in ("mild", "realistic", "severe")}
    keys = [(j, o, m) for j in range(50) for o in range(2) for m in range(3)]
    for k in keys:
        signs = {lv: np.sign(math.log(t.factor(*k))) for lv, t in tr.items()}
        # E1's sigma_job / sigma_op ratio differs between levels, so compare each component's sign via the job part
        assert len({np.sign(U.keyed_normal(*t.prefix, U._E1J, k[0], k[1])) for t in tr.values()}) == 1
        del signs
    late = {lv: {jid for jid in t.plan_by_id if U.keyed_uniform(*t.prefix, U._E3U, jid) < t._p["E3"]["q"]}
            for lv, t in tr.items()}
    assert late["mild"] <= late["realistic"] <= late["severe"] and late["severe"]
    canc = {lv: set(t.cancelled) for lv, t in tr.items()}
    assert canc["mild"] <= canc["realistic"] <= canc["severe"]
    rush = {lv: set(t.rush) for lv, t in tr.items()}
    assert rush["mild"] <= rush["realistic"] <= rush["severe"] and rush["severe"]
    d = {lv: t.report_delay(7, 1, QUEUED, 0) for lv, t in tr.items()}
    assert d["mild"] <= d["realistic"] <= d["severe"]


def test_ou_drift_is_stationary_with_its_time_constant():
    class F:
        dt = 0.02
        machines = [type("M", (), {"id": i})() for i in range(200)]
    tr = UncertaintyTrace.build({"jobs": []}, F(), UncertaintyConfig(levels={"E2d": "realistic"}), 11,
                                horizon=40 * 3600.0)
    x = np.array([tr.bdrift_x[m.id] for m in F.machines])
    sd, tau = 0.08, 4 * 3600.0
    assert abs(x.std() - sd) < 0.1 * sd
    lag = int(tau / U.BIAS_DT)
    r = np.mean(x[:, :-lag] * x[:, lag:]) / np.mean(x * x)       # autocorrelation about the known mean 0
    assert abs(r - math.exp(-1.0)) < 0.05
    b = np.array([[tr.bias_at(m.id, t) for t in np.arange(0, 40 * 3600, 600)] for m in F.machines])
    assert abs(b.mean() - 1.0) < 0.01                       # mean-one


def test_arrival_delay_and_dropout_distributions():
    from scipy import stats
    tr = _trace(5, "E3=mild")                               # sigma_a 2 min, no extra delays
    lat = [tr._lateness(j) for j in range(3000)]
    assert stats.kstest(np.array(lat) / 120.0, "norm").pvalue > 0.01
    tr = _trace(5, "O1=mild")                               # Exp(30 s)
    d = [tr.report_delay(j, 0, QUEUED, 0) for j in range(3000)]
    assert stats.kstest(np.array(d) / 30.0, "expon").pvalue > 0.01
    tr = _trace(5, "O2=realistic", horizon=200 * 3600.0)
    rate, mean = 1 / (4 * 3600.0), 900.0
    frac = np.mean([sum(min(e, 200 * 3600.0) - s for s, e in iv if s < 200 * 3600.0) / (200 * 3600.0)
                    for iv in tr.dropouts.values()])
    assert abs(frac - rate * mean / (1 + rate * mean)) < 0.015
    durs = [e - s for iv in tr.dropouts.values() for s, e in iv]
    assert stats.kstest(np.array(durs) / mean, "expon").pvalue > 0.01


def test_e2_static_bias_is_mean_one_per_floor():
    class F:
        dt = 0.02
        machines = [type("M", (), {"id": i})() for i in range(4000)]
    tr = UncertaintyTrace.build({"jobs": []}, F(), UncertaintyConfig(levels={"E2": "severe"}), 3)
    b = np.array([tr.bstat[m.id] for m in F.machines])
    assert abs(b.mean() - 1.0) < 4 * b.std() / math.sqrt(len(b))
    assert abs(np.log(b).std() - 0.20) < 0.01


# ── Engine: exactness at zero severity, execution hooks ─────────────────────────────────────────────────────

def test_zero_severity_is_bit_identical_including_the_view_and_observation():
    builder = ObservationBuilder(_floor(True), 15, 128)
    runs = []
    for spec in (None, UncertaintyConfig.zero(), UncertaintyConfig.zero(o1_stamp="booking", predispatch_on="actual")):
        tw = _twin(6, spec, frame=True, warmup=1800.0, cap=3600.0)
        S, O = (tw, tw) if spec is None else (tw.view, tw.observed)
        gen = tw.agent_decisions()
        rec = []
        try:
            dec = gen.send(None)
            i = 0
            while True:
                rec.append((dec, tuple(S.all_machine_loads().items()),
                            {k: v for k, v in builder.build(O, dec).items()} if i % 7 == 0 else None))
                dec = gen.send(("MDD", "PECT") if i % 3 else ("ATC", "PTECT"))
                i += 1
        except StopIteration:
            pass
        runs.append((_fp(tw), rec))
    for other in runs[1:]:
        assert other[0] == runs[0][0]
        for (d1, l1, o1), (d2, l2, o2) in zip(runs[0][1], other[1]):
            assert d1 == d2 and l1 == l2
            if o1 is not None:
                assert all(np.array_equal(o1[k], o2[k]) for k in o1)


def test_actual_durations_follow_the_trace_and_crn_holds_across_policies():
    rows = {}
    for rule in ("MDD_PECT", "SRT_SMPT"):
        tw = _twin(7, "E1=realistic", rule=rule).run()
        rows[rule] = {(j, o, m): (p, a) for j, o, m, t, p, a, e in tw.dur_rows}
        for (j, o, m), (p, a) in rows[rule].items():
            assert a == float(np.float32(p * tw.unc.factor(j, o, m)))
    common = set(rows["MDD_PECT"]) & set(rows["SRT_SMPT"])
    assert len(common) > 50
    assert all(rows["MDD_PECT"][k] == rows["SRT_SMPT"][k] for k in common)


def test_bias_uses_the_start_time_and_rules_read_planned_durations():
    tw = _twin(8, "E2d=severe").run()
    for j, o, m, t, p, a, e in tw.dur_rows[:200]:
        assert a == float(np.float32(p * tw.unc.bias_at(m, t)))
        assert p == float(tw.jobs[j].ops[o][m])            # Job.ops keeps the planned value
        assert e == p                                       # I0, no O3: the controller expects the plan


def test_predispatch_on_the_estimate_with_overruns_and_early_finishes():
    tw = _twin(9, "E1=severe,E2d=severe", rule="SRT_PTECT").run()
    assert all(j.state == EXITED for j in tw.jobs.values())
    late = [r for r in tw.dur_rows if r[5] > r[6] * 1.2]
    early = [r for r in tw.dur_rows if r[5] < r[6] * 0.8]
    assert late and early
    # the actual-countdown variant also completes
    tw2 = _twin(9, UncertaintyConfig.from_spec("E1=severe,E2d=severe,predispatch_on=actual"), rule="SRT_PTECT").run()
    assert all(j.state == EXITED for j in tw2.jobs.values())


def test_rush_orders_and_cancellations_change_only_the_world():
    tr = _trace(10, "E4=severe")
    world = {jd["id"] for jd in tr.world_jobs["jobs"]}
    plan = {jd["id"] for jd in tr.plan_jobs["jobs"]}
    assert tr.rush and set(tr.rush) <= world and not (set(tr.rush) & plan)
    assert tr.cancelled and not (set(tr.cancelled) & world)
    tw = Twin(_floor(), tr.world_jobs, TwinConfig(rule="MDD_PECT", agv_count=3, uncertainty=tr)).run()
    assert all(tw.jobs[r].state == EXITED for r in tr.rush)
    for r in tr.rush:                                       # tighter allowance c_rush in [1, 1.5]
        jd = tr.world_by_id[r]
        c = (jd["due"] - jd["arrival"]) / U._twk(jd)
        assert 0.99 <= c <= 1.51
    jid, notice = next(iter(tr.cancelled.items()))
    assert any(jd["id"] == jid for jd in tr.plan_view(notice - 1.0, set()))
    assert not any(jd["id"] == jid for jd in tr.plan_view(notice, set()))
    assert not any(jd["id"] in tr.rush for jd in tr.plan_view(0.0, set()))


def test_arrival_lateness_keeps_due_dates_on_the_plan():
    tr = _trace(11, "E3=severe")
    moved = [jd for jd in tr.world_jobs["jobs"] if jd["arrival"] != tr.plan_by_id[jd["id"]]["arrival"]]
    assert moved
    for jd in moved:
        p = tr.plan_by_id[jd["id"]]
        assert jd["due"] == p["due"]
        tw_op_due = Twin._make_job(p).op_due
        assert jd["op_due"] == tw_op_due                   # operation due dates from the PLANNED arrival
    pv = tr.plan_view(0.0, set())
    assert [jd["arrival"] for jd in pv] == [tr.plan_by_id[jd["id"]]["arrival"] for jd in pv]


# ── The view ─────────────────────────────────────────────────────────────────────────────────────────────────

def test_view_lags_by_exactly_the_booking_delay():
    D = 137.0
    tw = _twin(12, "O1=mild")
    tw.unc.report_delay = lambda *a: D
    log = []
    core = tw._vcore
    orig = core.on_event

    def rec(job, old, state):
        log.append((tw.now, job.id, state))
        orig(job, old, state)
    core.on_event = rec
    gen = tw.agent_decisions()
    checked = 0
    try:
        dec = gen.send(None)
        while True:
            core.sync()
            if len(log) > 50 and checked < 200:
                for jid in list(core.rep)[-15:]:
                    seen = [s for t, j, s in log if j == jid and t + D <= tw.now]
                    exp = seen[-1] if seen else NEEDS_ROUTING
                    assert core.rep[jid].state == exp
                    checked += 1
            dec = gen.send(("MDD", "PECT"))
    except StopIteration:
        pass
    assert checked >= 200 and core.stats["dropped_stale"] == 0


def test_local_truth_and_rules_read_the_view():
    import des_twin.engine as E
    tw = _twin(13, "O1=severe")
    seen = {"routing": 0, "stale": 0}
    orig = E.select_machine

    def spy(rule, cands, times, loads, utils, travel, **kw):
        for mid, q in zip(cands, loads):
            assert q == tw.view.machine_load(mid)
            if q != tw.machine_load(mid):
                seen["stale"] += 1
        seen["routing"] += 1
        return orig(rule, cands, times, loads, utils, travel, **kw)
    E.select_machine = spy
    try:
        gen = tw.agent_decisions()
        dec = gen.send(None)
        while True:
            V = tw.view
            if dec.kind == "dispatch":
                for j in dec.queue:                         # the machine's physical queue is true
                    assert V.jobs[j].state == QUEUED and V.jobs[j].location == dec.machine
                assert V.mach[dec.machine].idle
            else:
                for j in dec.job_candidates:                # the routing pool is true
                    assert V.jobs[j].state == NEEDS_ROUTING and V.jobs[j].cur_op == tw.jobs[j].cur_op
            dec = gen.send(("MDD", "PECT"))
    except StopIteration:
        pass
    finally:
        E.select_machine = orig
    assert seen["routing"] > 100 and seen["stale"] > 10


def test_dropout_freezes_the_machine_and_releases_at_the_end():
    tw = _twin(14, "O2=mild")
    tw.unc.report_delay = lambda *a: 0.0
    m0 = tw.machines[0].id
    s, e = 3000.0, 4800.0
    tw.unc.dropouts = {m.id: ([(s, e)] if m.id == m0 else []) for m in tw.machines}
    tw.unc._dstarts = {mid: [a for a, _ in iv] for mid, iv in tw.unc.dropouts.items()}
    gen = tw.agent_decisions()
    frozen, after = None, False
    try:
        dec = gen.send(None)
        while True:
            V = tw.view
            if s + 30 < tw.now < e:
                state = (V.mach[m0].idle, V.mach[m0].job)
                frozen = frozen or state
                if dec.kind != "dispatch" or dec.machine != m0:
                    assert state == frozen
            if tw.now >= e + 1 and not after:
                after = True
                assert V.mach[m0].idle == tw.mach[m0].idle and V.mach[m0].job == tw.mach[m0].job
            dec = gen.send(("MDD", "PECT"))
    except StopIteration:
        pass
    assert frozen is not None and after


def test_i1_estimator_converges_on_a_static_bias():
    tw = _twin(15, "E2=severe,info=I1").run()
    est = tw.view.bias_estimates()
    err = np.mean([abs(math.log(est[m][0]) - math.log(tw.unc.bias_at(m, tw.now))) for m in est])
    base = np.mean([abs(math.log(tw.unc.bias_at(m, tw.now))) for m in est])
    assert err < 0.25 * base


@pytest.mark.parametrize("seed,spec", [(15, "E2=severe,info=I1"), (16, "E2d=severe,info=I1"),
                                       (17, "E2=realistic,O1=realistic,info=I1")])
def test_i1_estimate_does_not_depend_on_who_synced_the_view(seed, spec):
    """The I1 estimate includes every completion visible by now, whoever read the view before. Until 2026-10-17
    ViewCore.bias_estimate / ObservedState._ops read it unsynced, so an episode changed with an extra ViewCore.sync()
    per decision (what TwinSchedulingEnv's observation build does; results/rq4-uncert-v5-policy, screen_diff.py)."""
    def play(extra_sync):
        tw = _twin(seed, spec, rule="MDD_PTECT")
        gen = tw.agent_decisions()
        try:
            gen.send(None)
            while True:
                if extra_sync:
                    tw._vcore.sync()
                gen.send(("MDD", "PTECT"))
        except StopIteration:
            pass
        return _fp(tw), list(tw.dur_rows), tw.view.bias_estimates()

    assert play(False) == play(True)


def test_i2_view_equals_a_plan_with_the_bias_baked_in():
    """With static bias only and perfect bias information the controllers see the true durations."""
    import copy
    tw = _twin(16, "E2=severe,info=I2").run()
    baked = copy.deepcopy(_plan(16))
    for jd in baked["jobs"]:
        for op in jd["ops"]:
            op["eligible"] = [[m, float(np.float32(float(np.float32(d)) * tw.unc.bstat[int(m)]))] for m, d in op["eligible"]]
    ref = Twin(_floor(), baked, TwinConfig(rule="MDD_PECT", agv_count=3)).run()
    assert tw.decision_rows == ref.decision_rows


# ── Branching, redraw, planner ─────────────────────────────────────────────────────────────────────────────

@fork_only
def test_branch_children_equal_replays_with_noise_on():
    spec = "mix:realistic"

    def replay(stop_at, opt):
        tw = _twin(17, spec, warmup=1200.0, cap=5400.0)
        gen = tw.agent_decisions()
        dec = gen.send(None)
        for _ in range(stop_at):
            dec = gen.send(("MDD", "PECT"))
        tw.force(*opt)
        branch.play_to_end(tw, gen, ("MDD", "PECT"), lambda t, d: ("MDD", "PECT"))
        return tw.summary(), list(tw.decision_rows)

    tw = _twin(17, spec, warmup=1200.0, cap=5400.0)
    gen = tw.agent_decisions()
    dec = gen.send(None)
    for _ in range(40):
        dec = gen.send(("MDD", "PECT"))
    if dec.kind == "dispatch":
        opts = [(j, dec.machine) for j in dec.queue][:3]
    else:
        opts = [(j, m) for j in dec.job_candidates for m in tw.candidate_machines(tw.jobs[j])][:3]
    res = branch.rollout(tw, gen, dec, opts, policy=lambda t, d: ("MDD", "PECT"),
                         score=lambda t: (t.summary(), list(t.decision_rows)), max_parallel=3)
    for opt, r in zip(opts, res):
        assert r == replay(40, opt)


@fork_only
def test_redraw_keeps_the_past_and_redraws_the_future():
    spec = "E1=realistic,E2d=severe,E3=realistic,E4=realistic,O2=realistic,O3=realistic"
    tw = _twin(18, spec, warmup=0.0, cap=0.0)
    tw.pause_at(4000.0)
    gen = tw.agent_decisions()
    dec = gen.send(None)
    while dec.kind != "pause":
        dec = gen.send(("MDD", "PECT"))
    t = tw.now
    new = tw.unc.redraw_after(tw, 50001, mode="oracle", rush_pool=_pool(18))
    i = int(t // U.BIAS_DT)
    for m in tw.machines:
        assert np.array_equal(new.bdrift_x[m.id][:i + 1], tw.unc.bdrift_x[m.id][:i + 1])
        assert not np.array_equal(new.bdrift_x[m.id][i + 5:], tw.unc.bdrift_x[m.id][i + 5:])
        assert [iv for iv in new.dropouts[m.id] if iv[0] <= t] == [iv for iv in tw.unc.dropouts[m.id] if iv[0] <= t]
    arrived = {j.id for j in tw.order}
    for jid in arrived:
        assert new.world_by_id[jid] == tw.unc.world_by_id[jid]
    started = [(j.id, r["op"]) for j in tw.jobs.values() for r in j.op_rows if r.get("start") is not None]
    assert started and all(new.est_factor(*k) == tw.unc.est_factor(*k) for k in started)
    fut_old = {jd["id"]: jd["arrival"] for jd in tw.unc.world_jobs["jobs"] if jd["id"] not in arrived}
    fut_new = {jd["id"]: jd["arrival"] for jd in new.world_jobs["jobs"] if jd["id"] not in arrived}
    assert fut_old != fut_new and all(a >= t for a in fut_new.values())
    for jid, notice in tw.unc.cancelled.items():
        if notice <= t:
            assert new.cancelled.get(jid) == notice           # a known cancellation stays

    def child(_):
        U.apply_trace(tw, new)
        gen.send(None)
        branch.play_to_end(tw, gen, ("MDD", "PECT"), lambda a, b: ("MDD", "PECT"))
        return all(j.state == EXITED for j in tw.jobs.values()), len(tw.jobs)
    done, n = branch.fork_map(child, [0], max_parallel=1)[0]
    assert done


def test_next_tick_arrival_is_the_next_tick_in_float32():
    """The clip-fix arrival (10-11, results/dev-uncert-clipfix): a des_jobs float32 value whose engine tick is exactly
    k + 1 (where float32 resolves dt), and always after tick k."""
    dt = _floor().dt
    for k in [0, 1, 7, 49, 50, 12345, 90000, 179999, 360000, 2 ** 17 * 50 - 1, 5_000_000, 9_999_999]:
        a = U.next_tick_arrival(k, dt)
        assert a == float(str(np.float32(a)))                    # as des_jobs.json writes arrivals
        assert U.arrival_tick(a, dt) == k + 1, (k, a)
        assert a > k * dt
    for k in (20_000_000, 50_000_000):                           # float32 spacing > dt: the earliest tick after k
        assert U.arrival_tick(U.next_tick_arrival(k, dt), dt) > k


def _pause_twin(seed, spec, t_pause):
    tw = _twin(seed, spec, warmup=0.0, cap=0.0)
    tw.pause_at(t_pause)
    gen = tw.agent_decisions()
    dec = gen.send(None)
    while dec.kind != "pause":
        dec = gen.send(("MDD", "PECT"))
    return tw, gen


@fork_only
def test_redraw_moves_arrivals_in_the_past_to_the_next_tick():
    """A redrawn E3 arrival at or before the current tick arrives at the next tick (before the fix: exactly t = now,
    which Twin.replace_pending rejected); other redrawn arrivals are unchanged, the redraw stays deterministic (CRN),
    and the forked child accepts the trace and plays to the end."""
    tw, gen = _pause_twin(18, "E3=severe,E1=realistic", 4000.0)
    t, k, dt = tw.now, tw.k, tw.dt
    assert t == k * dt
    arrived = {j.id for j in tw.order}
    nxt = U.next_tick_arrival(k, dt)
    found = None
    for seed_k in range(50000, 50040):
        new = tw.unc.redraw_after(tw, seed_k, mode="realizable", bias_est=lambda mid: (1.0, 0.0))
        raw = {}
        for jd in new.plan_jobs["jobs"]:
            jid, a = int(jd["id"]), float(jd["arrival"])
            if jid in arrived or jid in new.cancelled or a <= 0.0:
                continue
            r = max(0.0, a + new._lateness(jid))
            raw[jid] = U._f32r(r) if r != a else a
        past = [jid for jid, r in raw.items() if U.arrival_tick(r, dt) <= k]
        if past:
            found = (seed_k, new, raw, past)
            break
    assert found, "no redraw with an arrival in the past (pick another seed)"
    seed_k, new, raw, past = found
    fut = {int(jd["id"]): float(jd["arrival"]) for jd in new.world_jobs["jobs"] if int(jd["id"]) not in arrived}
    assert all(U.arrival_tick(a, dt) > k for a in fut.values())          # what replace_pending requires
    for jid, r in raw.items():
        want = nxt if jid in past else r                                  # only the past ones move
        assert fut[jid] == want and new.actual_arrival[jid] == want
        assert new.world_by_id[jid]["arrival"] == want
    assert all(U.arrival_tick(fut[jid], dt) == k + 1 for jid in past)
    # deterministic, and a caller without the tick in keep (realizable.py's _build_world) gets the same world
    again = tw.unc.redraw_after(tw, seed_k, mode="realizable", bias_est=lambda mid: (1.0, 0.0))
    assert again.world_jobs == new.world_jobs and again.actual_arrival == new.actual_arrival
    again._build_world(None, keep={"arrived": arrived, "t": t, "parent": tw.unc})
    assert again.world_jobs == new.world_jobs and again.actual_arrival == new.actual_arrival

    def child(_):
        belief_transform(tw, None, keep_trace=new)                        # apply_trace -> replace_pending
        assert tw.pending[tw.pending_i].arrival == nxt
        d = gen.send(None)
        try:
            while True:
                d = gen.send(("MDD", "PECT"))
        except StopIteration:
            pass
        return all(j.state == EXITED for j in tw.jobs.values()), [tw.jobs[j].arrival for j in past]
    done, arrivals = branch.fork_map(child, [0], max_parallel=1)[0]
    assert done and arrivals == [nxt] * len(past)


@fork_only
def test_planner_ensemble_runs_under_arrival_noise():
    """The noise-aware ensemble planner redraws E3 arrivals at every re-plan (it crashed under E3 / MIX before the
    clip fix when a redrawn arrival fell before the re-plan's current tick)."""
    pairs = ["MDD-PECT", "ATC-PTECT"]
    slot, warmup, window = 900.0, 1800.0, 3 * 900.0
    for seed, spec in ((22, "E3=severe"), (23, "mix:severe")):
        res = plan_episode(_twin(seed, spec, warmup=warmup, cap=window), pairs, slot, warmup, window,
                           max_parallel=2, ensemble=3, rush_pool=_pool(seed))
        assert len(res["schedule"]) == 3 and res["real"] >= 0.0


@fork_only
def test_pause_and_belief_transform_are_identity_without_noise():
    """Noise off: a child that pauses, applies the belief transform with the true future and plays on equals the
    uninterrupted run."""
    ref = _twin(19, None, warmup=1800.0, cap=5400.0)
    _play(ref, ["MDD-PECT", "ATC-PTECT"])
    for spec in (None, UncertaintyConfig.zero()):
        tw = _twin(19, spec, warmup=1800.0, cap=5400.0)
        tw.pause_at(3000.0)
        gen = tw.agent_decisions()
        t0, dec = None, gen.send(None)
        while dec.kind != "pause":
            t0 = t0 or tw.now
            dec = gen.send(tuple(["MDD-PECT", "ATC-PTECT"][int((tw.now - t0) // 900.0) % 2].split("-")))

        def child(_):
            fut = None if tw.unc is None else tw.unc.plan_view(3000.0, {j.id for j in tw.order})
            belief_transform(tw, fut)
            d = gen.send(None)
            try:
                while True:
                    d = gen.send(tuple(["MDD-PECT", "ATC-PTECT"][int((tw.now - t0) // 900.0) % 2].split("-")))
            except StopIteration:
                pass
            return _fp(tw)
        assert branch.fork_map(child, [0], max_parallel=1)[0] == _fp(ref)


@fork_only
def test_planner_v2_equals_a_replay_planner_without_noise():
    pairs = ["MDD-PECT", "ATC-PTECT", "SRT-PECT"]
    slot, warmup, window = 900.0, 1800.0, 4 * 900.0

    def replay(prefix):
        tw = _twin(20, None, warmup=warmup, cap=window)
        gen = tw.agent_decisions()
        t0 = None
        try:
            dec = gen.send(None)
            while True:
                t0 = t0 or tw.now
                k = int((tw.now - t0) // slot)
                dec = gen.send(tuple(prefix[min(k, len(prefix) - 1)].split("-")))
        except StopIteration:
            pass
        return window_tardiness(tw, t0)
    prefix = []
    for k in range(4):
        tails = [replay(prefix + [p]) for p in pairs]
        prefix.append(pairs[int(np.argmin(tails))])
    for spec in (None, UncertaintyConfig.zero()):
        res = plan_episode(_twin(20, spec, warmup=warmup, cap=window), pairs, slot, warmup, window, max_parallel=3)
        assert res["schedule"] == prefix and res["real"] == replay(prefix)


@fork_only
def test_planner_v2_runs_under_noise_with_belief_and_ensemble():
    pairs = ["MDD-PECT", "ATC-PTECT"]
    slot, warmup, window = 900.0, 1800.0, 2 * 900.0
    for spec, kw in (("mix:realistic", {}), ("E2d=severe,O1=realistic,info=I1", {}),
                     ("E1=realistic,E2d=realistic", {"ensemble": 2}), ("O1=severe", {"belief": False})):
        res = plan_episode(_twin(21, spec, warmup=warmup, cap=window), pairs, slot, warmup, window, max_parallel=2,
                           **kw)
        assert len(res["schedule"]) == 2 and res["real"] >= 0.0


@fork_only
def test_belief_transform_uses_the_reported_state():
    tw = _twin(22, "O1=severe,E1=realistic", warmup=1800.0, cap=3600.0)
    tw.pause_at(2700.0)
    gen = tw.agent_decisions()
    dec = gen.send(None)
    while dec.kind != "pause":
        dec = gen.send(("MDD", "PECT"))
    core = tw._vcore
    core.sync()
    expect = {}
    for m in tw.machines:
        rp = core.proc[m.id]
        if not m.idle and rp is not None and rp[0] == m.job:
            expect[m.id] = core.estimate(m.job, tw.jobs[m.job].cur_op, m.id, tw.jobs[m.job].ops[tw.jobs[m.job].cur_op][m.id])

    def child(_):
        stats = belief_transform(tw, tw.unc.plan_view(2700.0, {j.id for j in tw.order}))
        out = {mid: tw.mach[mid].dur for mid in expect if not tw.mach[mid].idle}
        return stats, out, tw.unc is None
    stats, durs, exact = branch.fork_map(child, [0], max_parallel=1)[0]
    assert exact
    for mid, d in durs.items():
        assert d == expect[mid]


# ── Plumbing ─────────────────────────────────────────────────────────────────────────────────────────────────

def test_lookahead_reads_the_plan():
    from des_twin.lookahead import lookahead_features
    tw = _twin(23, "E3=severe", warmup=1800.0, cap=3600.0)
    ref = Twin(_floor(), _plan(23), TwinConfig(rule="MDD_PECT", agv_count=3, warmup_seconds=1800.0,
                                               episode_duration_seconds=3600.0))
    g1, g2 = tw.agent_decisions(), ref.agent_decisions()
    g1.send(None)
    g2.send(None)
    # at the first decision both are at the warm-up end; the noisy twin's look-ahead equals the plan's
    tw.now = ref.now = 1800.0
    plan_feat = lookahead_features(ref)
    assert np.allclose(lookahead_features(tw), plan_feat, atol=0.05)


def test_twin_env_runs_an_episode_with_the_uncertainty_block():
    from env_wrappers.twin_env import TwinSchedulingEnv
    from scenarios.uncertainty import spec_from_arg, with_uncertainty

    def gen(seed):
        return _scenario(seed % 3, horizon=2 * 3600.0, warmup=600.0, cap=1800.0)
    g = with_uncertainty(gen, spec_from_arg("E2d=realistic,O1=realistic,E4n=mild"))
    sc = g(1)
    assert sc["uncertainty"]["noiseSeed"] == sc["seed"] and "rushPool" in sc["uncertainty"]
    env = TwinSchedulingEnv(_floor(True), scenario_generator=g, seed_rng=np.random.default_rng(0), obs_caps=(15, 128))
    obs = env.reset()
    assert env._twin.unc is not None and env._twin.observed is not None
    done, steps = False, 0
    while not done and steps < 3000:
        obs, r, done, info = env.step(np.array([1, 0]))
        steps += 1
    assert done and info["episode"]["length"] > 0
    with pytest.raises(ValueError):
        spec_from_arg("E9=mild")
    dr = spec_from_arg("dr:I1")
    assert dr(5).info == "I1" and dr(5) == dr(5)
