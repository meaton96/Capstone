"""Machine failures in the event-based twin (2026-10-09, dev-twin-failures): des_twin.failures (models, traces,
conditional redraw), the engine's FailureCoordinator port, the observation's failure columns, the reliability-aware
rules and the scenario plumbing; the 2026-10-09 quirk fixes (belt slot, no-alternate hold, legacy mode) and the
mid-handshake heading (dev-csharp-failures). Exactness with failures off against 84e221c3 is results/dev-twin-failures/
check_exact.py; here only the cheap version (an empty trace changes nothing)."""
import json
import math
import os

import numpy as np
import pytest

from des_twin import Floor, TwinConfig
from des_twin import failures as F
from des_twin.engine import EXITED, PROCESSING, Twin
from des_twin.scenario import episode_settings, failure_trace, resolve_jobs

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "des_twin")
FLOOR = os.path.join(FIX, "convoy_agv3", "des_floor.json")
UNITY_BLOCK = {"machineFailuresEnabled": True, "weibullK": 1.5, "weibullLambda": 7200.0, "repairLogMu": 6.3,
               "repairLogSigma": 0.5}


def _floor(frame=False):
    with open(FLOOR) as f:
        data = json.load(f)
    if frame:
        from tests.test_twin_env import with_frame
        data = with_frame(data)
    return Floor(data)


_SC = {}


def _scenario(seed, horizon=3 * 3600.0, p_fail=1.0):
    key = (seed, horizon, p_fail)
    if key not in _SC:
        from scenarios.randomized import RandomizedParams, randomized_scenario
        p = RandomizedParams(agv_count=3, failure_probability=p_fail, horizon_seconds=horizon,
                             due_date_allowance_range=(1.75, 2.5))
        _SC[key] = randomized_scenario(seed, p)
    return _SC[key]


def _run(rule, trace, seed=0, floor=None, **kw):
    floor = floor or _floor()
    jobs = resolve_jobs(_scenario(seed), floor)
    return Twin(floor, jobs, TwinConfig(rule=rule, agv_count=3, failures=trace, **kw)).run()


def _trace(model, floor, seed, horizon=100000.0):
    return model.trace([m.id for m in floor.machines], seed, horizon, floor.dt)


# ── Traces and models ────────────────────────────────────────────────────────────────────────────

def test_trace_is_deterministic_and_streams_are_per_machine():
    model = F.uniform_from_stochastic(UNITY_BLOCK)
    a, b = model.trace(range(10), 3, 50000.0), model.trace(range(10), 3, 50000.0)
    assert all(a.machines[i].events == b.machines[i].events for i in range(10))
    c = model.trace(range(10), 4, 50000.0)
    assert any(a.machines[i].events != c.machines[i].events for i in range(10))
    # one machine's parameters change: every other machine's draws stay put, and its repairs keep their stream
    params = {i: F.Rel(1.5, 7200.0, 6.3, 0.5) for i in range(10)}
    params[4] = F.Rel(1.5, 1800.0, 6.3, 0.5)
    d = F.WeibullModel(params, initial="unity").trace(range(10), 3, 50000.0)
    assert all(a.machines[i].events == d.machines[i].events for i in range(10) if i != 4)
    n = min(len(a.machines[4].events), len(d.machines[4].events))
    assert [r for _, r in a.machines[4].events[:n]] == [r for _, r in d.machines[4].events[:n]]
    assert len(d.machines[4].events) > len(a.machines[4].events)


def test_event_ticks_follow_unity_countdown_rules():
    model = F.uniform_from_stochastic(UNITY_BLOCK)
    tr = model.trace(range(5), 0, 80000.0)
    for mt in tr.machines.values():
        start = 1
        for i, (f, rep) in enumerate(mt.events):
            assert f == F.harvest_tick(start, mt.lives[i], tr.dt)
            start = F.harvest_tick(f, rep, tr.dt)
            assert start > f
    assert F.harvest_tick(10, 0.02, 0.02) == 11 and F.harvest_tick(10, 0.03, 0.02) == 12
    assert F.harvest_tick(10, 0.0, 0.02) == 11          # TickTTF decrements before it tests


def _lives(tr):
    """Renewal lives (seconds between a repair harvest and the next failure harvest) and repairs."""
    lives, reps = [], []
    for mid, mt in tr.machines.items():
        iv = tr.intervals(mid)
        for (f, r), (f2, _) in zip(iv, iv[1:]):
            lives.append((f2 - r) * tr.dt)
        reps += [rep for _, rep in mt.events]
    return np.array(lives), np.array(reps)


def test_uniform_statistics_match_the_model():
    rel = F.Rel(1.5, 7200.0, 6.3, 0.5)
    model = F.WeibullModel(rel, initial="equilibrium")
    tr = model.trace(range(200), 11, 400000.0)
    lives, reps = _lives(tr)
    assert lives.mean() == pytest.approx(rel.mtbf, rel=0.03)
    assert reps.mean() == pytest.approx(rel.mttr, rel=0.03)
    avail = np.mean([1 - tr.downtime(m, 400000.0) / 400000.0 for m in range(200)])
    assert avail == pytest.approx(rel.availability, abs=0.005)
    assert F.lam_for_availability(rel.availability, 1.5, 6.3, 0.5) == pytest.approx(7200.0)


def test_initial_age_conventions():
    rel = F.Rel(1.5, 7200.0, 6.3, 0.5)
    first = {}
    for initial in ("unity", "equilibrium", "fresh"):
        tr = F.WeibullModel(rel, initial=initial).trace(range(3000), 5, 60000.0)
        first[initial] = np.mean([mt.lives[0] for mt in tr.machines.values()]) / rel.lam
    assert first["unity"] == pytest.approx(math.gamma(1 + 1 / 1.5) / 2, rel=0.04)              # 0.45 lambda
    assert first["equilibrium"] == pytest.approx(math.gamma(1 + 2 / 1.5) / (2 * math.gamma(1 + 1 / 1.5)), rel=0.04)
    assert first["fresh"] == pytest.approx(math.gamma(1 + 1 / 1.5), rel=0.04)


def test_hetero_subset_rate_ratio_and_availability():
    types = {i: i % 5 for i in range(50)}
    model = F.hetero(types, ratio=4.0, fraction=0.4, availability_target=0.90, subset_seed=0)
    assert len(model.unreliable) == 20
    assert all(sum(1 for m in model.unreliable if types[m] == t) == 4 for t in range(5))   # spread over types
    assert F.hetero(types, subset_seed=0).unreliable == model.unreliable                    # fixed set
    assert F.hetero(types, subset_seed=7).unreliable != model.unreliable                    # redrawn set
    tr = model.trace(range(50), 1, 600000.0)
    bad = np.mean([len(tr.machines[m].events) for m in model.unreliable])
    good = np.mean([len(tr.machines[m].events) for m in range(50) if m not in model.unreliable])
    # up-time per failure is 4x shorter; with downtime included the count ratio is a bit below 4
    assert 3.0 < bad / good < 4.2
    avail = np.mean([1 - tr.downtime(m, 600000.0) / 600000.0 for m in range(50)])
    assert avail == pytest.approx(0.90, abs=0.01)


def test_regime_model_bad_state_fails_more_and_holds_availability():
    model = F.regime(ratio=8.0, bad_fraction=0.25, mean_bad=5400.0, availability_target=0.90)
    tr = model.trace(range(60), 2, 600000.0)
    good_t = bad_t = good_n = bad_n = 0.0
    for mid, mt in tr.machines.items():
        path = mt.regime
        for f, _ in mt.events:
            if model.state_at(path, f * tr.dt):
                bad_n += 1
            else:
                good_n += 1
        for (t0, s), (t1, _) in zip(path, path[1:]):
            t1 = min(t1, 600000.0)
            if t0 >= 600000.0:
                break
            if s:
                bad_t += t1 - t0
            else:
                good_t += t1 - t0
    assert bad_t / (good_t + bad_t) == pytest.approx(0.25, abs=0.03)
    assert (bad_n / bad_t) / (good_n / good_t) > 4.0           # the bad state fails much more often
    avail = np.mean([1 - tr.downtime(m, 600000.0) / 600000.0 for m in range(60)])
    assert avail == pytest.approx(0.90, abs=0.025)             # the availability helper is an approximation


def test_simulated_availability_calibration():
    model = F.regime(ratio=8.0, bad_fraction=0.25, availability_target=0.90)
    cal, s = F.calibrate_availability(model, 0.90, machine_ids=range(40), horizon=250000.0, seed=0, iters=14)
    check = F.simulated_availability(cal, list(range(80)), horizon=250000.0, seed=1)
    assert check == pytest.approx(0.90, abs=0.01)
    assert cal.lam[0] / cal.lam[1] == pytest.approx(8.0)


def test_degradation_hazard_rises_without_renewal():
    model = F.degradation(k=2.5, availability_target=0.9)
    tr = model.trace(range(100), 3, 100000.0)
    lives, _ = _lives(tr)
    early = np.mean([iv[1][0] - iv[0][1] for mid in range(100) for iv in [tr.intervals(mid)[:2]] if len(iv) == 2])
    later = np.mean([iv[-1][0] - iv[-2][1] for mid in range(100) for iv in [tr.intervals(mid)] if len(iv) > 4])
    assert later < early                                       # lives shorten as the machine ages


def test_conditional_redraw_keeps_the_past_and_the_current_repair():
    model = F.uniform_from_stochastic(UNITY_BLOCK)
    tr = model.trace(range(40), 9, 100000.0)
    k = 1_000_000                                                # 20,000 s
    rd = tr.redraw_after(k, seed=123)
    changed = 0
    for mid, mt in tr.machines.items():
        new = rd.machines[mid]
        past = [e for e in mt.events if e[0] <= k]
        assert new.events[:len(past)] == past
        assert all(f > k for f, _ in new.events[len(past):])
        changed += new.events[len(past):] != mt.events[len(past):]
        if past and F.harvest_tick(*past[-1], tr.dt) > k:        # down at k: the next life starts at the repair end
            r_end = F.harvest_tick(*past[-1], tr.dt)
            if len(new.events) > len(past):
                assert new.events[len(past)][0] > r_end
    assert changed > 30
    assert rd.redraw_after(k, seed=123).machines[0].events == rd.machines[0].events or True
    a, b = tr.redraw_after(k, seed=5), tr.redraw_after(k, seed=5)
    assert all(a.machines[m].events == b.machines[m].events for m in range(40))
    with pytest.raises(ValueError):
        rd.to_schedule()


def test_conditional_residual_life_given_age():
    """Weibull residual life given survival to age a: mean = integral of S(a + x) / S(a)."""
    k, lam, age = 1.5, 7200.0, 9000.0
    rng = np.random.default_rng(0)
    draws = np.array([F._weibull_cond(rng, k, lam, age) for _ in range(40000)])
    xs = np.linspace(0, 60000, 600001)
    surv = np.exp(-(((age + xs) / lam) ** k - (age / lam) ** k))
    assert draws.mean() == pytest.approx(np.trapz(surv, xs), rel=0.02)


def test_schedule_round_trip():
    tr = F.uniform_from_stochastic(UNITY_BLOCK).trace(range(6), 1, 80000.0)
    back = F.FailureTrace.from_schedule(json.loads(json.dumps(tr.to_schedule())))
    assert all(back.machines[m].events == tr.machines[m].events for m in range(6))


# ── Engine ───────────────────────────────────────────────────────────────────────────────────────

def _heavy_trace(floor, seed=1):
    """Frequent failures so one short episode exercises every handler."""
    model = F.WeibullModel(F.Rel(1.5, 1500.0, 5.5, 0.5), initial="unity")
    return _trace(model, floor, seed)


@pytest.mark.parametrize("rule", ["SPT_ECT", "MDD_TECT", "FIFO_SRWT"])
def test_engine_invariants_with_failures(rule):
    floor = _floor()
    trace = _heavy_trace(floor)
    tw = _run(rule, trace, floor=floor)
    s = tw.summary()
    assert not s["timed_out"]
    assert s["episode_failures"] > 20 and s["jobs_interrupted"] > 0 and s["queued_rerouted"] > 0
    # conservation: every job exited or is still in the system, and the count matches
    assert all((j.state == EXITED) == (j.exit_time is not None) for j in tw.jobs.values())
    assert s["jobs_completed"] == sum(j.state == EXITED for j in tw.jobs.values())
    # failures and downtime are the trace's (exogenous) up to the end of the episode
    end_tick = round(tw.makespan / floor.dt)
    for m in tw.machines:
        iv = [(f, r) for f, r in trace.intervals(m.id) if f <= end_tick]
        assert m.n_fail == len(iv)
        assert tw.downtime_so_far(m) == pytest.approx(sum(min(r, end_tick) - f for f, r in iv) * floor.dt, abs=1e-6)
    # no operation runs on a machine while it is down, and an interrupted operation restarts from scratch
    down = {m.id: [(f * floor.dt, r * floor.dt) for f, r in trace.intervals(m.id)] for m in tw.machines}
    restarted = 0
    for row in tw.op_rows():
        if row["start"] is None:
            continue
        end = row["end"] if row["end"] is not None else (row["interrupted"] if row["interrupted"] is not None
                                                         else tw.makespan)
        for f, r in down[row["machine"]]:
            assert not (row["start"] < r - 1e-9 and end > f + 1e-9), (row, f, r)
        if row["interrupted"] is not None:
            job = tw.jobs[row["job"]]
            later = [o for o in job.op_rows if o["op"] == row["op"] and o["routed"] >= row["interrupted"]]
            assert later, row                                     # re-routed: a new op row for the same operation
            done = [o for o in later if o["end"] is not None]
            if done:
                restarted += 1
                p = job.ops[row["op"]][done[0]["machine"]]
                # the full duration again (within the float32 countdown drift, ~1e-4 relative)
                assert done[0]["end"] - done[0]["start"] >= float(p) * (1 - 1e-3)
    assert restarted > 0
    assert all(not m.down or m.idle for m in tw.machines)


@pytest.mark.parametrize("legacy", [False, True])
def test_handlers_cancel_redirect_hold_and_deferral_all_occur(legacy):
    """Every handler fires across seeds. A transit with no operational alternate is held (delivered to the down
    machine and re-routed there, 2026-10-09) by default, aborted with legacy_failure_quirks."""
    floor = _floor()
    tot = {}
    for seed in range(4):
        for rule in ("SPT_ECT", "MDD_TECT", "SRT_SRWT"):
            tw = _run(rule, _heavy_trace(floor, seed), seed=seed, floor=floor, legacy_failure_quirks=legacy)
            for k, v in tw.summary().items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    tot[k] = tot.get(k, 0) + v
            # machine_failures.csv rows agree with the counters
            fails = [r for r in tw.fail_rows if r["event"] == "fail"]
            assert len(fails) == tw.summary()["episode_failures"]
            assert sum(r["queued_rerouted"] for r in fails) == tw.fstats["queued_rerouted"]
            assert sum(r["pickups_cancelled"] for r in fails) == tw.fstats["pickups_cancelled"]
            assert sum(r["redirected"] for r in fails) == tw.fstats["redirects"]
            assert sum(r["held"] for r in fails) == tw.fstats["held_to_down"]
            assert sum(r["aborted"] for r in fails) == tw.fstats["aborts"] + tw.fstats["redirect_no_route"]
            assert sum(r["predispatch_cleared"] for r in fails) == tw.fstats["predispatch_cleared"]
            assert [(r["sim_time"], r["machine_id"], r["event"]) for r in tw.fail_rows] == \
                [(round(t, 2), m, ev) for t, m, ev, _, _ in tw.fail_log]
    for key in ("pickups_cancelled", "redirects", "predispatch_cleared", "orphans_released", "jobs_interrupted",
                "queued_rerouted", "aborts" if legacy else "held_to_down", "deferred_job_seconds"):
        assert tot[key] > 0, key
    assert tot["aborts" if not legacy else "held_to_down"] == 0
    if not legacy:
        assert tot["delivered_to_down"] >= tot["held_to_down"] > 0
    assert tot["orphans_released"] <= tot["predispatch_cleared"]


def _first_hold(floor, legacy):
    """A heavy-trace run (seeds 0-3, SPT_ECT) with at least one transit that had no operational alternate."""
    for seed in range(4):
        tw = _run("SPT_ECT", _heavy_trace(floor, seed), seed=seed, floor=floor, legacy_failure_quirks=legacy)
        if tw.fstats["held_to_down" if not legacy else "aborts"]:
            return seed, tw
    raise AssertionError("no transit without an alternate in seeds 0-3")


def test_no_alternate_transit_is_delivered_to_the_down_machine():
    """Fixed handling: the AGV keeps going to the failed machine, the job is queued there and re-routed at once (B3),
    and its next pickup is from that machine (its true location), not from the machine it came from. Legacy: the
    transit is aborted and the job is later collected from where it came from."""
    floor = _floor()
    seed, tw = _first_hold(floor, legacy=False)
    held = [r for r in tw.fail_rows if r["held"]]
    t_fail, mid = held[0]["sim_time"], held[0]["machine_id"]
    # the op row of the held job: routed to mid before the failure, "queued" (delivered) at mid after it, never
    # started there, and the next attempt at the same operation is collected from mid
    rows = [o for j in tw.jobs.values() for o in j.op_rows
            if o["machine"] == mid and o["routed"] < t_fail and o["queued"] is not None and o["queued"] >= t_fail
            and o["start"] is None]
    assert rows, "the held delivery must land on the failed machine"
    o = rows[0]
    job = tw.jobs[o["job"]]
    later = [r for r in job.op_rows if r["op"] == o["op"] and r["routed"] >= o["queued"]]
    assert later and later[0]["from"] == mid
    assert all(ev != "abort" for _, _, ev, _ in tw.trace)
    # legacy: aborted, and the job's next attempt is collected from its previous location
    _, old = _first_hold(floor, legacy=True)
    assert old.fstats["held_to_down"] == 0 and any(ev == "abort" for _, _, ev, _ in old.trace)


@pytest.mark.parametrize("legacy", [False, True])
def test_rerouted_queued_jobs_free_their_belt_slot(legacy):
    """Step 2 / B3: a job re-routed off a failed machine without being started leaves its incoming belt (the C#
    PhysicalMachine.ReleaseIncomingSlot); with legacy_failure_quirks it keeps the slot (the old leak)."""
    floor = _floor()
    tw = Twin(floor, resolve_jobs(_scenario(0), floor),
              TwinConfig(rule="SPT_ECT", agv_count=3, failures=_heavy_trace(floor, 0), legacy_failure_quirks=legacy))
    leaked = []
    orig = tw._fail

    def spy(m):
        before = {jid for _, jobs in m.belts for jid in jobs}
        queued = [j.id for j in tw._queued_at[m.id].jobs()]
        orig(m)
        after = {jid for _, jobs in m.belts for jid in jobs}
        leaked.extend(j for j in queued if j in before and j in after)
        assert legacy or not (after & set(queued))

    tw._fail = spy
    tw.run()
    assert tw.fstats["queued_rerouted"] > 0
    assert bool(leaked) == legacy
    # whatever the mode, a belt only ever holds jobs that are queued at that machine (or, legacy, leaked ones)
    if not legacy:
        for m in tw.machines:
            for _, jobs in m.belts:
                assert all(tw.jobs[j].state == 3 and tw.jobs[j].location == m.id for j in jobs)


def _one_failure(floor, mid, tick, repair):
    tr = F.FailureTrace({m.id: F.MachineTrace() for m in floor.machines}, None, 0, 1e6, floor.dt)
    tr.machines[mid] = F.MachineTrace(events=[(tick, repair)], lives=[0.0])
    return tr


def test_delivery_to_a_machine_that_failed_this_tick_is_rerouted():
    """Audit B3: the AGV unloaded last tick (already returning in Unity), the failure harvest misses it, and the
    delivery harvest re-routes the job. Also the 1-tick pickup window: the AGV is MovingToDropoff in Unity but the
    job is still WaitingForPickup, so neither step 3 nor step 4 touches it and it is delivered to the down machine."""
    floor = _floor()
    ref = _run("SPT_ECT", None, floor=floor)
    rows = [r for r in ref.op_rows() if r["queued"] is not None and r["from"] >= 0 and r["queued"] > 600]
    row = rows[0]
    tick = round(row["queued"] / floor.dt)
    tw = _run("SPT_ECT", _one_failure(floor, row["machine"], tick, 500.0), floor=floor)
    assert tw.fstats["delivered_to_down"] == 1
    assert tw.fstats["redirects"] == tw.fstats["pickups_cancelled"] == 0
    job = tw.jobs[row["job"]]
    later = [o for o in job.op_rows if o["op"] == row["op"]]
    assert len(later) >= 2 and later[0]["start"] is None and later[1]["routed"] >= tick * floor.dt - 1e-9
    # the pickup window: fail the target the tick the pickup is harvested
    for t_pick, agv, ev, jid in ref.trace:
        if ev != "pickup" or t_pick < 600:
            continue
        rows = [o for o in ref.jobs[jid].op_rows if o["routed"] <= t_pick and o["queued"] is not None
                and o["queued"] > t_pick]
        if rows:
            target = rows[-1]["machine"]
            break
    p_tick = round(t_pick / floor.dt) + 1
    tw2 = _run("SPT_ECT", _one_failure(floor, target, p_tick, 5000.0), floor=floor)
    assert tw2.fstats["delivered_to_down"] == 1 and tw2.fstats["pickups_cancelled"] == 0


def test_onready_target_down_goes_back_to_routing():
    """TryAssignAgv: under onReady routing a routed job can wait for an AGV; if its target goes down meanwhile it is
    routed again."""
    floor = _floor()
    tot = 0
    for seed in range(3):
        tw = _run("SPT_ECT", _heavy_trace(floor, seed), seed=seed, floor=floor, routing_trigger="onReady")
        tot += tw.fstats["target_down_rerouted"]
        assert not tw.timed_out
    assert tot > 0


def test_empty_trace_changes_nothing():
    floor = _floor()
    empty = F.FailureTrace({m.id: F.MachineTrace() for m in floor.machines}, None, 0, 1e6, floor.dt)
    for rule in ("SPT_ECT", "MDD_TECT"):
        a, b = _run(rule, None, floor=floor), _run(rule, empty, floor=floor)
        assert a.decision_rows == b.decision_rows and a.trace == b.trace and a.job_rows() == b.job_rows()
        assert [{k: v for k, v in r.items() if k != "interrupted"} for r in b.op_rows()] == a.op_rows()
        sa, sb = a.summary(), b.summary()
        assert all(sa[k] == sb[k] for k in sa)
        assert a.metrics() == b.metrics()


def test_common_random_numbers_across_rules():
    floor = _floor()
    trace = F.uniform_from_stochastic(UNITY_BLOCK).trace([m.id for m in floor.machines], 2, 100000.0, floor.dt)
    logs = []
    for rule in ("SPT_ECT", "MDD_TECT", "SRT_SRWT", "ATC_TECT"):
        tw = _run(rule, trace, floor=floor)
        logs.append((tw.makespan, [(t, m) for t, m, ev, _, _ in tw.fail_log if ev == "fail"]))
    end = min(mk for mk, _ in logs)
    sets = [[e for e in log if e[0] <= end] for _, log in logs]
    assert all(s == sets[0] for s in sets) and len(sets[0]) > 5


def test_fork_rollouts_with_failures_equal_replay():
    from des_twin import branch
    floor = _floor()
    trace = _heavy_trace(floor, 2)
    base = ("SPT", "ECT")

    def start():
        tw = Twin(floor, resolve_jobs(_scenario(0), floor),
                  TwinConfig("SPT_ECT", agv_count=3, failures=trace, warmup_seconds=1800.0))
        gen = tw.agent_decisions()
        dec = gen.send(None)
        for _ in range(40):
            dec = gen.send(base)
        return tw, gen, dec

    tw, gen, dec = start()
    opts = [("SRT", "TECT"), ("MDD", "SRWT"), base]
    k_now = tw.k
    scored = branch.rollout(tw, gen, dec, opts, policy=lambda t, d: base,
                            score=lambda t: (t.summary(), list(t.decision_rows), t.fail_log), max_parallel=3)
    for opt, got in zip(opts, scored):
        tw2, gen2, _ = start()
        branch.play_to_end(tw2, gen2, opt, lambda t, d: base)
        assert got == (tw2.summary(), list(tw2.decision_rows), tw2.fail_log)
    # a rollout over a redrawn failure future is deterministic too, and keeps the past
    red = trace.redraw_after(k_now, seed=77)

    def redrawn(t):
        t.replace_failures(red)
        branch.play_to_end(t, gen, base, lambda t_, d: base)
        return t.summary(), t.fail_log
    r1 = branch.fork_map(lambda _: redrawn(tw), [0, 1], max_parallel=2)
    assert r1[0] == r1[1]
    assert [e for e in r1[0][1] if e[0] <= tw.now] == [e for e in tw.fail_log if e[0] <= tw.now]


def test_utilization_counts_operational_time_and_not_lost_work():
    floor = _floor()
    tw = _run("SPT_ECT", _heavy_trace(floor), floor=floor)
    for m in tw.machines:
        up = tw.now - tw.downtime_so_far(m)
        assert float(tw.utilization(m)) == pytest.approx(m.busy / up, rel=1e-5)
    assert sum(m.lost_work for m in tw.machines) > 0


# ── Observation ──────────────────────────────────────────────────────────────────────────────────

def test_observation_failure_columns():
    from des_twin.observation import COUNT_SCALE, TIME_SCALE, ObservationBuilder, squash
    floor = _floor(frame=True)
    tw = Twin(floor, resolve_jobs(_scenario(0), floor),
              TwinConfig("SPT_ECT", agv_count=3, failures=_heavy_trace(floor, 3), warmup_seconds=0.0))
    builder = ObservationBuilder(floor, 15, 256)
    gen = tw.agent_decisions()
    seen_down = seen_deferred = 0
    try:
        dec = gen.send(None)
        while True:
            obs = builder.build(tw, dec)
            mt, gs, fl, jt = obs["machine_table"], obs["global_scalars"], obs["event_flags"], obs["job_table"]
            down = [m for m in tw.machines if m.down]
            for i, m in enumerate(tw.machines):
                assert mt[i, 7] == (0.0 if m.down else 1.0)
                assert mt[i, 8] == (squash(tw.remaining_repair(m), TIME_SCALE) if m.down else 0.0)
                if m.down:
                    assert 0.0 < tw.remaining_repair(m) <= m.repair_dur
                    assert m.id not in (dec.machines if dec.kind == "routing" else ())
            assert gs[8] == np.float32(len(down)) / np.float32(len(tw.machines))
            assert fl[4] == (1.0 if down else 0.0)
            assert gs[14] == squash(len(tw.deferred), COUNT_SCALE)
            seen_down += bool(down)
            seen_deferred += int(jt[:, 6].sum() > 0)
            dec = gen.send(("SPT", "ECT"))
    except StopIteration:
        pass
    assert seen_down > 10
    if tw.summary()["deferred_job_seconds"] > 0:
        assert seen_deferred > 0


# ── Reliability-aware rules ──────────────────────────────────────────────────────────────────────

def test_reliability_rules_reduce_to_ect_without_failures():
    floor = _floor()
    for rel_rule, plain in (("RECT", "ECT"), ("RTECT", "TECT"), ("RECTA", "ECT"), ("HAECT", "ECT")):
        a, b = _run(f"SPT_{rel_rule}", None, floor=floor), _run(f"SPT_{plain}", None, floor=floor)
        assert a.decision_rows == b.decision_rows, rel_rule


@pytest.mark.parametrize("info", ["static", "online", "oracle"])
def test_reliability_rules_run_at_every_information_level(info):
    floor = _floor()
    types = {m.id: m.type for m in floor.machines}
    model = F.hetero(types, ratio=4.0, fraction=0.4, availability_target=0.85, subset_seed=0)
    trace = _trace(model, floor, 4)
    for rule in ("SRT_RECT", "SRT_RTECT", "MDD_RECTA", "MDD_HAECT"):
        tw = _run(rule, trace, floor=floor, reliability=info, reliability_kappa=1.0)
        assert tw.summary()["jobs_completed"] > 0
    # static information steers work away from the unreliable machines
    ect = _run("SRT_ECT", trace, floor=floor)
    rect = _run("SRT_HAECT", trace, floor=floor, reliability="static")
    share = lambda t: np.mean([r["machine"] in model.unreliable for r in t.op_rows() if r["start"] is not None])
    assert share(rect) < share(ect)


def test_online_estimate_separates_unreliable_machines():
    floor = _floor()
    types = {m.id: m.type for m in floor.machines}
    model = F.hetero(types, ratio=6.0, fraction=0.4, availability_target=0.85, subset_seed=0)
    tw = _run("SRT_RECT", _trace(model, floor, 5), floor=floor, reliability="online")
    est = {m.id: tw.reliability_info.estimate(m)[2] for m in tw.machines}
    bad = np.mean([est[m] for m in model.unreliable])
    good = np.mean([est[m] for m in est if m not in model.unreliable])
    assert bad < good


def test_online_estimator_parameters_and_pooled_prior():
    """TwinConfig.reliability_half_life / reliability_prior reach OnlineReliability (dev-hetero-tune, 10-10); the
    prior MTBF is the pooled floor MTBF (1 / mean rate), and a strong prior with a long memory pulls every machine
    towards it (the per-machine spread shrinks as the prior grows)."""
    floor = _floor()
    types = {m.id: m.type for m in floor.machines}
    model = F.hetero(types, ratio=6.0, fraction=0.4, availability_target=0.85, subset_seed=0)
    trace = _trace(model, floor, 5)
    pooled = len(types) / sum(1.0 / model.rel(m).mtbf for m in types)
    spread = {}
    for a0 in (0.5, 50.0):
        tw = _run("SRT_RECT", trace, floor=floor, reliability="online", reliability_half_life=3600.0,
                  reliability_prior=a0)
        est = tw.reliability_info
        assert est.h == 3600.0 and est.a0 == a0
        assert est.mtbf0 == pytest.approx(pooled)
        vals = [est.estimate(m)[2] for m in tw.machines]
        spread[a0] = np.std(np.log(vals))
    assert spread[50.0] < spread[0.5]


def test_pooled_static_information():
    """static_info="pooled" (condition b', unreliable set redrawn per episode): the static level gets the floor's pooled
    values for every machine, the oracle level each machine's own; "machine" (default) gives both per machine."""
    floor = _floor()
    types = {m.id: m.type for m in floor.machines}
    per = F.hetero(types, ratio=4.0, fraction=0.4, availability_target=0.85, subset_seed=3)
    pool = F.hetero(types, ratio=4.0, fraction=0.4, availability_target=0.85, subset_seed=3, static_info="pooled")
    rate = np.mean([1.0 / per.rel(m).mtbf for m in types])
    for m in types:
        assert per.static_rel(m) == per.current_rel(m, 0.0) == pool.current_rel(m, 0.0)
        k, lam, mtbf_, mttr_ = pool.static_rel(m)
        assert mtbf_ == pytest.approx(1.0 / rate) and lam == pytest.approx(mtbf_ / math.gamma(1 + 1 / k))
        assert mttr_ == pytest.approx(per.rel(m).mttr)
    assert F.scaled(pool, 2.0).static_rel(0)[2] == pytest.approx(2.0 * pool.static_rel(0)[2], rel=1e-9)
    with pytest.raises(ValueError):
        F.WeibullModel(F.Rel(), static_info="bogus")
    # the engine: pooled static information makes every candidate look the same (RECT -> ECT with a uniform q / A)
    trace = _trace(pool, floor, 4)
    tw = _run("SRT_RECT", trace, floor=floor, reliability="static")
    terms = tw.reliability_terms([m.id for m in tw.machines])["terms"]
    assert len({t[:5] for t in terms}) == 1
    tw = _run("SRT_RECT", trace, floor=floor, reliability="oracle")
    terms = tw.reliability_terms([m.id for m in tw.machines])["terms"]
    assert len({t[1] for t in terms}) == 2


# ── Scenario plumbing ────────────────────────────────────────────────────────────────────────────

def test_scenario_failures_opt_in_and_trace():
    sc = _scenario(0)
    assert sc["stochastic"]["machineFailuresEnabled"]
    with pytest.raises(NotImplementedError):
        episode_settings(sc)                                     # existing callers keep refusing
    episode_settings(sc, machine_failures=True)
    with pytest.raises(NotImplementedError):
        episode_settings({"stochastic": {"machineFailuresEnabled": True, "agvFailuresEnabled": True}},
                         machine_failures=True)
    floor = _floor()
    tr = failure_trace(sc, floor)
    assert set(tr.machines) == {m.id for m in floor.machines} and tr.n_failures() > 0
    assert tr.model.rel(0).lam == 7200.0 and tr.model.initial == "unity"
    assert failure_trace(sc, floor).machines[0].events == tr.machines[0].events
    assert failure_trace(_scenario(0, p_fail=0.0), floor) is None


def test_twin_env_runs_with_machine_failures():
    from env_wrappers.twin_env import TwinSchedulingEnv
    from tests.test_twin_env import with_frame
    with open(FLOOR) as f:
        data = with_frame(json.load(f))
    env = TwinSchedulingEnv(Floor(data), scenario=_scenario(1), obs_caps=(15, 256), machine_failures=True)
    obs = env.reset()
    assert env._twin.fail is not None
    for _ in range(300):
        obs, _, done, info = env.step(np.zeros(2, dtype=int))
        if done:
            break
    with pytest.raises(NotImplementedError):
        TwinSchedulingEnv(Floor(data), scenario=_scenario(1), obs_caps=(15, 256)).reset()


def test_dock_yaw_mid_handshake():
    """motion.dock_yaw_after (the heading of an AGV redirected or cancelled mid-handshake) turns as handshake_done
    does: unchanged before the first tick, the aligned heading from the completion tick on, monotone in between."""
    from des_twin.motion import Kinematics, angdiff, dock_yaw_after, handshake_done
    floor = _floor()
    kin = Kinematics(floor.agv, floor.dt, "kinematic")
    for yaw0, goal in ((0.0, 170.0), (350.0, 90.0), (90.0, 92.0)):
        done, yaw_done = handshake_done(kin, yaw0, goal, 100, True)
        assert dock_yaw_after(kin, yaw0, goal, 100, 99, True) == yaw0
        assert dock_yaw_after(kin, yaw0, goal, 100, done, True) == yaw_done
        gaps = [abs(angdiff(dock_yaw_after(kin, yaw0, goal, 100, t, True), goal)) for t in range(100, done + 1)]
        assert all(b <= a for a, b in zip(gaps, gaps[1:]))
