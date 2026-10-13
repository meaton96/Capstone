"""
@file test_ptime_rules.py
@brief Tests for the processing-time-weighted machine rules PECT / PTECT of des_twin.rules (dev-pect-rule, 10-10):
       alpha 0 is ECT / TECT, the exp shape reproduces the blind reliability rules RECT / RTECT exactly, the shapes
       behave as documented, and the engine passes TwinConfig.ptime_* through.
"""
import json
import math
import os
import random

import pytest

from des_twin import Floor, TwinConfig
from des_twin import failures as F
from des_twin.engine import Twin
from des_twin.rules import (PTIME_RULES, f32, parse_rule, ptime_scores, reliability_scores, select_machine)
from des_twin.scenario import resolve_jobs

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "des_twin")
FLOOR = os.path.join(FIX, "convoy_agv3", "des_floor.json")


def _pt(shape="lin", alpha=1.0, tau=5000.0, reroute=0.0, avail=1.0):
    return {"alpha": alpha, "shape": shape, "tau": tau, "reroute": reroute, "avail": avail}


def _random_cands(rng, n):
    times = [f32(rng.uniform(60.0, 900.0)) for _ in range(n)]
    loads = [f32(rng.choice([0.0, rng.uniform(0.0, 3000.0)])) for _ in range(n)]
    travel = [f32(rng.uniform(0.0, 60.0)) for _ in range(n)]
    return times, loads, travel


def test_parse_rule_accepts_ptime_halves():
    for m in PTIME_RULES:
        assert parse_rule(f"MDD_{m}") == ("MDD", m)


def test_alpha_zero_is_ect_and_tect_bit_for_bit():
    rng = random.Random(0)
    for _ in range(500):
        n = rng.randint(2, 6)
        times, loads, travel = _random_cands(rng, n)
        for shape in ("lin", "quad", "ratio", "exp"):
            pect = ptime_scores("PECT", times, loads, travel, _pt(shape, 0.0, reroute=900.0))
            ptect = ptime_scores("PTECT", times, loads, travel, _pt(shape, 0.0, reroute=900.0))
            assert pect == [f32(q + p) for q, p in zip(loads, times)]
            assert ptect == [f32(max(tr, q) + p) for tr, q, p in zip(travel, loads, times)]


def test_exp_shape_reproduces_blind_reliability_rules():
    """With every candidate at the same (A, MTBF) - the blind form, reliability_mix 0 - RECT / RTECT are PECT / PTECT
    with shape exp, tau = MTBF, R = the re-route cost, alpha = kappa and the queue divided by A."""
    rng = random.Random(1)
    mtbf, a = 5553.7, 0.9
    for _ in range(500):
        n = rng.randint(2, 6)
        times, loads, travel = _random_cands(rng, n)
        kappa, reroute = rng.choice([1.0, 4.0, 16.0, 64.0]), rng.choice([0.0, 90.0, 2880.0])
        rel = {"terms": [(a, mtbf, 600.0, 1.5, 6000.0, 0.0)] * n, "kappa": kappa, "reroute": reroute,
               "median_A": a}
        pt = _pt("exp", kappa, tau=mtbf, reroute=reroute, avail=a)
        assert ptime_scores("PECT", times, loads, travel, pt) == reliability_scores("RECT", times, loads, travel, rel)
        assert ptime_scores("PTECT", times, loads, travel, pt) == reliability_scores("RTECT", times, loads, travel, rel)


def test_shapes():
    times, loads, travel = [f32(100.0), f32(200.0)], [f32(0.0), f32(0.0)], [f32(0.0), f32(0.0)]
    assert ptime_scores("PECT", times, loads, travel, _pt("lin", 2.0)) == [f32(300.0), f32(600.0)]
    # quad: p^2 / p_mean, p_mean 150
    assert ptime_scores("PECT", times, loads, travel, _pt("quad", 1.0)) == [f32(100 + 1e4 / 150), f32(200 + 4e4 / 150)]
    # ratio: zero on the fastest candidate
    assert ptime_scores("PECT", times, loads, travel, _pt("ratio", 3.0)) == [f32(100.0), f32(200.0 + 3 * 200.0)]
    e = ptime_scores("PECT", times, loads, travel, _pt("exp", 1.0, tau=1000.0, reroute=50.0))
    assert e[0] == f32(100 + (1 - math.exp(-0.1)) * 100.0)
    with pytest.raises(ValueError):
        ptime_scores("PECT", times, loads, travel, _pt("cubic", 1.0))


def test_scale_free_shapes_rank_the_same_after_rescaling_time():
    """lin / quad / ratio have a dimensionless alpha: scaling every time by c keeps the choice."""
    rng = random.Random(2)
    for _ in range(300):
        n = rng.randint(2, 5)
        times, loads, travel = _random_cands(rng, n)
        c = rng.choice([0.5, 2.0, 3.0])
        for shape in ("lin", "quad", "ratio"):
            pt = _pt(shape, rng.choice([0.5, 2.0, 8.0]))
            s1 = ptime_scores("PTECT", times, loads, travel, pt)
            s2 = ptime_scores("PTECT", [f32(c * t) for t in times], [f32(c * q) for q in loads],
                              [f32(c * t) for t in travel], pt)
            b1, b2 = min(range(n), key=lambda i: s1[i]), min(range(n), key=lambda i: s2[i])
            assert b1 == b2 or math.isclose(float(s1[b1]), float(s1[b2]), rel_tol=1e-5)


def test_large_alpha_prefers_the_fast_machine():
    times, loads, travel = [f32(300.0), f32(200.0)], [f32(0.0), f32(150.0)], [f32(0.0), f32(0.0)]
    assert select_machine("ECT", [1, 2], times, loads, [f32(0)] * 2, travel) == 1          # 300 < 350
    assert select_machine("PECT", [1, 2], times, loads, [f32(0)] * 2, travel, pt=_pt("lin", 1.0)) == 2   # 600 > 500
    with pytest.raises(ValueError):
        select_machine("PECT", [1, 2], times, loads, [f32(0)] * 2, travel)


def _floor():
    with open(FLOOR) as f:
        return Floor(json.load(f))


def _jobs(floor, seed=0):
    from scenarios.randomized import RandomizedParams, randomized_scenario
    p = RandomizedParams(agv_count=3, failure_probability=0.0, horizon_seconds=3 * 3600.0,
                         due_date_allowance_range=(1.75, 2.5), op_machine_spread=0.3)
    return resolve_jobs(randomized_scenario(seed, p), floor)


def test_engine_alpha_zero_matches_tect_and_alpha_changes_routing():
    floor = _floor()
    jobs = _jobs(floor)
    run = lambda rule, **kw: Twin(floor, jobs, TwinConfig(rule=rule, agv_count=3, **kw)).run()
    tect = run("MDD_TECT")
    for shape in ("lin", "quad", "ratio"):
        assert run("MDD_PTECT", ptime_shape=shape, ptime_alpha=0.0).decision_rows == tect.decision_rows
    assert run("MDD_PECT", ptime_alpha=0.0).decision_rows == run("MDD_ECT").decision_rows
    heavy = run("MDD_PTECT", ptime_shape="lin", ptime_alpha=8.0)
    assert heavy.decision_rows != tect.decision_rows
    assert heavy.summary()["jobs_completed"] > 0


def test_engine_exp_equals_blind_rtect_under_ghost_trace():
    """dev-b2-pect-check's blind ghost RTECT (reliability_mix 0, a failure trace with every failure removed) and PTECT
    exp with that floor-mean MTBF / availability make the same decisions."""
    floor = _floor()
    jobs = _jobs(floor, 1)
    model = F.WeibullModel(F.Rel(1.5, F.lam_for_availability(0.90), 6.3, 0.5), initial="equilibrium")
    tr = model.trace([m.id for m in floor.machines], 1, 100000.0, floor.dt)
    for mt in tr.machines.values():
        mt.events = []
    rt = Twin(floor, jobs, TwinConfig(rule="MDD_RTECT", agv_count=3, failures=tr, reliability="static",
                                      reliability_kappa=16.0, reroute_cost=2880.0, reliability_mix=0.0))
    a, mtbf = rt.reliability_terms([m.id for m in floor.machines])["terms"][0][:2]
    rt.run()
    pt = Twin(floor, jobs, TwinConfig(rule="MDD_PTECT", agv_count=3, ptime_shape="exp", ptime_alpha=16.0,
                                      ptime_tau=mtbf, ptime_reroute=2880.0, ptime_avail=a)).run()
    assert pt.decision_rows == rt.decision_rows
    assert [(j.id, j.exit_time) for j in pt.jobs.values()] == [(j.id, j.exit_time) for j in rt.jobs.values()]
