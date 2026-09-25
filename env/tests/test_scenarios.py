"""
@file test_scenarios.py
@brief Tests for env/scenarios: seeded compound-scenario variants for RL training.

@par Usage
@code{.sh}
cd env && python -m pytest tests/test_scenarios.py -v
@endcode
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scenarios import REGISTRY, compound_generator, compound_variant


def test_pure_function_of_seed():
    """@brief The same seed must build byte-identical scenarios (reproducible training instances)."""
    a = compound_variant(7)
    b = compound_variant(7)
    assert a == b
    assert a is not b   # not literally cached/shared — a fresh dict each call


def test_different_seeds_differ():
    a, b = compound_variant(1), compound_variant(2)
    assert a["jobs"] != b["jobs"]
    assert a["seed"] == 1 and b["seed"] == 2


def test_matches_scenario_loader_schema():
    """@brief Must carry every field ScenarioLoader.BuildConfig/BuildJobs require."""
    scenario = compound_variant(42)
    assert scenario["name"]
    assert isinstance(scenario["agvCount"], int) and scenario["agvCount"] > 0
    assert scenario["machineTypeLayout"]
    assert scenario["jobs"]
    for job in scenario["jobs"][:5]:
        assert "id" in job and "arrivalTime" in job and job["operations"]
        for op in job["operations"]:
            assert {"machineType", "machineIndex", "duration"} <= op.keys()


def test_json_serializable():
    """@brief Must round-trip through the same JSON path EpisodeConfigChannel.queue_scenarios uses."""
    scenario = compound_variant(3)
    restored = json.loads(json.dumps(scenario))
    assert restored == scenario


def test_no_episode_duration_by_default():
    scenario = compound_variant(5)
    assert "stochastic" not in scenario


def test_episode_duration_seconds_adds_stochastic_block():
    scenario = compound_variant(5, episode_duration_seconds=1500.0)
    assert scenario["stochastic"] == {"episodeDurationSeconds": 1500.0}
    # Everything else about the instance (job data) must be untouched.
    assert scenario["jobs"] == compound_variant(5)["jobs"]


def test_compound_generator_is_a_seed_to_scenario_callable():
    generator = compound_generator(episode_duration_seconds=2000.0)
    scenario = generator(9)
    assert scenario == compound_variant(9, episode_duration_seconds=2000.0)


def test_registry_has_compound():
    # REGISTRY["compound"] is a variant-bound wrapper around compound_generator (not
    # compound_generator itself, now that it takes a `variant` param) -- check behavior:
    # it must produce the same scenario compound_generator(variant="compound") would.
    assert REGISTRY["compound"]()(7) == compound_generator(variant="compound")(7)


def test_registry_has_compound_v2():
    scenario = REGISTRY["compound_v2"]()(7)
    assert scenario["name"] == "compound_scenario_v2"
    assert scenario != REGISTRY["compound"]()(7)


# ── Randomized training family (scenarios/randomized.py) ──────────────────────────────────────

from scenarios.randomized import (  # noqa: E402
    DEFAULT_PARAMS, TYPES, randomized_generator, randomized_scenario, randomized_variant, summarize,
    warmup_offsets,
)


def test_randomized_pure_function_of_seed():
    assert randomized_scenario(11) == randomized_scenario(11)
    assert randomized_scenario(11)["jobs"] != randomized_scenario(12)["jobs"]


def test_randomized_schema_and_json_roundtrip():
    sc = randomized_variant(3, episode_duration_seconds=5400.0)
    assert json.loads(json.dumps(sc)) == sc
    assert sc["machineTypeLayout"] == [t for t in TYPES for _ in range(3)]
    assert (sc["agvCount"], sc["layout"], sc["reservationProtocol"], sc["parkingMethod"]) == (
        7, "D", "releasePrevious", "lane")
    for job in sc["jobs"]:
        assert 1 <= len(job["operations"]) <= 5
        for a, b in zip(job["operations"], job["operations"][1:]):
            assert a["machineType"] != b["machineType"]          # no immediate repeat
        for op in job["operations"]:
            assert op["machineIndex"] == [0, 1, 2] and len(op["duration"]) == 3
            assert all(d > 0 for d in op["duration"])
    arrivals = [j["arrivalTime"] for j in sc["jobs"]]
    assert arrivals == sorted(arrivals)
    assert [j["id"] for j in sc["jobs"]] == list(range(len(sc["jobs"])))


def test_randomized_failure_share_and_block():
    on = [randomized_scenario(s)["_meta"]["failures_on"] for s in range(300)]
    assert 0.55 < sum(on) / len(on) < 0.78                       # ~2/3
    seed = on.index(True)
    stoch = randomized_variant(seed, episode_duration_seconds=5400.0)["stochastic"]
    assert stoch["machineFailuresEnabled"] and stoch["weibullLambda"] == 7200.0
    assert stoch["episodeDurationSeconds"] == 5400.0
    off = on.index(False)
    assert "stochastic" not in randomized_scenario(off)


def test_randomized_load_is_spread_and_machine_bound():
    """@brief Unlike compound (~70% Weld), work spreads over all types; AGV demand stays under the cap."""
    p = DEFAULT_PARAMS
    cap = p.agv_count * p.agv_max_utilization / p.agv_seconds_per_move
    for s in range(20):
        st = summarize(randomized_scenario(s))
        assert max(st["work_share"].values()) < 0.35
        # The cap bounds each segment's expected rate; realized op counts and 60 s bursts add noise
        # (up to ~1.10x over a whole episode in the 2026-09-25 defaults; measured AGV busy stayed <= 0.51).
        assert st["moves_per_second"] <= cap * 1.15
        assert 2.5 < st["ops_per_job"] < 3.5
        assert p.op_mean_seconds[0] * 0.8 < st["mean_op_seconds"] < p.op_mean_seconds[1] * 1.2


def test_randomized_segments_cover_horizon():
    sc = randomized_scenario(4)
    ph = sc["_phases"]
    assert ph[0]["start"] == 0.0
    assert all(a["end"] == b["start"] for a, b in zip(ph, ph[1:]))
    assert ph[-1]["end"] <= DEFAULT_PARAMS.horizon_seconds
    assert all(j["arrivalTime"] < ph[-1]["end"] for j in sc["jobs"])


def test_randomized_warmup_leaves_a_full_window():
    gen = randomized_generator(5400.0, random_warmup=True)
    for s in range(30):
        sc = gen(s)
        w = sc["stochastic"]["warmupSeconds"]
        assert w in warmup_offsets(randomized_scenario(s), 5400.0)
        assert max(j["arrivalTime"] for j in sc["jobs"]) - w >= 5400.0 or w == 0.0
        assert sc["jobs"] == randomized_scenario(s)["jobs"]      # warm-up doesn't change the instance


def test_randomized_warmup_rule_rotates():
    rules = {randomized_variant(s)["dispatchingRule"] for s in range(8)}
    assert len(rules) == 4
    assert randomized_variant(0, warmup_dispatching_rule="FIFO_SRWT")["dispatchingRule"] == "FIFO_SRWT"


def test_registry_has_randomized():
    assert REGISTRY["randomized"](5400.0)(7) == randomized_generator(5400.0)(7)
