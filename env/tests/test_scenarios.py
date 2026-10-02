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


# ── Machine flexibility (2026-09-27) ──

_SCENARIO_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                             "linux_server", "BatchConfigs", "Scenarios")


def test_flexibility_zero_reproduces_saved_instances():
    """@brief At the default (0) the generator still builds the rnd_load instances every sweep used."""
    for s in (0, 5):
        path = os.path.join(_SCENARIO_DIR, f"rnd_load_s{s}.json")
        if not os.path.isfile(path):
            pytest.skip(f"{path} not present")
        with open(path) as f:
            saved = json.load(f)
        sc = randomized_scenario(s)
        assert sc["jobs"] == saved["jobs"]
        assert "machineFlexibilityProbability" not in sc


def test_flexibility_adds_keys_without_changing_the_instance():
    import dataclasses
    flex = dataclasses.replace(DEFAULT_PARAMS, machine_flexibility=0.3, secondary_time_multiplier=1.25)
    base, sc = randomized_scenario(3), randomized_scenario(3, flex)
    assert sc["machineFlexibilityProbability"] == 0.3 and sc["secondaryTimeMultiplier"] == 1.25
    assert len(sc["jobs"]) == len(base["jobs"])
    for jb, jf in zip(base["jobs"], sc["jobs"]):
        assert jb["arrivalTime"] == jf["arrivalTime"]
        for ob, of in zip(jb["operations"], jf["operations"]):
            assert of["allowSecondary"] is True and of["secondaryDuration"] > 0
            assert {k: v for k, v in of.items() if k not in ("allowSecondary", "secondaryDuration")} == ob
            # The nominal length sits inside the per-machine affinity band (±30%).
            assert min(ob["duration"]) / 1.31 <= of["secondaryDuration"] <= max(ob["duration"]) / 0.69


def test_registry_passes_flexibility():
    sc = REGISTRY["randomized"](5400.0, machine_flexibility=0.5, secondary_time_multiplier=1.5)(2)
    assert sc["machineFlexibilityProbability"] == 0.5 and sc["secondaryTimeMultiplier"] == 1.5
    comp = REGISTRY["compound"](3000.0, machine_flexibility=0.2)(1)
    assert comp["machineFlexibilityProbability"] == 0.2 and comp["secondaryTimeMultiplier"] == 1.0
    assert "machineFlexibilityProbability" not in REGISTRY["compound"](3000.0)(1)


def test_secondary_ops_without_a_level():
    """@brief secondary_ops writes the per-op opt-in only, so the player's -flex override sets the level."""
    import dataclasses
    sc = randomized_scenario(1, dataclasses.replace(DEFAULT_PARAMS, secondary_ops=True))
    assert "machineFlexibilityProbability" not in sc
    assert all(op["allowSecondary"] for job in sc["jobs"] for op in job["operations"])


# ── AGV breakdowns (scenarios.agv_failures) ─────────────────────────────────────────────────────────

def test_agv_failures_keep_the_instance():
    """@brief The breakdown block is added on top: same jobs and same machine-failure fields for a seed, so the
    run is paired with the no-breakdown baseline on the same instance."""
    from scenarios import AGV_FAILURE_DEFAULTS, with_agv_failures
    base = REGISTRY["randomized"](5400.0)
    wrapped = with_agv_failures(base)
    for seed in range(6):
        a, b = base(seed), wrapped(seed)
        assert a["jobs"] == b["jobs"]
        for key, value in (a.get("stochastic") or {}).items():
            assert b["stochastic"][key] == value
        for key, value in AGV_FAILURE_DEFAULTS.items():
            assert b["stochastic"][key] == value
        assert base(seed) == a      # the wrapper never mutates what the generator returns


def test_agv_failures_on_instances_without_machine_failures():
    """@brief Instances with machine failures off have no stochastic block; the wrapper creates one."""
    from scenarios import with_agv_failures
    base = REGISTRY["randomized"](None)
    off = next(s for s in range(50) if not (base(s).get("stochastic") or {}).get("machineFailuresEnabled"))
    block = with_agv_failures(base)(off)["stochastic"]
    assert block["agvFailuresEnabled"] is True
    assert not block.get("machineFailuresEnabled", False)


def test_agv_failure_overrides_and_validation():
    from channels.config_schema import validate_scenario
    from scenarios import agv_failure_block, with_agv_failures
    assert agv_failure_block({"agvWeibullLambda": 6000.0})["agvWeibullLambda"] == 6000.0
    with pytest.raises(KeyError):
        agv_failure_block({"weibullLambda": 6000.0})      # a machine field, not an AGV one
    validate_scenario(with_agv_failures(REGISTRY["randomized"](5400.0))(0))


# ── Floor overrides (scenarios.floor_override; evaluate.py --agvs / --layout) ──────────────────────────

def test_floor_override_changes_only_fleet_and_layout():
    """@brief Seed N is the same job set in every condition of the RQ3 sweep: only agvCount and layout change,
    and the arrival cap stays at the generator's own fleet."""
    from scenarios import with_floor
    base = REGISTRY["randomized"](5400.0)
    for agvs, layout in ((5, None), (9, "j"), (None, "G")):
        wrapped = with_floor(base, agvs, layout)
        for seed in range(4):
            a, b = base(seed), wrapped(seed)
            changed = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
            assert changed <= {"agvCount", "layout"}
            assert b["agvCount"] == (agvs if agvs is not None else a["agvCount"])
            assert b["layout"] == (layout.upper() if layout is not None else a["layout"])
            assert base(seed) == a      # the wrapper never mutates what the generator returns


def test_floor_override_composes_with_agv_failures():
    from scenarios import with_agv_failures, with_floor
    base = REGISTRY["randomized"](5400.0)
    assert with_floor(with_agv_failures(base), 9, "J")(3) == with_agv_failures(with_floor(base, 9, "J"))(3)


def test_floor_override_validation():
    from channels.config_schema import ConfigValidationError, validate_scenario
    from scenarios import floor_override_fields, with_floor
    assert floor_override_fields() == {}
    for agvs, layout in ((0, None), (True, None), (2.5, None), (None, "Z")):
        with pytest.raises(ValueError):
            floor_override_fields(agvs, layout)
    validate_scenario(with_floor(REGISTRY["randomized"](5400.0), 9, "J")(0))
    with pytest.raises(ConfigValidationError):       # above the gridlock-safe fleet for 15 machines
        validate_scenario(with_floor(REGISTRY["randomized"](5400.0), 16)(0))
