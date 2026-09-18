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
    assert REGISTRY["compound"] is compound_generator
