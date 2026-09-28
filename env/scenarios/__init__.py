"""
@file scenarios/__init__.py
@brief Python-side generators for scripted (ScenarioLoader-schema) training scenarios.

Unlike env/rewards, these produce the *instance* itself (job arrivals, durations, floor
layout) rather than a reward signal. A generator is a plain callable seed -> dict, suitable
for UnitySchedulingEnv/VectorizedUnityEnv's scenario_generator: given the same seed it must
always return the same scenario, so training instances stay reproducible.
"""

from scenarios.compound import REGISTRY, compound_generator, compound_variant
from scenarios.randomized import randomized_generator, randomized_scenario, randomized_variant


def scenario_machine_count(item) -> int:
    """@brief Machines on a scenario's floor: the length of its machineTypeLayout (required by ScenarioLoader).

    @param item  A scenario dict, or a path to a scenario JSON.
    """
    if not isinstance(item, dict):
        import json
        with open(item) as f:
            item = json.load(f)
    return len(item["machineTypeLayout"])


def row_caps_for(scenarios, max_machines: int = 0, max_jobs: int = 0):
    """@brief Observation row caps for a run that will queue @p scenarios (dicts or paths).

    @details Machine rows fit the largest floor; job rows follow config.obs_row_caps. A positive
    @p max_machines / @p max_jobs overrides that half. With no scenarios (the player's own default
    floor, size unknown here) the defaults MAX_MACHINES / MAX_JOBS are used.
    @return (machine rows, job rows)
    """
    from config import MAX_JOBS, MAX_MACHINES, obs_row_caps
    counts = [scenario_machine_count(s) for s in scenarios]
    machines, jobs = obs_row_caps(max(counts)) if counts else (MAX_MACHINES, MAX_JOBS)
    return (max_machines if max_machines > 0 else machines, max_jobs if max_jobs > 0 else jobs)
