"""
@file scenarios/__init__.py
@brief Python-side generators for scripted (ScenarioLoader-schema) training scenarios.

Unlike env/rewards, these produce the *instance* itself (job arrivals, durations, floor
layout) rather than a reward signal. A generator is a plain callable seed -> dict, suitable
for UnitySchedulingEnv/VectorizedUnityEnv's scenario_generator: given the same seed it must
always return the same scenario, so training instances stay reproducible.
"""

from scenarios.compound import REGISTRY, compound_generator, compound_variant
