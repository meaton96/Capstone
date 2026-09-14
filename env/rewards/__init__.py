"""
@file rewards/__init__.py
@brief Python-side reward functions for the scheduling agent.

@details
Unity does not compute a reward. On every agent step it sends a snapshot of raw
simulation metrics (see @ref rewards.metrics.METRIC_NAMES); the env wrapper calls the
reward function with the previous and current snapshot, so the reward for an action
covers everything that happened between that decision and the next one (or episode end).

A reward is selected by a JSON spec, so it can change without rebuilding Unity:

@code{.json}
{
  "name": "flow_time",
  "entry": "rewards/functions/flow_time.py:FlowTimeReward",
  "params": {"time_scale": 1000.0}
}
@endcode

@c entry is @c path/to/file.py:Name (relative paths resolve against the spec file, then
env/, then the working directory) or @c package.module:Name. @c Name is a
@ref rewards.base.RewardFunction subclass, or a plain function
@c fn(prev, curr, ctx, **params) returning a float or a dict of named terms. Named terms
are summed into the reward and logged separately. Any numeric parameter may be a
schedule over training steps (see @ref rewards.base.resolve).
"""

from rewards.metrics import METRIC_NAMES, SCHEMA_VERSION, SENSOR_NAME, MetricsSnapshot
from rewards.base import RewardContext, RewardFunction, resolve
from rewards.loader import LoadedReward, load_reward
