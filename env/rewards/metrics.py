"""
@file metrics.py
@brief Reward-metrics snapshot contract shared with Unity's RewardMetrics.cs.

@details
Unity sends one flat float vector per agent step through the observation sensor named
@ref SENSOR_NAME. @ref METRIC_NAMES must match @c RewardMetrics.Names in C# exactly —
same names, same order (tests/test_rewards.py checks this against the C# source). The
first value is the schema version, so running against a stale build fails loudly.
"_sum" metrics are running totals since episode start: a per-step term is curr - prev.
"""

from collections.abc import Mapping
from typing import Iterator

import numpy as np

## @brief Name of the Unity sensor carrying the metrics (see RewardMetricsSensor.cs).
SENSOR_NAME = "Z_RewardMetrics"

## @brief Must equal RewardMetrics.SchemaVersion in C#.
SCHEMA_VERSION = 3

METRIC_NAMES = (
    "schema_version",
    "sim_time",
    "episode_active",
    "decision_count",

    "jobs_total",
    "jobs_exited",
    "wip",
    "ops_total",
    "ops_completed",

    "flow_time_exited_sum",
    "time_in_system_sum",

    "jobs_needs_routing",
    "jobs_waiting_pickup",
    "jobs_in_transit",
    "jobs_queued",
    "jobs_processing",

    "time_needs_routing_sum",
    "time_waiting_pickup_sum",
    "time_in_transit_sum",
    "time_queued_sum",
    "time_processing_sum",

    "machines_total",
    "machines_busy",
    "machines_down",
    "machine_downtime_sum",

    "agvs_total",
    "agvs_idle",
    "agv_time_traveling_sum",
    "agv_time_waiting_route_sum",
    "agv_time_idle_sum",
    "agv_trips_total",

    "zone_block_events_sum",

    "deadlock",
    "timed_out",
    "all_jobs_exited",

    # v2
    "episode_seed",
    "episode_seed_index",

    # v3
    "truncated",
)

_INDEX = {name: i for i, name in enumerate(METRIC_NAMES)}


class MetricsSnapshot(Mapping):
    """@brief Read-only named view over one reward-metrics vector.

    @details Values are available as keys (@c snap["sim_time"]) or attributes
    (@c snap.sim_time).
    """

    def __init__(self, values):
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if values.shape[0] != len(METRIC_NAMES):
            raise ValueError(
                f"Expected {len(METRIC_NAMES)} reward metrics, got {values.shape[0]}: "
                "env/rewards/metrics.py METRIC_NAMES and RewardMetrics.Names in C# are out of sync."
            )
        version = int(round(values[0]))
        if version == 0 and not values.any():
            raise ValueError(
                "Unity sent an all-zero reward-metrics snapshot: the sensor is registered but was "
                "never filled (check RewardMetricsSensor.Write / FactoryOrchestrator.WriteRewardMetrics)."
            )
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"Unity sent reward-metrics schema v{version}, Python expects v{SCHEMA_VERSION}: "
                "rebuild the player or update env/rewards/metrics.py."
            )
        self._values = values

    @classmethod
    def from_dict(cls, values: Mapping) -> "MetricsSnapshot":
        """@brief Build a snapshot from named values; unspecified metrics are 0. For tests and replay."""
        unknown = set(values) - set(METRIC_NAMES)
        if unknown:
            raise KeyError(f"Unknown reward metrics: {sorted(unknown)}")
        arr = np.zeros(len(METRIC_NAMES))
        arr[0] = SCHEMA_VERSION
        for name, value in values.items():
            arr[_INDEX[name]] = value
        return cls(arr)

    def __getitem__(self, name: str) -> float:
        return float(self._values[_INDEX[name]])

    def __getattr__(self, name: str) -> float:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return float(self._values[_INDEX[name]])
        except KeyError:
            raise AttributeError(f"No reward metric named {name!r}") from None

    def __iter__(self) -> Iterator[str]:
        return iter(METRIC_NAMES)

    def __len__(self) -> int:
        return len(METRIC_NAMES)

    def delta(self, prev: "MetricsSnapshot", name: str) -> float:
        """@brief Change in metric @p name since snapshot @p prev."""
        return self[name] - prev[name]

    def to_array(self) -> np.ndarray:
        return self._values.copy()
