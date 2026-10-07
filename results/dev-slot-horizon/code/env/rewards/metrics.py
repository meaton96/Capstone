"""
@file metrics.py
@brief Reward-metrics snapshot contract shared with Unity's RewardMetrics.cs.

@details
Unity sends one flat float vector per agent step through the observation sensor named
@ref SENSOR_NAME. @ref METRIC_NAMES must match @c RewardMetrics.Names in C# exactly —
same names, same order (tests/test_rewards.py checks this against the C# source). The
first value is the schema version, so running against a stale build fails loudly.
"_sum" metrics are running totals since episode start: a per-step term is curr - prev.

A player built for an older schema (any version in @ref SCHEMA_LENGTHS) is still read: the metrics it does not send
are missing, and reading one raises, so a reward that needs them fails loudly instead of seeing zeros.
"""

from collections.abc import Mapping
from typing import Iterator, Optional

import numpy as np

## @brief Name of the Unity sensor carrying the metrics (see RewardMetricsSensor.cs).
SENSOR_NAME = "Z_RewardMetrics"

## @brief Must equal RewardMetrics.SchemaVersion in C#.
SCHEMA_VERSION = 5

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

    # v4
    "tick_error",   # episode ended early by an exception in the Unity tick (simulator failure, not a policy outcome)

    # v5 (2026-10-03): due dates; only jobs that carry one count, all 0 on scenarios without due dates
    "tardiness_sum",          # time past the due date over all jobs, live (integral of late WIP)
    "tardiness_exited_sum",   # exited jobs only: sum of max(0, exit - due)
    "jobs_late",              # open jobs already past their due date
    "jobs_exited_late",       # exited jobs that left after their due date
    "jobs_with_due_date",     # jobs in the system so far that carry a due date
)

## @brief Vector length each readable schema version sends (versions only ever append metrics).
SCHEMA_LENGTHS = {4: METRIC_NAMES.index("tick_error") + 1, 5: len(METRIC_NAMES)}

_INDEX = {name: i for i, name in enumerate(METRIC_NAMES)}


class MetricsSnapshot(Mapping):
    """@brief Read-only named view over one reward-metrics vector.

    @details Values are available as keys (@c snap["sim_time"]) or attributes
    (@c snap.sim_time).
    """

    def __init__(self, values):
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if values.size and not values.any():
            raise ValueError(
                "Unity sent an all-zero reward-metrics snapshot: the sensor is registered but was "
                "never filled (check RewardMetricsSensor.Write / FactoryOrchestrator.WriteRewardMetrics)."
            )
        version = int(round(values[0])) if values.size else -1
        if version not in SCHEMA_LENGTHS:
            raise ValueError(
                f"Unity sent reward-metrics schema v{version}, Python reads v{min(SCHEMA_LENGTHS)}-v{SCHEMA_VERSION}: "
                "rebuild the player or update env/rewards/metrics.py."
            )
        n = SCHEMA_LENGTHS[version]
        if values.shape[0] != n:
            raise ValueError(
                f"Expected {n} reward metrics for schema v{version}, got {values.shape[0]}: "
                "env/rewards/metrics.py METRIC_NAMES and RewardMetrics.Names in C# are out of sync."
            )
        self.version = version
        self._n = n
        self._values = np.concatenate([values, np.full(len(METRIC_NAMES) - n, np.nan)])

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
        i = _INDEX[name]
        if i >= self._n:
            raise KeyError(f"reward metric {name!r} needs schema v{_first_version(name)}; this player sends "
                           f"v{self.version} (rebuild it)")
        return float(self._values[i])

    def __getattr__(self, name: str) -> float:
        if name.startswith("_") or name == "version":
            raise AttributeError(name)
        if name not in _INDEX:
            raise AttributeError(f"No reward metric named {name!r}")
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(str(e)) from None

    def __iter__(self) -> Iterator[str]:
        return iter(METRIC_NAMES[:self._n])

    def __len__(self) -> int:
        return self._n

    def has(self, name: str) -> bool:
        """@brief True when this snapshot's schema carries metric @p name."""
        return _INDEX[name] < self._n

    def delta(self, prev: "MetricsSnapshot", name: str) -> float:
        """@brief Change in metric @p name since snapshot @p prev."""
        return self[name] - prev[name]

    def to_array(self) -> np.ndarray:
        """@brief All METRIC_NAMES values; metrics an older player does not send are NaN."""
        return self._values.copy()


def window_outcomes(first: Optional["MetricsSnapshot"], final: Optional["MetricsSnapshot"]) -> dict:
    """@brief Both scheduling objectives over an episode's agent window (first agent step to the end), whatever the
    reward was, so a run can be scored on either: the time-in-system integral (what flow_time sums) and the tardiness
    integral (what tardiness sums), plus the due-date counts at the end. Tardiness fields are None from a player
    older than metrics v5; everything is None when a snapshot is missing."""
    keys = ("window_time_in_system", "window_tardiness", "tardiness_exited_sum", "jobs_exited_late",
            "jobs_with_due_date")
    out = dict.fromkeys(keys)
    if first is None or final is None:
        return out
    out["window_time_in_system"] = final.delta(first, "time_in_system_sum")
    if final.has("tardiness_sum") and first.has("tardiness_sum"):
        out["window_tardiness"] = final.delta(first, "tardiness_sum")
        out["tardiness_exited_sum"] = final.tardiness_exited_sum
        out["jobs_exited_late"] = int(final.jobs_exited_late)
        out["jobs_with_due_date"] = int(final.jobs_with_due_date)
    return out


def _first_version(name: str) -> int:
    """@brief The schema version that introduced metric @p name."""
    i = _INDEX[name]
    return min(v for v, n in SCHEMA_LENGTHS.items() if i < n)


def elapsed_sim_time(prev: Optional["MetricsSnapshot"], curr: Optional["MetricsSnapshot"]) -> Optional[float]:
    """@brief Simulated seconds between two snapshots (Δτ of one decision step), or None when either is missing.

    @details The trainer discounts each step by gamma_s ** Δτ (PPOConfig.discount_horizon_s), so the env
    wrappers report this in info["dt"].
    """
    if prev is None or curr is None:
        return None
    return max(curr.sim_time - prev.sim_time, 0.0)
