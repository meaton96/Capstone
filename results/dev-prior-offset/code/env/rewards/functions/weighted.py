"""
@file weighted.py
@brief Build a reward from JSON alone: a weighted sum of metric terms.

@details
Each term reads one metric from @ref rewards.metrics.METRIC_NAMES:

@code{.json}
"terms": {
  "flow_time":  {"metric": "time_in_system_sum", "mode": "delta", "weight": -0.001},
  "congestion": {"metric": "agv_time_waiting_route_sum", "mode": "delta",
                 "weight": {"start": 0.0, "end": -0.001, "steps": 200000}},
  "deadlock":   {"metric": "deadlock", "mode": "terminal", "weight": -10.0}
}
@endcode

Modes: @c delta — weight × (curr − prev); @c value — weight × curr on every step;
@c terminal — weight × curr on the episode's final step only. Weights are schedulable.
"""

from rewards.base import RewardFunction, resolve
from rewards.metrics import METRIC_NAMES

MODES = ("delta", "value", "terminal")


class WeightedReward(RewardFunction):
    """@param terms  Mapping of term name → {"metric", "mode", "weight"}."""

    def __init__(self, terms):
        super().__init__(terms=terms)
        for name, term in terms.items():
            if term.get("metric") not in METRIC_NAMES:
                raise ValueError(f"Term {name!r}: unknown metric {term.get('metric')!r}")
            if term.get("mode", "delta") not in MODES:
                raise ValueError(f"Term {name!r}: mode must be one of {MODES}")
            if "weight" not in term:
                raise ValueError(f"Term {name!r}: missing 'weight'")

    def compute(self, prev, curr, ctx):
        out = {}
        for name, term in self.params["terms"].items():
            metric, mode = term["metric"], term.get("mode", "delta")
            if mode == "terminal" and not ctx.done:
                continue
            weight = resolve(term["weight"], ctx.global_step)
            value = curr.delta(prev, metric) if mode == "delta" else curr[metric]
            out[name] = weight * value
        return out
