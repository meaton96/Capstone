"""
@file agv_failures.py
@brief Adds an AGV breakdown block to generated scenarios (StochasticConfig.AGVFailuresEnabled in Unity).

@details Wraps any seed -> scenario generator. The block is merged into the scenario's "stochastic" object
after the generator has run, so the instance itself (jobs, machine failures, warm-up) is the same for a seed
with and without AGV failures, and Unity draws AGV breakdowns from per-AGV streams that leave the machine
failure stream untouched. The paired comparison against the fixed-rule baselines relies on both.

Model (AGVController, "Breakdowns"): a failed AGV stops in place and keeps its zones for a lognormal repair;
time to failure is Weibull in OPERATING seconds (travelling, loading, unloading), and the first life of an
episode is drawn from the equilibrium residual-life distribution.
"""

import copy
from typing import Callable, Dict, Optional

## @brief Working defaults (docs/THESIS_GAP_PLAN_2026-09-30.md, D2): mean life ~7,600 operating-s, which is
##        about 3 failures per 5,400 s window for 7 AGVs at ~60% busy; mean repair ~113 s. To be calibrated
##        against the AGV operating time measured in the regression run.
AGV_FAILURE_DEFAULTS: Dict = {
    "agvFailuresEnabled": True,
    "agvWeibullK": 1.5,
    "agvWeibullLambda": 8400.0,
    "agvRepairLogMu": 4.6,
    "agvRepairLogSigma": 0.5,
}


def agv_failure_block(overrides: Optional[Dict] = None) -> Dict:
    """@brief The AGV breakdown fields: AGV_FAILURE_DEFAULTS with @p overrides applied.

    @throws KeyError if @p overrides names a field that is not an AGV breakdown field.
    """
    overrides = dict(overrides or {})
    unknown = set(overrides) - set(AGV_FAILURE_DEFAULTS)
    if unknown:
        raise KeyError(f"Not AGV breakdown fields: {sorted(unknown)} (valid: {sorted(AGV_FAILURE_DEFAULTS)})")
    return {**AGV_FAILURE_DEFAULTS, **overrides}


def with_agv_failures(generator: Callable[[int], Dict],
                      overrides: Optional[Dict] = None) -> Callable[[int], Dict]:
    """@brief seed -> scenario callable that adds the AGV breakdown block to every scenario of @p generator.

    @details Applies to every instance, including ones whose machine failures are off (the block then
    creates the "stochastic" object). Existing stochastic fields (machine failures, episode cap, warm-up)
    are kept.
    """
    block = agv_failure_block(overrides)

    def _generate(seed: int) -> Dict:
        scenario = copy.deepcopy(generator(seed))
        stochastic = dict(scenario.get("stochastic") or {})
        stochastic.update(block)
        scenario["stochastic"] = stochastic
        return scenario

    return _generate
