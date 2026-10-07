"""
@file floor_override.py
@brief Runs generated scenarios on a different fleet size or floor layout (the RQ3 conditions sweep).

@details Wraps any seed -> scenario generator and overwrites "agvCount" / "layout" after the generator has run,
so seed N is the same job set in every condition. The overrides cannot go through the generator's own params:
randomized.RandomizedParams.agv_count also caps the arrival rate (transport_cap), so a generator built for
9 AGVs would draw different jobs than one built for 7. The arrival cap therefore stays at the generator's
fleet (7 for the randomized family), which is what makes the instances paired across conditions.
"""

import copy
from typing import Callable, Dict, Optional

from channels.config_schema import LAYOUTS


def floor_override_fields(agv_count: Optional[int] = None, layout: Optional[str] = None) -> Dict:
    """@brief The scenario fields to overwrite; empty when both arguments are None.

    @throws ValueError for a fleet below 1 or a layout not in config_schema.LAYOUTS.
    """
    fields: Dict = {}
    if agv_count is not None:
        if isinstance(agv_count, bool) or not isinstance(agv_count, int) or agv_count < 1:
            raise ValueError(f"AGV count must be an integer >= 1 (got {agv_count!r})")
        fields["agvCount"] = agv_count
    if layout is not None:
        name = str(layout).strip().upper()
        if name not in LAYOUTS:
            raise ValueError(f"unknown layout {layout!r} (valid: {', '.join(LAYOUTS)})")
        fields["layout"] = name
    return fields


def with_floor(generator: Callable[[int], Dict], agv_count: Optional[int] = None,
               layout: Optional[str] = None) -> Callable[[int], Dict]:
    """@brief seed -> scenario callable that runs @p generator's instances with @p agv_count AGVs on @p layout.

    @details Only "agvCount" and "layout" change; jobs, failures, warm-up and every other field are kept. The
    gridlock-safe fleet limit is still enforced when the scenario is validated on load.
    """
    fields = floor_override_fields(agv_count, layout)

    def _generate(seed: int) -> Dict:
        scenario = copy.deepcopy(generator(seed))
        scenario.update(fields)
        return scenario

    return _generate
