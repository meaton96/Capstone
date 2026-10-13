"""
@file buffers.py
@brief Runs generated scenarios with finite machine buffers (inputBufferCapacity / outputBufferCapacity).

@details Wraps any seed -> scenario generator and sets the two buffer sizes after the generator has run, so seed N
is the same job set at every buffer size (paired comparisons). 0 = unbounded, the default and the behaviour of every
run before 2026-10-02.

Semantics (Unity, FJSSPConfig / DecisionCoordinator):
- Input buffer: the most jobs that can wait at a machine (queued there, or routed there and not yet delivered),
  not counting the one it is processing. A slot is claimed at routing, so a full machine is not a routing
  candidate and a job with no up machine that has room waits in the routing pool. AGVs never wait at a full dock.
- Output buffer: finished jobs waiting at a machine for pickup. When it is full the next finished job stays on the
  machine (blocking after service) and the machine cannot start another operation until an AGV collects one.
Circular blocking can deadlock the floor; the 3,000 s watchdog then ends the episode (deadlock flag, reward
penalty), so report deadlocks next to any result from a bounded-buffer regime.
"""

import copy
from typing import Callable, Dict, Optional


def buffer_override_fields(input_capacity: Optional[int] = None, output_capacity: Optional[int] = None) -> Dict:
    """@brief The scenario fields to set; empty when both arguments are None.

    @throws ValueError for a negative or non-integer capacity.
    """
    fields: Dict = {}
    for key, value in (("inputBufferCapacity", input_capacity), ("outputBufferCapacity", output_capacity)):
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{key} must be an integer >= 0, 0 = unbounded (got {value!r})")
        fields[key] = value
    return fields


def with_buffers(generator: Callable[[int], Dict], input_capacity: Optional[int] = None,
                 output_capacity: Optional[int] = None) -> Callable[[int], Dict]:
    """@brief seed -> scenario callable that runs @p generator's instances with the given machine buffer sizes."""
    fields = buffer_override_fields(input_capacity, output_capacity)

    def _generate(seed: int) -> Dict:
        scenario = copy.deepcopy(generator(seed))
        scenario.update(fields)
        return scenario

    return _generate
