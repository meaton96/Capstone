"""
@file lookahead.py
@brief Critic-only look-ahead features (2026-10-07, handoff fix 6; asymmetric actor-critic, Pinto et al. 2017): what
       the instance will bring in the next 30 / 60 / 120 min, known to the twin from the scenario but never shown
       to the policy.

@details A slot's return varies with the arrivals after it far more than with the slot's choice (SNR 0.03-0.05,
docs/experiments/review_1007/CREDIT_ASSIGNMENT_TRACE_1007.md), and a critic that sees only the current state cannot
explain that variance. Arrivals are exogenous (the actions do not change them), so a value baseline that conditions
on them is still unbiased for the policy gradient (Mao et al. 2019). The policy never sees these features, so the
policy stays deployable without a forecast.

Per window [now, now + h) for h in HORIZONS (seconds), 5 features, all squashed to about [0, 1]:
  - arrivals per hour / 30
  - offered load: the arriving work (each op's mean time over its eligible machines) / (machines x h)
  - bottleneck load: the largest per-type share of that work / (machines of that type x h)
  - due-date slack of the arriving jobs: mean (due - arrival - work) / work, squashed (0.5 = no slack); 0 without due
    dates
  - mean active AGV fraction over the window (the scenario's agvSchedule; 1 without one)
"""
import numpy as np

HORIZONS = (1800.0, 3600.0, 7200.0)
FEATURES_PER_WINDOW = 5
LOOKAHEAD_DIM = len(HORIZONS) * FEATURES_PER_WINDOW


def _squash(x: float, scale: float) -> float:
    return float(x / (abs(x) + scale)) if scale > 0 else 0.0


def _fleet_fraction(schedule, fleet: int, t0: float, t1: float) -> float:
    """Mean active fleet / full fleet over [t0, t1) under agv_schedule ((start, count), ... sorted), as
    Twin._on_duty reads it (the full fleet before the first entry)."""
    if not schedule or fleet <= 0:
        return 1.0

    def active(t):
        n = fleet
        for start, k in schedule:
            if start > t:
                break
            n = k
        return min(n, fleet)

    points = [t0] + [s for s, _ in schedule if t0 < s < t1] + [t1]
    return sum((b - a) * active(a) for a, b in zip(points, points[1:])) / ((t1 - t0) * fleet)


def lookahead_features(twin) -> np.ndarray:
    """@brief The LOOKAHEAD_DIM features at the twin's current time (see the module docstring)."""
    now = twin.now
    machines = twin.floor.machines
    per_type = {}
    for m in machines:
        per_type[m.type] = per_type.get(m.type, 0) + 1
    pending = twin.pending[twin.pending_i:]
    out = []
    for h in HORIZONS:
        end = now + h
        work, by_type, slack, n = 0.0, {}, [], 0
        for job in pending:
            if job.arrival >= end:
                break
            if job.arrival < now:
                continue
            n += 1
            jw = 0.0
            for op, typ in zip(job.ops, job.types):
                w = float(np.mean(list(op.values())))
                jw += w
                by_type[typ] = by_type.get(typ, 0.0) + w
            work += jw
            if job.due is not None and jw > 0:
                slack.append((job.due - job.arrival - jw) / jw)
        bottleneck = max((w / (per_type.get(t, 1) * h) for t, w in by_type.items()), default=0.0)
        out += [
            _squash(n * 3600.0 / h, 30.0),
            _squash(work / (len(machines) * h), 1.0),
            _squash(bottleneck, 1.0),
            0.5 + 0.5 * _squash(float(np.mean(slack)), 2.0) if slack else 0.0,
            _fleet_fraction(twin.cfg.agv_schedule, len(twin.agvs), now, end),
        ]
    return np.asarray(out, dtype=np.float32)
