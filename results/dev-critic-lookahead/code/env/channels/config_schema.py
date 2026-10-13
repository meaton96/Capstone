"""
@file config_schema.py
@brief Schema and physical bounds for configs and scenarios sent to Unity (thesis section 7.1).

EpisodeConfigChannel validates every config / scenario here before it is sent, and raises
ConfigValidationError listing every problem instead of letting Unity fall back to a default.
Unity checks the same bounds again on receipt (ConfigValidator.cs, ScenarioLoader strict mode)
and stops the player on a failure, so this module is the early, readable error and the C# side
is the authority (it also sees the player's CLI overrides, e.g. -reservation). Keep the two in step.
"""

import math
from typing import Any, Dict, List

## @brief ScenarioLoader.KnownKeys (top level of a scripted scenario). "_"-prefixed keys are comments.
SCENARIO_KEYS = {
    "name", "seed", "agvCount", "agvMoveSpeed", "agvHandshakeDuration", "dispatchingRule",
    "jobs", "machineTypeLayout", "parkingMethod", "ioDocks", "reservationProtocol", "routingTrigger",
    "stochastic", "layout", "tiling", "machineFlexibilityProbability", "secondaryTimeMultiplier", "travelPrice",
    "inputBufferCapacity", "outputBufferCapacity", "agvSchedule",
}

## @brief EpisodeConfigChannel.ConfigKeys (top level of a generated-config message).
CONFIG_KEYS = {
    "name", "seed", "jobCount", "machinesPerType", "machineTypes", "minProcTime", "maxProcTime",
    "minOpsPerJob", "maxOpsPerJob", "agvCount", "agvMoveSpeed", "agvHandshakeDuration",
    "machineFlexibilityProbability", "secondaryTimeMultiplier", "parkingMethod", "ioDocks",
    "reservationProtocol", "routingTrigger", "tiling", "preDispatchingMethod", "stochastic", "procTimeParams",
    "travelPrice", "inputBufferCapacity", "outputBufferCapacity",
}

## @brief ScenarioLoader.StochasticKeys.
STOCHASTIC_KEYS = {
    "machineFailuresEnabled", "weibullK", "weibullLambda", "repairLogMu", "repairLogSigma",
    "agvFailuresEnabled", "agvWeibullK", "agvWeibullLambda", "agvRepairLogMu", "agvRepairLogSigma",
    "dynamicArrivalsEnabled", "arrivalLambda", "dynamicJobCap", "burstArrivalsEnabled", "burstSizeMean",
    "episodeDurationSeconds", "warmupSeconds",
}

TILING_KEYS = {"tiles", "machinesPerTile", "jobScope", "agvAssignment", "releaseRule", "releaseWeights", "seamGap"}

MACHINE_TYPES = ("Mill", "Lathe", "Weld", "Inspect", "Assemble")
LAYOUTS = tuple("ABCDEFGHIJKLMNO")
_JOB_RULES = ("SPT", "LPT", "SRT", "LRT", "FIFO", "PTWINQ", "EDD", "MDD", "ATC")
_MACHINE_RULES = ("SMPT", "SRWT", "MMUR", "ECT", "TECT")
DISPATCHING_RULES = {f"{j}_{m}" for j in _JOB_RULES for m in _MACHINE_RULES} | {"Random"}

## @brief Gridlock-safe fleet per 15-machine reference floor (ConfigValidator.cs, where the sources are cited):
##        holdPrevious gridlocks from 9 AGVs; releasePrevious measured deadlock-free to 15 on every layout.
REFERENCE_MACHINES = 15
MAX_AGVS_PER_REFERENCE = {"holdprevious": 8, "releaseprevious": 15}
DEFAULT_RESERVATION = "releasePrevious"


class ConfigValidationError(ValueError):
    """@brief A config or scenario outside the schema; @c errors lists every problem found."""

    def __init__(self, what: str, errors: List[str]):
        self.errors = list(errors)
        super().__init__(f"{what} rejected ({len(errors)} problem(s)):\n  - " + "\n  - ".join(errors))


def max_agv_count(machine_count: int, reservation_protocol: str = DEFAULT_RESERVATION) -> int:
    """@brief Largest accepted fleet: the protocol's measured threshold scaled by machines (ConfigValidator.MaxAgvCount)."""
    per_reference = MAX_AGVS_PER_REFERENCE[str(reservation_protocol).strip().lower()]
    return max(1, machine_count * per_reference // REFERENCE_MACHINES)


def validate_scenario(scenario: Dict[str, Any], name: str = "scenario") -> None:
    """@brief Raise ConfigValidationError unless @p scenario (ScenarioLoader schema) is fully valid."""
    v = _Checker()
    if not isinstance(scenario, dict):
        raise ConfigValidationError(name, ["a scenario must be a JSON object"])
    v.keys(scenario, SCENARIO_KEYS, "")

    layout = scenario.get("machineTypeLayout")
    if not isinstance(layout, list) or not layout:
        v.fail("machineTypeLayout must be a non-empty list of machine types")
        layout = []
    counts: Dict[str, int] = {}
    for i, t in enumerate(layout):
        canonical = _machine_type(t)
        if canonical is None:
            v.fail(f"machineTypeLayout[{i}]: unknown machine type {t!r} (valid: {', '.join(MACHINE_TYPES)})")
        else:
            counts[canonical] = counts.get(canonical, 0) + 1

    jobs = scenario.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        v.fail("jobs must be a non-empty list")
        jobs = []
    for j, job in enumerate(jobs):
        _check_job(v, job, j, counts)

    _check_common(v, scenario, len(layout))
    _check_agv_schedule(v, scenario)
    v.raise_if_any(name)


def _check_agv_schedule(v, scenario: Dict[str, Any]) -> None:
    """agvSchedule (2026-10-04): [{"start": t, "agvCount": n}], starts ascending >= 0, 1 <= n <= agvCount
    (ScenarioLoader.ReadAgvSchedule applies the same rules)."""
    sched = scenario.get("agvSchedule")
    if sched is None:
        return
    if not isinstance(sched, list) or not sched:
        v.fail("agvSchedule must be a non-empty list of {start, agvCount}")
        return
    fleet = scenario.get("agvCount")
    prev = -1.0
    for i, e in enumerate(sched):
        if not isinstance(e, dict) or set(e) != {"start", "agvCount"}:
            v.fail(f"agvSchedule[{i}] must have exactly the keys start, agvCount")
            continue
        t, n = e["start"], e["agvCount"]
        if isinstance(t, bool) or not isinstance(t, (int, float)) or t < 0 or t < prev:
            v.fail(f"agvSchedule[{i}].start must be a number >= 0 and >= the previous start")
        else:
            prev = float(t)
        if isinstance(n, bool) or not isinstance(n, int) or n < 1 or (isinstance(fleet, int) and n > fleet):
            v.fail(f"agvSchedule[{i}].agvCount must be an integer in 1..agvCount")


def validate_config(config: Dict[str, Any], name: str = "config") -> None:
    """@brief Raise ConfigValidationError unless @p config (EpisodeConfigChannel.send_config schema) is fully valid."""
    v = _Checker()
    if not isinstance(config, dict):
        raise ConfigValidationError(name, ["a config must be a JSON object"])
    v.keys(config, CONFIG_KEYS, "")

    types = config.get("machineTypes")
    if not isinstance(types, list) or not types:
        v.fail("machineTypes must be a non-empty list of machine types")
        types = []
    for i, t in enumerate(types):
        if t not in MACHINE_TYPES:   # Enum.Parse on this path is case-sensitive
            v.fail(f"machineTypes[{i}]: unknown machine type {t!r} (valid: {', '.join(MACHINE_TYPES)})")
    per_type = config.get("machinesPerType")
    v.integer(config, "jobCount", minimum=1, required=True)
    v.integer(config, "machinesPerType", minimum=1, required=True)
    v.number(config, "minProcTime", exclusive_min=0.0)
    v.number(config, "maxProcTime", exclusive_min=0.0)
    # A missing bound takes C#'s default (ConfigLoader: 15-90 s, 3-7 ops), so check against that too:
    # e.g. maxProcTime 10 alone fails ConfigValidator and stops the player.
    lo, hi = config.get("minProcTime", 15.0), config.get("maxProcTime", 90.0)
    if _is_number(lo) and _is_number(hi) and hi < lo:
        v.fail(f"maxProcTime must be >= minProcTime ({hi} < {lo}; a missing one defaults to 15 / 90 s)")
    v.integer(config, "minOpsPerJob", minimum=1)
    v.integer(config, "maxOpsPerJob", minimum=1)
    lo, hi = config.get("minOpsPerJob", 3), config.get("maxOpsPerJob", 7)
    if _is_int(lo) and _is_int(hi) and hi < lo:
        v.fail(f"maxOpsPerJob must be >= minOpsPerJob ({hi} < {lo}; a missing one defaults to 3 / 7)")

    ptp = config.get("procTimeParams")
    if ptp is not None:
        if not isinstance(ptp, dict):
            v.fail("procTimeParams must be an object keyed by machine type")
        else:
            for t, p in ptp.items():
                if t not in MACHINE_TYPES or not isinstance(p, dict):
                    v.fail(f"procTimeParams: unknown machine type or malformed entry {t!r}")
                    continue
                v.keys(p, {"mu", "sigma"}, f"procTimeParams.{t}.")
                v.number(p, "mu", exclusive_min=0.0, required=True, where=f"procTimeParams.{t}.")
                v.number(p, "sigma", minimum=0.0, required=True, where=f"procTimeParams.{t}.")

    machines = len(types) * per_type if _is_int(per_type) and per_type > 0 else 0
    _check_common(v, config, machines)
    v.raise_if_any(name)


# ── Shared checks ────────────────────────────────────────────────────────────────────────────────

def _check_common(v: "_Checker", cfg: Dict[str, Any], machine_count: int) -> None:
    v.integer(cfg, "seed", minimum=0)
    v.number(cfg, "agvMoveSpeed", exclusive_min=0.0)
    v.number(cfg, "agvHandshakeDuration", minimum=0.0)
    v.number(cfg, "machineFlexibilityProbability", minimum=0.0, maximum=1.0)
    v.number(cfg, "secondaryTimeMultiplier", exclusive_min=0.0)
    v.number(cfg, "travelPrice", minimum=0.0)
    v.integer(cfg, "inputBufferCapacity", minimum=0)
    v.integer(cfg, "outputBufferCapacity", minimum=0)
    v.choice(cfg, "reservationProtocol", {"holdprevious", "releaseprevious"})
    v.choice(cfg, "routingTrigger", {"ontransport", "onready"})
    v.choice(cfg, "parkingMethod", {"single", "multiple", "lane"})
    v.choice(cfg, "ioDocks", {"corner", "siding", "bypass"})
    if "dispatchingRule" in cfg and not any(str(cfg["dispatchingRule"]).lower() == r.lower() for r in DISPATCHING_RULES):
        v.fail(f"dispatchingRule: unknown rule {cfg['dispatchingRule']!r}")
    _check_layout(v, cfg.get("layout"))
    tiles = _check_tiling(v, cfg.get("tiling"))

    agvs = cfg.get("agvCount")
    if agvs is not None or "jobCount" in cfg:   # the config path defaults agvCount to 5 when absent
        agvs = 5 if agvs is None else agvs
        if not _is_int(agvs) or agvs < 1:
            v.fail(f"agvCount must be an integer >= 1 (got {agvs!r})")
        elif machine_count > 0:
            protocol = cfg.get("reservationProtocol") or DEFAULT_RESERVATION
            if str(protocol).strip().lower() in MAX_AGVS_PER_REFERENCE:
                limit = max_agv_count(machine_count, protocol)
                if agvs > limit:
                    v.fail(f"agvCount {agvs} is above the gridlock-safe maximum {limit} for {machine_count} machines "
                           f"under {protocol} (launch the player with -allowunsafefleet and send it without "
                           f"validation to run it anyway)")
            if tiles > 1 and agvs % tiles != 0:
                v.fail(f"tiling: agvCount {agvs} must be a positive multiple of the tile count {tiles}")

    s = cfg.get("stochastic")
    if s is not None:
        if not isinstance(s, dict):
            v.fail("stochastic must be an object")
        else:
            _check_stochastic(v, s)


def _check_job(v: "_Checker", job: Any, j: int, counts: Dict[str, int]) -> None:
    where = f"jobs[{j}]"
    if not isinstance(job, dict):
        v.fail(f"{where} must be an object")
        return
    v.number(job, "arrivalTime", minimum=0.0, where=f"{where}.")
    due = job.get("dueDate")
    if due is not None:
        arrival = job.get("arrivalTime", 0.0)
        if not _is_number(due) or not math.isfinite(due) or (_is_number(arrival) and due < arrival):
            v.fail(f"{where}.dueDate must be a finite number >= arrivalTime (got {due!r})")
    ops = job.get("operations")
    if not isinstance(ops, list) or not ops:
        v.fail(f"{where}.operations must be a non-empty list")
        return
    for o, op in enumerate(ops):
        at = f"{where}.operations[{o}]"
        if not isinstance(op, dict):
            v.fail(f"{at} must be an object")
            continue
        mtype = _machine_type(op.get("machineType"))
        if mtype is None:
            v.fail(f"{at}: unknown machineType {op.get('machineType')!r}")
            continue
        available = counts.get(mtype, 0)
        if available == 0:
            v.fail(f"{at}: no {mtype} machine in machineTypeLayout")
            continue
        index = op.get("machineIndex", "any")
        duration = op.get("duration")
        if isinstance(index, str):
            if index.lower() != "any":
                v.fail(f"{at}: machineIndex must be \"any\", an index or a list of indices (got {index!r})")
            if not _is_number(duration) or not duration > 0:
                v.fail(f"{at}: duration must be a number > 0 (got {duration!r})")
            continue
        indices = index if isinstance(index, list) else [index]
        if not indices:
            v.fail(f"{at}: machineIndex list is empty")
        for idx in indices:
            if not _is_int(idx) or not 0 <= idx < available:
                v.fail(f"{at}: machineIndex {idx!r} out of range for {mtype} (only {available} on the floor)")
        durations = duration if isinstance(duration, list) and isinstance(index, list) else [duration]
        if isinstance(duration, list) and len(duration) != len(indices):
            v.fail(f"{at}: duration list length {len(duration)} != machineIndex list length {len(indices)}")
        for d in durations:
            if not _is_number(d) or not d > 0:
                v.fail(f"{at}: duration must be a number > 0 (got {d!r})")
        if "secondaryDuration" in op and (not _is_number(op["secondaryDuration"]) or not op["secondaryDuration"] > 0):
            v.fail(f"{at}: secondaryDuration must be a number > 0")


def _check_stochastic(v: "_Checker", s: Dict[str, Any]) -> None:
    w = "stochastic."
    v.keys(s, STOCHASTIC_KEYS, w)
    for flag in ("machineFailuresEnabled", "agvFailuresEnabled", "dynamicArrivalsEnabled", "burstArrivalsEnabled"):
        if flag in s and not isinstance(s[flag], bool):
            v.fail(f"{w}{flag} must be true or false")
    if s.get("machineFailuresEnabled"):
        v.number(s, "weibullK", exclusive_min=0.0, where=w)
        v.number(s, "weibullLambda", exclusive_min=0.0, where=w)
        v.number(s, "repairLogSigma", minimum=0.0, where=w)
    v.number(s, "repairLogMu", where=w)
    if s.get("agvFailuresEnabled"):
        v.number(s, "agvWeibullK", exclusive_min=0.0, where=w)
        v.number(s, "agvWeibullLambda", exclusive_min=0.0, where=w)
        v.number(s, "agvRepairLogSigma", minimum=0.0, where=w)
    v.number(s, "agvRepairLogMu", where=w)
    if s.get("dynamicArrivalsEnabled"):
        v.number(s, "arrivalLambda", exclusive_min=0.0, where=w)
    v.integer(s, "dynamicJobCap", minimum=0, where=w)
    if s.get("burstArrivalsEnabled"):
        v.number(s, "burstSizeMean", minimum=1.0, where=w)
    v.number(s, "episodeDurationSeconds", minimum=0.0, where=w)
    v.number(s, "warmupSeconds", minimum=0.0, where=w)


def _check_layout(v: "_Checker", layout: Any) -> None:
    if layout is None:
        return
    if isinstance(layout, dict):
        v.keys(layout, {"preset"}, "layout.")
        layout = layout.get("preset")
        if layout is None:
            return
    if not isinstance(layout, str) or (layout.strip().upper() not in LAYOUTS and layout.strip().lower() != "legacy"):
        v.fail(f"layout: unknown layout {layout!r} (valid: {', '.join(LAYOUTS)})")


def _check_tiling(v: "_Checker", tiling: Any) -> int:
    if tiling is None:
        return 1
    if not isinstance(tiling, dict):
        v.fail("tiling must be an object, e.g. {\"tiles\": 7}")
        return 1
    v.keys(tiling, TILING_KEYS, "tiling.")
    v.integer(tiling, "tiles", minimum=1, where="tiling.")
    v.integer(tiling, "machinesPerTile", minimum=0, where="tiling.")
    v.number(tiling, "seamGap", minimum=0.0, where="tiling.")
    v.choice(tiling, "jobScope", {"tile", "open"}, where="tiling.")
    v.choice(tiling, "agvAssignment", {"tile", "pooled"}, where="tiling.")
    v.choice(tiling, "releaseRule", {"roundrobin", "leastwip", "weighted"}, where="tiling.")
    tiles = tiling.get("tiles", 1)
    tiles = tiles if _is_int(tiles) and tiles >= 1 else 1
    weighted = str(tiling.get("releaseRule", "")).strip().lower() == "weighted"
    weights = tiling.get("releaseWeights")
    if weights is not None:
        if not weighted:
            v.fail("tiling.releaseWeights needs releaseRule \"weighted\"")
        elif not isinstance(weights, list) or len(weights) != tiles:
            v.fail(f"tiling.releaseWeights must be a list with one weight per tile ({tiles}), got {weights!r}")
        elif not all(_is_number(w) and w > 0 for w in weights):
            v.fail(f"tiling.releaseWeights must all be finite numbers > 0 (got {weights!r})")
    elif weighted:
        v.fail(f"tiling.releaseRule \"weighted\" needs releaseWeights, one per tile ({tiles})")
    return tiles


def _machine_type(value: Any):
    """@brief Canonical machine type name (ScenarioLoader parses case-insensitively), or None if unknown."""
    if not isinstance(value, str):
        return None
    return next((t for t in MACHINE_TYPES if t.lower() == value.lower()), None)


def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


class _Checker:
    """@brief Collects every problem so one error lists them all."""

    def __init__(self):
        self.errors: List[str] = []

    def fail(self, message: str) -> None:
        self.errors.append(message)

    def raise_if_any(self, what: str) -> None:
        if self.errors:
            raise ConfigValidationError(what, self.errors)

    def keys(self, obj: Dict[str, Any], known: set, where: str) -> None:
        for key in obj:
            if not str(key).startswith("_") and key not in known:
                self.fail(f"{where}{key}: unknown key (typo? valid keys: {', '.join(sorted(known))})")

    def number(self, obj, key, minimum=None, maximum=None, exclusive_min=None, required=False, where=""):
        if key not in obj:
            if required:
                self.fail(f"{where}{key} is required")
            return
        x = obj[key]
        if not _is_number(x):
            self.fail(f"{where}{key} must be a finite number (got {x!r})")
        elif minimum is not None and x < minimum:
            self.fail(f"{where}{key} must be >= {minimum} (got {x})")
        elif exclusive_min is not None and not x > exclusive_min:
            self.fail(f"{where}{key} must be > {exclusive_min} (got {x})")
        elif maximum is not None and x > maximum:
            self.fail(f"{where}{key} must be <= {maximum} (got {x})")

    def integer(self, obj, key, minimum=None, required=False, where=""):
        if key not in obj:
            if required:
                self.fail(f"{where}{key} is required")
            return
        x = obj[key]
        if not _is_int(x):
            self.fail(f"{where}{key} must be an integer (got {x!r})")
        elif minimum is not None and x < minimum:
            self.fail(f"{where}{key} must be >= {minimum} (got {x})")

    def choice(self, obj, key, allowed: set, where=""):
        if key in obj and (not isinstance(obj[key], str) or obj[key].strip().lower() not in allowed):
            self.fail(f"{where}{key}: {obj[key]!r} is not one of {', '.join(sorted(allowed))}")
