"""Scenario JSON (ScenarioLoader schema) -> the twin's job list (des_jobs/1), without a Unity run.

Ports ScenarioLoader.BuildJobs for a typed floor: an op's "machineIndex" ("any", a single index, or an index
array, optionally with a per-machine duration array) counts machines of that PRIMARY type in ascending id order
(FactoryLayoutManager.PrimaryMachinesOfType), and each op's eligible list keeps Unity's dictionary insertion
order, which breaks rule ties. Arrival times and durations are float32 in Unity (JToken.Value<float>); they are
written here the way des_jobs.json writes them (shortest float32 repr), so a resolved instance and an exported
one run identically.

Machine flexibility (machineFlexibilityProbability > 0) is not supported: Unity draws each machine's secondary
capabilities from UnityEngine.Random at floor build, which the twin does not reproduce.
"""
import numpy as np

TYPES = ("Mill", "Lathe", "Weld", "Inspect", "Assemble")


def _f32(x):
    """float32 value as des_jobs.json writes it (shortest round-trip repr of the float32)."""
    return float(str(np.float32(x)))


def _canon_type(name):
    for t in TYPES:
        if t.lower() == str(name).lower():
            return t
    raise ValueError(f"unknown machine type {name!r}")


def check_floor(scenario, floor):
    """Raises when the scenario cannot run on this exported floor (different machine types, flexibility)."""
    # The floor layout places the types in its own order (D interleaves them), so compare counts per type.
    layout = sorted(_canon_type(t) for t in scenario["machineTypeLayout"])
    types = sorted(m.type for m in floor.machines)
    if layout != types:
        raise ValueError(f"scenario machineTypeLayout {layout} does not match the floor's machine types {types}")
    if float(scenario.get("machineFlexibilityProbability", 0.0) or 0.0) > 0.0:
        raise NotImplementedError("machine flexibility is not modelled by the twin (capabilities are drawn from "
                                  "UnityEngine.Random at floor build)")


def resolve_jobs(scenario, floor):
    """ScenarioLoader.BuildJobs (strict) on a typed floor. Returns a des_jobs/1 dict for Twin / run_twin."""
    check_floor(scenario, floor)
    primary = {}
    for m in sorted(floor.machines, key=lambda m: m.id):
        primary.setdefault(m.type, []).append(m.id)

    jobs = []
    next_auto = 0
    for j, raw in enumerate(scenario["jobs"]):
        jid = int(raw["id"]) if raw.get("id") is not None else next_auto
        next_auto = max(next_auto, jid + 1)
        ops = []
        for o, op in enumerate(raw["operations"]):
            t = _canon_type(op["machineType"])
            ids = primary.get(t, [])
            if not ids:
                raise ValueError(f"job {jid} op {o}: no machines of type {t} on the floor")
            idx = op.get("machineIndex")
            dur = op["duration"]
            if idx is None or (isinstance(idx, str) and idx.lower() == "any"):
                eligible = [[mid, _f32(dur)] for mid in ids]
            elif isinstance(idx, list):
                if isinstance(dur, list) and len(dur) != len(idx):
                    raise ValueError(f"job {jid} op {o}: duration array length {len(dur)} != machineIndex "
                                     f"length {len(idx)}")
                eligible = {}
                for k, i in enumerate(idx):
                    if not 0 <= int(i) < len(ids):
                        raise ValueError(f"job {jid} op {o}: machineIndex {i} out of range for {t} ({len(ids)})")
                    eligible[ids[int(i)]] = _f32(dur[k] if isinstance(dur, list) else dur)
                eligible = [[mid, d] for mid, d in eligible.items()]   # a repeated index keeps its first slot
            else:
                i = int(idx)
                if not 0 <= i < len(ids):
                    raise ValueError(f"job {jid} op {o}: machineIndex {i} out of range for {t} ({len(ids)})")
                eligible = [[ids[i], _f32(dur)]]
            ops.append({"type": t, "eligible": eligible})
        due = raw.get("dueDate")
        jobs.append({"id": jid, "arrival": _f32(raw.get("arrivalTime", 0.0)), "ops": ops,
                     "due": None if due is None else _f32(due)})

    jobs.sort(key=lambda jd: (np.float32(jd["arrival"]), jd["id"]))
    return {"schema": "des_jobs/1", "instance": scenario.get("name"), "seed": scenario.get("seed", 42),
            "jobs": jobs}


def episode_settings(scenario):
    """Episode-control fields of a scenario: (warm-up seconds, episode duration cap, warm-up rule).

    Failures, dynamic (Poisson) arrivals and bursts in the "stochastic" block are not modelled and raise."""
    s = scenario.get("stochastic") or {}
    unsupported = [k for k in ("machineFailuresEnabled", "agvFailuresEnabled", "dynamicArrivalsEnabled",
                               "burstArrivalsEnabled") if s.get(k)]
    if unsupported:
        raise NotImplementedError(f"the twin does not model {', '.join(unsupported)}; generate scenarios with "
                                  "these off (e.g. RandomizedParams(failure_probability=0))")
    return (float(s.get("warmupSeconds", 0.0) or 0.0), float(s.get("episodeDurationSeconds", 0.0) or 0.0),
            scenario.get("dispatchingRule", "SRT_SRWT"))


def assign_due_dates(jobs_data, allowance):
    """Total-work-content (TWK) due dates (Blackstone, Phillips & Hogg 1982), the method of Rajendran & Holthaus
    1999, Holthaus & Rajendran 1997 and Sels et al. 2012: d_i = r_i + c * TWK_i, with TWK_i the job's work content
    estimated as the thesis's remaining work at arrival (fastest eligible machine per operation). Operation due
    dates for MOD split the allowance in proportion to work (Baker & Kanet 1983): d_io = r_i + c * (work of
    operations 0..o). Returns a copy of the des_jobs/1 dict with "due" and "op_due" on every job.

    @param allowance  the allowance factor c (> 0); larger is looser."""
    if not allowance > 0:
        raise ValueError(f"allowance factor must be > 0 (got {allowance})")
    out = dict(jobs_data)
    out["jobs"] = []
    for jd in jobs_data["jobs"]:
        cum, op_due = 0.0, []
        for op in jd["ops"]:
            cum += min(float(np.float32(d)) for _, d in op["eligible"])
            op_due.append(float(jd["arrival"]) + allowance * cum)
        out["jobs"].append(dict(jd, due=op_due[-1], op_due=op_due))
    out["due_date_allowance"] = float(allowance)
    return out


def agv_schedule(scenario):
    """TwinConfig.agv_schedule from the scenario's "agvSchedule" ([{"start": t, "agvCount": n}, ...], written by the
    randomized generator's regime blocks with a fleet_mix); () when it has none."""
    return tuple((float(e["start"]), int(e["agvCount"])) for e in sorted(scenario.get("agvSchedule") or [],
                                                                         key=lambda e: e["start"]))
