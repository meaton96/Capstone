#!/usr/bin/env python3
"""
generate_scenarios.py

Generates hand-crafted scenario JSON files consumed by ScenarioLoader.cs
(-scenario / -scenariodir). Unlike FJSSPJobGenerator's i.i.d.-random job
generation, these instances place contention deliberately -- a forced
bottleneck machine, two machines standing off against each other, an
identical-job batch, a mix of short and long jobs sharing one resource,
periodic arrival waves -- so dispatching/routing rule choice actually has
something to prove itself against. i.i.d.-random jobs tend to spread
contention thin enough across enough machines that rule choice rarely
matters, which is the working theory for why the generated l008 scenarios
showed so little rule differentiation outside outright failure collapse.

Each generator function returns a scenario dict (matching ScenarioLoader's
JSON schema); main() writes them all to this directory. Run directly:

    python3 generate_scenarios.py

ScenarioLoader schema reminder — "machineIndex" per operation is one of:
  "any"        -- eligible = every runtime machine of that type (today's
                  default generation behavior; no forced contention)
  <int>        -- eligible = exactly that one machine (0-based index within
                  the type); pins the op, removing routing choice entirely
  [<int>, ...] -- eligible = exactly that subset; routing rule has to choose
                  among a small, specific set of contended candidates
"""
import json
import os
import random

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Same floor as the l008 scenario matrix (3 machines per type x 5 types),
# so results are visually/scale-comparable to the existing generated runs.
DEFAULT_FLOOR = (
    ["Mill"] * 3 + ["Lathe"] * 3 + ["Weld"] * 3 + ["Inspect"] * 3 + ["Assemble"] * 3
)
DEFAULT_AGV_COUNT = 7
MACHINES_PER_TYPE = 3  # matches DEFAULT_FLOOR -- used by zero_contention's disjointness check

# 3-3-3-3-1: every type keeps its 3 instances except Weld, which drops to a
# single machine -- a permanent, unscripted structural bottleneck (as opposed
# to single_bottleneck/two_machine_standoff, which force contention onto the
# DEFAULT_FLOOR via machineIndex pinning even though 3 Weld instances exist).
# Tests whether real capacity scarcity reproduces the same routing-rule
# tiering (SRWT > MMUR > SMPT) two_machine_standoff showed for a scripted
# 2-machine standoff -- see sparse_bottleneck_floor().
SPARSE_FLOOR = (
    ["Mill"] * 3 + ["Lathe"] * 3 + ["Weld"] * 1 + ["Inspect"] * 3 + ["Assemble"] * 3
)
SPARSE_AGV_COUNT = 6


def op(machine_type, machine_index, duration):
    """duration is normally a scalar (same cost for every eligible machine).
    When machine_index is a list, duration may instead be a same-length list,
    pairing duration[k] with machine_index[k] -- ScenarioLoader then gives
    each candidate machine a genuinely different cost for this job, instead
    of a tie. Without that, a routing rule scoring on per-job processing
    time (SMPT) sees identical candidates and its tie-break deterministically
    always picks the same one -- total collapse onto one machine, not graded
    imbalance. See two_machine_standoff() for the graded version.
    """
    if isinstance(duration, (list, tuple)):
        return {"machineType": machine_type, "machineIndex": machine_index,
                "duration": [round(d, 2) for d in duration]}
    return {"machineType": machine_type, "machineIndex": machine_index, "duration": round(duration, 2)}


def job(job_id, arrival_time, ops):
    return {"id": job_id, "arrivalTime": round(arrival_time, 2), "operations": ops}


def scenario(name, jobs, comment, seed=42, agv_count=DEFAULT_AGV_COUNT, floor=None):
    return {
        "_comment": comment,
        "name": name,
        "seed": seed,
        "agvCount": agv_count,
        "machineTypeLayout": floor if floor is not None else DEFAULT_FLOOR,
        "jobs": jobs,
    }


def write(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"wrote {path}  ({len(data['jobs'])} jobs)")


# ─────────────────────────────────────────────────────────────────────────
#  Archetype generators
# ─────────────────────────────────────────────────────────────────────────

def zero_contention(n_jobs=3, seed=42):
    """Control scenario: every job's operations run on a private machine no
    other concurrently-active job touches. Capped at MACHINES_PER_TYPE (3) --
    beyond that, jobs would have to start sharing a type's instances and this
    stops being genuinely contention-free. Expect every rule to converge:
    if this scenario shows differentiation, something else is going on
    (arrival-order effects, AGV contention) rather than dispatch/routing
    choice, since there's no queue to choose from.
    """
    if n_jobs > MACHINES_PER_TYPE:
        print(f"[zero_contention] WARNING: n_jobs={n_jobs} > {MACHINES_PER_TYPE} "
              f"machines/type -- jobs will start sharing machines, this is no "
              f"longer a true zero-contention control.")
    rng = random.Random(seed)
    jobs = []
    types = ["Mill", "Lathe", "Weld"]
    for i in range(n_jobs):
        idx = i % MACHINES_PER_TYPE
        ops = [op(t, idx, rng.uniform(20, 50)) for t in types]
        jobs.append(job(i, 0.0, ops))
    return scenario(
        "zero_contention", jobs, seed=seed,
        comment=(f"Control: {n_jobs} jobs, each pinned to its own private machine "
                 f"(index {{i}} % {MACHINES_PER_TYPE}) for every op -- no two jobs ever "
                 f"compete for the same machine. All rules should converge; this is "
                 f"the baseline that any other scenario's differentiation is measured against."),
    )


def single_bottleneck(n_jobs=8, bottleneck_type="Weld", bottleneck_index=1,
                       feeder_type="Mill", feeder_duration=20.0,
                       duration_range=(10.0, 90.0), seed=42):
    """n_jobs all arrive together, do a quick uncontended feeder op, then
    every one of them converges on the SAME single bottleneck_type[bottleneck_index]
    machine with a spread of durations. That queue is where dispatch-rule
    choice (which waiting job gets pulled next -- SPT/LPT/SRT/FIFO) should
    produce visibly different clearing orders and flow-time distributions.
    Routing rule is irrelevant here (eligibility is a single machine, no
    choice) -- this isolates the SelectJob half of a rule, not SelectMachine.
    """
    rng = random.Random(seed)
    jobs = []
    for i in range(n_jobs):
        d = rng.uniform(*duration_range)
        ops = [
            op(feeder_type, "any", feeder_duration),
            op(bottleneck_type, bottleneck_index, d),
        ]
        jobs.append(job(i, 0.0, ops))
    return scenario(
        "single_bottleneck", jobs, seed=seed,
        comment=(f"{n_jobs} jobs arrive at t=0, quick feeder op on any {feeder_type}, "
                 f"then ALL converge on the single {bottleneck_type}[{bottleneck_index}] "
                 f"machine with durations spread over {duration_range}. Isolates dispatch-"
                 f"order effects (SelectJob) -- no routing choice exists here."),
    )


def two_machine_standoff(n_jobs=10, contested_type="Weld", indices=(0, 1),
                          feeder_type="Mill", feeder_duration=20.0,
                          duration_range=(15.0, 60.0), cost_gap=(1.3, 2.0), seed=42):
    """n_jobs all arrive together, quick feeder op, then every job is eligible
    for EXACTLY the two machines in `indices` of contested_type (not "any" --
    that would dilute contention across all 3 instances of the type; not a
    single pinned index -- that removes routing choice entirely).

    v2: each job's two candidate machines have GENUINELY DIFFERENT costs --
    one index is randomly picked as the "cheap" one (duration d) and the
    other as "expensive" (d * cost_gap), independently per job, so which
    machine is actually better varies job to job. v1 gave both candidates
    the identical duration, which meant SMPT's per-job-processing-time score
    was permanently tied between the two machines -- its ArgMinIdx tie-break
    then deterministically sent 100% of jobs to the same one (confirmed via
    decision_log.csv: 180-183 decisions to machine index 0, zero to index 1,
    every SMPT-family run). That's a real failure mode (SMPT has no
    congestion-awareness at all) but a different, more extreme one than
    "picks badly" -- total collapse from a tie, not a bad greedy choice. This
    version gives SMPT a real per-job decision to get right or wrong, so its
    routing choices can be checked directly (does it pick the actually-
    cheaper machine, and does that add up to a good aggregate outcome, or
    does ignoring the other machine's queue depth still cost it against
    SRWT/MMUR).
    """
    rng = random.Random(seed)
    jobs = []
    for i in range(n_jobs):
        d = rng.uniform(*duration_range)
        gap = rng.uniform(*cost_gap)
        durations = [d, d * gap] if rng.random() < 0.5 else [d * gap, d]
        ops = [
            op(feeder_type, "any", feeder_duration),
            op(contested_type, list(indices), durations),
        ]
        jobs.append(job(i, 0.0, ops))
    return scenario(
        "two_machine_standoff", jobs, seed=seed,
        comment=(f"{n_jobs} jobs arrive at t=0, quick feeder op, then every job is "
                 f"eligible for EXACTLY {contested_type}{list(indices)} (a 2-machine "
                 f"subset) with a GENUINELY different cost per candidate machine (one "
                 f"index is {cost_gap[0]}-{cost_gap[1]}x cheaper than the other, "
                 f"randomly per job) -- not the identical-cost tie v1 had. Isolates "
                 f"routing-rule effects (SelectMachine): does the rule pick the "
                 f"actually-cheaper machine, and does that add up to a good outcome "
                 f"once the other machine's queue depth is accounted for or ignored."),
    )


def speed_trap_standoff(n_jobs=63, contested_type="Weld", indices=(0, 1),
                         feeder_type="Mill", feeder_duration=20.0,
                         mean_duration=40.0, cost_gap=1.5, seed=42):
    """Same 2-machine-standoff shape as two_machine_standoff, but the cost
    asymmetry is CONSISTENT across every job (index 0 is always cheaper) --
    not two_machine_standoff v2's per-job-randomized 50/50 gap, and not v1's
    degenerate identical-cost tie. Both of those designs accidentally let
    -SMPT rules off the hook: v1's tie-break artifact isn't a real greedy
    mistake, and v2's random 50/50 cheap-machine assignment self-balances
    load across both machines even if SMPT always "correctly" picks the
    cheaper one for that job. Here, SMPT's greedy per-job choice is
    genuinely correct every single time (index 0 truly is cheaper for every
    job) -- and that's exactly the trap: it funnels every job onto machine 0
    while machine 1 sits idle, never noticing machine 0's queue is
    saturating. n_jobs is sized so that funneled onto ONE machine alone,
    utilization is ~1.4 (guaranteed unbounded backlog growth); balanced
    across both, it's ~0.7 (comfortable) -- SRWT/MMUR rules should balance
    load and stay near the comfortable number, -SMPT rules should not.
    """
    rng = random.Random(seed)
    jobs = []
    for i in range(n_jobs):
        d = mean_duration * rng.uniform(0.85, 1.15)
        ops = [
            op(feeder_type, "any", feeder_duration),
            op(contested_type, list(indices), [d, d * cost_gap]),
        ]
        jobs.append(job(i, 0.0, ops))
    return scenario(
        "speed_trap_standoff", jobs, seed=seed,
        comment=(f"{n_jobs} jobs arrive at t=0, quick feeder op, then every job is "
                 f"eligible for EXACTLY {contested_type}{list(indices)}, with index "
                 f"{indices[0]} CONSISTENTLY {cost_gap}x cheaper than index "
                 f"{indices[1]} for every job (not two_machine_standoff's per-job-"
                 f"randomized gap). Tests whether a routing rule that greedily always "
                 f"picks the genuinely-faster machine (SMPT) loses to one that balances "
                 f"queue depth or utilization (SRWT/MMUR) once the faster machine "
                 f"saturates -- a 'locally correct, globally wrong' trap, distinct from "
                 f"two_machine_standoff's per-job speed-gap test."),
    )


def sparse_bottleneck_floor(n_jobs=15, seed=42):
    """Same shape of job as single_bottleneck/two_machine_standoff -- a batch
    that arrives together and needs Mill, then Weld, then Assemble -- but run
    on SPARSE_FLOOR instead of DEFAULT_FLOOR, with every op using "any"
    eligibility (no machineIndex pinning at all). Weld only has ONE instance
    on this floor, so "any" resolves to exactly that one machine by
    construction -- the bottleneck is a property of the floor, not a scripted
    pin. If routing-rule tiering (SRWT > MMUR > SMPT) shows up here despite
    nothing being scripted, that's evidence the mechanism two_machine_standoff
    isolated is a real property of capacity scarcity, not an artifact of how
    the pinned scenarios were constructed. Mill/Assemble stay at 3 instances
    (unscarce), so any differentiation should trace specifically to the Weld
    step, not general floor pressure.
    """
    rng = random.Random(seed)
    jobs = []
    for i in range(n_jobs):
        ops = [
            op("Mill", "any", rng.uniform(15, 40)),
            op("Weld", "any", rng.uniform(15, 60)),
            op("Assemble", "any", rng.uniform(15, 40)),
        ]
        jobs.append(job(i, 0.0, ops))
    return scenario(
        "sparse_bottleneck_floor", jobs, seed=seed,
        floor=SPARSE_FLOOR, agv_count=SPARSE_AGV_COUNT,
        comment=(f"{n_jobs} jobs arrive at t=0, each needs any Mill -> the single Weld "
                 f"machine (SPARSE_FLOOR has only 1, vs 3 on DEFAULT_FLOOR) -> any "
                 f"Assemble, all eligibility is \"any\" -- nothing pinned or scripted. "
                 f"The Weld step is a bottleneck purely because the floor only has one "
                 f"such machine. Compare against single_bottleneck (same job shape, "
                 f"scripted pin, DEFAULT_FLOOR) to see whether a real floor-topology "
                 f"bottleneck reproduces the same rule tiering as a scripted one."),
    )


def identical_batch(n_jobs=6, bottleneck_type="Weld", bottleneck_index=1,
                     feeder_type="Mill", feeder_duration=20.0,
                     op_duration=40.0, seed=42):
    """n_jobs COPIES of the exact same job (same op count, same durations,
    same pinned bottleneck machine), all arriving together. Every score-based
    rule (SPT/LPT/SRT/LRT) is degenerate here -- every candidate has an
    identical score -- so whatever the queue-selection code does on a tie
    (currently: lowest job ID, i.e. de facto FIFO) is ALL that determines
    outcome. Exposes whether a rule's apparent behavior on diverse instances
    is really its scoring logic or just how it breaks ties, which diverse
    instances rarely expose since exact score ties are rare there.
    """
    jobs = []
    for i in range(n_jobs):
        ops = [
            op(feeder_type, "any", feeder_duration),
            op(bottleneck_type, bottleneck_index, op_duration),
        ]
        jobs.append(job(i, 0.0, ops))
    return scenario(
        "identical_batch", jobs, seed=seed,
        comment=(f"{n_jobs} IDENTICAL jobs (same feeder + same {op_duration}s op on "
                 f"{bottleneck_type}[{bottleneck_index}]) all arrive at t=0. Every score-"
                 f"based rule ties on every candidate -- outcome is determined entirely "
                 f"by tie-break behavior (currently lowest job ID / de facto FIFO)."),
    )


def bimodal_mix(n_long=2, long_duration=150.0,
                 n_short=40, short_interval=5.0, short_duration=12.0,
                 short_start_delay=5.0,
                 bottleneck_type="Weld", bottleneck_index=1, seed=42):
    """v3 -- sharper imbalance than v2. v1 had all jobs arrive together
    (differences washed out in the mean); v2 streamed short jobs in but only
    at a 1.25x arrival/service imbalance (8s arrivals vs 10s service), which
    was too close to parity -- early in the stream there were still brief
    gaps where the queue emptied and a long job could sneak through, so the
    observed effect (SPT: 668s vs FIFO: 384.8s flow-time for the 2nd long
    job) was real but modest, not the stark number intended.

    v3 widens the imbalance to ~2.4x (5s arrivals vs 12s service): the short-
    job queue is now robustly non-empty almost immediately and stays that
    way for the whole stream, plus accumulates a substantial excess backlog
    ((12-5) * n_short =~ 280s worth) that still has to drain AFTER the stream
    ends before the long jobs get a turn under SPT.

    Arrival STRUCTURE is still what makes this work, not just the duration
    mix: n_long jobs arrive at t=0 (so FIFO/age-based ordering puts them
    first, unstarved, as a baseline); n_short jobs then stream in afterward
    (short_start_delay, then every short_interval seconds), all single-op,
    all pinned to the same bottleneck machine as the long jobs. FIFO
    processes both long jobs first (they arrived before any short job
    existed) and never starves them. SRT/LRT are single-op-remaining for
    every job here, so they degenerate to SPT/LPT exactly -- expect SRT_* to
    match SPT_* and LRT_MMUR to match LPT_*. LPT prioritizes the long jobs
    immediately (like FIFO here, since there are only n_long of them) but
    then makes the SHORT jobs wait behind both long jobs -- a much smaller,
    bounded delay than SPT's deferral, not a symmetric tradeoff.

    Read the flow-time of job IDs 0..n_long-1 specifically (not mean/p95 --
    with n_short vastly outnumbering n_long, the long jobs may not even land
    in the p95 tail) to see the starvation effect directly.
    """
    rng = random.Random(seed)
    jobs = []
    jid = 0
    for _ in range(n_long):
        jobs.append(job(jid, 0.0, [op(bottleneck_type, bottleneck_index, long_duration)]))
        jid += 1
    for i in range(n_short):
        arrival = short_start_delay + i * short_interval
        d = short_duration * rng.uniform(0.9, 1.1)
        jobs.append(job(jid, arrival, [op(bottleneck_type, bottleneck_index, d)]))
        jid += 1
    return scenario(
        "bimodal_mix", jobs, seed=seed,
        comment=(f"{n_long} long jobs (single {long_duration}s op) arrive at t=0 on "
                 f"{bottleneck_type}[{bottleneck_index}]; starting at t={short_start_delay}, "
                 f"{n_short} short jobs (~{short_duration}s) stream in every "
                 f"{short_interval}s on the SAME machine -- arrival rate (1/{short_interval}) "
                 f"exceeds single-machine service rate for short jobs alone (1/{short_duration}), "
                 f"so the queue never empties for the whole {n_short * short_interval:.0f}s "
                 f"stream. Under SPT the long jobs never look shortest and starve for the "
                 f"whole stream; under FIFO they ran first at t=0 and never starve. Look at "
                 f"job IDs 0..{n_long - 1}'s individual flow-time, not the mean/p95 -- they're "
                 f"outnumbered {n_short}:{n_long} so aggregate stats can dilute the effect."),
    )


def convoy_waves(n_waves=4, jobs_per_wave=5, wave_interval=400.0,
                  bottleneck_type="Weld", bottleneck_index=1,
                  feeder_type="Mill", feeder_duration=20.0,
                  duration_range=(15.0, 60.0), seed=42):
    """Periodic bursts of jobs_per_wave jobs every wave_interval seconds
    (instead of steady Poisson arrivals), each wave converging on the same
    bottleneck machine. Tests whether a rule fully clears each wave's backlog
    during the quiet gap before the next wave lands (steady recovery) or
    accumulates residual queue wave-over-wave (resonance/beat-frequency
    buildup) -- a different failure mode than steady-state saturation.
    """
    rng = random.Random(seed)
    jobs = []
    jid = 0
    for w in range(n_waves):
        arrival = w * wave_interval
        for _ in range(jobs_per_wave):
            d = rng.uniform(*duration_range)
            ops = [
                op(feeder_type, "any", feeder_duration),
                op(bottleneck_type, bottleneck_index, d),
            ]
            jobs.append(job(jid, arrival, ops))
            jid += 1
    return scenario(
        "convoy_waves", jobs, seed=seed,
        comment=(f"{n_waves} waves of {jobs_per_wave} jobs every {wave_interval}s, each "
                 f"wave converging on {bottleneck_type}[{bottleneck_index}]. Tests whether "
                 f"a rule clears each wave's backlog before the next lands, or accumulates "
                 f"residual queue wave-over-wave."),
    )


def _quiet_phase_jobs(rng, id_start, t0, duration, n_jobs=10):
    """Light, spread-out load across Mill/Lathe/Inspect (never Weld, so it
    never touches the bottleneck machine at all) -- a genuine recovery/
    baseline window between stress phases, not just a short gap.
    """
    types = ["Mill", "Lathe", "Inspect"]
    jobs = []
    interval = duration / n_jobs
    for i in range(n_jobs):
        t = t0 + i * interval
        d = rng.uniform(20, 40)
        jobs.append(job(id_start + i, t, [op(types[i % len(types)], "any", d)]))
    return jobs, n_jobs


def _bottleneck_phase_jobs(rng, id_start, t0, duration, mean_duration=55.0,
                            bottleneck_type="Weld", index=1, utilization=0.85):
    """Streamed (not lump-sum) arrivals onto ONE pinned machine, job count
    picked from a target utilization so the phase is sustained pressure for
    roughly its whole nominal duration rather than a burst that either drains
    in seconds or overflows for 3x the window. n_jobs = utilization *
    duration / mean_duration (a 1-server queue's job count for that
    utilization over that time).
    """
    n_jobs = max(1, int(utilization * duration / mean_duration))
    jobs = []
    interval = duration / n_jobs
    for i in range(n_jobs):
        t = t0 + i * interval
        d = mean_duration * rng.uniform(0.7, 1.3)
        jobs.append(job(id_start + i, t, [op("Mill", "any", 15.0), op(bottleneck_type, index, d)]))
    return jobs, n_jobs


def _standoff_phase_jobs(rng, id_start, t0, duration, mean_duration=40.0,
                          contested_type="Weld", indices=(0, 1), utilization=0.7,
                          cost_gap=(1.3, 2.0)):
    """Same idea as _bottleneck_phase_jobs but for a 2-server subset --
    combined capacity is ~2x a single server, so utilization is measured
    against len(indices) servers. Each job's two candidates have a
    genuinely different cost (one is cost_gap x more expensive, randomly
    per job) -- see two_machine_standoff()'s docstring for why an identical
    cost across candidates makes SMPT's tie-break collapse to one machine
    rather than showing graded imbalance.
    """
    n_servers = len(indices)
    n_jobs = max(1, int(utilization * duration * n_servers / mean_duration))
    jobs = []
    interval = duration / n_jobs
    for i in range(n_jobs):
        t = t0 + i * interval
        d = mean_duration * rng.uniform(0.7, 1.3)
        gap = rng.uniform(*cost_gap)
        durations = [d, d * gap] if rng.random() < 0.5 else [d * gap, d]
        jobs.append(job(id_start + i, t,
                         [op("Mill", "any", 15.0), op(contested_type, list(indices), durations)]))
    return jobs, n_jobs


def _speed_trap_phase_jobs(rng, id_start, t0, duration, mean_duration=40.0,
                            contested_type="Weld", indices=(0, 1), cost_gap=1.5,
                            funneled_utilization=1.4):
    """Streamed version of speed_trap_standoff() (see that function's docstring for the
    mechanism) -- same 2-machine contest as _standoff_phase_jobs, but index[0] is
    CONSISTENTLY cost_gap x cheaper for every job, never randomized/swapped. A routing rule
    that greedily always picks the genuinely-cheaper machine (SMPT) is making the locally
    correct choice every time, and that's the trap: it funnels the whole stream onto one
    machine while the other sits idle, never noticing the queue it's building.

    n_jobs is sized against funneled_utilization (>1.0 = if concentrated onto machine 0
    alone, that machine is oversaturated for the whole phase -- guaranteed unbounded
    backlog under -SMPT rules) rather than the balanced-capacity utilization
    _standoff_phase_jobs uses, since here the two candidate machines are not
    interchangeable by design -- one is unambiguously the "attractive" one.
    """
    n_jobs = max(1, int(funneled_utilization * duration / mean_duration))
    jobs = []
    interval = duration / n_jobs
    for i in range(n_jobs):
        t = t0 + i * interval
        d = mean_duration * rng.uniform(0.85, 1.15)
        jobs.append(job(id_start + i, t,
                         [op("Mill", "any", 15.0),
                          op(contested_type, list(indices), [d, d * cost_gap])]))
    return jobs, n_jobs


def _burst_phase_jobs(rng, id_start, t0, n_jobs=30, mean_duration=40.0,
                       bottleneck_type="Weld", index=1):
    """A deliberate shock -- every job lands in the same few seconds, unlike
    every other phase here. This one is SUPPOSED to overflow its nominal
    window; the point is watching how long recovery takes afterward.
    """
    jobs = []
    for i in range(n_jobs):
        t = t0 + rng.uniform(0.0, 5.0)
        d = mean_duration * rng.uniform(0.7, 1.3)
        jobs.append(job(id_start + i, t, [op("Mill", "any", 15.0), op(bottleneck_type, index, d)]))
    return jobs, n_jobs


def _starvation_phase_jobs(rng, id_start, t0, n_long=3, long_duration=180.0,
                            n_short=80, short_interval=5.0, short_duration=12.0,
                            bottleneck_type="Weld", index=1):
    """Same mechanism as bimodal_mix v3: n_long jobs land at t0 (unstarved
    under FIFO/age-based ordering), then a continuous short-job stream at a
    rate (~2.4x) that outruns single-machine service capacity, so the queue
    never empties for the whole stream and keeps accumulating excess backlog
    -- guarantees SPT-family rules defer the long jobs for the entire stream
    plus whatever's left to drain afterward.
    """
    jobs = [job(id_start + i, t0, [op(bottleneck_type, index, long_duration)])
            for i in range(n_long)]
    for i in range(n_short):
        t = t0 + 5.0 + i * short_interval
        d = short_duration * rng.uniform(0.9, 1.1)
        jobs.append(job(id_start + n_long + i, t, [op(bottleneck_type, index, d)]))
    span = 5.0 + n_short * short_interval
    return jobs, n_long + n_short, span


def compound_scenario(seed=42):
    """A long, cyclical episode -- not a quick tour of six phases, a
    sustained one. Every stress phase is sized (via a target-utilization
    job count, not a fixed job count) so it actually reaches sustained
    pressure for close to its full nominal duration instead of draining in
    seconds or overflowing for 3x the window -- the failure mode the first
    version of this scenario had, where "most of the WIP was from jobs added
    at the start" because phases were too small to show anything but a
    single arrival spike.

    bottleneck and standoff are each visited TWICE (with two other phases
    and quiet buffers in between), specifically so a rule's second encounter
    with the same regime can be compared against its first -- does prior
    exposure change anything, or is behavior stable per-regime.

    Timeline (t=start-end, ~10,600s total, ~360 jobs):
      quiet_1              ~400s     light background load, no Weld at all
      bottleneck_1        ~1800s     sustained single-machine pressure, Weld[1] pinned
      quiet_2               400s
      standoff_1          ~1800s     sustained 2-machine contention, Weld[0,1]
      quiet_3               400s
      burst_1              instant   sudden shock -- deliberately overflows its window
      quiet_4               400s
      starvation_1         ~650s     long-job-starvation stream
      quiet_5               400s
      bottleneck_2 (repeat) ~1800s   compare directly against bottleneck_1
      quiet_6               400s
      standoff_2 (repeat)  ~1800s    compare directly against standoff_1
      quiet_7               400s     final cooldown

    "_phases" metadata (start/end/n_jobs per phase) is emitted so the
    analysis script can draw phase-boundary bands without hardcoding the
    schedule a second time.
    """
    rng = random.Random(seed)
    jobs = []
    phases = []
    jid = 0
    t = 0.0

    def record(name, t0, span, n):
        phases.append({"name": name, "start": t0, "end": t0 + span, "n_jobs": n})

    js, n = _quiet_phase_jobs(rng, jid, t, 400.0, n_jobs=15); jobs += js; jid += n
    record("quiet_1", t, 400.0, n); t += 400.0 + 50.0

    js, n = _bottleneck_phase_jobs(rng, jid, t, 1800.0); jobs += js; jid += n
    record("bottleneck_1", t, 1800.0, n); t += 1800.0 + 50.0

    js, n = _quiet_phase_jobs(rng, jid, t, 400.0, n_jobs=12); jobs += js; jid += n
    record("quiet_2", t, 400.0, n); t += 400.0 + 50.0

    js, n = _standoff_phase_jobs(rng, jid, t, 1800.0); jobs += js; jid += n
    record("standoff_1", t, 1800.0, n); t += 1800.0 + 50.0

    js, n = _quiet_phase_jobs(rng, jid, t, 400.0, n_jobs=12); jobs += js; jid += n
    record("quiet_3", t, 400.0, n); t += 400.0 + 50.0

    js, n = _burst_phase_jobs(rng, jid, t, n_jobs=30); jobs += js; jid += n
    record("burst_1", t, 5.0, n); t += 300.0  # extra gap -- this one is meant to overflow

    js, n = _quiet_phase_jobs(rng, jid, t, 400.0, n_jobs=12); jobs += js; jid += n
    record("quiet_4", t, 400.0, n); t += 400.0 + 50.0

    js, n, span = _starvation_phase_jobs(rng, jid, t, n_short=80); jobs += js; jid += n
    record("starvation_1", t, span, n); t += span + 50.0

    js, n = _quiet_phase_jobs(rng, jid, t, 400.0, n_jobs=12); jobs += js; jid += n
    record("quiet_5", t, 400.0, n); t += 400.0 + 50.0

    js, n = _bottleneck_phase_jobs(rng, jid, t, 1800.0); jobs += js; jid += n
    record("bottleneck_2", t, 1800.0, n); t += 1800.0 + 50.0

    js, n = _quiet_phase_jobs(rng, jid, t, 400.0, n_jobs=12); jobs += js; jid += n
    record("quiet_6", t, 400.0, n); t += 400.0 + 50.0

    js, n = _standoff_phase_jobs(rng, jid, t, 1800.0); jobs += js; jid += n
    record("standoff_2", t, 1800.0, n); t += 1800.0 + 50.0

    js, n = _quiet_phase_jobs(rng, jid, t, 400.0, n_jobs=15); jobs += js; jid += n
    record("quiet_7_cooldown", t, 400.0, n); t += 400.0

    s = scenario(
        "compound_scenario", jobs, seed=seed,
        comment=(f"A long (~{t:.0f}s), cyclical episode -- {len(jobs)} jobs across "
                 f"{len(phases)} phases: quiet -> bottleneck -> quiet -> standoff -> quiet -> "
                 f"burst -> quiet -> starvation -> quiet -> bottleneck (repeat) -> quiet -> "
                 f"standoff (repeat) -> quiet_cooldown. Every stress phase's job count is "
                 f"picked from a target utilization so it sustains real pressure for close "
                 f"to its full nominal duration, not a spike that drains in seconds -- v1 of "
                 f"this scenario was too small for any of this to show up as more than an "
                 f"arrival spike. bottleneck/standoff each appear twice so a rule's second "
                 f"encounter with the same regime can be compared directly against its first. "
                 f"See _phases for exact start/end/n_jobs per phase."),
    )
    s["_phases"] = phases
    return s


def compound_scenario_v2(seed=42):
    """compound_scenario plus a new regime: speed_trap (see
    speed_trap_standoff()/_speed_trap_phase_jobs docstrings). Every phase compound_scenario
    had is unchanged and in the same order -- this is additive, not a replacement, so
    compound_scenario stays available as-is for anything that already depends on it
    (existing PDR baselines, prior training runs). speed_trap is inserted right after each
    standoff (thematically closest: both are 2-machine routing contests, but standoff's
    per-job-randomized cost gap happens to self-balance load under greedy-SMPT routing,
    while speed_trap's consistent gap does not -- see speed_trap_standoff's docstring for
    why that distinction turned out to matter empirically: SPT_SMPT loses to SRWT-family
    rules by ~30-40% mean flow time here, the first tested regime in this scenario family
    where it isn't at or near the top).

    Timeline additions vs. compound_scenario (~15,855s total, 533 jobs):
      ... quiet_3a, standoff_1, quiet_3b, speed_trap_1 ~1800s, quiet_3c, burst_1 ...
      ... quiet_6a, standoff_2, quiet_6b, speed_trap_2 ~1800s, quiet_6c, quiet_7_cooldown

    "_phases" metadata unchanged in shape/meaning from compound_scenario.
    """
    rng = random.Random(seed)
    jobs = []
    phases = []
    jid = 0
    t = 0.0

    def record(name, t0, span, n):
        phases.append({"name": name, "start": t0, "end": t0 + span, "n_jobs": n})

    js, n = _quiet_phase_jobs(rng, jid, t, 400.0, n_jobs=15); jobs += js; jid += n
    record("quiet_1", t, 400.0, n); t += 400.0 + 50.0

    js, n = _bottleneck_phase_jobs(rng, jid, t, 1800.0); jobs += js; jid += n
    record("bottleneck_1", t, 1800.0, n); t += 1800.0 + 50.0

    js, n = _quiet_phase_jobs(rng, jid, t, 400.0, n_jobs=12); jobs += js; jid += n
    record("quiet_2", t, 400.0, n); t += 400.0 + 50.0

    js, n = _standoff_phase_jobs(rng, jid, t, 1800.0); jobs += js; jid += n
    record("standoff_1", t, 1800.0, n); t += 1800.0 + 50.0

    js, n = _quiet_phase_jobs(rng, jid, t, 400.0, n_jobs=12); jobs += js; jid += n
    record("quiet_2b", t, 400.0, n); t += 400.0 + 50.0

    js, n = _speed_trap_phase_jobs(rng, jid, t, 1800.0); jobs += js; jid += n
    record("speed_trap_1", t, 1800.0, n); t += 1800.0 + 50.0

    js, n = _quiet_phase_jobs(rng, jid, t, 400.0, n_jobs=12); jobs += js; jid += n
    record("quiet_3", t, 400.0, n); t += 400.0 + 50.0

    js, n = _burst_phase_jobs(rng, jid, t, n_jobs=30); jobs += js; jid += n
    record("burst_1", t, 5.0, n); t += 300.0  # extra gap -- this one is meant to overflow

    js, n = _quiet_phase_jobs(rng, jid, t, 400.0, n_jobs=12); jobs += js; jid += n
    record("quiet_4", t, 400.0, n); t += 400.0 + 50.0

    js, n, span = _starvation_phase_jobs(rng, jid, t, n_short=80); jobs += js; jid += n
    record("starvation_1", t, span, n); t += span + 50.0

    js, n = _quiet_phase_jobs(rng, jid, t, 400.0, n_jobs=12); jobs += js; jid += n
    record("quiet_5", t, 400.0, n); t += 400.0 + 50.0

    js, n = _bottleneck_phase_jobs(rng, jid, t, 1800.0); jobs += js; jid += n
    record("bottleneck_2", t, 1800.0, n); t += 1800.0 + 50.0

    js, n = _quiet_phase_jobs(rng, jid, t, 400.0, n_jobs=12); jobs += js; jid += n
    record("quiet_6", t, 400.0, n); t += 400.0 + 50.0

    js, n = _standoff_phase_jobs(rng, jid, t, 1800.0); jobs += js; jid += n
    record("standoff_2", t, 1800.0, n); t += 1800.0 + 50.0

    js, n = _quiet_phase_jobs(rng, jid, t, 400.0, n_jobs=12); jobs += js; jid += n
    record("quiet_6b", t, 400.0, n); t += 400.0 + 50.0

    js, n = _speed_trap_phase_jobs(rng, jid, t, 1800.0); jobs += js; jid += n
    record("speed_trap_2", t, 1800.0, n); t += 1800.0 + 50.0

    js, n = _quiet_phase_jobs(rng, jid, t, 400.0, n_jobs=15); jobs += js; jid += n
    record("quiet_7_cooldown", t, 400.0, n); t += 400.0

    s = scenario(
        "compound_scenario_v2", jobs, seed=seed,
        comment=(f"compound_scenario plus a new regime -- {len(jobs)} jobs across "
                 f"{len(phases)} phases, ~{t:.0f}s total: quiet -> bottleneck -> quiet -> "
                 f"standoff -> quiet -> speed_trap -> quiet -> burst -> quiet -> "
                 f"starvation -> quiet -> bottleneck (repeat) -> quiet -> standoff "
                 f"(repeat) -> quiet -> speed_trap (repeat) -> quiet_cooldown. speed_trap "
                 f"is the first tested regime where SPT_SMPT (the rule that otherwise wins "
                 f"or ties almost everywhere in compound_scenario) loses decisively to "
                 f"SRWT-family routing -- see speed_trap_standoff()/_speed_trap_phase_jobs "
                 f"for the mechanism. See _phases for exact start/end/n_jobs per phase."),
    )
    s["_phases"] = phases
    return s


# ─────────────────────────────────────────────────────────────────────────
#  Scaled floors (E4: machine count x AGV ratio) -- docs/EXPERIMENT_PLAN_2026-09-23.md
# ─────────────────────────────────────────────────────────────────────────
# The 15-machine scenarios above pin phases to fixed machines (Weld[1], Weld[0,1]) and fix job counts. Scaling
# to M machines keeps each regime's per-machine load: machines per type k = M/5, s = k/3 (the capacity ratio to the
# 15-machine floor), g = round(s) "lanes". A 1-machine bottleneck becomes g independent single-machine bottlenecks
# (Weld[1..g], each at the same utilisation, jobs still pinned so there is no routing choice); the 2-machine
# standoff / speed-trap pool becomes 2g machines (Weld[0..2g-1]). Job counts scale with the same capacity, so
# utilisation per machine is unchanged. At M=15 (s=1, g=1) every function reproduces the original scenario
# EXACTLY (same jobs, same rng draws) -- checked by check_scaled_matches_original() -- so the 15-machine point
# of an E4 sweep is directly comparable with the E1 data. For M not a multiple of 15, g is rounded, so the
# per-machine load of the pinned phases is off by at most 1/(2g)-ish (e.g. M=100: g=7 vs the exact 6.67).

TYPES = ["Mill", "Lathe", "Weld", "Inspect", "Assemble"]


def scaled_floor(machines):
    if machines % len(TYPES):
        raise ValueError(f"machines={machines} must be a multiple of {len(TYPES)} (equal machines per type)")
    return [t for t in TYPES for _ in range(machines // len(TYPES))]


def _scale(machines):
    """(s, g): capacity ratio to the 15-machine floor, and the number of parallel pinned 'lanes'."""
    k = machines // len(TYPES)
    s = k / MACHINES_PER_TYPE
    g = max(1, round(s))
    if 2 * g > k:
        raise ValueError(f"machines={machines}: standoff pool of {2 * g} exceeds {k} Weld machines")
    return s, g


def _bottleneck_phase_scaled(rng, id_start, t0, duration, targets, mean_duration=55.0, utilization=0.85):
    """g single-machine bottlenecks, each _bottleneck_phase_jobs' load, arrivals staggered across the lanes."""
    n = max(1, int(utilization * duration / mean_duration))
    interval = duration / n
    jobs = []
    for c, idx in enumerate(targets):
        for i in range(n):
            t = t0 + i * interval + c * interval / len(targets)
            d = mean_duration * rng.uniform(0.7, 1.3)
            jobs.append(job(id_start + len(jobs), t, [op("Mill", "any", 15.0), op("Weld", idx, d)]))
    return jobs, len(jobs)


def _standoff_phase_scaled(rng, id_start, t0, duration, pool, mean_duration=40.0, utilization=0.7, cost_gap=(1.3, 2.0)):
    """_standoff_phase_jobs over a pool of any size: half of a job's candidates are cost_gap x dearer, chosen at random."""
    m = len(pool)
    n_jobs = max(1, int(utilization * duration * m / mean_duration))
    interval = duration / n_jobs
    jobs = []
    for i in range(n_jobs):
        d = mean_duration * rng.uniform(0.7, 1.3)
        gap = rng.uniform(*cost_gap)
        if m == 2:
            durations = [d, d * gap] if rng.random() < 0.5 else [d * gap, d]
        else:
            cheap = set(rng.sample(range(m), m // 2))
            durations = [d if k in cheap else d * gap for k in range(m)]
        jobs.append(job(id_start + i, t0 + i * interval,
                        [op("Mill", "any", 15.0), op("Weld", list(pool), durations)]))
    return jobs, n_jobs


def _speed_trap_phase_scaled(rng, id_start, t0, duration, pool, mean_duration=40.0, cost_gap=1.5,
                             funneled_utilization=1.4):
    """_speed_trap_phase_jobs over a pool: the first half is consistently cheaper; sized so funnelling onto it saturates it."""
    m = len(pool)
    n_cheap = m // 2
    n_jobs = max(1, int(funneled_utilization * duration * n_cheap / mean_duration))
    interval = duration / n_jobs
    jobs = []
    for i in range(n_jobs):
        d = mean_duration * rng.uniform(0.85, 1.15)
        durations = [d] * n_cheap + [d * cost_gap] * (m - n_cheap)
        jobs.append(job(id_start + i, t0 + i * interval,
                        [op("Mill", "any", 15.0), op("Weld", list(pool), durations)]))
    return jobs, n_jobs


# Phase order and quiet-phase sizes copied from compound_scenario / compound_scenario_v2 above.
_TIMELINE_V1 = [("quiet", "quiet_1", 15), ("bottleneck", "bottleneck_1"), ("quiet", "quiet_2", 12),
                ("standoff", "standoff_1"), ("quiet", "quiet_3", 12), ("burst", "burst_1"),
                ("quiet", "quiet_4", 12), ("starvation", "starvation_1"), ("quiet", "quiet_5", 12),
                ("bottleneck", "bottleneck_2"), ("quiet", "quiet_6", 12), ("standoff", "standoff_2"),
                ("quiet", "quiet_7_cooldown", 15)]
_TIMELINE_V2 = [("quiet", "quiet_1", 15), ("bottleneck", "bottleneck_1"), ("quiet", "quiet_2", 12),
                ("standoff", "standoff_1"), ("quiet", "quiet_2b", 12), ("speed_trap", "speed_trap_1"),
                ("quiet", "quiet_3", 12), ("burst", "burst_1"), ("quiet", "quiet_4", 12),
                ("starvation", "starvation_1"), ("quiet", "quiet_5", 12), ("bottleneck", "bottleneck_2"),
                ("quiet", "quiet_6", 12), ("standoff", "standoff_2"), ("quiet", "quiet_6b", 12),
                ("speed_trap", "speed_trap_2"), ("quiet", "quiet_7_cooldown", 15)]


def compound_scaled(machines, version=1, seed=42, agv_count=None):
    """compound_scenario (version=1) or compound_scenario_v2 (version=2) on a floor of `machines` machines."""
    s, g = _scale(machines)
    rng = random.Random(seed)
    timeline = _TIMELINE_V1 if version == 1 else _TIMELINE_V2
    bn_targets = list(range(1, 1 + g))          # g=1 -> Weld[1], as in the original
    pool = list(range(0, 2 * g))                # g=1 -> Weld[0,1]
    jobs, phases, jid, t = [], [], 0, 0.0
    for entry in timeline:
        kind, name = entry[0], entry[1]
        if kind == "quiet":
            js, n = _quiet_phase_jobs(rng, jid, t, 400.0, n_jobs=max(1, int(round(entry[2] * s))))
            span, adv = 400.0, (400.0 if name == "quiet_7_cooldown" else 450.0)
        elif kind == "bottleneck":
            js, n = _bottleneck_phase_scaled(rng, jid, t, 1800.0, bn_targets); span, adv = 1800.0, 1850.0
        elif kind == "standoff":
            js, n = _standoff_phase_scaled(rng, jid, t, 1800.0, pool); span, adv = 1800.0, 1850.0
        elif kind == "speed_trap":
            js, n = _speed_trap_phase_scaled(rng, jid, t, 1800.0, pool); span, adv = 1800.0, 1850.0
        elif kind == "burst":
            js, n = [], 0
            for idx in bn_targets:
                j2, n2 = _burst_phase_jobs(rng, jid + n, t, n_jobs=30, index=idx); js += j2; n += n2
            span, adv = 5.0, 300.0
        else:  # starvation: an independent long-job-starvation stream on each lane
            js, n = [], 0
            for idx in bn_targets:
                j2, n2, span = _starvation_phase_jobs(rng, jid + n, t, n_short=80, index=idx); js += j2; n += n2
            adv = span + 50.0
        jobs += js; jid += n
        phases.append({"name": name, "start": t, "end": t + span, "n_jobs": n})
        t += adv
    name = f"e4_compound{'_v2' if version == 2 else ''}_m{machines}"
    out = scenario(
        name, jobs, seed=seed, agv_count=agv_count or max(1, round(machines / 3)), floor=scaled_floor(machines),
        comment=(f"E4 scaled {'compound_scenario_v2' if version == 2 else 'compound_scenario'} on {machines} machines "
                 f"({machines // len(TYPES)} per type): {len(jobs)} jobs, ~{t:.0f}s, {len(phases)} phases. Same regimes "
                 f"as the 15-machine scenario at the same per-machine load: {g} parallel single-machine bottleneck "
                 f"lane(s) Weld{bn_targets}, a {len(pool)}-machine standoff/speed-trap pool Weld{pool}; quiet, burst "
                 f"and starvation load scaled with capacity (x{s:.2f}). At 15 machines identical to the original."))
    out["_phases"] = phases
    return out


def steady_scaled(machines, utilization=0.7, seed=42, agv_count=None, mean_duration=55.0, horizon=3600.0):
    """The _mfsweep steady load on `machines` machines: evenly spaced jobs, Mill then any Weld, `utilization` across all
    Weld machines. Regime-matched to _mfsweep_control (137 jobs at 15 machines), not byte-identical (fresh draws)."""
    k = machines // len(TYPES)
    n = max(1, int(utilization * k * horizon / mean_duration))
    rng = random.Random(seed)
    jobs = [job(i, i * horizon / n, [op("Mill", "any", 15.0),
                                     op("Weld", "any", mean_duration * rng.uniform(0.7, 1.3))]) for i in range(n)]
    return scenario(
        f"e4_steady_m{machines}", jobs, seed=seed, agv_count=agv_count or max(1, round(machines / 3)),
        floor=scaled_floor(machines),
        comment=(f"E4 steady load on {machines} machines ({k} Weld): {n} jobs evenly spaced over {horizon:.0f}s, each "
                 f"Mill then any Weld (~{mean_duration:.0f}s +-30%), utilisation {utilization} across the Weld machines "
                 f"-- the _mfsweep_control regime scaled with capacity."))


def check_scaled_matches_original():
    """At 15 machines the scaled compound generators must reproduce compound_scenario / v2 exactly."""
    for version, orig in ((1, compound_scenario()), (2, compound_scenario_v2())):
        sc = compound_scaled(15, version)
        assert sc["jobs"] == orig["jobs"], f"v{version}: scaled jobs differ from the original"
        assert sc["_phases"] == orig["_phases"], f"v{version}: scaled phases differ from the original"
        assert sc["machineTypeLayout"] == orig["machineTypeLayout"]
    print("scaled compound generators reproduce compound_scenario[_v2] exactly at 15 machines")


def write_scaled(machine_counts):
    check_scaled_matches_original()
    for m in machine_counts:
        for s in (steady_scaled(m), compound_scaled(m, 1), compound_scaled(m, 2)):
            write(os.path.join(OUT_DIR, f"{s['name']}.json"), s)
            write(os.path.join(OUT_DIR, f"{s['name']}_fail.json"), with_failures(s))


# ─────────────────────────────────────────────────────────────────────────

def main():
    generators = [
        zero_contention(),
        single_bottleneck(),
        two_machine_standoff(),
        speed_trap_standoff(),
        sparse_bottleneck_floor(),
        identical_batch(),
        bimodal_mix(),
        convoy_waves(),
        compound_scenario(),
        compound_scenario_v2(),
    ]
    for s in generators:
        write(os.path.join(OUT_DIR, f"{s['name']}.json"), s)
    # Failures-on variants of the phased scenarios, for the load-regime sweep (docs/EXPERIMENT_PLAN_2026-09-23.md
    # E1). Same job stream; only machine failures differ (same Weibull/lognormal parameters as _mfsweep_shard0).
    for s in (compound_scenario(), compound_scenario_v2()):
        write(os.path.join(OUT_DIR, f"{s['name']}_fail.json"), with_failures(s))


# Machine-failure block shared by every failures-on scenario (matches _mfsweep_shard0.json).
FAILURES = {
    "machineFailuresEnabled": True,
    "weibullK": 1.5,
    "weibullLambda": 900.0,
    "repairLogMu": 4.0,
    "repairLogSigma": 0.5,
}


def with_failures(s):
    """Copy of scenario s with machine failures on. episodeDurationSeconds covers the last arrival plus drain."""
    s = dict(s)
    last = max(j["arrivalTime"] for j in s["jobs"])
    s["stochastic"] = dict(FAILURES, episodeDurationSeconds=float(round(last * 2 + 5000, -3)))
    s["name"] = s["name"] + "_fail"
    s["_comment"] = s["_comment"] + " [FAILURES ON: Weibull k=1.5 lambda=900 s, lognormal repair mu=4.0 sigma=0.5.]"
    return s


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--scaled", type=int, nargs="+", metavar="MACHINES",
                    help="write only the E4 scaled scenarios (steady, compound, compound_v2, each +_fail) for these "
                         "machine counts, e.g. --scaled 15 30 60 100; without it the standard set is regenerated")
    args = ap.parse_args()
    write_scaled(args.scaled) if args.scaled else main()
