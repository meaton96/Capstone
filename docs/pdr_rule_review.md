# PDR Rule Set Review — Literature Grounding & Structural Audit

Prompted by discovering `LPT_MMUR`/`LRT_MMUR` were coded as max-queue (load-concentrating)
routing instead of the "Minimum Machine Utilization" the paper describes. Fixed in
`DispatchingEngine.cs` (2026-09-03). This doc is the deeper audit requested afterward:
are the remaining 9 rules structurally distinct, and does each have a literature-backed
reason to be in the DRL agent's action space.

**Note:** section 1's table is a snapshot from the initial audit (9 rules, `MRT_SRWT` still
present). See §4 for what actually shipped — `MRT_SRWT` was subsequently removed, leaving 8
real rules + Random.

## 1. What the code currently does (post-MMUR-fix, pre-MRT_SRWT-removal — see §4)

Every rule is a **(job-selection, machine-selection)** pair. `SelectJob`/`SelectRoutingJob`
picks which job goes next; `SelectMachine` picks which candidate machine it goes to.

| Rule | Job selection | Machine selection | Job-select basis | Machine-select basis |
|---|---|---|---|---|
| `SPT_SMPT` | min proc time (this machine) | min proc time (candidate) | `ArgMin(ProcTime)` | `ArgMin(CandidateJobTimes)` |
| `SPT_SRWT` | min proc time (this machine) | min queued workload | `ArgMin(ProcTime)` | `ArgMin(CandidateQueueLengths)` |
| `LPT_MMUR` | max proc time (this machine) | min queued workload | `ArgMax(ProcTime)` | `ArgMin(CandidateQueueLengths)` *(fixed)* |
| `LPT_SMPT` | max proc time (this machine) | min proc time (candidate) | `ArgMax(ProcTime)` | `ArgMin(CandidateJobTimes)` |
| `SRT_SRWT` | min total remaining work | min queued workload | `ArgMin(RemainingWork)` | `ArgMin(CandidateQueueLengths)` |
| `SRT_SMPT` | min total remaining work | min proc time (candidate) | `ArgMin(RemainingWork)` | `ArgMin(CandidateJobTimes)` |
| `LRT_MMUR` | max total remaining work | min queued workload | `ArgMax(RemainingWork)` | `ArgMin(CandidateQueueLengths)` *(fixed)* |
| `SDT_SRWT` | FIFO — oldest arrival first | min queued workload | `ArgMax(wait time)` | `ArgMin(CandidateQueueLengths)` |
| `MRT_SRWT` | max total remaining work | min queued workload | `ArgMax(RemainingWork)` | `ArgMin(CandidateQueueLengths)` |

One important implementation detail: `CandidateQueueLengths` (despite the name) is **not**
a job headcount — `JobStore.GetMachineLoad()` sums the *processing time* of every job
queued or already committed to that machine. So "SRWT" is already a workload/congestion
measure in the WINQ sense (Holthaus & Rajendran), not a raw queue-length count.

### Two problems this surfaces

1. **`LRT_MMUR` ≡ `MRT_SRWT`, in every field.** Both use `ArgMax(RemainingWork)` for job
   selection and, after the MMUR fix, both use `ArgMin(CandidateQueueLengths)` for machine
   selection. They are now the same rule under two names. This was actually foreseeable —
   the code comment on `MRT_SRWT` says it was added specifically as *"a 'standard' MRT
   baseline distinct from [LRT_MMUR's] deliberately load-concentrating [routing]"* — i.e.
   someone already built the corrected version of MMUR, under a different name, without
   connecting it back to the MMUR bug itself.

2. **`SDT_SRWT` isn't a due-date rule.** The paper table calls it "Shortest Due Time," but
   there is no due-date field anywhere in the job model (`grep -rn "DueDate"` returns
   nothing in `Simulation/`). The implementation is FIFO by arrival time — a legitimate,
   well-known rule in its own right (starvation/fairness), but not what the name or the
   paper claims it is.

## 2. Literature grounding

### Job-selection rules (which job to dispatch next)

Priority dispatching rules for job shops are cataloged in the classic surveys —
Panwalkar & Iskander's *A Survey of Scheduling Rules* (Operations Research 25(1), 1977,
>100 rules, 1200+ citations) and Blackstone, Phillips & Hogg's 1982 survey — and refined
for dynamic shops by Holthaus & Rajendran (1997, 2000) and Rajendran & Holthaus (1999),
who showed that combining processing time with a congestion/workload term (their
`PT+WINQ` and `2PT+WINQ+NPT` rules) beats plain SPT on mean flow time and tardiness.

| Rule | Literature basis | Distinct effect |
|---|---|---|
| **SPT** (shortest proc time) | Classic; in every PDR survey since Conway et al. 1967 | Minimizes WIP/flow time, but starves long jobs |
| **LPT** (longest proc time) | Panwalkar & Iskander; Graham's LPT bound for parallel-machine makespan | Reduces machine idle-fragmentation, but tail-latency risk for short jobs |
| **LWKR / SRT** (least work remaining) | Holthaus & Rajendran | Clears near-complete jobs fast, reduces jobs-in-system |
| **MWKR / LRT** (most work remaining) | Holthaus & Rajendran; used as a baseline rule in most DRL-for-FJSP papers (e.g. Luo 2020, *Applied Soft Computing* 91:106208) | Front-loads heavy jobs early, avoids tail-end congestion pile-up |
| **FIFO** (oldest arrival) | Classic fairness/starvation-avoidance baseline | The only rule with an arrival-time (not work-content) basis — genuinely orthogonal to the SPT/LPT/SRT/LRT family |

*(Note: the code's `MRT_SRWT` comment cites Luo (2020) for an "MRT" rule by name — I could
not verify that paper's exact rule table since it's paywalled (ScienceDirect 403'd the
fetch); treat that specific attribution as unconfirmed provenance, not a checked citation.
MWKR itself is a real, independently well-established rule regardless.)*

### Machine-selection rules (which candidate machine to route to)

FJSP surveys (e.g. the 2023 *European Journal of Operational Research* FJSP review) split
this into workload-based and time-based routing:

| Rule | Literature basis | Distinct effect |
|---|---|---|
| **SMPT** (min proc time on candidate) | Classic — local greedy, matches job to its fastest machine | Doesn't look at machine congestion at all |
| **SRWT / WINQ-style** (min queued workload) | Holthaus & Rajendran's WINQ term | Congestion-avoidance — the correct, intended behavior of what "MMUR" was named for |
| **Least Utilized Machine (LUM)** — cumulative busy-time ratio, not instantaneous queue | Workload-based routing (WRW) literature in FJSP reviews | Genuinely different signal from SRWT: a machine can have zero queue right now but still be the most-utilized machine over the horizon, or vice versa |
| ~~Max-queue / load-concentrating~~ | No literature support found as a *sequencing* rule — it's the mechanism you'd use to *stress-test* congestion, not to schedule against it | This was the actual old MMUR behavior; not a rule any FJSP survey recommends as a real dispatching policy |

## 3. What this means for the current 9-rule action space

- `SDT_SRWT` should either (a) be renamed to reflect what it actually does (FIFO), keeping
  it as the fairness/starvation-avoidance rule, or (b) become a real due-date rule (EDD),
  which requires adding a `DueDate` field to the job model — a bigger change, not just a
  rename.
- `LRT_MMUR` and `MRT_SRWT` need to stop being identical. Two real options:
  - Retire one of them (they contribute zero additional coverage to the action space today).
  - Give `MMUR` its own real, literature-grounded signal — cumulative machine utilization
    (LUM, item above) instead of instantaneous queued workload — which would make it
    genuinely mean "Minimum Machine Utilization" for the first time, and restore it as a
    distinct rule from `SRWT`. This needs new live per-machine utilization tracking wired
    into `DecisionRequest` (the sim already computes `UtilizationRate` in `EpisodeRecord`,
    but only as a post-episode aggregate for CSV logging — not exposed live at decision time).
- No other rule pairs are duplicates; SPT/LPT/SRT/LRT/FIFO job-selection and SMPT/SRWT
  machine-selection are each backed by an independent, distinct literature source.

## 4. Resolution (2026-09-03)

1. **`SDT_SRWT` → `FIFO_SRWT`.** Renamed everywhere (enum, comments, the `Scheduling/Core`
   validation-reference enum, `-rules` CLI sweep scripts, plot rule-order lists, `high_level.md`,
   `sim_flow.html`). Behavior unchanged — still FIFO by arrival order. No due-date modeling added;
   revisit only if a genuine EDD rule becomes worth the job-generation changes it needs.
2. **`LRT_MMUR` / `MRT_SRWT` duplicate → `MRT_SRWT` removed entirely.** Two changes landed here,
   in order:
   - First, MMUR was given a real, distinct signal: live per-machine cumulative utilization
     (busy time / operational time so far this episode), wired into
     `DecisionRequest.CandidateUtilization`.
     - `EpisodeTracker` gained `ProcessingTimeSoFar(machineId)` and
       `DowntimeSoFar(machineId, simTime)` accessors on its existing live accumulators.
     - `DecisionCoordinator` now takes the `EpisodeTracker` and the machine's
       in-progress-operation start-time map (both already live in `FactoryOrchestrator`), and
       computes `MachineUtilization()` = (closed processing time + in-flight partial) /
       (simTime − downtime so far) per candidate machine.
     - `DispatchingEngine.SelectMachine` now routes `LPT_MMUR`/`LRT_MMUR` via
       `ArgMinIdx(CandidateUtilization)` instead of `CandidateQueueLengths`.
     - Logged to `decision_log.csv`'s previously-unused `CandidateStatC` column for routing rows.
   - With that fix in place, `LRT_MMUR` and `MRT_SRWT` no longer compute the same thing
     (utilization vs. instantaneous queued workload), but they still share the exact same
     job-priority half (`ArgMax` total remaining work) — the only difference between them is a
     subtle machine-selection nuance. Given `MRT_SRWT` was never part of the paper's rule set to
     begin with (it only existed as an ad-hoc workaround for the buggy MMUR — see point 1 above),
     the call was to drop it rather than keep two rules that are this close. `MRT_SRWT` is now
     removed from the enum, both switch statements in `DispatchingEngine`, the `-rules` CLI sweep
     scripts, and the plotting scripts' rule-order lists. `LPT_MMUR`/`LRT_MMUR` remain as the
     paper's actual, now-correctly-implemented Load-Balancing pair. Action space: 9 rules total
     (8 real + Random), down from 10.

   Not yet done: no sweep has been re-run against the fixed/reduced rule set. All prior results
   (including everything in `results/0902f` and earlier) were generated under the old buggy
   max-queue MMUR and the now-removed `MRT_SRWT`, and should not be read as reflecting the
   current rule set. A small smoke-test config is at
   `linux_server/BatchConfigs/mmur_smoketest.json`.
