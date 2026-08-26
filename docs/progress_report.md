# Capstone Progress Overview — Summer 2026

**Project:** Dynamic Flexible Job Shop Scheduling (DFJSP) via Deep Reinforcement Learning, simulated in Unity (C#/ML-Agents) with a physically modeled AGV transport layer, trained with a Python/PPO agent.

---

## 1. Where things stood in spring (context, not the focus of this update)

By late April the core simulation existed end-to-end: DES-style job/machine logic driven inside Unity, AGVs with real pathing and traffic zones, a config-driven job/machine generator, an observation/action bridge to ML-Agents, and a headless Linux batch runner for running many configs unattended. Baseline dispatching-rule (PDR) runs against Brandimarte instances were producing results. That foundation is what the summer's work builds on.

## 2. Summer work, in chronological order

### May: realism and failure modeling
- Machine failure → repair → fix cycle added, with AGV rerouting around failed machines (`9fad1b8`, `12b6b06`).
- Round-trip testing for failure events, FJSSP bug fixes (`9a822ab`).
- Logging expanded to track machine utilization and per-instance identifiers (`3197703`, `f35c281`).
- Deterministic vs. dynamic failure results collected and visualized (`b4d5098`, `3b298a6`).
- Logging moved into its own assembly and reworked for stochastic (multi-seed) runs (`00512c2`).

### Late May–June: dynamic arrivals and AGV logistics realism
- Switched from static Brandimarte instances to **randomly generated job data** driven by a seed + config, so instance size/mix is fully parameterized rather than fixed to a benchmark set (`c7f030b`, `9d8113b`+).
- Added **Poisson job arrivals** so the shop floor stays continuously saturated instead of draining a fixed batch — much closer to a real dynamic shop (`4d0b660`, `8c649fc`, merged in PR #7 `poisson_arrivals`).
- Added per-arrival and per-operation logging needed to analyze the new dynamic regime (`bbfc694`, `e7840c6`).
- AGV parking behavior reworked: multiple parking zones, "dispatch nearest AGV," and a configurable pre-dispatching method, giving more realistic idle-AGV behavior instead of a single fixed home position (`78eec17`, `7fbba3f`, `56643c1`, `aadea70`).
- Job generation seed bug fixed, aisle direction corrected (`4d7ff37`, `c28131b`).
- Parking-behavior test results collected (`20d0e2b`, June 21) — **this is the last commit before a ~7-week gap**, during which the next two items were worked through primarily in Claude web sessions rather than local commits.

### The two big bugs (diagnosed over the gap, landed in the Aug 13 commit)
This is the most important story for your advisor — two independent bugs were both making every dispatching rule look identical, for completely different reasons, and both had to be found and fixed before the results meant anything:

1. **PDR convergence bug.** All 9 dispatching rules (8 priority rules + random) were producing the *exact same* makespan. Root cause diagnosis (documented 2026-07-25) found the shop was running in a heavily overloaded regime (~5× arrival rate vs. system capacity) where makespan is dictated almost entirely by arrival volume divided by system capacity — any work-conserving rule gives the same answer under saturation, so the rules genuinely were indistinguishable at that load level, not just similar.
2. **Decisions-per-frame artifact.** Independently, the simulation was serializing to *one dispatching decision per simulated frame*. At high simulation speed this meant decision throughput was frame-rate-bound rather than logic-bound, which silently inflated makespan across the board — a pure artifact of the sim loop, not the scheduling policy. Fixed by draining **all** ready heuristic decisions within a frame (`BaselineDrainMode` in `FactoryOrchestrator`), landed in the Aug 13 mega-commit.

**After both fixes**, rules stopped converging to a single number — makespan differentiation now shows up correctly in flow rate, AGV congestion, and AGV travel time, i.e., the rules are now differentiating on transport-layer behavior rather than being masked by a loop artifact or drowned out by overload. Further analysis (see `results/cap_control_unbound`) showed the actual system bottleneck is AGV transport capacity, not machine capacity — machines run ~25% utilized while AGVs spend 22–42% of time waiting on route congestion. Under light transport load rules diverge by ~10%; under heavy AGV load some configurations gridlock entirely for every rule except the max-queue (MMUR) variants.

### Aug 13: the "job completion run data" commit — closing the flow-time gap
The July diagnosis had flagged a specific gap: there was no per-job realized completion-time logging, only static job specs, which made it impossible to compute flow time or tardiness — the metrics that actually matter under Poisson arrivals (makespan is a poor metric once the shop never fully drains). This commit closed that gap:
- New `EpisodeTracker.cs`, extended `JobData`/`Jobstore` to record realized per-job completion times.
- `ResultsLogger`/`EpisodeRecord` extended with flow-time statistics (mean/p95/max flow time, mean transport wait, censored-job counts) and dynamic-arrival stats (theoretical vs. realized inter-arrival time).
- This is also where the frame-drain fix (`BaselineDrainMode`) and related orchestrator changes landed.

### Aug 24: corrected job-completion test + figures
- Reran the job-completion logging with corrections (`f69fe2d`, `eca3a5b`) and generated the first flow-time analysis figures (`6b1fd9f`): makespan vs. arrival rate (λ), flow time vs. λ, flow-time rank heatmap across rules, and wait-time decomposition by load — for both single- and multiple-parking configurations.

## 3. Where the project stands right now (uncommitted, in progress)

There's active work on the `pre-dispatch` branch not yet committed:
- **Deadlock watchdog** added to `FactoryOrchestrator`: a circular-wait deadlock in the AGV traffic-zone reservation logic (`TrafficZoneManager.TryReserve`) never self-resolves, so the sim was previously running such cases all the way to the 500,000s episode timeout before giving up — very expensive and it muddies results (a deadlocked run isn't a "slow" run, it's a stuck one). The new watchdog sums traversal counts across all traffic zones and declares deadlock if *zero* AGVs anywhere complete a zone entry for 3,000 consecutive sim-seconds while jobs remain unfinished, terminating the episode immediately and flagging it (`DeadlockDetected`/`DeadlockSimTime` on the episode record) instead of silently reporting an inflated makespan.
- Episode timeout also lowered from 500,000s to 100,000s for faster iteration.
- Two new sweep configs staged (untracked): `agv_congestion_sweep.json` and `agv_deadlock_threshold_sweep.json`. The congestion sweep already found a "deadlock cliff": at 8 AGVs, only the two MMUR (max-queue) rules survive — everything else gridlocks; by 12 AGVs, *every* rule gridlocks (more AGVs made congestion worse, not better). The threshold sweep is narrowing in on exactly where that cliff sits (6–7 AGVs for non-MMUR rules, 9–11 for MMUR).

This is directly in service of the "fix deadlock" next step — the detection/early-termination side is built; the open question is whether to also fix the underlying reservation logic to prevent circular waits, or to treat the deadlock boundary as a known operating limit and stay under it.

## 4. Next steps (this semester)

1. **Finish simulation testing** — run the deadlock-threshold and congestion sweeps to completion, confirm the flow-time/makespan results are stable post-fix, decide on a standard AGV-count operating range that avoids the gridlock cliff.
2. **Fix deadlock** — watchdog/early-termination is in place; decide whether the reservation logic itself needs a fix (e.g., ordered lock acquisition, timeout-and-retry) or whether staying below the AGV-count cliff is an acceptable constraint to document.
3. **Update the reward function** — needs to reflect the corrected metrics (flow time, transport wait, deadlock avoidance) now that makespan-under-overload is understood to be a poor training signal.
4. **Update the neural network architecture** — `env/models/` (encoder, actor-critic, network) hasn't been touched since the observation-bridge was first stood up in spring; needs to be brought in line with the expanded observation/logging schema (flow-time stats, deadlock flag, AGV congestion signals) before training can start.
5. **Begin model training** — once reward and architecture are updated, start actual PPO training runs against the corrected simulation.

---
