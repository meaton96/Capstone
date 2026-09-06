# Recent Updates & Fixes

This document tracks the scheduling/simulation-engine work done in the `pdr_scheduling` branch, covering the PDR routing fix, the scenario-coverage sweep it enabled, and the bugs that sweep surfaced. Written as a technical changelog for whoever picks this branch up next.

## Summary

The PDR routing fix (`pdr scheduling applied to routing jobs`) measurably improved the two MMUR dispatching rules — cutting their flow-time variance by ~40% at steady load — but didn't change the bigger picture: makespan and throughput still converge across rules under normal operating conditions. That pushed the investigation toward a different question: **under what factory-floor scenarios (machine breakdowns, burst arrivals) does rule choice actually matter, and can a DRL agent learn to switch rules accordingly?**

Building the scenario-coverage sweep needed to answer that surfaced two real, previously-unknown simulation bugs — one of which was silently corrupting the arrival stream (and, in the worst case, the entire back half of an episode) any time machine failures and dispatching-rule comparisons were combined. Both are fixed here.

---

## 1. PDR fix impact: verified and quantified

**What we did:** Compared `results/0903b` (post-fix) against `results/pre_pdr_fix/0903_steady_l008` (pre-fix) — same instance, same seeds, same λ=0.008 steady load, 9 rules × 3 seeds.

**Result:** The two MMUR rules (`LPT_MMUR`, `LRT_MMUR`) went from a clear outlier bucket to only mildly worse than the rest of the pack:

| | old mean flow time | old CV | new mean flow time | new CV |
|---|---|---|---|---|
| LPT_MMUR / LRT_MMUR | ~1500–1520 | 0.35–0.36 | ~1326–1331 | 0.20–0.21 |
| everyone else | ~1100–1175 (unchanged) | 0.19–0.21 (unchanged) | ~1095–1175 | 0.18–0.20 |

WIP peaks for the MMUR rules dropped from 20+ to ~17. Conventional rules (SPT/SRT/LPT × SMPT/SRWT, random) were unaffected in both mean and variance — confirms the fix is routing-specific and MMUR rules are the ones sensitive to routing decisions. Completion rate and machine utilization were unchanged (~95–96% completion, ~20–21% utilization, transport-bound system in both regimes).

**Conclusion drawn from this:** rule differentiation isn't going to show up as "one rule dominates" under normal load — that would actually be a bad sign (over-fit to a narrow scenario). The real question is scenario-dependent performance, which motivated the sweep below.

## 2. Analysis pipeline

`results/scripts/run_all.py` is now the single entry point for a results folder — `python run_all.py <folder>` runs the full pipeline (AGV decomposition, utilization diagnosis, throughput/WIP plots, time-in-system variability, rule-variability-by-config, steady-state WIP dial). It previously only wired up the older generic scripts; it now also runs `analyze_time_in_system.py`, `analyze_rule_variability_by_config.py`, and `plot_steady_state_dial.py`.

For a single-λ "steady state" WIP comparison (as opposed to a multi-λ sweep), `throughput.py`'s per-rule isolated WIP plot (`wip_isolated_<instance>.png`) reads more clearly than `plot_steady_state_dial.py`'s `wip_by_lambda.png` — the latter alpha-blends all 9 rule lines into one panel and becomes illegible when only one λ is being examined.

No scripts were stale enough to move to `scripts/old/` — everything present still serves a distinct purpose (lambda sweeps, generated-instance plots, and the steady-state/variability scripts all cover different result shapes).

## 3. Burst-arrival feature (new)

The simulator only supported a single-job-per-event homogeneous Poisson arrival process — no way to model "many jobs arrive at once" (e.g. a truck dropping off a multi-item order). Added a real feature rather than approximating it with existing knobs:

- **`StochasticConfig.cs`** — new `BurstArrivalsEnabled` (bool) / `BurstSizeMean` (float) fields. Default (`false` / `1.0`) is a no-op; existing single-job behavior is unchanged unless explicitly enabled.
- **`StochasticEventManager.cs`** — new `SampleBurstSize()`: burst size = `1 + Poisson(BurstSizeMean - 1)`, so at least one job always arrives. Implemented via Knuth's algorithm (`SamplePoisson`).
- **`FactoryOrchestrator.cs`** (`TickPoissonClock`) — each arrival event now spawns a whole burst cluster at the same sim-time instead of exactly one job, still respecting `DynamicJobCap`.
- **`ConfigLoader.cs`** — wired `burstArrivalsEnabled` / `burstSizeMean` through the JSON config DTO.

## 4. RNG stream split: arrivals vs. failures

**Bug found:** `StochasticEventManager` sampled machine/AGV failure TTF, repair duration, *and* arrival inter-arrival time all from one shared `System.Random`. Since failure events fire based on real simulation dynamics (which depend on the dispatching rule in effect), each rule consumed a different number of RNG draws before its next arrival draw — desyncing the arrival sequence **per rule**, even under an identical seed. Confirmed empirically: with `machineFailuresEnabled=true`, the same seed produced anywhere from 5 to 233 dynamic arrivals depending only on which rule was running. Config combos without machine failures (e.g. burst-only) were unaffected — proof the RNG design, not the rule itself, was the cause.

This silently broke the core assumption behind every "same seed → same environment, compare rules" analysis done with machine failures enabled.

**Fix:** `StochasticEventManager` now keeps two independent `System.Random` streams — `_rng` for failure/repair sampling, `_arrivalRng` (seeded via `seed ^ 0x9E3779B9`) for arrival/burst sampling. Every distribution sampler explicitly takes the RNG it should draw from.

## 5. Arrival-clock freeze crash (critical fix)

**Symptom:** Under `machineFailuresEnabled=true`, the dynamic-arrival clock would go completely and permanently silent partway through an otherwise-normal episode — not a detected deadlock, not an early episode end (makespan still reached the full fixed duration). Example: one run got only 5 arrivals total, the last at sim-time 867s, then nothing for the remaining ~29,000s of a "normal-looking" episode.

**Root cause** (confirmed from the actual Unity player log, not just static analysis):

1. A machine failure triggers AGV traffic congestion; AGVs stall on zone reservations and self-recover via `AGVController.HandleZoneStall()`.
2. That recovery marks whichever job the AGV was carrying as stalled — correct for a job mid-route to its *next* operation, but `FlagHarvester.HarvestStalledAGVs()` did the same thing unconditionally for a job on its **final exit trip** (already done all operations, `TargetMachineId == -1` by convention).
3. Sending a finished job back to `NeedsRouting` has no next machine to route it to. `DecisionCoordinator.FindNextDecision()` then indexes `job.EligibleMachinesPerOp[job.CurrentOpIndex]` for it — **`IndexOutOfRangeException`**.
4. That exception is thrown from `DrainHeuristicDecisions()`, called *before* `TickPoissonClock()` in the same `Update()` method — so once it starts, it recurs every single frame (nothing clears the bad job state on its own), and the arrival clock — along with *every other* dispatch/routing decision — never runs again for the rest of the episode. The deadlock/timeout checks run earlier in `Update()` and keep succeeding, which is why the episode still limps to its normal fixed-duration end with `deadlock_detected=0`.

**Impact:** this means any prior sweep combining machine failures with AGV congestion was at risk of silently losing the entire back half of an episode's data — no error surfaced in `results.csv`, no deadlock flag, a normal-looking makespan.

**Fix:**
- `FlagHarvester.cs` (`HarvestStalledAGVs`) — a stalled job on its last operation now goes back to `WaitingForPickup` (retry the exit pickup) instead of `NeedsRouting`.
- `DecisionCoordinator.cs` (`FindNextDecision`) — added a defensive bounds check + log-and-skip at the actual indexing site, so any *other* future path that makes the same mistake degrades to a logged warning instead of crashing the episode.

## 6. Orphaned job on failed AGV redirect

**Bug found:** `FailureCoordinator.HandleMachineFailure` (Step 4) redirects an in-transit AGV to an alternate machine when its original destination fails, via `agv.RedirectDropoff(...)` — without checking whether the redirect actually succeeded. If `RedirectDropoff` couldn't find a route to the new target, it reset the AGV (`FullReset()`) but the job — already mutated to target the alternate machine — was left permanently stuck `InTransit` with no owning AGV. This also inflated `JobStore.GetMachineLoad()` for the alternate machine for the rest of the episode (any `InTransit` job targeting it counts toward load).

**Fix:**
- `AGVController.RedirectDropoff` now returns `bool` (success/failure) instead of `void`.
- `FailureCoordinator.cs` checks that return value; on failure, resets the job to `NeedsRouting` with `TargetMachineId`/`AssignedAgvId` cleared — mirroring the existing "no alternate machine found" branch already in the same method.

## 7. Scenario-coverage matrix

Built out `linux_server/BatchConfigs/scenario_matrix_l008.json` on top of the `results/0903b` baseline (same factory, same λ=0.008) to test rule performance under stressors not previously covered post-PDR-fix:

| config | stressor |
|---|---|
| `l008_baseline` | none (= `results/0903b`) |
| `l008_mf_low` | machine failures, mean TTF ≈ 3× episode |
| `l008_mf_high` | machine failures, mean TTF ≈ 1× episode |
| `l008_burst` | burst arrivals, mean 3 jobs/event |
| `l008_mf_high_burst` | high machine failures + bursts combined |

**AGV failures are deliberately excluded.** `StochasticConfig.AGVFailuresEnabled` / `StochasticEventManager.SampleAGVTTF()`/`SampleAGVRepair()` exist, but nothing in `AGVController.cs` (or anywhere else) ever consumes them — there's no AGV breakdown/repair state machine. `EpisodeRecord.cs` even has `AGVFailureCount`/`AGVRepairTime` commented out (`// Phase 3`, never finished). Enabling that flag today would silently no-op rather than fail loudly, so it's left out until the feature is actually built and tested in isolation.

**Sweep status:**
- `l008_baseline` / `l008_burst` (no machine failures): clean, trustworthy data (`results/0904a`, `results/0904b`). Burst overload flattens completion to ~45–47% for *every* rule — no differentiation, the system is just saturated (bursts push the realized arrival rate to ~3× baseline).
- `l008_mf_low` / `l008_mf_high` / `l008_mf_high_burst`: results from `0904a`/`0904b` are **not usable** — generated before fixes #4–#6 above landed, so they're confounded by the RNG desync and/or the arrival-freeze crash. Needs a rebuild + rerun to get trustworthy data.

## 8. Python NN observation-shape check (no bug found)

Investigated a suspected observation-tensor shape mismatch between the C# `ObservationBuilder` and the Python training code. Traced the full pipeline (`ObservationBuilder.cs`, the Unity scene's `BehaviorParameters` asset, `env/config.py`, `env/env_wrappers/unity_env.py`, `env/models/encoder.py`) — everything matches at 13,328 floats end to end, and `unity_env.py` asserts this at runtime. Also checked the RL action space (discrete branch count): C# and Python both agree on 8 real PDR actions.

Likely explanation: remembered from an earlier, since-reverted experiment (AGV pre-dispatch methods). Cleaned up along the way:
- Stale `(3, 100, 40)` scheduling-matrix shape comments/dummy tensors in `encoder.py`, `placeholder_env.py`, and `test_architecture.py` (real shape is `(3, 20, 16)`) — cosmetic only, the encoder is shape-agnostic.
- `env/config.py`'s `PDR_ACTIONS` list had a stale `"SDT-SRWT"` label (old rule name) — updated to `"FIFO-SRWT"` to match the current C# rule name at the same index.

## Outstanding / known issues (not yet fixed)

- **`FJSSPConfig.CloneWithSeed` shares `StochasticConfig` by reference** (`FJSSPConfig.cs`) across every repeat/seed cloned from one base config. Currently inert — nothing mutates `StochasticConfig` fields at runtime — but a latent landmine if runtime state is ever added to that class.
- **`env/tests/test_architecture.py` can't currently run** in this environment — `mlagents_envs` isn't installed, so pytest fails at collection. Unrelated to the shape-comment fix; needs the package installed to actually exercise the test suite.

## Next steps

1. Rebuild `linux_server`'s headless player (picks up sections 3–6 above).
2. Rerun `l008_mf_low`, `l008_mf_high`, `l008_mf_high_burst` from `scenario_matrix_l008.json`.
3. Verify the arrival-freeze fix: check `job_completions.csv` for any job left permanently `InTransit`/incomplete-but-not-censored after a run with machine failures enabled.
4. Redo the per-scenario PDR comparison (completion rate, flow-time distribution, WIP) now that the confounds are removed.
