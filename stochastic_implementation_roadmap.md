# DFJSP Stochastic Implementation Roadmap — Phases 2–5

**Context:** Unity 6.3 + PhysX + NavMesh · PPO via Stable-Baselines3 · ML-Agents side-channel  
**Infrastructure complete:** `StochasticEventManager`, `StochasticConfig`, updated `FJSSPConfig` (Phase 1)  
**Approach:** Implement and validate one disruption type at a time before combining.

---

## Phase 2 — Machine Failure Model

**Goal:** Weibull(k=1.5) time-to-failure, log-normal repair times, full machine state machine.

### Implementation Tasks

**1. Machine state machine**  
Add three states: `OPERATIONAL → FAILED → REPAIRING → OPERATIONAL`.  
On entry to `FAILED`: cancel the current operation and return it to the job's pending queue as if it never started (do not partially credit processing time). Sample repair duration from `StochasticEventManager.Instance.SampleMachineRepair()` immediately on failure so the repair timer is known before the observation is built.

**2. Time-to-failure sampling**  
On episode reset and after each repair completes, call `StochasticEventManager.Instance.SampleMachineTTF()` to schedule the next failure. Store the result as a countdown timer on each machine controller. Decrement each frame by `Time.deltaTime`. Reset the machine's age counter to zero after each repair — the Weibull is applied fresh from the repaired state, not accumulated lifetime.

**3. Initial age randomisation**  
Do not start all machines at age zero — this would let a policy exploit the fact that no failures occur in the first portion of every episode. On episode reset, sample each machine's initial TTF from `Uniform(0, sampled_TTF)` to randomise where each machine sits in its wear-out curve. Concretely: sample a full TTF, then draw the starting countdown from `Uniform(0, full_TTF)`.

**4. Observation space extension**  
Extend the Global Scalars vector (currently 10D) with three new channels:
- Fraction of machines currently in `FAILED` state
- Fraction of machines currently in `REPAIRING` state  
- Mean normalised remaining repair time across repairing machines (0 if none)

Add a 4th channel to the 64×64 spatial occupancy tensor encoding machine health state: `0` = operational, `0.5` = repairing, `1.0` = failed. The CNN-SPPF encoder accepts variable input channels — verify the first conv layer handles 4 channels.

**5. Predictive pre-dispatch update**  
Your existing pre-dispatch logic sends an idle AGV to a machine dock before the current operation finishes. Extend it: if the target machine is in `REPAIRING` state, factor the mean remaining repair time into the ETA comparison. Do not dispatch to a machine whose expected ready time exceeds that of the next-best alternative.

**6. AGV job re-routing on machine failure**  
If a machine fails while an AGV is in transit carrying a job destined for it, the AGV must be redirected. On machine entry to `FAILED`, broadcast a `MachineFailedEvent(machineId)`. Any AGV controller whose current destination matches `machineId` should call the scheduler for a re-routing decision: find the next-best eligible machine for that operation and issue a new NavMesh destination.

### Reward note  
Your existing reward (`−ΔMakespan / N_ops`) already penalises the makespan spike caused by a failure. No immediate reward change is needed. Consider a small additive per-step penalty for each machine stuck in `FAILED` before repair begins if you find the policy learns to ignore single-machine dependency risk — but validate the base signal first before adding terms.

### Testing checkpoint  
Before moving to Phase 3: run 50 deterministic episodes (`StochasticMode = false`) and confirm makespan results match your Phase 0 baseline logs within a small margin. Then run 50 stochastic episodes at low failure rate (`WeibullLambda = 2700`) and confirm failures are occurring at the expected frequency. Log `total_failures`, `total_repair_time`, and `mean_TTF_observed` per episode to validate against theoretical Weibull mean.

> **Critical:** Verify the deadlock manager handles the case where a machine failure leaves an AGV stranded at a dock with no valid destination. The zone routing system must not deadlock waiting for a machine that is in a long repair cycle.

---

## Phase 3 — AGV Failure Model

**Goal:** Same Weibull/log-normal distributions as machines, handling mid-route failure and stranded job recovery.

### Implementation Tasks

**1. AGV state extension**  
Add `BROKEN` and `REPAIRING` to the existing AGV state machine alongside `IDLE / MOVING / LOADING / UNLOADING`. Keep battery depletion (→ planned dock return) strictly separate from Weibull failure (→ unplanned in-place breakdown). They are different failure modes and the observation should encode both.

**2. Time-to-failure sampling**  
Same pattern as machines: call `StochasticEventManager.Instance.SampleAGVTTF()` on episode reset and after each repair. Apply the same initial age randomisation (`Uniform(0, sampled_TTF)`) so AGVs are not all fresh at episode start.

**3. In-transit failure handling**  
On `BROKEN` during `MOVING`: freeze the AGV at its current NavMesh position. If carrying a job, mark it as `STRANDED` with the AGV's current world position. Immediately register the AGV's position as a static obstacle in the directed zone routing manager — treat it identically to a permanent obstacle for the duration of repair. Remove the obstacle registration when repair completes.

**4. Stranded job recovery**  
On job entering `STRANDED` state, fire a `JobStrandedEvent(jobId, worldPosition)`. The scheduler should evaluate whether to dispatch a rescue AGV: find the nearest idle AGV and compare the cost of rescue dispatch vs waiting for the broken AGV to repair. A simple heuristic for now: dispatch rescue if `EstimatedRepairTime > RescueTransitTime * 1.5`. The rescue AGV picks up the job from the breakdown position and continues to the originally assigned machine (or re-evaluates if that machine has also failed).

**5. Fleet health observation**  
Extend Global Scalars further:
- Fraction of AGVs currently operational
- Number of stranded jobs (normalised by total job count)
- Mean AGV age normalised by `AGVWeibullLambda`

**6. Battery vs failure distinction in observation**  
Ensure the spatial occupancy grid and/or event flags distinguish AGV-at-charger (battery) from AGV-broken-in-place (Weibull). The policy needs to know whether an AGV will return to service in a short, predictable time (battery recharge) vs an uncertain, potentially long repair.

### Testing checkpoint  
Before moving to Phase 4: confirm that a broken AGV occupying a bottleneck zone does not deadlock the fleet. Run a stress test at high AGV failure rate (`AGVWeibullLambda = 300`, well below normal operating range) and verify all episodes terminate without hanging. Check that stranded job recovery fires correctly and rescued jobs complete.

> **Critical:** The mid-route failure → static obstacle → deadlock interaction is the highest-risk integration point in the entire stochastic extension. Test this in isolation before combining with machine failures.

---

## Phase 4 — Dynamic Job Arrivals

**Goal:** Homogeneous Poisson process injects new jobs mid-episode, shifting factory utilisation unpredictably.

### Implementation Tasks

**1. Poisson arrival clock**  
Create a `PoissonClock` MonoBehaviour. On each episode start (when `DynamicArrivalsEnabled = true`), draw the first inter-arrival time via `StochasticEventManager.Instance.SampleInterArrivalTime()`. Maintain a countdown timer. When it reaches zero: fire `JobArrivedEvent`, inject the new job into the scheduler, and immediately draw the next inter-arrival time.

**2. Dynamic job generation**  
On `JobArrivedEvent`: generate a new `FJSSPJobDefinition` using the existing procedural job generator (same `minOps/maxOps`, `procTimeParams` as the episode config). Assign a due date sampled from `N(current_makespan_estimate × 1.2, σ)` — this creates realistic urgency without knowing the true optimum. Assign `ArrivalTime = current_simulation_time`.

**3. Scheduling matrix resize strategy**  
Your scheduling matrix is `n × 2m × 3`. Dynamic arrivals grow `n` mid-episode. Use a fixed max-job buffer with a padding mask rather than dynamic resizing — define `MaxJobs = initial_job_count + arrival_buffer` (e.g. `initial + 20`). Pad unused rows with zeros. The CNN-SPPF encoder already handles variable-length padded inputs via size-agnostic inference; confirm at inference time that the SPPF pooling layer is not sensitive to the specific padding pattern.

**4. Due-date urgency encoding**  
Add a per-job urgency scalar to the scheduling matrix's 3rd channel (currently used for completion flags): `urgency = (due_date − current_time) / episode_horizon`. Negative values indicate a tardy job. This gives the SDT/SRWT PDR rules the signal they need when late-arrival jobs become urgent.

**5. Arrival rate curriculum**  
Do not start training with high `ArrivalLambda`. Begin with near-zero arrivals (essentially deterministic load) and gradually increase `ArrivalLambda` via a curriculum schedule tied to training progress or a manual sweep parameter. Very high lambda creates a continuously saturated factory — a qualitatively different scheduling regime that may require a separate policy regime or extended training.

**6. Tardiness reward term (optional — validate first)**  
With due dates now meaningful, a tardiness term can be added: `−α × max(0, completion_time − due_date) / N_ops` alongside the existing makespan delta. Keep `α` small (start at 0.1) and run the reward ablation in Phase 5 before committing. The base makespan-delta signal should remain dominant.

### Testing checkpoint  
Confirm that the scheduling matrix padding does not cause the SPPF encoder to produce degenerate outputs on episodes with varying job counts. Run short inference tests with `n = initial_count`, `n = initial_count + 10`, `n = initial_count + 20` and compare output distributions. Also verify that `PoissonClock` produces the correct empirical arrival rate: log total arrivals per episode and check against `λ × episode_duration`.

> **Note:** The homogeneous Poisson assumption means constant `λ` throughout the episode. If you later want shift-pattern demand (time-varying λ), upgrade to a non-homogeneous process. Keeping `ArrivalLambda` as a runtime parameter costs nothing and avoids a refactor.

---

## Phase 5 — Observation & Reward Consolidation

**Goal:** Integrate all new channels, validate the full observation vector, and confirm no silent regressions before training.

### Implementation Tasks

**1. Observation vector audit**  
After Phases 2–4, enumerate every scalar added to Global Scalars. Target ≤ 18D to stay within the same FC encoder capacity range. Normalise all new channels to `[0, 1]` or standard-normal using running statistics from a fresh environment rollout — do not use hardcoded normalisation constants for channels whose range depends on config parameters like `WeibullLambda`.

| Original (10D) | Added Phase 2 | Added Phase 3 | Added Phase 4 |
|---|---|---|---|
| Utilisation metrics ×10 | Frac. machines failed | Frac. AGVs operational | — |
| | Frac. machines repairing | Frac. jobs stranded | — |
| | Mean remaining repair time | Mean AGV age / λ_agv | — |

**2. Spatial grid 4th channel**  
Confirm the spatial occupancy tensor is `64×64×4` (machines, jobs, AGVs, health-state). Verify CNN-SPPF kernel sizes and receptive field accommodate 4-channel input. A shallow adjustment to the first conv layer may be needed; this is a one-line change but requires a full retrain.

**3. Reward ablation**  
Before combining reward terms, run short training sweeps (≤ 500k steps each) with each term in isolation:
- Makespan-delta only (existing)
- Tardiness penalty only (Phase 4)
- Per-step failure-idle penalty only (Phase 2, if added)

Check that each term alone produces a non-degenerate learning signal. A poorly scaled term will dominate and mask the others in the combined reward. Adjust `α` weights accordingly before the full training run.

**4. Stochastic mode regression test**  
Re-run your existing Taillard/Brandimarte instances with `StochasticConfig = null` (deterministic) after all code changes. Confirm makespan results are within a small margin (< 1%) of your Phase 0 baseline logs. Any divergence indicates a silent bug introduced during Phases 2–4.

**5. Retrain vs fine-tune decision**  
The observation space change (new channels, 4th spatial grid channel) means the existing PPO checkpoint's input layer dimensions no longer match. A full retrain is required. Consider warm-starting with a low-stochasticity curriculum config (`StochasticBatchConfigs.json`: `mf_low` regime) and annealing toward full disruption over training rather than starting at the hardest config.

### Before moving to Phase 6 (baseline re-run)  
- [ ] All three disruption types (`mf`, `agv`, `arr`) individually validated in isolation  
- [ ] Combined disruption (`full` config) runs 50+ episodes without hanging or deadlocking  
- [ ] Observation vector fully normalised and audited  
- [ ] Reward ablation complete, `α` weights set  
- [ ] Deterministic regression test passes against Phase 0 baseline logs  
- [ ] `StochasticDistributionValidator` passes (`-validatestochastic` exit code 0)

---

## Reference: Parameter Starting Points

| Parameter | Value | Rationale |
|---|---|---|
| `WeibullK` | 1.5 | Wear-out regime (increasing failure rate with age) |
| `WeibullLambda` (machine, low) | 2700 | Mean TTF ≈ 3× typical episode length |
| `WeibullLambda` (machine, high) | 900 | Mean TTF ≈ 1× typical episode length |
| `AGVWeibullLambda` (low) | 2100 | ~78% of machine λ — AGVs fail slightly more often |
| `AGVWeibullLambda` (high) | 700 | High disruption AGV sweep |
| `RepairLogMu` | 4.0 | Real-space mean repair ≈ 62 sim-seconds |
| `RepairLogSigma` | 0.5 | Moderate heavy tail on repair duration |
| `AGVRepairLogMu` | 3.4 | Real-space mean AGV repair ≈ 30 sim-seconds |
| `AGVRepairLogSigma` | 0.4 | Slightly tighter tail than machine repair |
| `ArrivalLambda` (low) | 0.003 | ≈ 1 arrival per 333 sim-seconds |
| `ArrivalLambda` (high) | 0.008 | ≈ 1 arrival per 125 sim-seconds |

> All `WeibullLambda` values should be calibrated against your actual mean episode length once measured. The ratios (low = 3×, high = 1×) are the target, not the absolute numbers.
