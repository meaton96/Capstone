# DJFSP Simulation — Structural Analysis Notes
*Generated: 2026-05-28 | Brandimarte Benchmark, 15 instances, 8 PDRs + RANDOM, 2 regimes*

---

## Context

Simulation of a Dynamic Job-shop Flexible Scheduling Problem (DJFSP) with AGV transport, run across the full Brandimarte benchmark suite (MK01–MK15). Two regimes tested: **deterministic** (no machine failures) and **stochastic_low** (Weibull TTF + exponential repair). PDRs evaluated: SPT_SMPT, SPT_SRWT, LPT_MMUR, LPT_SMPT, SRT_SRWT, SRT_SMPT, LRT_MMUR, SDT_SRWT, RANDOM.

Key anomaly motivating this analysis: several instances showed **40× makespan inflation** under stochastic_low for specific PDRs, while other instances were nearly flat across regimes.

---

## Key Finding: The Cascade Failure Mechanism

### The positive feedback loop

Machine failure rate is governed by a Weibull TTF calibrated to mean processing time per instance. Critically, TTF is fixed at episode start but **the number of failures that occur scales with episode duration**. This creates a positive feedback loop:

```
Bad PDR choice → longer initial makespan
→ more time for failures to occur
→ failures cause rerouting / queue buildup
→ even longer makespan
→ yet more failures
→ runaway cascade
```

Expected total failures across all machines = `n_machines × T / meanTTF`. For MK04 with 8 machines and meanTTF ≈ 25,750 sim-units:
- At deterministic makespan (~8,000): ~2.5 expected failures — manageable
- At SPT_SRWT stochastic outlier (~335,000): ~104 expected failures — catastrophic

### Why this is PDR-dependent (not just stochastic noise)

The failure count heatmap in the MK04 deep dive shows LPT_SMPT averaging ~0.7 failures per machine vs SPT_SRWT/SDT_SRWT averaging ~13–15. This is not independent variance — it is a **direct reflection of makespan length**. The same PDRs that produce long makespans deterministically produce exponentially longer makespans stochastically. The failure counts are a symptom, not a cause.

---

## Structural Root Cause: "SPT Bait" at Bottleneck Machines

### The key metric: Vulnerability × 1/AvgMandatoryProcTime (SPT Danger Index)

Computed per machine across MK01, MK04, MK08 (representative comparison set).

**MK04 — Machine 5:**
- 11 mandatory operations (no alternative machine) → 12.2% of all ops stranded if M5 fails
- Average processing time of mandatory ops: **1.45 base units** (range 1–2)
- SPT and SDT rules preferentially dispatch to M5 because its ops are the shortest in the instance
- Result: SPT/SDT converge a disproportionate share of in-flight jobs onto M5, then M5 fails under load, and the queued jobs have no escape route

**MK08 — Machine 0 (contrast case):**
- 43 mandatory operations → 19.1% vulnerability (structurally worse than MK04-M5)
- Average processing time of mandatory ops: **12.16 base units** (range 7–19)
- SPT/SDT do NOT preferentially route to M0 — its ops are long, so SPT avoids it
- Result: M0's queue stays distributed; failures are absorbed without cascade

**MK01 — Machine 1:**
- 6 mandatory ops, avg proc = 6.0 base units
- Moderate vulnerability, moderate SPT attraction — moderate stochastic inflation (~8×)

### Summary table

| Instance | Critical Machine | Vulnerability | Avg Mandatory Proc | SPT Attraction | Stoch. Inflation |
|----------|-----------------|---------------|--------------------|----------------|-----------------|
| MK01     | M1              | 10.9%         | 6.0 (medium)       | Medium         | ~8×             |
| MK04     | M5              | 12.2%         | **1.45 (very low)**| **HIGH**       | **~42×**        |
| MK08     | M0              | 19.1%         | 12.2 (high)        | Low            | ~1×             |

---

## AGV Blocking Connection

The stochastic AGV time budget charts show some instances transitioning from ~80% idle (deterministic) to heavily blocked (stochastic). This is mechanistically downstream of the cascade:

1. SPT floods bottleneck machine → queue builds
2. Machine fails → queued jobs stuck at failed machine
3. AGVs dispatched to those jobs cannot complete pickup/dropoff
4. AGVs enter waiting-blocked state or attempt reroutes
5. AGV pool congestion propagates to unrelated jobs
6. Row-aisle block rate rises, compounding the delay

The reroute outliers in MK03 and MK13 likely share a similar bottleneck-machine structure — worth verifying with deep dive plots on those instances.

---

## Implications for DRL Training

### Why DRL has a structural advantage over PDRs here

A PDR like SPT is locally greedy and topology-blind: it sees a short operation and dispatches without awareness that the target machine is a mandatory choke point already under load. A DRL agent with access to the full job schedule matrix can in principle learn a composite decision signal:

> *"This operation is short AND mandatory AND routes to a machine with high queue depth AND that machine has no alternatives for many pending jobs → deprioritise despite short processing time"*

No single-rule PDR can express this. It requires cross-referencing operation duration, machine topology, and current queue state simultaneously.

### Concrete training considerations

**Curriculum design:**
- Easy stages (stochastic ≈ deterministic): MK08, MK10, MK11, MK12 — good for initial policy learning
- Hard stages (high cascade risk): MK01, MK04, MK07 — introduce after the agent has a stable baseline policy

**Evaluation / ablation:**
- Train on deterministic only → evaluate stochastic → expect SPT-like cascade behavior to re-emerge; use as ablation baseline
- Train on stochastic → check whether policy avoids M5-type queue concentration on MK04; this tests genuine structural awareness vs episode memorisation

**Interpretability probe:**
- MK04 M5 is a clean test case: does the trained policy route short-op mandatory jobs to M5 at lower rates than SPT under load? If yes, the agent has learned something genuinely useful
- The cascade threshold (~2 expected failures = danger zone) could serve as an auxiliary reward signal or safety constraint during training

**Observation space note:**
- The full job schedule matrix (which machines are eligible for each pending operation, and their processing times) is the key input that enables this structural awareness
- At minimum the agent needs: current queue depth per machine, eligible machines per pending operation, processing times, and a binary flag for machine health state

### Open questions for later

1. Would an inverse-SPT heuristic (explicitly avoid short mandatory ops on high-vulnerability machines) close most of the gap without DRL? Good baseline to test.
2. Does the cascade threshold vary predictably with instance size (n_jobs, n_machines, flexibility ratio)?
3. Do MK03/MK13 reroute spikes come from a similar SPT-bait structure?
4. How does fleet size (AGV count) interact with cascade severity — more AGVs could accelerate queue buildup at bottleneck machines.

---

## Instance Structural Quick-Reference

From JSON analysis of Brandimarte instances:

| Instance | Jobs | Machines | Ops | Avg Flex | % Single-Machine | Bottleneck Ratio | Max Vuln |
|----------|------|----------|-----|----------|-----------------|-----------------|----------|
| MK01     | 10   | 6        | 55  | 2.09     | 29%             | 1.24×           | 10.9%    |
| MK04     | 15   | 8        | 90  | 1.91     | 29%             | 1.37×           | 12.2%    |
| MK08     | 20   | 10       | 225 | 1.43     | 57%             | 2.00×           | 19.1%    |

*MK08 looks structurally worse on most metrics yet is the most resilient — confirming that raw vulnerability numbers are insufficient without the SPT-attraction component.*

---

*See also: structural_analysis.png (figure), 05_mk04_deep_dive.png, 23_cascade_chain.png*
