Why this is the right lever

Everything we've tested so far (proc-time multipliers, machine failures, burst arrivals) varies load, not structure. With i.i.d.-random job generation, every job's op sequence, machine eligibility, and durations are independently sampled — at any real batch size, contention averages out across machines and rules converge by law of large numbers. That's almost certainly the deeper reason baseline/burst show zero rule differentiation even with the reproducibility bug fixed: there's rarely a genuine queue to choose from. A dispatch rule can only prove itself when multiple jobs are actually contending for the same resource at the same instant — Brandimarte differentiates rules precisely because its instances have that structure baked in, not because the load level is different.

Scenario menu

Contention topology (your core idea, and the highest-value one to build first)

Zero-contention — construct a window where each concurrently-active job's ops route to disjoint machines by design. Expect near-total convergence across all 9 rules; this is the "control" that proves any differentiation elsewhere is real, not an artifact.
Single bottleneck — force most jobs' operations through one specific machine during a window (e.g. everyone needs Weld-2 for op 3). This is the single best scenario for isolating dispatch-rule differences (SPT/LPT/SRT/FIFO — which job gets pulled off a backed-up queue), since that choice is invisible whenever queues stay short.
Two-machine standoff — exactly your proposal: pin a batch of jobs so their contention is split between two specific machines. This isolates routing-rule differences (SMPT/SRWT/MMUR — which of the two absorbs the load) in a way we can't currently see, since natural contention spreads across 3 same-type machines.

Job homogeneity

Identical-job batch — N copies of the exact same job definition arriving together. Every score-based rule (SPT, LPT, SRT, LRT) degenerates to a tie on identical jobs, and the code breaks ties by lowest job ID — meaning right now, a "diverse enough" instance can be masking the fact that most rules silently collapse to FIFO whenever job diversity is low. Worth exposing directly.
Bimodal mix — short single-op jobs and long multi-op jobs injected in the same wave. This is where SPT-starves-long-jobs / LPT-starves-short-jobs tradeoffs actually show up, and ties directly into the tardiness idea from before — mean flow-time can look great while a tail of jobs starves indefinitely; only p95/max or a tardiness metric would catch it.

Arrival structure

Synchronized-need burst — a wave where every job's first op needs the same machine type (immediate routing contention at t=0) vs. a wave where each job needs a different type first (no routing contention, only downstream dispatch contention). Separates routing-time from dispatch-time effects cleanly.
Convoy waves — periodic bursts with quiet gaps, instead of steady Poisson. Tests whether a rule structurally recovers during the gap or accumulates backlog wave-over-wave (resonance rather than steady-state saturation).

Capacity asymmetry

True single-instance bottleneck — give one machine type only 1 instance instead of 3, so it's a bottleneck by construction rather than probabilistically. Directly tests which rule protects a genuinely scarce resource.

Adversarial (pathology-hunting)

SPT-starvation trap — one long job injected, then a continuous stream of short jobs on the same machine. SPT should starve the long job indefinitely; mean flow-time will look fine while max flow-time (or tardiness) blows up. Good stress test for whichever metric we trust to catch it.

Diagnostic decomposition (not a "scenario" so much as an experiment design)

Narrow eligibility (each op has exactly one legal machine) isolates pure dispatch-order effects — there's no routing choice left to make.
Wide eligibility + single-op jobs (no queue ever forms at a job level) isolates pure routing effects — there's no dispatch choice left to make.
Running both against the same rule set tells you whether a rule's failure mode is in SelectJob (dispatch) or SelectMachine (routing) — right now every result conflates both.