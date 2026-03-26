using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.Events;
using Assets.Scripts.Scheduling.Core;
using Assets.Scripts.Scheduling.Data;
using Newtonsoft.Json;

namespace Assets.Scripts.Simulation
{
    /// @file SimulationBridge.cs
    /// @brief Stepped DES orchestrator that pauses at every dispatch decision point,
    /// accepts an action (from a fixed rule or RL agent), then advances to the next.
    ///
    /// @details The simulation loop is:
    /// @code
    ///   bridge.StartEpisode(instance);
    ///   while (!bridge.IsDone)
    ///   {
    ///       DecisionRequest req = bridge.CurrentDecision;
    ///       int action = agent.GetAction(req);   // or fixedRuleIndex
    ///       StepResult result = bridge.Step(action);
    ///       // result.Reward, result.Done, result.Observation...
    ///   }
    /// @endcode
    ///
    /// This is the Gymnasium @c step() pattern. The bridge owns the DES, the visual
    /// layer, and the reward computation. It exposes the same interface whether the
    /// caller is a fixed-rule validator (Phase 1), an ML-Agents policy (Phase 3),
    /// or a Python training loop over a socket (Phase 3 alt).
    ///
    /// @par Execution modes
    /// - **Training mode** (default): No visuals. @ref Step returns synchronously.
    ///   Maximum throughput for PPO data collection.
    /// - **Visual debug mode**: Visuals are driven between decision points via a
    ///   coroutine. @ref Step still returns immediately, but visual playback catches
    ///   up asynchronously. Toggle with @ref enableVisuals.
    ///
    /// @par Phase 1 coverage
    /// - Step 5  — DES-to-Unity bridge (stepped)
    /// - Step 7  — Job lifecycle counters, episode-end stats
    /// - Step 8  — Decision-point detection, logging, flash
    /// - Step 9  — Inspector controls, HUD data
    /// - Step 10 — @ref RunEpisodeWithFixedRule for batch validation

    // ═════════════════════════════════════════════════════════════
    //  Supporting types
    // ═════════════════════════════════════════════════════════════

    /// @brief Snapshot of the DES state at a decision point, presented to the
    /// agent (or fixed rule) so it can choose an action.
    [Serializable]
    public struct DecisionRequest
    {
        /// @brief ID of the machine that just became free and has a non-empty queue.
        public int MachineId;

        /// @brief Current simulation time.
        public double SimTime;

        /// @brief IDs of jobs waiting in the machine's queue.
        public int[] QueuedJobIds;

        /// @brief Processing times of the queued operations (parallel to QueuedJobIds).
        public double[] QueuedDurations;

        /// @brief How many decisions have been made so far this episode.
        public int DecisionIndex;

        /// @brief Total jobs in the instance.
        public int TotalJobs;

        /// @brief Jobs completed so far.
        public int CompletedJobs;

        // TODO Phase 2: Add full observation dict (grid, sched_matrix, scalars,
        //               distances, event_flags) once ObservationBuilder exists.
    }

    /// @brief Returned by @ref SimulationBridge.Step after the agent's action is applied.
    [Serializable]
    public struct StepResult
    {
        /// @brief Reward signal for this step.
        public float Reward;

        /// @brief True if the episode is finished (all jobs complete).
        public bool Done;

        /// @brief The next decision request, if not done. Check @ref Done first.
        public DecisionRequest NextDecision;

        /// @brief Current makespan estimate (max end-time so far).
        public double CurrentMakespan;

        /// @brief Number of operations completed so far.
        public int OperationsCompleted;
    }

    /// @brief Final episode statistics, emitted via @ref SimulationBridge.OnEpisodeFinished.
    [Serializable]
    public struct EpisodeResult
    {
        public string InstanceName;
        public string RuleName;
        public double Makespan;
        public double OptimalMakespan;
        public int TotalJobs;
        public int TotalOperations;
        public int CompletedJobs;
        public int DecisionPoints;
        public double TotalReward;
        public int[] PerMachineDecisions;

        /// @brief Percentage gap from optimal: (makespan - optimal) / optimal × 100.
        public double OptimalityGap =>
            OptimalMakespan > 0
                ? (Makespan - OptimalMakespan) / OptimalMakespan * 100.0
                : 0;
    }

    // ═════════════════════════════════════════════════════════════
    //  SimulationBridge
    // ═════════════════════════════════════════════════════════════

    public class SimulationBridge : MonoBehaviour
    {
        // ─────────────────────────────────────────────────────────
        //  Inspector — Scene references
        // ─────────────────────────────────────────────────────────

        [Header("Scene References")]
        [SerializeField] private FactoryLayoutManager layoutManager;

        // TODO Step 3: [SerializeField] private JobManager jobManager;
        // TODO Step 4: [SerializeField] private AGVPool agvPool;

        // ─────────────────────────────────────────────────────────
        //  Inspector — Episode configuration (Step 9)
        // ─────────────────────────────────────────────────────────

        [Header("Episode Configuration")]
        [Tooltip("JSON asset from Resources/Instances.")]
        [SerializeField] private TextAsset taillardJson;

        [Tooltip("Dispatching rule index (0-7) for fixed-rule mode. Ignored when an RL agent is connected.")]
        [Range(0, 7)]
        [SerializeField] private int fixedRuleIndex = 0;

        [Tooltip("If true, use the fixed rule for all decisions (Phase 1 validation).")]
        [SerializeField] private bool useFixedRule = true;

        [Tooltip("Auto-start an episode on Play.")]
        [SerializeField] private bool autoStartOnPlay = true;

        // ─────────────────────────────────────────────────────────
        //  Inspector — Visual mode
        // ─────────────────────────────────────────────────────────

        [Header("Visual Playback")]
        [Tooltip("Enable visual playback between decision points. Disable for training throughput.")]
        [SerializeField] private bool enableVisuals = true;

        [Tooltip("Speed multiplier for visual playback between decisions.")]
        [Range(0.1f, 100f)]
        [SerializeField] private float speedMultiplier = 5f;

        [Tooltip("Flash machines at decision points.")]
        [SerializeField] private bool flashOnDecision = true;

        // ─────────────────────────────────────────────────────────
        //  Inspector — Events
        // ─────────────────────────────────────────────────────────

        [Header("Events")]
        public UnityEvent<DecisionRequest> OnDecisionRequired;
        public UnityEvent<StepResult> OnStepCompleted;
        public UnityEvent<EpisodeResult> OnEpisodeFinished;

        // ─────────────────────────────────────────────────────────
        //  Action-space mapping
        // ─────────────────────────────────────────────────────────

        /// @brief Maps agent action index (0-7) to the composite PDR enum.
        ///
        /// @details Matches the action table in Environment.md:
        /// | 0 | SPT-SMPT | 1 | SPT-SRWT | 2 | LPT-MMUR | 3 | SRT-SRWT |
        /// | 4 | LRT-SMPT | 5 | LRT-MMUR | 6 | SRT-SMPT | 7 | SDT-SRWT |
        private static readonly DispatchingRule[] ActionToRule = new DispatchingRule[]
        {
            DispatchingRule.SPT_SMPT,   // 0
            DispatchingRule.SPT_SRWT,   // 1
            DispatchingRule.LPT_MMUR,   // 2
            DispatchingRule.SRT_SRWT,   // 3
            DispatchingRule.LPT_SMPT,   // 4
            DispatchingRule.LRT_MMUR,   // 5
            DispatchingRule.SRT_SMPT,   // 6
            DispatchingRule.SDT_SRWT,   // 7

        };

        /// @brief Total number of discrete actions in the action space.
        public static int ActionCount => ActionToRule.Length;

        // ─────────────────────────────────────────────────────────
        //  Runtime state
        // ─────────────────────────────────────────────────────────

        private DESSimulator simulator;
        private TaillardInstance currentInstance;
        private bool episodeActive;
        private int decisionCount;
        private double totalReward;
        private double previousMakespan;
        private int[] perMachineDecisions;

        /// @brief Visual command buffer: events that occurred between the
        /// previous decision point and the current one, replayed visually
        /// when @ref enableVisuals is true.
        private readonly List<VisualEvent> pendingVisualEvents = new List<VisualEvent>();

        // ─────────────────────────────────────────────────────────
        //  Public read-only accessors (Step 9 HUD)
        // ─────────────────────────────────────────────────────────

        /// @brief True while an episode is in progress.
        public bool IsEpisodeActive => episodeActive;

        /// @brief True when the episode has ended (all jobs scheduled).
        public bool IsDone => !episodeActive;

        /// @brief The current decision request. Only valid when the DES is
        /// waiting and the episode is active.
        public DecisionRequest CurrentDecision { get; private set; }

        /// @brief True if the DES is paused waiting for an action.
        public bool IsWaitingForAction => episodeActive && simulator.WaitingForDecision;

        /// @brief Current simulation time.
        public double SimTime => simulator?.CurrentTime ?? 0;

        /// @brief Running makespan estimate (max operation end-time so far).
        public double CurrentMakespan => ComputeRunningMakespan();

        /// @brief Visual playback speed.
        public float SpeedMultiplier
        {
            get => speedMultiplier;
            set => speedMultiplier = Mathf.Max(0.01f, value);
        }

        /// @brief Decisions made so far this episode.
        public int DecisionCount => decisionCount;

        /// @brief The underlying DES simulator for advanced queries.
        public DESSimulator Simulator => simulator;

        // ─────────────────────────────────────────────────────────
        //  Unity lifecycle
        // ─────────────────────────────────────────────────────────

        private void Awake()
        {
            simulator = new DESSimulator();
        }

        private void Start()
        {
            if (autoStartOnPlay && taillardJson != null)
            {
                StartEpisode();

                // In fixed-rule mode, auto-play the whole episode.
                if (useFixedRule)
                {
                    StartCoroutine(RunFixedRuleCoroutine());
                }
                // Otherwise, the RL agent calls Step() externally.
            }
        }

        // ─────────────────────────────────────────────────────────
        //  Public API — Episode lifecycle
        // ─────────────────────────────────────────────────────────

        /// @brief Initialises a new episode using the Inspector-configured instance.
        ///
        /// @details Loads the Taillard JSON, resets the DES, optionally builds the
        /// visual floor, and advances to the first decision point. After this call,
        /// @ref CurrentDecision is populated and @ref IsWaitingForAction is true
        /// (assuming the instance produces at least one queuing conflict).
        public void StartEpisode()
        {
            StartEpisode(taillardJson);
        }

        /// @brief Starts an episode with an explicit Taillard JSON asset.
        ///
        /// @param json TextAsset containing the Taillard instance JSON.
        public void StartEpisode(TextAsset json)
        {
            if (json == null)
            {
                Debug.LogError("[SimBridge] No Taillard instance assigned.");
                return;
            }

            taillardJson = json;
            currentInstance = LoadInstance(json);
            if (currentInstance == null) return;

            // ── Initialise DES (LoadInstance calls Reset internally) ──
            simulator.LoadInstance(currentInstance);

            // ── Build visual floor ───────────────────────────────
            if (enableVisuals && layoutManager != null)
            {
                layoutManager.BuildFloor(simulator);
                // TODO Step 3: jobManager.SpawnJobs(simulator);
            }

            // ── Reset episode tracking ───────────────────────────
            episodeActive = true;
            decisionCount = 0;
            totalReward = 0;
            previousMakespan = 0;
            perMachineDecisions = new int[simulator.Machines.Length];
            pendingVisualEvents.Clear();

            Debug.Log($"[SimBridge] Episode started: {currentInstance.Name} " +
                      $"({currentInstance.JobCount}J × {currentInstance.MachineCount}M)");

            // ── Advance to the first decision point ──────────────
            AdvanceToNextDecision();
        }

        /// @brief The core Gymnasium-style step function.
        ///
        /// @details
        /// 1. Validates that the DES is actually waiting for a decision.
        /// 2. Maps the action index (0-7) to a composite @ref DispatchingRule.
        /// 3. Snapshots the queue state for logging/visuals.
        /// 4. Calls @ref DESSimulator.ApplyDecision with the chosen rule.
        /// 5. Advances the DES event-by-event to the next decision point (or end).
        /// 6. Computes the step reward from delta-makespan.
        /// 7. If visuals are enabled, plays back the intermediate events.
        /// 8. Returns a @ref StepResult.
        ///
        /// @param action Index into the composite PDR action space (0-7).
        /// @returns A @ref StepResult with reward, done flag, and next decision.
        public StepResult Step(int action)
        {
            if (!episodeActive || !simulator.WaitingForDecision)
            {
                Debug.LogWarning("[SimBridge] Step() called but no decision is pending.");
                return new StepResult { Done = true };
            }

            if (action < 0 || action >= ActionToRule.Length)
            {
                Debug.LogError($"[SimBridge] Invalid action {action}. Must be 0-{ActionToRule.Length - 1}.");
                return new StepResult { Done = true };
            }

            // ── 1. Resolve the action ───────────────────────────
            DispatchingRule chosenRule = ActionToRule[action];
            int machineId = simulator.PendingDecisionMachineId;
            Machine machine = simulator.Machines[machineId];

            // Snapshot queue before the decision modifies it.
            int[] queueSnapshot = machine.WaitingQueue
                .Select(op => op.JobId).ToArray();

            // ── 2. Apply the decision to the DES ────────────────
            simulator.ApplyDecision(chosenRule);
            decisionCount++;
            perMachineDecisions[machineId]++;

            // ── 3. Visual: record decision flash/log ────────────
            if (enableVisuals && layoutManager != null)
            {
                EmitDecisionVisual(machineId, queueSnapshot, chosenRule, action);
            }

            // ── 4. Advance to next decision or episode end ──────
            AdvanceToNextDecision();

            // ── 5. Compute reward ───────────────────────────────
            double currentMakespan = ComputeRunningMakespan();
            float reward = ComputeReward(currentMakespan);
            totalReward += reward;
            previousMakespan = currentMakespan;

            // ── 6. Build result ─────────────────────────────────
            bool done = !episodeActive;

            StepResult result = new StepResult
            {
                Reward = reward,
                Done = done,
                CurrentMakespan = currentMakespan,
                OperationsCompleted = simulator.TotalOperationsCompleted,
            };

            if (!done)
                result.NextDecision = CurrentDecision;

            // ── 7. Visual playback ──────────────────────────────
            if (enableVisuals && pendingVisualEvents.Count > 0)
                FlushVisualEvents();

            // ── 8. Notify listeners ─────────────────────────────
            OnStepCompleted?.Invoke(result);

            if (done)
                FinaliseEpisode();

            return result;
        }

        /// @brief Runs an entire episode synchronously with a fixed rule.
        /// No visuals. Returns the result immediately.
        ///
        /// @details Used for Phase 1 Step 10 batch validation:
        /// @code
        ///   foreach (TextAsset ta in taillardAssets)
        ///       foreach (int rule = 0; rule < 8; rule++)
        ///           EpisodeResult r = bridge.RunEpisodeWithFixedRule(ta, rule);
        /// @endcode
        ///
        /// @param json The Taillard instance.
        /// @param ruleIndex Action index (0-7).
        /// @returns Final @ref EpisodeResult.
        public EpisodeResult RunEpisodeWithFixedRule(TextAsset json, int ruleIndex)
        {
            bool savedVisuals = enableVisuals;
            enableVisuals = false;

            StartEpisode(json);

            while (episodeActive && simulator.WaitingForDecision)
            {
                Step(ruleIndex);
            }

            enableVisuals = savedVisuals;
            return BuildEpisodeResult();
        }

        // ─────────────────────────────────────────────────────────
        //  Core stepping logic
        // ─────────────────────────────────────────────────────────

        /// @brief Advances the DES event-by-event until it either hits a
        /// decision point or runs out of events.
        ///
        /// @details Between decision points, many events may fire (job arrivals
        /// that auto-start on idle machines, operation completions on machines
        /// with empty queues). These are all processed automatically. Only when
        /// @ref TryStartNextOnMachineStepped finds a non-empty queue does the
        /// DES pause and return @ref SimStepResult.DecisionRequired.
        ///
        /// Visual events are buffered into @ref pendingVisualEvents for
        /// optional playback after the step completes.
        private void AdvanceToNextDecision()
        {
            pendingVisualEvents.Clear();

            while (true)
            {
                SimStepResult result = simulator.ProcessNextEvent();

                switch (result)
                {
                    case SimStepResult.Continue:
                        // Another event processed, no decision needed yet.
                        SnapshotVisualState();
                        continue;

                    case SimStepResult.DecisionRequired:
                        // DES paused. Build the decision request for the caller.
                        CurrentDecision = BuildDecisionRequest();
                        OnDecisionRequired?.Invoke(CurrentDecision);

                        Debug.Log($"[SimBridge] Decision #{decisionCount}: " +
                                  $"t={simulator.CurrentTime:F1}, " +
                                  $"M{simulator.PendingDecisionMachineId}, " +
                                  $"queue=[{string.Join(", ", CurrentDecision.QueuedJobIds)}]");
                        return;

                    case SimStepResult.Done:
                        // All events exhausted. Episode over.
                        episodeActive = false;
                        return;
                }
            }
        }

        /// @brief Builds a @ref DecisionRequest from the current DES state.
        ///
        /// @details Captures the pending machine's queue contents and
        /// episode-level counters so the agent has everything it needs
        /// to choose an action. The full observation tensor (grid, sched
        /// matrix, etc.) will be added in Phase 2 via an ObservationBuilder.
        private DecisionRequest BuildDecisionRequest()
        {
            int machineId = simulator.PendingDecisionMachineId;
            Machine machine = simulator.Machines[machineId];

            int[] jobIds = machine.WaitingQueue
                .Select(op => op.JobId).ToArray();
            double[] durations = machine.WaitingQueue
                .Select(op => (double)op.Duration).ToArray();

            return new DecisionRequest
            {
                MachineId = machineId,
                SimTime = simulator.CurrentTime,
                QueuedJobIds = jobIds,
                QueuedDurations = durations,
                DecisionIndex = decisionCount,
                TotalJobs = simulator.Jobs.Length,
                CompletedJobs = simulator.TotalJobsCompleted,
            };
        }

        // ─────────────────────────────────────────────────────────
        //  Reward computation
        // ─────────────────────────────────────────────────────────

        /// @brief Computes the step reward from delta-makespan.
        ///
        /// @details The primary signal is the negative change in running
        /// makespan since the last decision. This gives a dense, per-step
        /// gradient that directly penalises schedule elongation:
        ///
        /// @code
        ///   reward = -(currentMakespan - previousMakespan) / normaliser
        /// @endcode
        ///
        /// A completion bonus proportional to the optimality ratio is added
        /// on the final step to reinforce good terminal performance.
        ///
        /// @param currentMakespan Running makespan after this step.
        /// @returns Scalar reward.
        private float ComputeReward(double currentMakespan)
        {
            // Normalise by total operations so reward magnitude is stable
            // across instance sizes (ta01=225 ops, ta11=400 ops, etc.).
            int totalOps = simulator.Jobs.Sum(j => j.Operations.Length);
            double normaliser = Math.Max(totalOps, 1);

            double delta = currentMakespan - previousMakespan;
            float reward = (float)(-delta / normaliser);

            // Terminal bonus: ratio of optimal to achieved makespan.
            if (!simulator.EventQueue.HasEvents &&
                currentInstance != null &&
                currentInstance.metadata.optimum > 0)
            {
                double ratio = currentInstance.metadata.optimum / Math.Max(currentMakespan, 1);
                reward += (float)ratio;
            }

            return reward;
        }

        /// @brief Max end-time across all operations that have started processing.
        private double ComputeRunningMakespan()
        {
            if (simulator?.Jobs == null) return 0;

            double max = 0;
            foreach (Job job in simulator.Jobs)
            {
                foreach (Operation op in job.Operations)
                {
                    // EndTime is -1 for unscheduled operations.
                    if (op.EndTime > max)
                        max = op.EndTime;
                }
            }
            return max;
        }

        // ─────────────────────────────────────────────────────────
        //  Fixed-rule auto-play (Phase 1 validation)
        // ─────────────────────────────────────────────────────────

        /// @brief Coroutine that auto-plays every decision with the
        /// configured @ref fixedRuleIndex, yielding between steps
        /// so visuals can update.
        private IEnumerator RunFixedRuleCoroutine()
        {
            yield return null; // let layout settle one frame

            while (episodeActive && simulator.WaitingForDecision)
            {
                Step(fixedRuleIndex);

                if (enableVisuals)
                {
                    // Let visual playback breathe between decisions.
                    yield return StartCoroutine(WaitForVisualPlayback());
                }
                else
                {
                    // Even without visuals, yield every N steps to
                    // avoid freezing the editor on large instances.
                    if (decisionCount % 50 == 0)
                        yield return null;
                }
            }

            Debug.Log("[SimBridge] Fixed-rule auto-play complete.");
        }

        // ─────────────────────────────────────────────────────────
        //  Visual layer
        // ─────────────────────────────────────────────────────────

        /// @brief Lightweight record of a single visual event for buffered playback.
        private struct VisualEvent
        {
            public enum Kind
            {
                OperationStarted,
                OperationCompleted,
            }

            public Kind Type;
            public double Time;
            public int JobId;
            public int MachineId;
            public double Duration;
        }

        /// @brief Snapshots any new operation starts/completions from the DES.
        ///
        /// @details Called after each @ref ProcessNextEvent in the advance loop.
        /// Checks every machine for state transitions at the current time.
        private void SnapshotVisualState()
        {
            if (!enableVisuals) return;

            double t = simulator.CurrentTime;

            foreach (Machine m in simulator.Machines)
            {
                // Detect freshly started operations.
                if (m.State == MachineState.Busy &&
                    m.CurrentOperation != null &&
                    Math.Abs(m.CurrentOperation.StartTime - t) < 0.001)
                {
                    pendingVisualEvents.Add(new VisualEvent
                    {
                        Type = VisualEvent.Kind.OperationStarted,
                        Time = t,
                        JobId = m.CurrentOperation.JobId,
                        MachineId = m.Id,
                        Duration = m.CurrentOperation.Duration,
                    });
                }

                // Detect freshly completed operations (machine just went Idle
                // and an operation ended at this time).
                if (m.State == MachineState.Idle && m.CurrentOperation == null)
                {
                    // Check if any job's operation on this machine ended now.
                    foreach (Job job in simulator.Jobs)
                    {
                        foreach (Operation op in job.Operations)
                        {
                            if (op.MachineId == m.Id &&
                                op.EndTime > 0 &&
                                Math.Abs(op.EndTime - t) < 0.001)
                            {
                                pendingVisualEvents.Add(new VisualEvent
                                {
                                    Type = VisualEvent.Kind.OperationCompleted,
                                    Time = t,
                                    JobId = op.JobId,
                                    MachineId = m.Id,
                                });
                            }
                        }
                    }
                }
            }
        }

        /// @brief Records a decision-point flash and log on the machine visual.
        private void EmitDecisionVisual(int machineId, int[] queueSnapshot,
                                        DispatchingRule rule, int actionIndex)
        {
            MachineVisual mv = layoutManager?.GetMachineVisual(machineId);
            if (mv == null) return;

            Machine coreMachine = simulator.Machines[machineId];
            int chosenJobId = coreMachine.CurrentOperation?.JobId ?? -1;

            mv.RecordDecisionPoint(
                (float)simulator.CurrentTime,
                queueSnapshot,
                chosenJobId,
                $"{rule} (action {actionIndex})",
                flash: flashOnDecision
            );
        }

        /// @brief Pushes all buffered visual events to the machine visuals.
        private void FlushVisualEvents()
        {
            if (layoutManager == null) return;

            foreach (VisualEvent ve in pendingVisualEvents)
            {
                MachineVisual mv = layoutManager.GetMachineVisual(ve.MachineId);
                if (mv == null) continue;

                switch (ve.Type)
                {
                    case VisualEvent.Kind.OperationStarted:
                        mv.BeginOperation(ve.JobId, (float)ve.Time, (float)ve.Duration);
                        break;

                    case VisualEvent.Kind.OperationCompleted:
                        mv.CompleteOperation(ve.JobId);
                        break;
                }
            }

            UpdateAllProgressBars();
        }

        /// @brief Waits for visual playback to display between decision points.
        ///
        /// @details Computes the sim-time span of buffered events and converts
        /// to a real-time wait via the speed multiplier. Replace with
        /// event-driven AGV completion in Phase 1 Step 4.
        private IEnumerator WaitForVisualPlayback()
        {
            if (pendingVisualEvents.Count == 0)
                yield break;

            double earliest = pendingVisualEvents[0].Time;
            double latest = pendingVisualEvents[pendingVisualEvents.Count - 1].Time;
            double span = Math.Max(latest - earliest, 0.1);

            float realWait = (float)(span / speedMultiplier);
            realWait = Mathf.Clamp(realWait, 0.02f, 5f);

            float elapsed = 0f;
            while (elapsed < realWait)
            {
                elapsed += Time.deltaTime;
                UpdateAllProgressBars();
                yield return null;
            }
        }

        /// @brief Drives all busy machine progress bars to the current sim time.
        private void UpdateAllProgressBars()
        {
            if (layoutManager?.MachineVisuals == null) return;

            float t = (float)(simulator?.CurrentTime ?? 0);
            foreach (MachineVisual mv in layoutManager.MachineVisuals)
            {
                if (mv != null && mv.CurrentState == MachineState.Busy)
                    mv.UpdateProgress(t);
            }
        }

        // ─────────────────────────────────────────────────────────
        //  Episode finalisation
        // ─────────────────────────────────────────────────────────

        /// @brief Logs final stats and fires @ref OnEpisodeFinished.
        private void FinaliseEpisode()
        {
            EpisodeResult result = BuildEpisodeResult();

            Debug.Log($"[SimBridge] Episode complete: " +
                      $"makespan={result.Makespan:F1}, " +
                      $"optimal={result.OptimalMakespan:F1}, " +
                      $"gap={result.OptimalityGap:F1}%, " +
                      $"decisions={result.DecisionPoints}, " +
                      $"reward={result.TotalReward:F3}");

            OnEpisodeFinished?.Invoke(result);
        }

        private EpisodeResult BuildEpisodeResult()
        {
            return new EpisodeResult
            {
                InstanceName = currentInstance?.Name ?? "unknown",
                RuleName = useFixedRule
                    ? ActionToRule[fixedRuleIndex].ToString()
                    : "agent",
                Makespan = simulator.Makespan > 0
                    ? simulator.Makespan
                    : ComputeRunningMakespan(),
                OptimalMakespan = currentInstance?.metadata.optimum ?? 0,
                TotalJobs = simulator.Jobs?.Length ?? 0,
                TotalOperations = simulator.TotalOperationsCompleted,
                CompletedJobs = simulator.TotalJobsCompleted,
                DecisionPoints = decisionCount,
                TotalReward = totalReward,
                PerMachineDecisions = (int[])perMachineDecisions?.Clone(),
            };
        }

        // ─────────────────────────────────────────────────────────
        //  Instance loading
        // ─────────────────────────────────────────────────────────

        /// @brief Deserialises a Taillard JSON TextAsset.
        ///
        /// @note Adapt the deserialisation call to match your TaillardInstance API.
        private TaillardInstance LoadInstance(TextAsset json)
        {
            try
            {
                TaillardInstance instance = JsonConvert.DeserializeObject<TaillardInstance>(json.text);

                if (instance == null)
                {
                    Debug.LogError($"[SimBridge] Failed to parse: {json.name}");
                    return null;
                }

                Debug.Log($"[SimBridge] Loaded: {instance.Name} " +
                          $"({instance.JobCount}J × {instance.MachineCount}M), " +
                          $"optimum={instance.metadata?.optimum ?? 0}");
                return instance;
            }
            catch (Exception ex)
            {
                Debug.LogError($"[SimBridge] Parse error: {ex.Message}");
                return null;
            }
        }
    }
}