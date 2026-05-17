using System;
using System.Linq;
using UnityEngine;
using UnityEngine.Events;
using System.Collections.Generic;
using Assets.Scripts.Logging;
using Assets.Scripts.Simulation.Machines;
using Assets.Scripts.Simulation.AGV;
using Assets.Scripts.Simulation.FactoryLayout;
using Assets.Scripts.Simulation.Jobs;
using Assets.Scripts.Simulation.Stochastic;
using Assets.Scripts.Simulation.Types;
using Assets.Scripts.Simulation.Channels;
using Unity.MLAgents;

namespace Assets.Scripts.Simulation
{
    /// @brief The central orchestrator responsible for driving the factory simulation.
    ///
    /// @details SimulationBridge implements a strictly centralized state machine.
    /// In a single @c Update tick, it harvests status flags from physical components
    /// (Machines and AGVs), manages job transitions, resolves AGV assignments,
    /// and interfaces with the @c SchedulingAgent to resolve scheduling conflicts.
    /// No other component is permitted to mutate @c JobData state.
    ///
    /// Phase 2 additions:
    ///   - @c HarvestMachineFailureFlags() runs first each frame, handling machine
    ///     failures and repair completions before normal processing flags are read.
    ///   - @c BuildRoutingDecision() and @c FindNextDecision() exclude machines that
    ///     are Failed or Repairing from candidate lists and dispatch checks.
    ///   - @c StartEpisode() seeds per-machine TTF countdowns via @c InitializeStochastic().
    ///   - Public observation helpers expose fleet health scalars for the RL agent.
    public class SimulationBridge : MonoBehaviour
    {
        public static SimulationBridge Instance;

        [Header("Scene References")]
        [SerializeField] private FactoryLayoutManager layoutManager;
        [SerializeField] private TrafficZoneManager trafficZoneManager;
        [SerializeField] private AGVPool agvPool;
        [SerializeField] private SchedulingAgent agent;
        public JobStore Jobs;

        [Header("Episode Configuration")]
        public int PreDispatchLeadTime = 15;
        public bool AutoStartOnPlay = false;

        private FJSSPJobDefinition[] prebuiltJobs;

        private FJSSPConfig currentConfig;
        private Dictionary<MachineType, List<int>> cachedMachinesByType;
        public Dictionary<MachineType, List<int>> CachedMachinesByType => cachedMachinesByType;

        private bool episodeActive;
        private int decisionCount;
        public int DecisionCount => decisionCount;
        private double totalReward;
        private double previousMakespan;
        private float startTime;

        public bool IsEpisodeActive => episodeActive;
        public bool IsFactoryReady { get; set; }
        public double SimTime => Time.time - startTime;
        public FJSSPConfig CurrentConfig => currentConfig;

        public DecisionRequest CurrentDecision { get; private set; }
        public bool IsWaitingForAction { get; private set; }
        public string LastAppliedRule { get; private set; } = "Waiting...";

        [Header("Events")]
        public UnityEvent<DecisionRequest> OnDecisionRequired;
        public UnityEvent<StepResult> OnStepCompleted;
        public UnityEvent<EpisodeResult> OnEpisodeFinished;
        public UnityEvent OnFactorySpawned;

        private static readonly DispatchingRule[] ActionToRule = new DispatchingRule[]
        {
            DispatchingRule.SPT_SMPT,
            DispatchingRule.SPT_SRWT,
            DispatchingRule.LPT_MMUR,
            DispatchingRule.LPT_SMPT,
            DispatchingRule.SRT_SRWT,
            DispatchingRule.SRT_SMPT,
            DispatchingRule.LRT_MMUR,
            DispatchingRule.SDT_SRWT
        };

        // ── Stochastic Tracking Variables ──────────────────────────────────────
        private int _episodeFailureCount;
        private float _episodeTotalRepairTime;
        private float _episodeTotalTtfObserved;
        private Dictionary<int, double> _machineLastOperationalTime = new Dictionary<int, double>();

        // ── Per-machine utilization tracking (deterministic + stochastic) ─────
        // Keyed by MachineId. Initialized in StartEpisode() for every machine.
        private Dictionary<int, double> _machineProcessingStartTime = new();
        private Dictionary<int, double> _machineTotalProcessingTime = new();
        private Dictionary<int, int> _machineOpsCompleted = new();
        // Downtime = time in Failed or Repairing state. Zero for deterministic runs.
        private Dictionary<int, double> _machineDowntimeStart = new();
        private Dictionary<int, double> _machineTotalDowntime = new();

        public static int ActionCount => ActionToRule.Length;
        public int GetRuleIndex(DispatchingRule rule) => Array.IndexOf(ActionToRule, rule);

        private void Awake()
        {
            if (Instance != null) { Destroy(this); return; }
            Instance = this;
        }

        private void Start()
        {
            if (AutoStartOnPlay && agent != null)
                agent.IsArmed = true;
        }

        /// @brief Updates the active configuration for the next simulation run.
        ///
        /// @details Also seeds the @c StochasticEventManager so the entire stochastic
        /// stream is deterministic given config.Seed. Per-machine TTF initialisation
        /// happens later in @c StartEpisode() once the factory floor is built.
        public void LoadConfig(FJSSPConfig config)
        {
            currentConfig = config;
            IsFactoryReady = false;

            // Seed the stochastic RNG from the config so the failure stream is
            // reproducible. Per-machine TTF countdown init happens in StartEpisode.
            StochasticEventManager.Instance?.Initialize(config);
        }

        /// @brief Injects pre-built job definitions (e.g. from Brandimarte benchmarks).
        public void LoadPrebuiltJobs(FJSSPJobDefinition[] jobs)
        {
            prebuiltJobs = jobs;
        }

        /// @brief Physically instantiates the factory floor, machines, and AGV fleet.
        ///
        /// @details Initializes the @c layoutManager, @c trafficZoneManager, and
        /// @c agvPool based on the current @c FJSSPConfig. This must be called
        /// before @c StartEpisode.
        public void SpawnFactory()
        {
            if (currentConfig == null) return;
            if (IsFactoryReady || episodeActive) StopEpisode();

            UnityEngine.Random.InitState(currentConfig.Seed);
            cachedMachinesByType = layoutManager.BuildFloor(currentConfig);
            trafficZoneManager.BuildZoneGraph();
            agvPool.InitializeFleet(currentConfig.AGVCount);

            IsFactoryReady = true;
            OnFactorySpawned?.Invoke();
        }

        /// @brief Initiates a new simulation episode.
        ///
        /// @details Generates a new set of job definitions, initializes the @c JobStore,
        /// resets performance metrics, and — when stochastic mode is active — seeds
        /// each machine's TTF countdown with an age-randomised starting value.
        public void StartEpisode()
        {
            // ── Consume Python-sent config if available ────────────────────────
            // EpisodeConfigChannel.Instance is null during headless batch runs
            // (no ML-Agents connection), so this is a safe no-op in that mode.
            var pythonConfig = EpisodeConfigChannel.Instance?.ConsumeConfig();
            if (pythonConfig != null)
            {
                currentConfig = pythonConfig;
                IsFactoryReady = false;  // force SpawnFactory with the new config
                SimLogger.Low($"[Bridge] Applied Python config: {currentConfig.Name}");
            }
            // if (currentConfig == null)
            //     currentConfig = BuildDefaultConfig();

            if (currentConfig == null)
            {
                currentConfig = BuildDefaultStochasticConfig();
            }
            if (!IsFactoryReady)
                SpawnFactory();

            agent.SetHeuristicRule(currentConfig.dispatchingRule);

            // ── Use prebuilt jobs if injected, otherwise generate ──────────────
            FJSSPJobDefinition[] jobDefs;
            if (prebuiltJobs != null)
            {
                jobDefs = prebuiltJobs;
                prebuiltJobs = null;
                SimLogger.Low("[Orchestrator] Using prebuilt benchmark jobs");
            }
            else
            {
                jobDefs = FJSSPJobGenerator.Generate(currentConfig, cachedMachinesByType);
            }

            Jobs.Initialize(jobDefs, spawnVisuals: true);

            if (currentConfig.Stochastic != null && currentConfig.Stochastic.AnyEnabled)
            {

                // ── Reset Stochastic Tracking ──────────────────────────────────────
                _episodeFailureCount = 0;
                _episodeTotalRepairTime = 0f;
                _episodeTotalTtfObserved = 0f;
                _machineLastOperationalTime.Clear();

                // ── Stochastic: arm per-machine TTF countdowns ─────────────────────
                StochasticEventManager.Instance?.Initialize(currentConfig);
                foreach (var machine in layoutManager.Machines)
                {
                    machine.InitializeStochastic();
                    _machineLastOperationalTime[machine.MachineId] = 0.0; // Starts operational at t=0
                }
            }

            // ── Per-machine utilization init ───────────────────────────────────────
            _machineProcessingStartTime.Clear();
            _machineTotalProcessingTime.Clear();
            _machineOpsCompleted.Clear();
            _machineDowntimeStart.Clear();
            _machineTotalDowntime.Clear();
            foreach (var machine in layoutManager.Machines)
            {
                _machineTotalProcessingTime[machine.MachineId] = 0.0;
                _machineOpsCompleted[machine.MachineId] = 0;
                _machineTotalDowntime[machine.MachineId] = 0.0;
            }
            episodeActive = true;
            decisionCount = 0;
            totalReward = 0;
            previousMakespan = 0;
            IsWaitingForAction = false;
            startTime = Time.time;

            SimLogger.Low($"[Orchestrator] Episode started: {currentConfig.JobCount} jobs, " +
                          $"{layoutManager.MachineCount} machines, " +
                          $"stochastic={StochasticEventManager.Instance?.IsActive}");
        }

        /// @brief Aborts the current episode and cleans up all runtime data.
        public void StopEpisode()
        {
            episodeActive = false;
            IsWaitingForAction = false;
            IsFactoryReady = false;
            layoutManager.ClearFloor();
            Jobs.Cleanup();
            agvPool.ClearFleet();
        }

        // ── Update loop ───────────────────────────────────────────────────────

        /// @brief The core execution loop of the simulation.
        ///
        /// @details Processes the simulation in six distinct phases:
        ///   0. Harvest machine failure / repair-complete flags  ← Phase 2 addition
        ///   1. Harvest normal completion flags from machines.
        ///   2. Harvest delivery/pickup flags from AGVs.
        ///   3. Predictive pre-dispatch of AGVs for near-complete operations.
        ///   4. Assignment of AGVs to jobs awaiting transport.
        ///   5. Identification and triggering of the next scheduling decision.
        ///
        /// Failure flags are processed first so that jobs returned to NeedsRouting
        /// by a failure are immediately eligible for routing in the same frame.
        private void Update()
        {
            if (!episodeActive) return;

            HarvestMachineFailureFlags();   // Phase 2: must run before normal flags
            HarvestMachineFlags();
            HarvestAGVFlags();
            HarvestAlmostDoneFlags();
            AssignAGVs();

            if (!IsWaitingForAction)
                FindNextDecision();

            if (Jobs.AreAllExited())
                FinaliseEpisode();
        }

        // ── Phase 2: Machine failure harvesting ───────────────────────────────

        /// @brief Polls all machines for @c FailedFlag and @c RepairCompleteFlag.
        ///
        /// @details This is a no-op when @c MachineFailuresEnabled is false, adding
        /// zero overhead in deterministic mode.
        private void HarvestMachineFailureFlags()
        {
            if (StochasticEventManager.Instance == null ||
                !StochasticEventManager.Instance.MachineFailuresEnabled)
                return;



            foreach (var machine in layoutManager.Machines)
            {
                if (machine.FailedFlag)
                    HandleMachineFailure(machine);
                else if (machine.RepairCompleteFlag)
                    HandleMachineRepairComplete(machine);
            }
        }

        /// @brief Handles job return and AGV re-routing when a machine fails.
        ///
        /// @details Execution order:
        ///   1. Return any active processing job to NeedsRouting (no partial credit).
        ///   2. Return all jobs Queued at this machine to NeedsRouting.
        ///   3. Re-route any AGV whose cargo is destined for this machine.
        ///   4. Cancel any pre-dispatched AGV headed for this machine.
        ///   5. Call @c AcknowledgeFailure() to begin the repair countdown.
        ///   6. Invalidate any pending scheduling decision for this machine.
        private void HandleMachineFailure(PhysicalMachine machine)
        {
            int machineId = machine.MachineId;
            SimLogger.Low($"[Orchestrator] Machine {machineId} FAILED. " +
                          $"RepairTime={machine.SampledRepairDuration:F1}s");

            _episodeFailureCount++;
            // Discard any partial processing interval — the operation restarts in
            // full after repair, consistent with the existing job-return logic.
            _machineProcessingStartTime.Remove(machine.MachineId);

            // Begin tracking downtime from the moment of failure.
            _machineDowntimeStart[machine.MachineId] = SimTime;
            _episodeTotalRepairTime += machine.SampledRepairDuration;

            if (_machineLastOperationalTime.TryGetValue(machineId, out double lastOpTime))
            {
                _episodeTotalTtfObserved += (float)(SimTime - lastOpTime);
            }


            // 1. Return the actively processing job to NeedsRouting.
            //    Do not credit any processing time — the operation is restarted in full.
            if (machine.ActiveJobId >= 0)
            {
                JobData processingJob = Jobs.Get(machine.ActiveJobId);
                if (processingJob != null && processingJob.State == JobState.Processing)
                {
                    processingJob.State = JobState.NeedsRouting;
                    processingJob.LocationMachineId = machineId;
                    processingJob.StateEntryTime = SimTime;
                    SimLogger.Low($"[Orchestrator] Job {processingJob.JobId} returned to " +
                                  $"NeedsRouting (was Processing on failed machine {machineId}).");
                }
            }

            // 2. Return jobs Queued at this machine to NeedsRouting.
            //    They may have been waiting for the machine and should now be re-routed
            //    to an operational alternative rather than sitting through a long repair.
            foreach (var job in Jobs.AllJobs)
            {
                if (job.LocationMachineId == machineId && job.State == JobState.Queued)
                {
                    job.State = JobState.NeedsRouting;
                    job.StateEntryTime = SimTime;
                    SimLogger.Low($"[Orchestrator] Queued job {job.JobId} re-routed " +
                                  $"from failed machine {machineId}.");
                }
            }

            // 3. Re-route AGVs carrying jobs destined for this machine.
            //    The job is returned to NeedsRouting; the AGV will continue its
            //    transit and arrive at the failed machine, but since the machine is
            //    now Repairing (IsIdle=true, IsAvailableForWork=false) FindNextDecision
            //    will not dispatch anything to it. The job stays Queued there until
            //    a subsequent HarvestMachineFailureFlags pass (if re-routed above)
            //    or until repair completes.
            //
            //    TODO: For tighter control, implement AGVController.AbortMission() to
            //    halt the AGV in place and return it to Idle. This prevents the AGV
            //    from wasting transit time heading to a dead machine. The job would
            //    then stay InTransit with AssignedAgvId cleared, waiting for
            //    AssignAGVs() to re-dispatch once routing assigns a new target.
            foreach (var agv in agvPool.AllAGVs)
            {
                int agvJobId = agv.CurrentJobId;
                if (agvJobId < 0) continue;

                JobData transitJob = Jobs.Get(agvJobId);
                if (transitJob == null) continue;
                if (transitJob.State != JobState.InTransit) continue;
                if (transitJob.TargetMachineId != machineId) continue;

                transitJob.State = JobState.NeedsRouting;
                transitJob.TargetMachineId = -1;
                transitJob.AssignedAgvId = -1;
                transitJob.StateEntryTime = SimTime;

                SimLogger.Low($"[Orchestrator] AGV {agv.AgvId} carrying job {agvJobId} " +
                              $"re-routed: destination machine {machineId} has failed. " +
                              $"(Implement AGVController.AbortMission() for immediate halt.)");
            }

            // 4. Cancel any pre-dispatched AGV headed for this machine.
            foreach (var job in Jobs.AllJobs)
            {
                if (job.PreDispatchedAgvId < 0) continue;
                if (job.TargetMachineId != machineId) continue;

                AGVController preAgv = agvPool.GetPreDispatchedAGV(job.JobId);
                if (preAgv != null)
                {
                    // TODO: Replace with preAgv.CancelPreDispatch() once AGVController
                    // exposes cancellation. For now we release the booking and the AGV
                    // will idle when it arrives with no finalization pending.
                    SimLogger.Low($"[Orchestrator] Pre-dispatch for job {job.JobId} to " +
                                  $"machine {machineId} cancelled. " +
                                  $"(Implement AGVController.CancelPreDispatch() for clean halt.)");
                }
                job.PreDispatchedAgvId = -1;
            }

            // 5. Transition machine to Repairing and start the repair countdown.
            machine.AcknowledgeFailure();
            RefreshMachineLabels(machineId);

            // 6. Invalidate any pending decision that was built for this machine.
            //    A routing decision's MachineId is -1 (job-centric), so only
            //    dispatch decisions targeting this specific machine are affected.
            if (IsWaitingForAction &&
                CurrentDecision.Type == DecisionType.Dispatch &&
                CurrentDecision.MachineId == machineId)
            {
                IsWaitingForAction = false;
                SimLogger.Low($"[Orchestrator] Pending dispatch decision for machine " +
                              $"{machineId} invalidated (machine failed).");
            }
        }

        /// @brief Handles returning a machine to operational status after repair.
        private void HandleMachineRepairComplete(PhysicalMachine machine)
        {
            SimLogger.Low($"[Orchestrator] Machine {machine.MachineId} repair complete — " +
                          $"returning to OPERATIONAL.");

            machine.AcknowledgeRepairComplete();
            // Close the downtime interval that opened in HandleMachineFailure.
            if (_machineDowntimeStart.TryGetValue(machine.MachineId, out double dtStart))
            {
                _machineTotalDowntime[machine.MachineId] += SimTime - dtStart;
                _machineDowntimeStart.Remove(machine.MachineId);
            }
            RefreshMachineLabels(machine.MachineId);

            _machineLastOperationalTime[machine.MachineId] = SimTime;

            // No further action required: FindNextDecision will naturally pick up
            // any Queued jobs at this machine on the next frame now that
            // IsAvailableForWork is true again.
        }

        // Add to SimulationBridge — call from FinaliseEpisode()
        private void LogStochasticEpisodeSummary()
        {
            if (StochasticEventManager.Instance == null ||
                !StochasticEventManager.Instance.MachineFailuresEnabled)
                return;

            int totalFailures = _episodeFailureCount;
            float totalRepairTime = _episodeTotalRepairTime;
            float meanTtfObs = totalFailures > 0 ? _episodeTotalTtfObserved / totalFailures : 0f;

            // Theoretical Weibull mean: λ × Γ(1 + 1/k)
            // k=1.5 → Γ(1.667) ≈ 0.9027, so mean ≈ λ × 0.9027
            float theoreticalMeanTtf = currentConfig.Stochastic.WeibullLambda * 0.9027f;

            Debug.Log($"[StochasticValidation] Failures={totalFailures} " +
                      $"MeanTTF_obs={meanTtfObs:F1}s  MeanTTF_theory={theoreticalMeanTtf:F1}s  " +
                      $"TotalRepairTime={totalRepairTime:F1}s");
        }

        // ── Observation helpers (Phase 2 Global Scalars additions) ───────────

        /// @brief Fraction of machines currently in the @c Failed state. [0, 1]
        /// New Global Scalars channel 11 (0-indexed from 10).
        public float GetFractionMachinesFailed()
        {
            int total = layoutManager.MachineCount;
            if (total == 0) return 0f;
            int count = layoutManager.Machines.Count(m => m.HealthState == MachineHealthState.Failed);
            return (float)count / total;
        }

        /// @brief Fraction of machines currently in the @c Repairing state. [0, 1]
        /// New Global Scalars channel 12.
        public float GetFractionMachinesRepairing()
        {
            int total = layoutManager.MachineCount;
            if (total == 0) return 0f;
            int count = layoutManager.Machines.Count(m => m.HealthState == MachineHealthState.Repairing);
            return (float)count / total;
        }

        /// @brief Mean normalised remaining repair time across all repairing machines. [0, 1]
        /// Returns 0 when no machine is currently repairing.
        /// New Global Scalars channel 13.
        public float GetMeanNormalisedRepairTime()
        {
            var repairing = layoutManager.Machines
                .Where(m => m.HealthState == MachineHealthState.Repairing &&
                            m.SampledRepairDuration > 0f)
                .ToList();

            if (repairing.Count == 0) return 0f;

            float sum = repairing.Sum(m => m.RemainingRepairTime / m.SampledRepairDuration);
            return sum / repairing.Count;
        }

        // ── Unchanged: HarvestMachineFlags ───────────────────────────────────

        /// @brief Processes machines that have finished their current processing timer.
        ///
        /// @details Unchanged from original. Advances the operation index of the
        /// associated job, updates conveyor visuals, and transitions the job state to
        /// either @c NeedsRouting or @c WaitingForPickup (if all operations are complete).
        private void HarvestMachineFlags()
        {
            foreach (var machine in layoutManager.Machines)
            {
                if (!machine.FinishedFlag) continue;

                int jobId = machine.ActiveJobId;
                machine.ClearFinished();
                // Accumulate processing time and increment op counter for this machine.
                int mid = machine.MachineId;
                if (_machineProcessingStartTime.TryGetValue(mid, out double procStart))
                {
                    _machineTotalProcessingTime[mid] += SimTime - procStart;
                    _machineProcessingStartTime.Remove(mid);
                }
                if (_machineOpsCompleted.ContainsKey(mid))
                    _machineOpsCompleted[mid]++;

                JobData job = Jobs.Get(jobId);
                if (job == null) continue;

                job.CompletedOps++;
                if (job.CurrentOpIndex < job.TotalOperations)
                    job.CurrentOpIndex++;

                machine.PlaceOnOutgoing(jobId, job.Visual);
                RefreshMachineLabels(machine.MachineId);

                if (job.IsLastOperation)
                {
                    job.State = JobState.WaitingForPickup;
                    job.TargetMachineId = -1;
                    job.LocationMachineId = machine.MachineId;
                    job.StateEntryTime = SimTime;

                    if (job.PreDispatchedAgvId >= 0)
                    {
                        AGVController preAgv = agvPool.GetPreDispatchedAGV(job.JobId);
                        if (preAgv != null)
                        {
                            preAgv.FinalizePreDispatch(job.JobId, layoutManager.OutgoingBeltPosition, null, job.Visual);
                            job.AssignedAgvId = preAgv.AgvId;
                        }
                        job.PreDispatchedAgvId = -1;
                    }
                }
                else
                {
                    job.State = JobState.NeedsRouting;
                    job.LocationMachineId = machine.MachineId;
                    job.StateEntryTime = SimTime;
                }
            }
        }

        // ── Unchanged: HarvestAlmostDoneFlags ────────────────────────────────

        /// @brief Triggers predictive AGV movement for jobs nearing completion.
        ///
        /// @details Pre-dispatch sends an AGV to the *source* machine's pickup dock
        /// (the machine that is nearly done). The destination machine for the *next*
        /// operation is not yet decided at this point, so no health-state filtering
        /// of the future destination is required here. That filtering happens in
        /// @c BuildRoutingDecision when the job transitions to NeedsRouting.
        private void HarvestAlmostDoneFlags()
        {
            foreach (var machine in layoutManager.Machines)
            {
                if (!machine.AlmostDoneFlag) continue;

                int jobId = machine.AlmostDoneJobId;
                machine.ClearAlmostDone();

                JobData job = Jobs.Get(jobId);
                if (job == null || job.State != JobState.Processing || job.PreDispatchedAgvId >= 0) continue;
                if (job.CompletedOps == job.TotalOperations - 1) continue;

                AGVController agv = agvPool.GetAvailableAGV();
                if (agv == null) continue;

                agv.PreDispatch(jobId, machine.GetPickupPosition(), machine);
                job.PreDispatchedAgvId = agv.AgvId;
            }
        }

        // ── Unchanged: HarvestAGVFlags ────────────────────────────────────────

        /// @brief Processes AGV completion flags to transition job states.
        private void HarvestAGVFlags()
        {
            foreach (var agv in agvPool.AllAGVs)
            {
                if (agv.PickedUpFlag)
                {
                    int jobId = agv.CurrentJobId;
                    JobData job = Jobs.Get(jobId);
                    if (job != null && job.State == JobState.WaitingForPickup)
                    {
                        job.State = JobState.InTransit;
                        job.StateEntryTime = SimTime;
                    }
                }

                if (agv.DeliveredFlag)
                {
                    int jobId = agv.DeliveredJobId;
                    int machineId = agv.DeliveredMachineId;
                    JobData job = Jobs.Get(jobId);

                    if (job != null)
                    {
                        if (machineId < 0)
                        {
                            job.State = JobState.Exited;
                            job.LocationMachineId = -1;
                            job.StateEntryTime = SimTime;
                            if (job.Visual != null) job.Visual.gameObject.SetActive(false);
                        }
                        else
                        {
                            job.State = JobState.Queued;
                            job.LocationMachineId = machineId;
                            job.StateEntryTime = SimTime;

                            PhysicalMachine targetMachine = layoutManager.GetMachine(machineId);
                            targetMachine.PlaceOnIncoming(jobId, job.Visual);
                            RefreshMachineLabels(machineId);
                        }
                        job.TotalTransitTime += (SimTime - job.StateEntryTime);
                        job.AssignedAgvId = -1;
                    }
                }

                if (agv.PickedUpFlag || agv.DeliveredFlag)
                    agv.ClearFlags();
            }
        }

        // ── Unchanged: AssignAGVs ─────────────────────────────────────────────

        /// @brief Pairs unassigned jobs with available AGV units.
        private void AssignAGVs()
        {
            var candidates = new List<JobData>();
            foreach (var job in Jobs.AllJobs)
            {
                if (job.State == JobState.WaitingForPickup
                    && job.AssignedAgvId == -1
                    && job.PreDispatchedAgvId < 0)
                {
                    candidates.Add(job);
                }
            }

            foreach (var job in candidates)
            {
                AGVController agv = agvPool.GetAvailableAGV();
                if (agv == null) break;

                PhysicalMachine sourceMachine = job.LocationMachineId >= 0 ? layoutManager.GetMachine(job.LocationMachineId) : null;
                Vector3 pickupPos = sourceMachine != null ? sourceMachine.GetPickupPosition() : layoutManager.IncomingBeltPosition;

                PhysicalMachine targetMachine = job.TargetMachineId >= 0 ? layoutManager.GetMachine(job.TargetMachineId) : null;
                Vector3 dropoffPos = targetMachine != null ? targetMachine.GetDropoffPosition() : layoutManager.OutgoingBeltPosition;

                job.AssignedAgvId = agv.AgvId;
                agv.Dispatch(job.JobId, pickupPos, dropoffPos, sourceMachine, targetMachine, job.Visual);
                agv.SetCarryVisual(job.Visual);
            }
        }

        // ── Modified: FindNextDecision ────────────────────────────────────────

        /// @brief Evaluates the factory state to determine if a new decision is required.
        ///
        /// @details Phase 2 change: dispatch candidates are now gated by
        /// @c machine.IsAvailableForWork in addition to @c machine.IsIdle, so that
        /// Failed and Repairing machines are never offered to the scheduler.
        private void FindNextDecision()
        {
            JobData routingJob = Jobs.GetNextNeedsRouting();
            if (routingJob != null)
            {
                // Guard: if all eligible machines for this job's required type are
                // currently Failed/Repairing, we cannot build a valid routing decision.
                // Skip and try again next frame — the job stays NeedsRouting.
                var eligibleIds = new HashSet<int>(
                    routingJob.EligibleMachinesPerOp[routingJob.CurrentOpIndex].Keys);

                bool anyAvailable = layoutManager.Machines
                    .Any(m => eligibleIds.Contains(m.MachineId) && m.IsAvailableForWork);

                if (!anyAvailable)
                {
                    SimLogger.Low($"[Orchestrator] Job {routingJob.JobId} needs {eligibleIds.ToString()} " +
                                  $"but all machines of that type are Failed/Repairing. " +
                                  $"Deferring routing decision.");
                    // Fall through to check for other decisions rather than blocking entirely.
                }
                else
                {
                    CurrentDecision = BuildRoutingDecision(routingJob);
                    IsWaitingForAction = true;
                    OnDecisionRequired?.Invoke(CurrentDecision);
                    return;
                }
            }

            foreach (var machine in layoutManager.Machines)
            {
                // Phase 2 change: gate on IsAvailableForWork so Failed/Repairing
                // machines with a queued job do not trigger a dispatch decision.
                if (machine.IsIdle && machine.IsAvailableForWork && Jobs.HasDispatchableJob(machine.MachineId))
                {
                    CurrentDecision = BuildDispatchDecision(machine.MachineId);
                    IsWaitingForAction = true;
                    OnDecisionRequired?.Invoke(CurrentDecision);
                    return;
                }
            }
        }

        // ── Unchanged: Step ───────────────────────────────────────────────────

        /// @brief Applies an agent's chosen action to the simulation.
        ///
        /// @param actionIndex The discrete index of the dispatching rule to apply.
        /// @return A @c StepResult containing the immediate reward and episode status.
        public StepResult Step(int actionIndex)
        {
            IsWaitingForAction = false;

            if (CurrentDecision.Type == DecisionType.Routing)
                ExecuteRoutingDecision(actionIndex);
            else if (CurrentDecision.Type == DecisionType.Dispatch)
                ExecuteDispatchDecision(actionIndex);

            float reward = CalculateReward();
            totalReward += reward;

            return new StepResult
            {
                Reward = reward,
                Done = false,
                CurrentMakespan = SimTime
            };
        }

        // ── Modified: ExecuteRoutingDecision ─────────────────────────────────

        /// @brief Finalizes a machine assignment for a job based on the chosen rule.
        private void ExecuteRoutingDecision(int actionIndex)
        {
            int chosenMachineId = ApplyMachineSelectionRule(actionIndex, CurrentDecision);
            JobData job = Jobs.Get(CurrentDecision.JobId);
            if (job == null) return;

            job.TargetMachineId = chosenMachineId;
            job.State = JobState.WaitingForPickup;
            job.StateEntryTime = SimTime;

            if (job.PreDispatchedAgvId >= 0)
            {
                AGVController preAgv = agvPool.GetPreDispatchedAGV(job.JobId);
                if (preAgv != null)
                {
                    PhysicalMachine targetMachine = layoutManager.GetMachine(chosenMachineId);
                    Vector3 dropoffPos = targetMachine != null ? targetMachine.GetDropoffPosition() : layoutManager.OutgoingBeltPosition;
                    preAgv.FinalizePreDispatch(job.JobId, dropoffPos, targetMachine, job.Visual);
                    job.AssignedAgvId = preAgv.AgvId;
                    job.PreDispatchedAgvId = -1;
                    return;
                }
                job.PreDispatchedAgvId = -1;
            }
        }

        // ── Unchanged: ExecuteDispatchDecision ───────────────────────────────

        /// @brief Commences machine processing for a job selected by the chosen rule.
        private void ExecuteDispatchDecision(int actionIndex)
        {
            int machineId = CurrentDecision.MachineId;
            int chosenJobId = ApplyDispatchingRule(actionIndex, machineId);

            JobData job = Jobs.Get(chosenJobId);
            if (job == null || job.State != JobState.Queued || job.LocationMachineId != machineId) return;

            float duration = job.GetProcessingTime(machineId);
            job.State = JobState.Processing;
            job.TotalWaitTime += (SimTime - job.StateEntryTime);
            job.StateEntryTime = SimTime;

            PhysicalMachine machine = layoutManager.GetMachine(machineId);
            machine.StartJob(chosenJobId, duration, job.Visual);
            // Record when this machine began processing so HarvestMachineFlags can
            // accumulate the exact elapsed SimTime when FinishedFlag fires.
            _machineProcessingStartTime[machineId] = SimTime;
            RefreshMachineLabels(machineId);

            LastAppliedRule = ActionToRule[actionIndex].ToString();
        }

        // ── Modified: BuildRoutingDecision ───────────────────────────────────

        /// @brief Constructs a request for a machine routing decision.
        ///
        /// @details Phase 2 change: candidate machines are filtered to
        /// @c IsAvailableForWork == true so that Failed or Repairing machines are
        /// never presented to the agent as valid routing targets.
        private DecisionRequest BuildRoutingDecision(JobData job)
        {
            MachineType required = job.NextRequiredType;

            var eligibleIds = new HashSet<int>(
                job.EligibleMachinesPerOp[job.CurrentOpIndex].Keys);

            var candidates = layoutManager.Machines
                .Where(m => eligibleIds.Contains(m.MachineId) && m.IsAvailableForWork)
                .Select(m => m.MachineId)
                .ToList();

            float[] queueLengths = candidates.Select(id => Jobs.GetMachineLoad(id)).ToArray();
            float[] jobTimes = candidates.Select(id => job.GetProcessingTime(id)).ToArray();

            return new DecisionRequest
            {
                Type = DecisionType.Routing,
                SimTime = SimTime,
                DecisionIndex = decisionCount++,
                TotalJobs = Jobs.JobCount,
                CompletedJobs = Jobs.CountInState(JobState.Exited),
                JobId = job.JobId,
                SourceMachineId = job.LocationMachineId,
                RequiredType = required,
                CandidateMachineIds = candidates.ToArray(),
                CandidateQueueLengths = queueLengths,
                CandidateJobTimes = jobTimes,
            };
        }

        // ── Unchanged: BuildDispatchDecision ─────────────────────────────────

        /// @brief Constructs a request for a job dispatch decision.
        private DecisionRequest BuildDispatchDecision(int machineId)
        {
            List<int> queue = Jobs.GetDispatchableJobs(machineId);
            double[] durations = queue.Select(id => (double)Jobs.GetProcessingTime(id, machineId)).ToArray();

            return new DecisionRequest
            {
                Type = DecisionType.Dispatch,
                MachineId = machineId,
                SimTime = SimTime,
                DecisionIndex = decisionCount++,
                TotalJobs = Jobs.JobCount,
                CompletedJobs = Jobs.CountInState(JobState.Exited),
                QueuedJobIds = queue.ToArray(),
                QueuedDurations = durations,
            };
        }

        // ── rule application helpers ──────────────────────────────

        private int ApplyDispatchingRule(int actionIndex, int machineId)
        {
            DispatchingRule rule = ActionToRule[actionIndex];
            List<int> queue = Jobs.GetDispatchableJobs(machineId);

            if (queue.Count == 0) return -1;
            if (queue.Count == 1) return queue[0];

            return rule switch
            {
                DispatchingRule.SPT_SMPT or DispatchingRule.SPT_SRWT => ArgMin(queue, id => Jobs.Get(id).GetProcessingTime(machineId)),
                DispatchingRule.LPT_MMUR or DispatchingRule.LPT_SMPT => ArgMax(queue, id => Jobs.Get(id).GetProcessingTime(machineId)),
                DispatchingRule.SRT_SRWT or DispatchingRule.SRT_SMPT => ArgMin(queue, id => GetRemainingWork(id)),
                DispatchingRule.LRT_MMUR => ArgMax(queue, id => GetRemainingWork(id)),
                DispatchingRule.SDT_SRWT => ArgMin(queue, id => (float)(SimTime - Jobs.Get(id).ArrivalTime)),
                _ => queue[0]
            };
        }

        private int ApplyMachineSelectionRule(int actionIndex, DecisionRequest req)
        {
            DispatchingRule rule = ActionToRule[actionIndex];
            int[] candidates = req.CandidateMachineIds;

            if (candidates.Length == 1) return candidates[0];

            return rule switch
            {
                DispatchingRule.SPT_SMPT or DispatchingRule.LPT_SMPT or DispatchingRule.SRT_SMPT => candidates[ArgMinIdx(req.CandidateJobTimes)],
                DispatchingRule.SPT_SRWT or DispatchingRule.SRT_SRWT or DispatchingRule.SDT_SRWT => candidates[ArgMinIdx(req.CandidateQueueLengths)],
                DispatchingRule.LPT_MMUR or DispatchingRule.LRT_MMUR => candidates[ArgMaxIdx(req.CandidateQueueLengths)],
                _ => candidates[0]
            };
        }

        // ── Unchanged: label refresh, reward, finalise, utilities ─────────────

        private void RefreshMachineLabels(int machineId)
        {
            PhysicalMachine machine = layoutManager.GetMachine(machineId);
            if (machine == null) return;

            int inCount = Jobs.GetDispatchableJobs(machineId).Count;
            int outCount = Jobs.AllJobs.Count(j => j.LocationMachineId == machineId && j.State == JobState.WaitingForPickup);
            machine.RefreshQueueLabels(inCount, outCount);
        }

        private float GetRemainingWork(int jobId)
        {
            JobData j = Jobs.Get(jobId);
            if (j == null) return 0f;
            float total = 0f;
            for (int o = j.CurrentOpIndex; o < j.TotalOperations; o++)
            {
                float min = j.EligibleMachinesPerOp[o].Values.Min();
                total += min;
            }
            return total;
        }

        private float CalculateReward()
        {
            float current = (float)SimTime;
            float delta = current - (float)previousMakespan;
            previousMakespan = current;

            int totalOps = Jobs.AllJobs.Sum(j => j.TotalOperations);
            return -delta / (Mathf.Max(totalOps, 1) * Time.timeScale);
        }

        private void FinaliseEpisode()
        {
            episodeActive = false;
            LogStochasticEpisodeSummary();

            var telemetry = EpisodeTelemetryChannel.Instance;
            if (telemetry != null)
            {
                telemetry.RecordEpisodeResult(
                    makespan: SimTime,
                    jobCount: currentConfig.JobCount,
                    machineCount: layoutManager.MachineCount,
                    totalOps: Jobs.AllJobs.Sum(j => j.TotalOperations),
                    decisions: decisionCount,
                    totalReward: totalReward,
                    ruleName: LastAppliedRule,
                    stochasticTag: currentConfig.Stochastic?.Tag ?? "none"
                );
                telemetry.Flush();
            }

            // Close any downtime intervals still open (machines still repairing at
            // episode end — only possible in stochastic mode).
            foreach (var machine in layoutManager.Machines)
            {
                if (_machineDowntimeStart.TryGetValue(machine.MachineId, out double dtStart))
                {
                    _machineTotalDowntime[machine.MachineId] += SimTime - dtStart;
                    _machineDowntimeStart.Remove(machine.MachineId);
                }
            }

            // Log one row per machine to machine_utilization.csv.
            foreach (var machine in layoutManager.Machines)
            {
                int mid = machine.MachineId;
                double timeProcessing = _machineTotalProcessingTime.TryGetValue(mid, out double tp) ? tp : 0.0;
                double totalDowntime = _machineTotalDowntime.TryGetValue(mid, out double td) ? td : 0.0;
                double timeOperational = SimTime - totalDowntime;   // == SimTime in deterministic runs

                ResultsLogger.LogMachineUtilization(
                    ruleName: LastAppliedRule,
                    seed: currentConfig.Seed,
                    makespan: SimTime,
                    machineId: mid,
                    machineType: machine.PrimaryType.ToString(),
                    opsCompleted: _machineOpsCompleted.TryGetValue(mid, out int ops) ? ops : 0,
                    timeProcessing: timeProcessing,
                    timeOperational: timeOperational
                );
            }

            OnEpisodeFinished?.Invoke(new EpisodeResult
            {
                Makespan = SimTime,
                DecisionPoints = decisionCount,
                TotalReward = totalReward,
                AGVCount = agvPool.AllAGVs.Count,
            });
        }

        private int ArgMin(List<int> ids, Func<int, float> score)
        {
            int best = ids[0]; float bestS = float.MaxValue;
            foreach (int id in ids) { float s = score(id); if (s < bestS) { bestS = s; best = id; } }
            return best;
        }

        private int ArgMax(List<int> ids, Func<int, float> score)
        {
            int best = ids[0]; float bestS = float.MinValue;
            foreach (int id in ids) { float s = score(id); if (s > bestS) { bestS = s; best = id; } }
            return best;
        }

        private int ArgMinIdx(float[] v)
        {
            int b = 0; for (int i = 1; i < v.Length; i++) if (v[i] < v[b]) b = i; return b;
        }

        private int ArgMaxIdx(float[] v)
        {
            int b = 0; for (int i = 1; i < v.Length; i++) if (v[i] > v[b]) b = i; return b;
        }

        private StochasticConfig BuildStochasticConfig()
        {
            return new StochasticConfig
            {
                MachineFailuresEnabled = true,
                WeibullK = 1.5f,
                WeibullLambda = 2000.0f,
                RepairLogMu = 2.0f,
                RepairLogSigma = 0.5f,
                // AgvFailuresEnabled = false,
                DynamicArrivalsEnabled = false
            };
        }

        private FJSSPConfig BuildDefaultStochasticConfig()
        {
            MachineType[] types = (MachineType[])Enum.GetValues(typeof(MachineType));
            var layout = new MachineType[types.Length];
            for (int i = 0; i < types.Length; i++) layout[i] = types[i];

            var procParams = new Dictionary<MachineType, (float mu, float sigma)>
            {
                { MachineType.Mill,     (mu:  9f, sigma: 1f) },
                { MachineType.Lathe,    (mu:  7f, sigma: 1f) },
                { MachineType.Weld,     (mu: 15f, sigma: 2f) },
                { MachineType.Inspect,  (mu:  6f, sigma: 1f) },
                { MachineType.Assemble, (mu: 24f, sigma: 4f) },
            };

            return new FJSSPConfig
            {
                Seed = 42,
                JobCount = 5,
                MachinesPerType = 1,
                MachineTypeLayout = layout,
                MinProcTime = 1f,
                MaxProcTime = 30f,
                MinOpsPerJob = 2,
                MaxOpsPerJob = 4,
                MaxArrivalTime = 0f,
                ProcTimeParams = procParams,
                AGVCount = 3,
                dispatchingRule = DispatchingRule.SRT_SRWT,
                Stochastic = BuildStochasticConfig(),
                MachineFlexibilityProbability = 0f,
            };
        }

        private FJSSPConfig BuildDefaultConfig()
        {
            MachineType[] types = (MachineType[])Enum.GetValues(typeof(MachineType));
            var layout = new MachineType[types.Length];
            for (int i = 0; i < types.Length; i++) layout[i] = types[i];

            var procParams = new Dictionary<MachineType, (float mu, float sigma)>
            {
                { MachineType.Mill,     (mu:  9f, sigma: 1f) },
                { MachineType.Lathe,    (mu:  7f, sigma: 1f) },
                { MachineType.Weld,     (mu: 15f, sigma: 2f) },
                { MachineType.Inspect,  (mu:  6f, sigma: 1f) },
                { MachineType.Assemble, (mu: 24f, sigma: 4f) },
            };

            return new FJSSPConfig
            {
                Seed = 42,
                JobCount = 15,
                MachinesPerType = 1,
                MachineTypeLayout = layout,
                MinProcTime = 1f,
                MaxProcTime = 30f,
                MinOpsPerJob = 2,
                MaxOpsPerJob = 4,
                MaxArrivalTime = 0f,
                ProcTimeParams = procParams,
                AGVCount = 3,
                dispatchingRule = DispatchingRule.SRT_SRWT,
                MachineFlexibilityProbability = 0f,
            };
        }
    }
}