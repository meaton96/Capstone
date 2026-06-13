using System;
using System.Linq;
using UnityEngine;
using UnityEngine.Events;
using System.Collections.Generic;
using Assets.Scripts.Simulation.Logging;
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
    /// <summary>
    /// Central orchestrator for the factory simulation. Manages episode lifecycle,
    /// coordinates machine/AGV/job systems, and interfaces with the learning agent
    /// for dispatch decision-making.
    /// </summary>
    public class FactoryOrchestrator : MonoBehaviour
    {
        /// <summary>
        /// Singleton instance of the FactoryOrchestrator.
        /// </summary>
        public static FactoryOrchestrator Instance;

        /// <summary>
        /// Manager responsible for laying out and manipulating the factory floor.
        /// </summary>
        [Header("Scene References")]
        [SerializeField] private FactoryLayoutManager layoutManager;

        /// <summary>
        /// Manager responsible for traffic zones and AGV pathfinding.
        /// </summary>
        [SerializeField] private TrafficZoneManager trafficZoneManager;

        /// <summary>
        /// Pool managing all AGV instances in the simulation.
        /// </summary>
        [SerializeField] private AGVPool agvPool;

        private string _configuredRuleName = "unknown";

        /// <summary>
        /// Scheduling agent that makes dispatch decisions during episodes.
        /// </summary>
        [SerializeField] private SchedulingAgent agent;

        /// <summary>
        /// Collection of all job data instances managed during the episode.
        /// </summary>
        public JobStore Jobs;

        /// <summary>
        /// Number of seconds before a job's required arrival time to dispatch an AGV in advance.
        /// </summary>
        [Header("Episode Configuration")]
        public int PreDispatchLeadTime = 15;

        /// <summary>
        /// Whether to automatically start an episode when the scene plays.
        /// </summary>
        public bool AutoStartOnPlay = false;

        /// <summary>
        /// Pre-built job definitions used for benchmark scenarios. Set before SpawnFactory.
        /// </summary>
        private FJSSPJobDefinition[] prebuiltJobs;

        /// <summary>
        /// The current simulation configuration loaded for this episode.
        /// </summary>
        private FJSSPConfig currentConfig;

        /// <summary>
        /// Cached mapping of machine types to their machine IDs, built during factory setup.
        /// </summary>
        private Dictionary<MachineType, List<int>> cachedMachinesByType;

        /// <summary>
        /// Read-only access to the cached machine type mapping.
        /// </summary>
        public Dictionary<MachineType, List<int>> CachedMachinesByType => cachedMachinesByType;

        /// <summary>
        /// Whether a simulation episode is currently active.
        /// </summary>
        private bool episodeActive;

        /// <summary>
        /// Number of dispatch decisions made during the current episode.
        /// </summary>
        private int decisionCount;

        /// <summary>
        /// Total number of dispatch decisions made in the current episode.
        /// </summary>
        public int DecisionCount => decisionCount;

        /// <summary>
        /// Cumulative reward accumulated during the current episode.
        /// </summary>
        private double totalReward;

        /// <summary>
        /// Makespan value from the previous step, used for reward calculation.
        /// </summary>
        private double previousMakespan;

        /// <summary>
        /// Simulation start time, used to compute elapsed simulation time.
        /// </summary>
        private float startTime;

        /// <summary>
        /// Whether a simulation episode is currently running.
        /// </summary>
        public bool IsEpisodeActive => episodeActive;

        /// <summary>
        /// Whether the factory layout has been spawned and is ready for episode start.
        /// </summary>
        public bool IsFactoryReady { get; set; }

        /// <summary>
        /// Elapsed simulation time since episode start.
        /// </summary>
        public double SimTime => Time.time - startTime;

        /// <summary>
        /// The current FJSSP configuration for this episode.
        /// </summary>
        public FJSSPConfig CurrentConfig => currentConfig;

        /// <summary>
        /// The most recent decision request awaiting an action from the agent.
        /// </summary>
        public DecisionRequest CurrentDecision { get; private set; }

        /// <summary>
        /// Whether the orchestrator is currently waiting for the agent to provide an action.
        /// </summary>
        public bool IsWaitingForAction { get; private set; }

        /// <summary>
        /// Name of the dispatching rule applied in the last step.
        /// </summary>
        public string LastAppliedRule { get; private set; } = "Waiting...";

        /// <summary>
        /// Tracks episode-level statistics including makespan, machine failures, and repair times.
        /// </summary>
        private readonly EpisodeTracker _tracker = new();

        /// <summary>
        /// Maps machine IDs to the simulation time when processing started, used for flag harvesting.
        /// </summary>
        private readonly Dictionary<int, double> _machineProcessingStartTime = new();

        /// <summary>
        /// Harvests machine and AGV state flags during each simulation step.
        /// </summary>
        private FlagHarvester _flags;

        /// <summary>
        /// Coordinates machine failure events and repair scheduling.
        /// </summary>
        private FailureCoordinator _failures;

        /// <summary>
        /// Manages decision requests and coordinates dispatch/routing decisions.
        /// </summary>
        private DecisionCoordinator _decisions;

        // ── Poisson arrival clock ─────────────────────────────────────────────

        /// <summary>
        /// SimTime at which the next dynamic job should be injected.
        /// float.MaxValue when the clock is disarmed (deterministic mode or cap reached).
        /// </summary>
        private float _nextArrivalSimTime = float.MaxValue;

        /// <summary>
        /// Next job ID to assign to a dynamically-arrived job.
        /// Initialised to config.JobCount so IDs never collide with the initial batch.
        /// </summary>
        private int _nextDynamicJobId;

        /// <summary>
        /// Count of dynamic jobs spawned so far in the current episode. Used to enforce
        /// DynamicJobCap and to populate EpisodeRecord.DynamicArrivals.
        /// </summary>
        private int _dynamicJobsSpawned;

        /// <summary>
        /// SimTime of the most recent Poisson job injection. -1 if none have fired this episode.
        /// Copied to EpisodeRecord.LastDynamicArrivalTime in FinaliseEpisode.
        /// </summary>
        private float _lastDynamicArrivalSimTime = -1f;

        /// <summary>
        /// Fired when a new dispatch or routing decision is required. Passes the DecisionRequest.
        /// </summary>
        [Header("Events")]
        public UnityEvent<DecisionRequest> OnDecisionRequired;

        /// <summary>
        /// Fired after each simulation step completes. Passes the StepResult.
        /// </summary>
        public UnityEvent<StepResult> OnStepCompleted;

        /// <summary>
        /// Fired when an episode finishes. Passes the final EpisodeRecord.
        /// </summary>
        public UnityEvent<EpisodeRecord> OnEpisodeFinished;

        /// <summary>
        /// Fired after the factory layout is spawned.
        /// </summary>
        public UnityEvent OnFactorySpawned;

        /// <summary>
        /// Total number of discrete actions available to the decision engine.
        /// </summary>
        public static int ActionCount => DispatchingEngine.ActionCount;

        /// <summary>
        /// Converts a DispatchingRule enum value to its corresponding action index.
        /// </summary>
        /// <param name="rule">The dispatching rule to convert.</param>
        /// <returns>The zero-based index for the given rule.</returns>
        public int GetRuleIndex(DispatchingRule rule) => DispatchingEngine.IndexForRule(rule);

        /// <summary>
        /// Maximum allowed simulation time in seconds before an episode is forcibly terminated.
        /// </summary>
        private const double MAX_EPISODE_SIM_SECONDS = 500_000.0;

        /// <summary>
        /// Singleton initialization. Destroys duplicate instances if one already exists.
        /// </summary>
        private void Awake()
        {
            if (Instance != null) { Destroy(this); return; }
            Instance = this;
        }

        /// <summary>
        /// Called on startup. Arms the scheduling agent if AutoStartOnPlay is enabled.
        /// </summary>
        private void Start()
        {
            if (AutoStartOnPlay && agent != null)
                agent.IsArmed = true;
        }

        /// <summary>
        /// Loads a simulation configuration and initializes the stochastic event manager.
        /// Resets factory readiness so SpawnFactory must be called before starting an episode.
        /// </summary>
        /// <param name="config">The FJSSP configuration to load.</param>
        public void LoadConfig(FJSSPConfig config)
        {
            currentConfig = config;
            IsFactoryReady = false;
            StochasticEventManager.Instance?.Initialize(config);
        }

        /// <summary>
        /// Stores pre-built job definitions for use during the next episode start.
        /// Prebuilt jobs override procedural generation.
        /// </summary>
        /// <param name="jobs">Array of job definitions to use for benchmark scenarios.</param>
        public void LoadPrebuiltJobs(FJSSPJobDefinition[] jobs)
        {
            prebuiltJobs = jobs;
        }

        /// <summary>
        /// Spawns the factory layout by building the floor plan, constructing the traffic
        /// zone graph, and initializing the AGV fleet. Resets any existing episode first.
        /// </summary>
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

        /// <summary>
        /// Starts a new simulation episode. Initializes jobs, stochastic events, failure
        /// coordination, and decision systems. Resets all agents and machines to their
        /// initial state. If a Python-provided config exists, it overrides the current config.
        /// </summary>
        public void StartEpisode()
        {
            var pythonConfig = EpisodeConfigChannel.Instance?.ConsumeConfig();
            if (pythonConfig != null)
            {
                currentConfig = pythonConfig;
                IsFactoryReady = false;
                SimLogger.Low($"[Bridge] Applied Python config: {currentConfig.Name}");
            }

            currentConfig ??= DefaultConfigFactory.BuildDefault();

            if (!IsFactoryReady)
                SpawnFactory();

            trafficZoneManager.ResetEpisodeStats();
            foreach (var agv in agvPool.AllAGVs)
                agv.ResetEpisodeStats();

            agent.SetHeuristicRule(currentConfig.dispatchingRule);
            _configuredRuleName = currentConfig.dispatchingRule.ToString();

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
                StochasticEventManager.Instance?.Initialize(currentConfig);
                foreach (var machine in layoutManager.Machines)
                    machine.InitializeStochastic();
            }

            _tracker.Reset();
            _machineProcessingStartTime.Clear();

            _flags = new FlagHarvester();
            _flags.Initialize(Jobs, agvPool, layoutManager, _tracker, _machineProcessingStartTime);

            _failures = new FailureCoordinator();
            _failures.Initialize(
                Jobs, agvPool, layoutManager, _tracker, _machineProcessingStartTime,
                onMachineFailedInvalidateDecision: (machineId) =>
                {
                    if (IsWaitingForAction &&
                        CurrentDecision.Type == DecisionType.Dispatch &&
                        CurrentDecision.MachineId == machineId)
                    {
                        IsWaitingForAction = false;
                        SimLogger.Medium($"[Orchestrator] Pending dispatch decision for machine " +
                                      $"{machineId} invalidated (machine failed).");
                    }
                },
                refreshLabels: _flags.RefreshMachineLabels
            );

            _decisions = new DecisionCoordinator();
            _decisions.Initialize(
                Jobs, layoutManager,
                getSimTime: () => SimTime,
                getDecisionCount: () => decisionCount,
                incrementDecisionCount: () => decisionCount++
            );

            episodeActive = true;
            decisionCount = 0;
            totalReward = 0;
            previousMakespan = 0;
            IsWaitingForAction = false;
            startTime = Time.time;

            // ── Arm Poisson arrival clock ──────────────────────────────────────
            _dynamicJobsSpawned = 0;
            _lastDynamicArrivalSimTime = -1f;
            _nextDynamicJobId = currentConfig.JobCount; // dynamic IDs start after initial batch
            bool arrivalsEnabled = StochasticEventManager.Instance?.DynamicArrivalsEnabled ?? false;
            if (arrivalsEnabled)
            {
                // SimTime ≈ 0 here, so first arrival is sampled from t=0
                _nextArrivalSimTime = StochasticEventManager.Instance.SampleInterArrivalTime();
                int cap = currentConfig.Stochastic?.DynamicJobCap ?? 0;
                SimLogger.Low($"[Orchestrator] Poisson clock armed — " +
                              $"first arrival ≈ t={_nextArrivalSimTime:F1}s " +
                              $"λ={currentConfig.Stochastic?.ArrivalLambda} " +
                              $"cap={(cap == 0 ? "∞" : cap.ToString())}");
            }
            else
            {
                _nextArrivalSimTime = float.MaxValue;
            }

            SimLogger.Low($"[Orchestrator] Episode started: {currentConfig.JobCount} jobs, " +
                          $"{layoutManager.MachineCount} machines, " +
                          $"stochastic={StochasticEventManager.Instance?.IsActive}");
        }

        /// <summary>
        /// Stops the current episode and cleans up all simulation state. Clears the factory
        /// floor, job store, and AGV fleet.
        /// </summary>
        public void StopEpisode()
        {
            episodeActive = false;
            IsWaitingForAction = false;
            IsFactoryReady = false;
            layoutManager.ClearFloor();
            Jobs.Cleanup();
            agvPool.ClearFleet();
        }

        /// <summary>
        /// Called every frame. Processes simulation flags, checks for new decisions, and
        /// terminates the episode when all jobs have exited or the time limit is reached.
        /// </summary>
        private void Update()
        {
            if (!episodeActive) return;

            if (SimTime > MAX_EPISODE_SIM_SECONDS)
            {
                SimLogger.Low($"[Orchestrator] Episode timeout at {SimTime:F0}s — terminating.");
                FinaliseEpisode();
                return;
            }

            _failures.SetSimTime(SimTime);
            _flags.SetSimTime(SimTime);

            _failures.HarvestFailureFlags();
            _flags.HarvestMachineFlags();
            _flags.HarvestAGVFlags();
            _flags.HarvestAlmostDoneFlags(PreDispatchLeadTime);
            _flags.AssignAGVs();

            if (!IsWaitingForAction)
            {
                var req = _decisions.FindNextDecision();
                if (req != null)
                {
                    CurrentDecision = req;
                    IsWaitingForAction = true;
                    OnDecisionRequired?.Invoke(CurrentDecision);
                }
            }

            // ── Tick Poisson arrival clock ─────────────────────────────────────
            // Runs regardless of IsWaitingForAction — arrivals are asynchronous events.
            TickPoissonClock();

            if (Jobs.AreAllExited())
                FinaliseEpisode();
        }

        /// <summary>
        /// Checks whether the Poisson arrival clock has fired and, if so, injects a new
        /// dynamic job and schedules the next arrival. Called every frame from Update.
        /// </summary>
        private void TickPoissonClock()
        {
            if (SimTime < _nextArrivalSimTime) return;

            int cap = currentConfig.Stochastic?.DynamicJobCap ?? 0;
            if (cap != 0 && _dynamicJobsSpawned >= cap) return;

            FJSSPJobDefinition def = FJSSPJobGenerator.GenerateSingle(
                _nextDynamicJobId++, currentConfig, cachedMachinesByType);

            Jobs.AddDynamicJob(def, spawnVisuals: true);
            _dynamicJobsSpawned++;
            _lastDynamicArrivalSimTime = (float)SimTime;

            def.ArrivalTime = (float)SimTime;

            SimLogger.Medium($"[Orchestrator] Dynamic job {def.JobId} arrived at " +
                             $"t={SimTime:F1}s — " +
                             $"{_dynamicJobsSpawned}/{(cap == 0 ? "∞" : cap.ToString())} dynamic jobs");

            bool moreExpected = cap == 0 || _dynamicJobsSpawned < cap;
            _nextArrivalSimTime = moreExpected
                ? (float)SimTime + StochasticEventManager.Instance.SampleInterArrivalTime()
                : float.MaxValue;
        }

        /// <summary>
        /// Executes a single simulation step given an action index from the agent.
        /// Applies the dispatch or routing decision, calculates reward, and returns the result.
        /// </summary>
        /// <param name="actionIndex">The index of the action to execute.</param>
        /// <returns>A StepResult containing the reward and simulation status.</returns>
        public StepResult Step(int actionIndex)
        {
            IsWaitingForAction = false;

            if (CurrentDecision.Type == DecisionType.Routing)
                ExecuteRoutingDecision(actionIndex);
            else if (CurrentDecision.Type == DecisionType.Dispatch)
                ExecuteDispatchDecision(actionIndex);

            float reward = CalculateReward();
            totalReward += reward;

            return new StepResult { Reward = reward, Done = false, CurrentMakespan = SimTime };
        }

        /// <summary>
        /// Executes a routing decision by assigning the specified job to the selected machine
        /// and preparing AGV pickup if applicable.
        /// </summary>
        /// <param name="actionIndex">The action index encoding the machine selection.</param>
        private void ExecuteRoutingDecision(int actionIndex)
        {
            int chosenMachineId = DispatchingEngine.SelectMachine(actionIndex, CurrentDecision);
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
                    Vector3 dropoffPos = targetMachine != null
                        ? targetMachine.GetDropoffPosition() : layoutManager.OutgoingBeltPosition;
                    preAgv.FinalizePreDispatch(job.JobId, dropoffPos, targetMachine, job.Visual);
                    job.AssignedAgvId = preAgv.AgvId;
                    job.PreDispatchedAgvId = -1;
                    return;
                }
                job.PreDispatchedAgvId = -1;
            }
        }

        /// <summary>
        /// Executes a dispatch decision by selecting a job from the specified machine's queue
        /// and starting its processing on the physical machine.
        /// </summary>
        /// <param name="actionIndex">The action index encoding the job selection.</param>
        private void ExecuteDispatchDecision(int actionIndex)
        {
            int machineId = CurrentDecision.MachineId;
            int chosenJobId = DispatchingEngine.SelectJob(actionIndex, machineId, Jobs, SimTime);

            JobData job = Jobs.Get(chosenJobId);
            if (job == null || job.State != JobState.Queued || job.LocationMachineId != machineId) return;

            float duration = job.GetProcessingTime(machineId);
            job.State = JobState.Processing;
            job.TotalWaitTime += (SimTime - job.StateEntryTime);
            job.StateEntryTime = SimTime;

            PhysicalMachine machine = layoutManager.GetMachine(machineId);
            machine.StartJob(chosenJobId, duration, job.Visual);

            _machineProcessingStartTime[machineId] = SimTime;

            _flags.RefreshMachineLabels(machineId);
            LastAppliedRule = _configuredRuleName;
        }

        /// <summary>
        /// Finalizes the current episode by collecting telemetry, building the episode record,
        /// logging results, and firing the OnEpisodeFinished event. Includes AGV performance
        /// and segment congestion data collection.
        /// </summary>
        private void FinaliseEpisode()
        {
            episodeActive = false;

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

            EpisodeRecord record = _tracker.Build(
                config: currentConfig,
                simTime: SimTime,
                ruleName: LastAppliedRule,
                completedJobs: Jobs.CountInState(JobState.Exited),
                totalOps: Jobs.AllJobs.Sum(j => j.TotalOperations),
                decisionPoints: decisionCount,
                totalReward: totalReward,
                agvCount: agvPool.AllAGVs.Count,
                machines: layoutManager.Machines,
                averageTimeScale: Time.timeScale
            );

            // Patch dynamic-arrival fields — EpisodeTracker.Build() derives JobCount from
            // config, which only reflects the initial batch. Override if dynamic jobs were spawned.
            SimLogger.Low($"[Orchestrator] Finalising episode — dynamic jobs spawned: {_dynamicJobsSpawned}");
            SimLogger.Low($"[Orchestrator] Last dynamic arrival at t={_lastDynamicArrivalSimTime:F1}s");
            record.DynamicArrivals = _dynamicJobsSpawned;
            record.LastDynamicArrivalTime = _lastDynamicArrivalSimTime;
            if (_dynamicJobsSpawned > 0)
                record.JobCount = Jobs.JobCount;  // true total = initial + dynamic

            // Collect AGV performance records
            foreach (var agv in agvPool.AllAGVs)
                record.AGVRecords.Add(agv.GetRecord(record.Makespan));

            // Skip parking alcove (Capacity=64) — it's intentionally unconstrained and
            // would inflate the zone count without diagnostic value.
            foreach (TrafficZone zone in trafficZoneManager.Zones)
            {
                if (zone.Name == "Parking_Alcove") continue;
                record.SegmentRecords.Add(new SegmentRecord
                {
                    ZoneId = zone.ZoneId,
                    ZoneName = zone.Name,
                    AisleType = zone.AisleType.ToString(),
                    FlowDirection = zone.Flow.ToString(),
                    TraversalCount = zone.TraversalCount,
                    BlockEvents = zone.BlockEvents,
                    TotalBlockTime = zone.TotalBlockTime,
                });
            }

            // Populate per-job operation records for job_operations.csv.
            // Dynamic jobs have IDs >= the initial batch size (set in StartEpisode).
            int initialBatchSize = currentConfig.JobCount;
            foreach (var job in Jobs.AllJobs)
            {
                bool isDynamic = job.JobId >= initialBatchSize;
                for (int i = 0; i < job.TotalOperations; i++)
                {
                    var eligible = job.EligibleMachinesPerOp[i];
                    float min = float.MaxValue, max = 0f, sum = 0f;
                    foreach (float t in eligible.Values)
                    {
                        if (t < min) min = t;
                        if (t > max) max = t;
                        sum += t;
                    }
                    record.JobOperationRecords.Add(new JobOperationRecord
                    {
                        JobId = job.JobId,
                        IsDynamic = isDynamic,
                        ArrivalTime = job.ArrivalTime,
                        OpIndex = i,
                        MachineTypeRequired = job.OperationTypes[i].ToString(),
                        EligibleMachineCount = eligible.Count,
                        MinProcTime = eligible.Count > 0 ? min : 0f,
                        MaxProcTime = eligible.Count > 0 ? max : 0f,
                        MeanProcTime = eligible.Count > 0 ? sum / eligible.Count : 0f,
                    });
                }
            }

            if (!Academy.Instance.IsCommunicatorOn)
            {
                SimLogger.Low($"[Orchestrator] Logging episode results");
                //ResultsLogger.LogAll(record);
            }

            if (record.MachineFailureCount > 0)
            {
                float theoreticalMeanTtf = currentConfig.Stochastic != null
                    ? EpisodeTracker.TheoreticalMeanTTF(currentConfig.Stochastic.WeibullLambda)
                    : 0f;
                SimLogger.Low($"[StochasticSummary] Failures={record.MachineFailureCount} " +
                              $"TotalRepairTime={record.MachineRepairTime:F1}s " +
                              $"MeanTTF_theory={theoreticalMeanTtf:F1}s");
            }

            OnEpisodeFinished?.Invoke(record);
        }

        /// <summary>
        /// Calculates the per-step reward as the negative normalized change in makespan.
        /// Penalizes increases in completion time relative to the number of remaining operations.
        /// </summary>
        /// <returns>The computed reward value (typically negative).</returns>
        private float CalculateReward()
        {
            float current = (float)SimTime;
            float delta = current - (float)previousMakespan;
            previousMakespan = current;
            int totalOps = Jobs.AllJobs.Sum(j => j.TotalOperations);
            return -delta / (Mathf.Max(totalOps, 1) * Time.timeScale);
        }

        /// <summary>
        /// Returns the fraction of machines currently in a failed state.
        /// </summary>
        /// <returns>Ratio of failed machines to total machines (0.0 to 1.0).</returns>
        public float GetFractionMachinesFailed()
        {
            int total = layoutManager.MachineCount;
            if (total == 0) return 0f;
            return (float)layoutManager.Machines.Count(m => m.HealthState == MachineHealthState.Failed) / total;
        }

        /// <summary>
        /// Returns the fraction of machines currently in a repairing state.
        /// </summary>
        /// <returns>Ratio of repairing machines to total machines (0.0 to 1.0).</returns>
        public float GetFractionMachinesRepairing()
        {
            int total = layoutManager.MachineCount;
            if (total == 0) return 0f;
            return (float)layoutManager.Machines.Count(m => m.HealthState == MachineHealthState.Repairing) / total;
        }

        /// <summary>
        /// Returns the mean normalized remaining repair time across all repairing machines.
        /// Normalized as the ratio of remaining time to the originally sampled repair duration.
        /// </summary>
        /// <returns>Average normalized repair time (0.0 to 1.0).</returns>
        public float GetMeanNormalisedRepairTime()
        {
            var repairing = layoutManager.Machines
                .Where(m => m.HealthState == MachineHealthState.Repairing && m.SampledRepairDuration > 0f)
                .ToList();

            if (repairing.Count == 0) return 0f;
            return repairing.Sum(m => m.RemainingRepairTime / m.SampledRepairDuration) / repairing.Count;
        }
    }
}