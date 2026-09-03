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
        /// When true, heuristic (fixed-PDR) decisions are drained within a single frame
        /// instead of one-per-frame. ONLY valid for baseline batch runs with no neural
        /// policy — it bypasses the ml-agents request cycle and applies the configured
        /// rule directly. MUST stay false for agent training/inference, or the policy
        /// never sees observations.
        /// </summary>
        public bool BaselineDrainMode = false;

        private int _baselineRuleIndex;
        private bool _baselineRuleIsRandom;
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
        /// Live per-decision log (routing + dispatch), cleared each episode and copied onto
        /// the EpisodeRecord at FinaliseEpisode. See DecisionRecord for schema.
        /// </summary>
        private readonly List<DecisionRecord> _decisionLog = new List<DecisionRecord>();

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

        // ── Throughput window clock ───────────────────────────────────────────
        /// <summary>Length of each throughput window (sim-seconds), from config; 60 default.</summary>
        private float _throughputWindowLength = 60f;

        /// <summary>SimTime at which the next throughput window closes.</summary>
        private float _nextThroughputBoundary = 60f;

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
        private const double MAX_EPISODE_SIM_SECONDS = 100_000.0;

        /// <summary>
        /// Deadlock watchdog: if zero AGVs anywhere in the traffic-zone network complete a
        /// zone entry (TrafficZone.TraversalCount, summed across all zones) for this many
        /// consecutive sim-seconds while jobs remain incomplete, the episode is declared
        /// deadlocked and terminated immediately instead of running to MAX_EPISODE_SIM_SECONDS.
        /// A circular-wait deadlock in TrafficZoneManager.TryReserve never self-resolves (no
        /// AGV in the cycle can ever move), so ANY sustained system-wide stall is conclusive —
        /// no legitimate congestion (even the heaviest surviving runs) goes this long without
        /// a traversal completing somewhere in the network.
        /// </summary>
        private const double DEADLOCK_STALL_SECONDS = 3_000.0;

        private int _lastZoneTraversalTotal = -1;
        private double _lastTraversalChangeSimTime;
        private bool _deadlockDetected;
        private double _deadlockSimTime = -1.0;

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
            agvPool.InitializeFleet(currentConfig);

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

            _baselineRuleIsRandom = currentConfig.dispatchingRule == DispatchingRule.Random;
            _baselineRuleIndex = GetRuleIndex(currentConfig.dispatchingRule);

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
            _throughputWindowLength = currentConfig.ThroughputTimingWindow <= 0f ? 60f : currentConfig.ThroughputTimingWindow;
            _nextThroughputBoundary = _throughputWindowLength;   // first window closes at t = windowLength
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
                incrementDecisionCount: () => decisionCount++,
                // -1 in RL-agent/interactive mode (no fixed rule yet at decision-assembly time —
                // job-priority pre-selection wouldn't make sense before the agent has acted).
                // Re-resolves "random" per call, matching DrainHeuristicDecisions' own resolution
                // (line ~615) -- independent draws for job-selection vs. the eventual machine/job
                // Execute-time choice, an accepted minor inconsistency specific to the Random PDR.
                getBaselineActionIndex: () => BaselineDrainMode
                    ? (_baselineRuleIsRandom ? UnityEngine.Random.Range(0, DispatchingEngine.ActionCount) : _baselineRuleIndex)
                    : -1
            );

            episodeActive = true;
            decisionCount = 0;
            _decisionLog.Clear();
            totalReward = 0;
            previousMakespan = 0;
            IsWaitingForAction = false;
            startTime = Time.time;

            // ── Arm deadlock watchdog ────────────────────────────────────────────
            _lastZoneTraversalTotal = -1;
            _lastTraversalChangeSimTime = 0.0;
            _deadlockDetected = false;
            _deadlockSimTime = -1.0;

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

            double fixedDuration = currentConfig?.Stochastic?.EpisodeDurationSeconds ?? 0.0;
            if (fixedDuration > 0.0 && SimTime > fixedDuration)
            {
                SimLogger.Low($"[Orchestrator] Fixed episode duration reached at {SimTime:F0}s — " +
                              $"terminating (steady-state mode; in-flight jobs recorded as censored).");
                FinaliseEpisode();
                return;
            }

            if (CheckForDeadlock())
            {
                _deadlockDetected = true;
                _deadlockSimTime = SimTime;
                SimLogger.Error($"[Orchestrator] Deadlock detected — no AGV completed a traffic-zone " +
                                 $"entry anywhere in the network for {DEADLOCK_STALL_SECONDS:F0}s " +
                                 $"(stalled since {_lastTraversalChangeSimTime:F0}s, now {SimTime:F0}s). " +
                                 $"Terminating early instead of running to timeout.");
                FinaliseEpisode();
                return;
            }

            _failures.SetSimTime(SimTime);
            _flags.SetSimTime(SimTime);

            _failures.HarvestFailureFlags();
            _flags.HarvestMachineFlags();
            _flags.HarvestAGVFlags();
            _flags.HarvestStalledAGVs();
            _flags.HarvestAlmostDoneFlags(PreDispatchLeadTime);
            _flags.AssignAGVs();

            if (!IsWaitingForAction)
            {
                if (BaselineDrainMode)
                    DrainHeuristicDecisions();
                else
                {
                    var req = _decisions.FindNextDecision();
                    if (req != null)
                    {
                        CurrentDecision = req;
                        IsWaitingForAction = true;
                        OnDecisionRequired?.Invoke(CurrentDecision);
                    }
                }
            }

            // ── Tick Poisson arrival clock ─────────────────────────────────────
            // Runs regardless of IsWaitingForAction — arrivals are asynchronous events.
            TickPoissonClock();
            TickThroughputClock();
            // Guard against ending the episode while arrivals are still pending: if the
            // currently-spawned job pool drains to zero before the next scheduled Poisson
            // arrival lands, AreAllExited() alone would end the episode early and silently
            // drop the remaining arrivals — and since which rule races ahead fastest varies,
            // this made the realized workload differ across rules for the "same" seed.
            if (Jobs.AreAllExited() && AllArrivalsExhausted())
                FinaliseEpisode();
        }

        /// <summary>
        /// True once the system-wide sum of TrafficZone.TraversalCount has gone unchanged for
        /// DEADLOCK_STALL_SECONDS while jobs remain incomplete. A summed, network-wide signal
        /// is used (rather than watching any single zone or AGV) because heavy-but-resolving
        /// congestion routinely stalls individual zones for a while — only a true circular-wait
        /// deadlock stops EVERY zone in the network from ever admitting another AGV.
        /// </summary>
        private bool CheckForDeadlock()
        {
            if (Jobs.AreAllExited()) return false;

            int total = 0;
            foreach (var zone in trafficZoneManager.Zones) total += zone.TraversalCount;

            if (total != _lastZoneTraversalTotal)
            {
                _lastZoneTraversalTotal = total;
                _lastTraversalChangeSimTime = SimTime;
                return false;
            }

            return (SimTime - _lastTraversalChangeSimTime) > DEADLOCK_STALL_SECONDS;
        }

        /// <summary>
        /// True if no further Poisson arrivals can occur this episode: arrivals are disabled,
        /// or a finite cap has already been reached. False if arrivals are enabled and either
        /// uncapped or the cap hasn't been hit yet — in both cases more jobs may still spawn.
        /// </summary>
        private bool AllArrivalsExhausted()
        {
            bool arrivalsEnabled = StochasticEventManager.Instance?.DynamicArrivalsEnabled ?? false;
            if (!arrivalsEnabled) return true;

            int cap = currentConfig.Stochastic?.DynamicJobCap ?? 0;
            return cap != 0 && _dynamicJobsSpawned >= cap;
        }
        /// <summary>
        /// Drains all CURRENTLY-READY decisions this frame, applying the configured heuristic
        /// rule directly. Each Step() commits its routing/dispatch state before the next
        /// FindNextDecision() runs, so dependent decisions resolve in order. The loop terminates
        /// naturally when no ready decision remains — freshly-routed jobs are NOT re-pickable
        /// here because they must physically travel before becoming Queued/dispatchable, so the
        /// ready set is bounded by current floor state, not unbounded.
        ///
        /// This removes the engine-imposed one-decision-per-frame serialization that made the
        /// system appear decision-bound. It models a fast scheduler that clears ready work
        /// immediately — the correct baseline behaviour. Only runs when BaselineDrainMode is set.
        /// </summary>
        private void DrainHeuristicDecisions()
        {
            const int guard = 1_000_000;   // paranoia; real count bounded by ready events
            int n = 0;
            while (n++ < guard)
            {
                DecisionRequest req = _decisions.FindNextDecision();
                if (req == null) break;

                CurrentDecision = req;
                IsWaitingForAction = true;   // Step() expects to be clearing a pending decision

                int action = _baselineRuleIsRandom
                    ? UnityEngine.Random.Range(0, ActionCount)
                    : _baselineRuleIndex;

                Step(action);                // commits the decision, clears IsWaitingForAction
            }

            if (n >= guard)
                SimLogger.Error("[Orchestrator] DrainHeuristicDecisions hit guard — possible " +
                                "decision that doesn't change state. Investigate FindNextDecision.");
        }
        /// <summary>
        /// Closes every throughput window boundary that SimTime has crossed this frame. The while-loop
        /// handles a frame whose dt spans more than one window (e.g. high timescale), mirroring how the
        /// Poisson clock catches up.
        /// </summary>
        private void TickThroughputClock()
        {
            while (SimTime >= _nextThroughputBoundary)
            {
                double start = _nextThroughputBoundary - _throughputWindowLength;
                _tracker.CloseThroughputWindow(start, _nextThroughputBoundary, WorkInProgress());
                _nextThroughputBoundary += _throughputWindowLength;
            }
        }

        /// <summary>Jobs currently in the system (spawned but not yet Exited).</summary>
        private int WorkInProgress() => Jobs.JobCount - Jobs.CountInState(JobState.Exited);

        /// <summary>
        /// Checks whether the Poisson arrival clock has fired and, if so, injects a new
        /// dynamic job and schedules the next arrival. Called every frame from Update.
        /// </summary>
        private void TickPoissonClock()
        {
            int cap = currentConfig.Stochastic?.DynamicJobCap ?? 0;
            // Spawn ALL arrivals whose scheduled time has passed this frame, not just one.
            while (SimTime >= _nextArrivalSimTime)
            {
                if (cap != 0 && _dynamicJobsSpawned >= cap) { _nextArrivalSimTime = float.MaxValue; break; }

                FJSSPJobDefinition def = FJSSPJobGenerator.GenerateSingle(
                    _nextDynamicJobId++, currentConfig, cachedMachinesByType);
                def.ArrivalTime = (float)SimTime;   // note: see caveat below
                Jobs.AddDynamicJob(def, spawnVisuals: true);
                _dynamicJobsSpawned++;
                _lastDynamicArrivalSimTime = (float)SimTime;

                bool moreExpected = cap == 0 || _dynamicJobsSpawned < cap;
                _nextArrivalSimTime = moreExpected
                    ? _nextArrivalSimTime + StochasticEventManager.Instance.SampleInterArrivalTime()
                    : float.MaxValue;
            }
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
            LogRoutingDecision(chosenMachineId);
            JobData job = Jobs.Get(CurrentDecision.JobId);
            if (job == null) return;

            job.TargetMachineId = chosenMachineId;
            job.TransitionTo(JobState.WaitingForPickup, SimTime);

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
            LogDispatchDecision(chosenJobId);

            JobData job = Jobs.Get(chosenJobId);
            if (job == null || job.State != JobState.Queued || job.LocationMachineId != machineId) return;

            float duration = job.GetProcessingTime(machineId);
            job.OpProcStartTimes[job.CurrentOpIndex] = (float)SimTime;
            job.TransitionTo(JobState.Processing, SimTime);

            PhysicalMachine machine = layoutManager.GetMachine(machineId);
            machine.StartJob(chosenJobId, duration, job.Visual);

            _machineProcessingStartTime[machineId] = SimTime;

            _flags.RefreshMachineLabels(machineId);
            LastAppliedRule = _configuredRuleName;
        }

        /// <summary>
        /// Records a routing decision (job -> machine) to _decisionLog. Fires for every routing
        /// decision including the candidates.Length &lt;= 1 degenerate case DispatchingEngine
        /// short-circuits on, so the log can directly show how often the rule never actually ran.
        /// </summary>
        private void LogRoutingDecision(int chosenMachineId)
        {
            var req = CurrentDecision;
            int count = req.CandidateMachineIds?.Length ?? 0;
            int jobCandidateCount = req.JobCandidateIds?.Length ?? 0;
            _decisionLog.Add(new DecisionRecord
            {
                SimTime = SimTime,
                DecisionIndex = decisionCount,
                IsRouting = true,
                SubjectId = req.JobId,
                ChosenId = chosenMachineId,
                CandidateCount = count,
                IsDegenerate = count <= 1,
                CandidateIds = string.Join("|", req.CandidateMachineIds ?? Array.Empty<int>()),
                CandidateStatA = string.Join("|", req.CandidateJobTimes ?? Array.Empty<float>()),
                CandidateStatB = string.Join("|", req.CandidateQueueLengths ?? Array.Empty<float>()),
                JobCandidateCount = jobCandidateCount,
                IsJobSelectionDegenerate = jobCandidateCount <= 1,
                JobCandidateIds = string.Join("|", req.JobCandidateIds ?? Array.Empty<int>()),
            });
        }

        /// <summary>
        /// Records a dispatch decision (machine picks a queued job) to _decisionLog. Fires for
        /// every dispatch decision including the queue.Count &lt;= 1 degenerate case.
        /// </summary>
        private void LogDispatchDecision(int chosenJobId)
        {
            var req = CurrentDecision;
            int[] queuedIds = req.QueuedJobIds ?? Array.Empty<int>();
            int count = queuedIds.Length;
            var remainingWork = new float[count];
            var arrivalTimes = new float[count];
            for (int i = 0; i < count; i++)
            {
                JobData qJob = Jobs.Get(queuedIds[i]);
                remainingWork[i] = DispatchingEngine.GetRemainingWork(queuedIds[i], Jobs);
                arrivalTimes[i] = qJob?.ArrivalTime ?? -1f;
            }
            _decisionLog.Add(new DecisionRecord
            {
                SimTime = SimTime,
                DecisionIndex = decisionCount,
                IsRouting = false,
                SubjectId = req.MachineId,
                ChosenId = chosenJobId,
                CandidateCount = count,
                IsDegenerate = count <= 1,
                CandidateIds = string.Join("|", queuedIds),
                CandidateStatA = string.Join("|", req.QueuedDurations ?? Array.Empty<double>()),
                CandidateStatB = string.Join("|", remainingWork),
                CandidateStatC = string.Join("|", arrivalTimes),
            });
        }

        /// <summary>
        /// Finalizes the current episode by collecting telemetry, building the episode record,
        /// logging results, and firing the OnEpisodeFinished event. Includes AGV performance
        /// and segment congestion data collection.
        /// </summary>
        private void FinaliseEpisode()
        {
            episodeActive = false;
            // Close the trailing partial window so completions after the last full boundary
            // still land in throughput.csv. Skipped if SimTime sits exactly on a closed boundary.
            double lastBoundary = _nextThroughputBoundary - _throughputWindowLength;
            if (SimTime > lastBoundary)
                _tracker.CloseThroughputWindow(lastBoundary, SimTime, WorkInProgress());

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
            record.DynamicArrivals = _dynamicJobsSpawned;
            record.LastDynamicArrivalTime = _lastDynamicArrivalSimTime;
            if (_dynamicJobsSpawned > 0)
                record.JobCount = Jobs.JobCount;  // true total = initial + dynamic

            record.DecisionRecords = new List<DecisionRecord>(_decisionLog);

            // Configuration snapshot fields
            record.ParkingMethod = currentConfig.parkingMethod;
            record.PreDispatchingMethod = currentConfig.preDispatchingMethod;

            // Deadlock watchdog outcome — see CheckForDeadlock
            record.DeadlockDetected = _deadlockDetected;
            record.DeadlockSimTime = _deadlockDetected ? _deadlockSimTime : -1.0;

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
                        TravelTime = job.OperationTravelTimes[i],
                        QueueEntryTime = job.OpQueueEntryTimes[i],
                        ProcStartTime = job.OpProcStartTimes[i],
                        ProcEndTime = job.OpProcEndTimes[i],
                    });
                }

                bool completed = job.State == JobState.Exited;

                // Work content: sum of realized proc durations for completed ops; for ops never
                // finished (censored jobs), fall back to that op's mean estimate across eligible machines.
                float workContent = 0f;
                for (int i = 0; i < job.TotalOperations; i++)
                {
                    float realized = (job.OpProcStartTimes[i] >= 0 && job.OpProcEndTimes[i] >= 0)
                        ? job.OpProcEndTimes[i] - job.OpProcStartTimes[i] : -1f;
                    if (realized >= 0)
                    {
                        workContent += realized;
                    }
                    else
                    {
                        var eligible = job.EligibleMachinesPerOp[i];
                        workContent += eligible.Count > 0 ? eligible.Values.Average() : 0f;
                    }
                }

                record.JobCompletionRecords.Add(new JobCompletionRecord
                {
                    JobId = job.JobId,
                    IsDynamic = isDynamic,
                    Completed = completed,
                    ArrivalTime = job.ArrivalTime,
                    ExitTime = job.ExitTime,
                    TotalOperations = job.TotalOperations,
                    CompletedOps = job.CompletedOps,
                    WorkContent = workContent,
                    TimeNeedsRouting = job.TimeNeedsRouting,
                    TimeWaitingPickup = job.TimeWaitingPickup,
                    TimeInTransit = job.TimeInTransit,
                    TimeQueued = job.TimeQueued,
                    TimeProcessingState = job.TimeProcessing,
                });
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