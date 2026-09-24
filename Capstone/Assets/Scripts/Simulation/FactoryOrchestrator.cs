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

        /// <summary>
        /// When true, RL decisions are drained within a single FixedUpdate tick by manually
        /// forcing one <see cref="Academy.EnvironmentStep"/> call per ready decision, instead
        /// of relying on ML-Agents' default one-step-per-FixedUpdate automatic cadence. This
        /// is the neural-policy analogue of <see cref="BaselineDrainMode"/> -- it removes the
        /// same engine-imposed decision-per-tick throttle, but for a live agent instead of a
        /// fixed heuristic, so each drained decision pays a real policy round-trip (forward
        /// pass, and a full RPC hop while training) rather than an instant lookup.
        ///
        /// Toggle this off if it destabilizes training/inference or blows the per-tick time
        /// budget -- it's a config flag specifically so it can be A/B'd without a code change.
        /// Also settable via the "-rldecisiondrain" CLI flag (read in Awake, since
        /// HeadlessBatchRunner -- and its own CLI parsing -- disables itself whenever the
        /// ML-Agents communicator is on, i.e. exactly when this flag matters).
        /// Mutually exclusive with BaselineDrainMode; if both are set, BaselineDrainMode wins.
        /// </summary>
        public bool RLDecisionDrainMode = false;

        /// <summary>Last value applied to Academy.AutomaticSteppingEnabled, so
        /// SyncAcademyManualStepping only touches it when RLDecisionDrainMode actually changes.</summary>
        private bool? _lastSyncedAutomaticStepping;

        private int _baselineRuleIndex;
        private bool _baselineRuleIsRandom;

        // ── Per-episode wall-clock profiling (see LogEpisodeTiming) ─────────────
        private int _episodeIndex;
        private double _episodeWallStart;
        private int _episodeTicks;
        private int _episodeEnvSteps;
        private int _episodeGcStart;
        private readonly System.Diagnostics.Stopwatch _envStepWatch = new System.Diagnostics.Stopwatch();

        // ── Per-episode instance seed from EpisodeSeedChannel; -1 when none was queued ─────
        private int _episodeSeed = -1;
        private int _episodeSeedIndex = -1;

        // ── Optional decision_log.csv in RL mode (-decisionlogdir <dir>); null when off ─────
        private string _decisionLogDir;
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
        /// Rebuild the factory (floor, traffic zones, AGV fleet) at the start of every episode, as
        /// HeadlessBatchRunner does before each run. Without it, consecutive episodes share AGV
        /// positions/state and machine state, so the same instance plays out differently depending
        /// on the previous episode. Forced on when an ML-Agents communicator is connected.
        /// SpawnFactory re-seeds UnityEngine.Random with the config seed, so episodes without a
        /// Python-queued seed (EpisodeSeedChannel) all replay that seed's instance.
        /// </summary>
        public bool RespawnFactoryEachEpisode = false;

        /// <summary>
        /// Pre-built job definitions used for benchmark scenarios. Set before SpawnFactory.
        /// </summary>
        private FJSSPJobDefinition[] prebuiltJobs;

        /// <summary>
        /// Job builder for a scripted scenario sent by Python (EpisodeConfigChannel). Unlike
        /// prebuiltJobs, which applies to a single episode, this re-runs every episode until
        /// Python sends a different config, so training can replay the scenario.
        /// </summary>
        private Func<Dictionary<MachineType, List<int>>, FJSSPJobDefinition[]> _scenarioJobBuilder;

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
        /// Elapsed simulation time, accumulated once per FixedUpdate tick by a constant
        /// Time.fixedDeltaTime instead of read from Time.time. Time.time is real wall-clock
        /// time scaled by Time.timeScale — under headless batch runs (multiple rule processes
        /// competing for CPU, see run_generated.sh) its per-frame granularity is at the mercy
        /// of OS scheduling and isn't reproducible run-to-run even for an identical seed.
        /// Accumulating a fixed per-tick increment instead makes SimTime a pure function of
        /// tick count, independent of how long each tick actually took in wall-clock terms.
        /// </summary>
        private double _simTime;

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
        public double SimTime => _simTime;

        /// <summary>
        /// Cumulative utilization (busy / operational time so far) of a machine as of now — the same
        /// signal the MMUR rules route on. 0 before the episode's decision coordinator exists.
        /// </summary>
        public float MachineUtilization(int machineId) =>
            _decisions != null ? _decisions.MachineUtilization(machineId, _simTime) : 0f;

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

        // ── Scripted arrivals (jobs with an explicit ArrivalTime > 0 in the initial batch,
        //    from a hand-crafted scenario or a jittered generated batch) ───────────────────

        /// <summary>
        /// Jobs from the initial batch (prebuilt or generated) with ArrivalTime > 0, held
        /// back from JobStore.Initialize and injected individually as SimTime reaches each
        /// one's arrival time. Sorted ascending by ArrivalTime; TickScriptedArrivals only
        /// ever needs to look at the front of the list.
        /// </summary>
        private readonly List<FJSSPJobDefinition> _pendingScriptedArrivals = new List<FJSSPJobDefinition>();

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
        /// Gridlock watchdog: if the plant makes no PRODUCTIVE progress for this many consecutive
        /// sim-seconds while jobs remain incomplete, the episode is declared deadlocked and
        /// terminated instead of running to MAX_EPISODE_SIM_SECONDS. Productive progress is an AGV
        /// delivery or a machine finishing an operation (FlagHarvester.ProgressCount).
        ///
        /// This deliberately does NOT watch zone entries (TrafficZone.TraversalCount). Stall
        /// recovery (AGVController.RetreatFromStall) re-reserves the previous zone every
        /// zoneStallTimeoutSeconds, and each successful reservation bumps that counter, so a fully
        /// gridlocked fleet kept the old watchdog fed forever: 0 of 120 gridlocked runs in the
        /// 2026-09-19 sweep ever tripped it, 61 of them sat idle to the 20,000 s cap.
        /// </summary>
        private const double DEADLOCK_STALL_SECONDS = 3_000.0;

        private int _lastProgressCount = -1;
        private double _lastProgressChangeSimTime;
        private bool _deadlockDetected;
        private double _deadlockSimTime = -1.0;
        private double _firstStallSimTime = -1.0;
        private AGVCollisionMonitor _collisions = new AGVCollisionMonitor();

        /// <summary>
        /// Set when the episode ends because it hit its deliberate steady-state time cap
        /// (Stochastic.EpisodeDurationSeconds) with jobs still in flight — a cutoff, not a real
        /// failure. Reported to Python via RewardMetrics so RL training bootstraps the value
        /// function past it instead of treating it like a real terminal.
        /// </summary>
        private bool _truncatedByTimeLimit;

        /// <summary>
        /// SimTime at which the warm-up window (Stochastic.WarmupSeconds) ends and the RL agent
        /// takes over. 0 (the default) means no warm-up -- the agent controls from t=0, unchanged
        /// from every episode before this field existed.
        /// </summary>
        private double _warmupEndSimTime;

        /// <summary>
        /// True while SimTime is still inside the warm-up window. Decisions in this window are
        /// resolved by <see cref="DrainHeuristicDecisions"/> exactly like BaselineDrainMode --
        /// the fixed rule already resolved into _baselineRuleIndex/_baselineRuleIsRandom every
        /// episode (see StartEpisode) drives the floor to a realistic mid-scenario state (busy
        /// machines, in-transit AGVs, populated queues) with no Python round-trips, before the
        /// agent's first real decision hands off from that state.
        /// </summary>
        private bool InWarmup => SimTime < _warmupEndSimTime;

        /// <summary>
        /// True from StartEpisode() until the tick FixedUpdate observes InWarmup go false --
        /// a one-shot latch so the decisionCount/_decisionLog reset at hand-off fires exactly
        /// once per episode instead of every tick after warm-up ends.
        /// </summary>
        private bool _warmupActive;

        /// <summary>
        /// Singleton initialization. Destroys duplicate instances if one already exists.
        /// </summary>
        private void Awake()
        {
            if (Instance != null) { Destroy(this); return; }
            Instance = this;

            if (GetCLIArg("-rldecisiondrain") != null)
            {
                RLDecisionDrainMode = true;
                SimLogger.Low("[Orchestrator] RLDecisionDrainMode ENABLED via CLI flag.");
            }

            // HeadlessBatchRunner (which normally writes decision logs) disables itself under
            // ML-Agents, so RL evaluation opts in here to get the same per-decision CSV.
            _decisionLogDir = GetCLIArg("-decisionlogdir");
            if (!string.IsNullOrEmpty(_decisionLogDir))
            {
                ResultsLogger.SetSubdirectory(_decisionLogDir);
                SimLogger.Low($"[Orchestrator] Writing decision_log.csv to {ResultsLogger.OutputDirectory}.");
            }
            if (RLDecisionDrainMode && BaselineDrainMode)
                SimLogger.LogWarning("[Orchestrator] Both RLDecisionDrainMode and BaselineDrainMode " +
                                      "are set -- BaselineDrainMode takes priority and RL drain will " +
                                      "be ignored this run.");
        }

        /// <summary>
        /// Minimal CLI flag lookup, mirroring HeadlessBatchRunner.GetCLIArg. Duplicated rather
        /// than shared because HeadlessBatchRunner disables itself whenever the ML-Agents
        /// communicator is on -- exactly the case RLDecisionDrainMode needs to be read in.
        /// </summary>
        private static string GetCLIArg(string key)
        {
            string[] args = Environment.GetCommandLineArgs();
            for (int i = 0; i < args.Length - 1; i++)
                if (args[i] == key)
                    return args[i + 1];
            return null;
        }

        /// <summary>
        /// Called on startup. Arms the scheduling agent if AutoStartOnPlay is enabled.
        /// </summary>
        /// <remarks>
        /// A live ML-Agents communicator (mlagents-learn, or any external trainer) means
        /// there's no human present to click "Start Sim" -- HeadlessBatchRunner, the other
        /// thing that can arm an episode, disables itself in exactly that situation (see
        /// its own IsCommunicatorOn check). Without this, episodes never start under a real
        /// trainer: Academy still ticks and reports steps, but FixedUpdate's episodeActive
        /// gate means nothing in the simulation ever runs and no episode ever completes.
        /// So a connected communicator forces AutoStartOnPlay regardless of the Inspector
        /// checkbox, which also keeps SchedulingAgent looping episodes for the whole run
        /// (see the AutoStartOnPlay checks in OnEpisodeBegin/HandleEpisodeFinished).
        /// </remarks>
        private void Start()
        {
            if (Academy.Instance.IsCommunicatorOn && !RespawnFactoryEachEpisode)
            {
                RespawnFactoryEachEpisode = true;
                SimLogger.Low("[Orchestrator] ML-Agents communicator detected — rebuilding the factory " +
                              "at every episode start so episodes are independent.");
            }

            if (Academy.Instance.IsCommunicatorOn && !AutoStartOnPlay)
            {
                AutoStartOnPlay = true;
                SimLogger.Low("[Orchestrator] ML-Agents communicator detected — forcing " +
                              "AutoStartOnPlay so episodes start and loop without manual arming.");
            }

            if (AutoStartOnPlay && agent != null)
                agent.IsArmed = true;
        }

        /// <summary>
        /// The single point where command-line overrides (ConfigOverrides) are applied to a config,
        /// so no batch / scenario / Python entry path can miss one. Mutates and returns the config.
        /// </summary>
        private static FJSSPConfig ApplyConfigOverrides(FJSSPConfig config)
        {
            if (config != null && ConfigOverrides.ReservationProtocol != null)
                config.reservationProtocol = ConfigOverrides.ReservationProtocol;
            if (config != null && ConfigOverrides.ParkingMethod != null)
                config.parkingMethod = ConfigOverrides.ParkingMethod;
            if (config != null && ConfigOverrides.Layout != null)
                config.Layout = LayoutSpec.FromPreset(ConfigOverrides.Layout);
            if (config != null && ConfigOverrides.IoDocks != null)
                config.ioDocks = ConfigOverrides.IoDocks;
            return config;
        }

        /// <summary>
        /// Loads a simulation configuration and initializes the stochastic event manager.
        /// Resets factory readiness so SpawnFactory must be called before starting an episode.
        /// </summary>
        /// <param name="config">The FJSSP configuration to load.</param>
        public void LoadConfig(FJSSPConfig config)
        {
            currentConfig = ApplyConfigOverrides(config);
            _scenarioJobBuilder = null;
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
            _episodeIndex++;
            _episodeWallStart = Time.realtimeSinceStartupAsDouble;
            _episodeTicks = 0;
            _episodeEnvSteps = 0;
            _envStepWatch.Reset();
            _episodeGcStart = GC.CollectionCount(0);

            var pythonScenario = EpisodeConfigChannel.Instance?.ConsumeScenario();
            if (pythonScenario != null)
            {
                var (scenarioConfig, buildJobs) =
                    ScenarioLoader.LoadDeferredFromJson(pythonScenario.Json, pythonScenario.Name);
                if (scenarioConfig != null)
                {
                    currentConfig = ApplyConfigOverrides(scenarioConfig);
                    _scenarioJobBuilder = buildJobs;
                    IsFactoryReady = false;
                    SimLogger.Low($"[Bridge] Applied Python scenario: {scenarioConfig.Name} " +
                                  $"({scenarioConfig.JobCount} jobs, {scenarioConfig.AGVCount} AGVs)");
                }
                else
                {
                    SimLogger.LogError($"[Bridge] Python scenario '{pythonScenario.Name}' failed to load; " +
                                        "keeping the previous config.");
                }
            }

            var pythonConfig = EpisodeConfigChannel.Instance?.ConsumeConfig();
            if (pythonConfig != null)
            {
                currentConfig = ApplyConfigOverrides(pythonConfig);
                _scenarioJobBuilder = null;
                IsFactoryReady = false;
                SimLogger.Low($"[Bridge] Applied Python config: {currentConfig.Name}");
            }

            currentConfig ??= DefaultConfigFactory.BuildDefault();

            if (RespawnFactoryEachEpisode)
                IsFactoryReady = false;

            if (!IsFactoryReady)
                SpawnFactory();

            trafficZoneManager.ResetEpisodeStats();
            foreach (var agv in agvPool.AllAGVs)
                agv.ResetEpisodeStats();

            agent.SetHeuristicRule(currentConfig.dispatchingRule);
            _configuredRuleName = currentConfig.dispatchingRule.ToString();

            _baselineRuleIsRandom = currentConfig.dispatchingRule == DispatchingRule.Random;
            _baselineRuleIndex = GetRuleIndex(currentConfig.dispatchingRule);

            // A seed queued by Python makes this episode's instance (generated jobs, and the
            // stochastic streams below) a pure function of that seed. Without one, generation
            // continues the UnityEngine.Random stream seeded once in SpawnFactory, so the instance
            // depends on everything drawn during earlier episodes.
            _episodeSeed = -1;
            _episodeSeedIndex = -1;
            if (EpisodeSeedChannel.Instance != null &&
                EpisodeSeedChannel.Instance.TryDequeue(out _episodeSeed, out _episodeSeedIndex))
            {
                UnityEngine.Random.InitState(_episodeSeed);
            }

            FJSSPJobDefinition[] jobDefs;
            if (prebuiltJobs != null)
            {
                jobDefs = prebuiltJobs;
                prebuiltJobs = null;
                SimLogger.Low("[Orchestrator] Using prebuilt benchmark jobs");
            }
            else if (_scenarioJobBuilder != null)
            {
                jobDefs = _scenarioJobBuilder(cachedMachinesByType);
            }
            else
            {
                jobDefs = FJSSPJobGenerator.Generate(currentConfig, cachedMachinesByType);
            }

            // JobStore.Initialize marks every job it's given as immediately routable — it
            // treats ArrivalTime as metadata, not a gate. That's fine for every existing
            // caller (Brandimarte instances and the un-jittered generated batch both use
            // ArrivalTime=0 throughout), but a hand-crafted scenario (or a generated batch
            // with InitialArrivalSpread>0) can specify jobs due later in the episode, which
            // Initialize would silently make available at t=0 anyway. Split the batch here:
            // anything due now loads normally; anything due later is held and injected via
            // AddDynamicJob (the same call the Poisson clock already uses correctly) once
            // SimTime reaches its arrival time.
            var immediateJobs = new List<FJSSPJobDefinition>();
            _pendingScriptedArrivals.Clear();
            foreach (var def in jobDefs)
            {
                if (def.ArrivalTime > 0f) _pendingScriptedArrivals.Add(def);
                else immediateJobs.Add(def);
            }
            _pendingScriptedArrivals.Sort((a, b) => a.ArrivalTime.CompareTo(b.ArrivalTime));

            Jobs.Initialize(immediateJobs, spawnVisuals: true);

            if (currentConfig.Stochastic != null && currentConfig.Stochastic.AnyEnabled)
            {
                StochasticEventManager.Instance?.Initialize(
                    currentConfig, _episodeSeed >= 0 ? _episodeSeed : currentConfig.Seed);
                foreach (var machine in layoutManager.Machines)
                    machine.InitializeStochastic();
            }
            _throughputWindowLength = currentConfig.ThroughputTimingWindow <= 0f ? 60f : currentConfig.ThroughputTimingWindow;
            _nextThroughputBoundary = _throughputWindowLength;   // first window closes at t = windowLength
            _tracker.Reset();
            _machineProcessingStartTime.Clear();
            foreach (var machine in layoutManager.Machines) machine.ResetHandoffStats();

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
                tracker: _tracker,
                machineProcessingStartTime: _machineProcessingStartTime,
                // -1 in RL-agent/interactive mode (no fixed rule yet at decision-assembly time —
                // job-priority pre-selection wouldn't make sense before the agent has acted).
                // Re-resolves "random" per call, matching DrainHeuristicDecisions' own resolution
                // (line ~615) -- independent draws for job-selection vs. the eventual machine/job
                // Execute-time choice, an accepted minor inconsistency specific to the Random PDR.
                getBaselineActionIndex: () => (BaselineDrainMode || InWarmup)
                    ? (_baselineRuleIsRandom ? UnityEngine.Random.Range(0, DispatchingEngine.ActionCount) : _baselineRuleIndex)
                    : -1
            );

            episodeActive = true;
            decisionCount = 0;
            _decisionLog.Clear();
            IsWaitingForAction = false;
            _simTime = 0.0;
            _warmupEndSimTime = currentConfig.Stochastic?.WarmupSeconds ?? 0.0;
            _warmupActive = _warmupEndSimTime > 0.0;
            if (_warmupActive)
                SimLogger.Low($"[Orchestrator] Warm-up armed: {_warmupEndSimTime:F0}s under " +
                              $"{currentConfig.dispatchingRule} before the agent takes over.");

            // ── Arm deadlock watchdog ────────────────────────────────────────────
            _lastProgressCount = -1;
            _lastProgressChangeSimTime = 0.0;
            _deadlockDetected = false;
            _deadlockSimTime = -1.0;
            _firstStallSimTime = -1.0;
            _collisions = new AGVCollisionMonitor();
            _truncatedByTimeLimit = false;

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
                          $"stochastic={StochasticEventManager.Instance?.IsActive}, " +
                          $"seed={_episodeSeed} (queue index {_episodeSeedIndex})");
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
        /// Called every fixed-timestep tick. Advances the deterministic simulation clock,
        /// processes simulation flags, checks for new decisions, and terminates the episode
        /// when all jobs have exited or the time limit is reached.
        ///
        /// Runs on FixedUpdate (constant Time.fixedDeltaTime per call) rather than Update
        /// (variable, real-wall-clock-dependent per call) so the sequence and timing of
        /// simulated events is a pure function of tick count — reproducible for a given
        /// seed regardless of real-world CPU scheduling/contention during the tick.
        /// </summary>
        private void FixedUpdate()
        {
            if (!episodeActive) return;

            SyncAcademyManualStepping();

            _simTime += Time.fixedDeltaTime;
            _episodeTicks++;

            if (SimTime > MAX_EPISODE_SIM_SECONDS)
            {
                SimLogger.Low($"[Orchestrator] Episode timeout at {SimTime:F0}s — terminating.");
                FinaliseEpisode();
                return;
            }

            // Relative to _warmupEndSimTime (0 when WarmupSeconds is unset, so this is exactly
            // "SimTime > fixedDuration" for every episode before WarmupSeconds existed) --
            // EpisodeDurationSeconds caps the agent's counted window, not absolute SimTime.
            // Without the offset, a warm-up cutoff past the cap (e.g. an 8555s random-phase
            // offset with a 1800s cap) truncates the episode while still inside warm-up: zero
            // decisions ever reach the agent, Academy.EnvironmentStep() never fires, and Python's
            // step() blocks with no response until its own RPC timeout kills the run.
            double fixedDuration = currentConfig?.Stochastic?.EpisodeDurationSeconds ?? 0.0;
            if (fixedDuration > 0.0 && SimTime > _warmupEndSimTime + fixedDuration)
            {
                SimLogger.Low($"[Orchestrator] Fixed episode duration reached at {SimTime:F0}s — " +
                              $"terminating (steady-state mode; in-flight jobs recorded as censored).");
                _truncatedByTimeLimit = true;
                FinaliseEpisode();
                return;
            }

            CheckFirstStall();
            _collisions.Tick(agvPool.AllAGVs, SimTime, Time.fixedDeltaTime,
                             pos => trafficZoneManager.GetZoneAtPosition(pos)?.Name);

            if (CheckForDeadlock())
            {
                _deadlockDetected = true;
                _deadlockSimTime = SimTime;
                SimLogger.Error($"[Orchestrator] Deadlock detected — no delivery or operation completion " +
                                 $"for {DEADLOCK_STALL_SECONDS:F0}s " +
                                 $"(last progress at {_lastProgressChangeSimTime:F0}s, now {SimTime:F0}s). " +
                                 $"Terminating early instead of running to timeout.");
                DumpTrafficSnapshot("watchdog fired");
                FinaliseEpisode();
                return;
            }

            _failures.SetSimTime(SimTime);
            _flags.SetSimTime(SimTime);

            _failures.HarvestFailureFlags();
            _flags.HarvestMachineFlags();
            _flags.HarvestAGVFlags();
            _flags.HarvestStalledAGVs();
            _flags.ReleaseOrphanedPreDispatches();
            _flags.HarvestAlmostDoneFlags(PreDispatchLeadTime);
            _flags.AssignAGVs();

            if (_warmupActive && !InWarmup)
            {
                // Warm-up just ended this tick -- hand off to the RL agent with a clean slate.
                // decisionCount/_decisionLog are episode-scoped counters consumed by CSV/decision-
                // log output; without this reset they'd carry the warm-up's fixed-rule decisions,
                // which aren't the agent's and would pollute both.
                _warmupActive = false;
                decisionCount = 0;
                _decisionLog.Clear();
                SimLogger.Low($"[Orchestrator] Warm-up complete at {SimTime:F0}s — RL agent now in " +
                              "control (decisionCount and decision log reset).");
            }

            if (!IsWaitingForAction)
            {
                if (BaselineDrainMode || InWarmup)
                    DrainHeuristicDecisions();
                else if (RLDecisionDrainMode)
                    DrainRLDecisions();
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
            TickScriptedArrivals();
            TickThroughputClock();
            // Guard against ending the episode while arrivals are still pending: if the
            // currently-spawned job pool drains to zero before the next scheduled Poisson
            // arrival lands, AreAllExited() alone would end the episode early and silently
            // drop the remaining arrivals — and since which rule races ahead fastest varies,
            // this made the realized workload differ across rules for the "same" seed. Same
            // reasoning applies to scripted arrivals still waiting in _pendingScriptedArrivals.
            if (Jobs.AreAllExited() && AllArrivalsExhausted() && _pendingScriptedArrivals.Count == 0)
                FinaliseEpisode();
        }

        /// <summary>
        /// Injects any scripted-arrival job (explicit ArrivalTime > 0 from the initial batch)
        /// whose time has come. Mirrors TickPoissonClock's structure but walks a pre-sorted
        /// list instead of sampling from a distribution — see _pendingScriptedArrivals.
        /// </summary>
        private void TickScriptedArrivals()
        {
            while (_pendingScriptedArrivals.Count > 0
                   && SimTime >= _pendingScriptedArrivals[0].ArrivalTime)
            {
                FJSSPJobDefinition def = _pendingScriptedArrivals[0];
                _pendingScriptedArrivals.RemoveAt(0);
                Jobs.AddDynamicJob(def, spawnVisuals: true);
            }
        }

        /// <summary>
        /// True once FlagHarvester.ProgressCount (deliveries + operation completions) has gone
        /// unchanged for DEADLOCK_STALL_SECONDS while jobs remain incomplete. A network-wide
        /// productive-output signal is used (rather than watching any zone or AGV) because
        /// heavy-but-resolving congestion routinely stalls individual zones for a while, and
        /// because movement counters are fed by stall recovery itself — see DEADLOCK_STALL_SECONDS.
        /// While every known job has exited (arrivals still pending) the clock is held at "now",
        /// so a long arrival gap cannot trip the watchdog the moment the next job appears.
        /// </summary>
        private bool CheckForDeadlock()
        {
            if (_flags == null) return false;

            if (Jobs.AreAllExited())
            {
                _lastProgressChangeSimTime = SimTime;
                return false;
            }

            int progress = _flags.ProgressCount;
            if (progress != _lastProgressCount)
            {
                _lastProgressCount = progress;
                _lastProgressChangeSimTime = SimTime;
                return false;
            }

            return (SimTime - _lastProgressChangeSimTime) > DEADLOCK_STALL_SECONDS;
        }

        /// <summary>
        /// Records the sim-time of the first AGV zone-stall recovery (gridlock onset) and dumps a
        /// snapshot of who holds what at that moment, which is the cleanest picture of the jam:
        /// by the time the watchdog fires the fleet has been shuffled by hundreds of recoveries.
        /// </summary>
        private void CheckFirstStall()
        {
            if (_firstStallSimTime >= 0.0) return;
            foreach (var agv in agvPool.AllAGVs)
            {
                if (agv.StallRecoveryCount <= 0) continue;
                _firstStallSimTime = SimTime;
                DumpTrafficSnapshot("first stall recovery");
                return;
            }
        }

        /// <summary>
        /// Logs every AGV's state and every occupied traffic zone. Grep the sim log for
        /// "[GridlockSnapshot]". Pre-dispatched AGVs are cross-checked against their job so an
        /// orphaned claim (job no longer names the AGV) is called out explicitly.
        /// </summary>
        private void DumpTrafficSnapshot(string reason)
        {
            var sb = new System.Text.StringBuilder();
            sb.AppendLine($"[GridlockSnapshot] {reason} at sim={SimTime:F0}s seed={_episodeSeed}");

            foreach (var agv in agvPool.AllAGVs)
            {
                sb.Append("  ").Append(agv.DebugSummary());
                if (agv.IsPreDispatched)
                {
                    JobData pj = Jobs.Get(agv.PreDispatchedJobId);
                    if (pj == null) sb.Append(" [preJob missing]");
                    else
                    {
                        PhysicalMachine src = layoutManager.GetMachine(pj.LocationMachineId >= 0 ? pj.LocationMachineId : pj.TargetMachineId);
                        sb.Append($" [preJob state={pj.State} jobClaimsAgv={pj.PreDispatchedAgvId}" +
                                  $"{(pj.PreDispatchedAgvId != agv.AgvId ? " ORPHANED" : "")}" +
                                  $" srcMachine={(src != null ? src.MachineId : -1)}" +
                                  $" srcHealth={(src != null ? src.HealthState.ToString() : "?")}]");
                    }
                }
                sb.AppendLine();
            }

            sb.Append("  occupied zones:");
            foreach (var zone in trafficZoneManager.Zones)
                if (!zone.IsEmpty)
                    sb.Append($" {zone.Name}[{string.Join(",", zone.OccupantAgvIds)}]");
            sb.AppendLine();

            sb.Append("  non-operational machines:");
            foreach (var m in layoutManager.Machines)
                if (m.HealthState != MachineHealthState.Operational)
                    sb.Append($" M{m.MachineId}={m.HealthState}");

            SimLogger.Error(sb.ToString());
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
        /// Keeps Academy.AutomaticSteppingEnabled in sync with RLDecisionDrainMode. ML-Agents
        /// normally steps once per FixedUpdate via a hidden stepper object; DrainRLDecisions
        /// needs to be the only thing calling Academy.EnvironmentStep() while it's draining a
        /// tick's ready decisions, so automatic stepping must be off for that entire duration
        /// -- not just during the drain call -- or the hidden stepper could fire mid-drain and
        /// double-resolve a decision. Only touches Academy when the desired state actually
        /// changes, so this is cheap to call unconditionally every tick.
        /// </summary>
        private void SyncAcademyManualStepping()
        {
            bool desiredAutomatic = !RLDecisionDrainMode || BaselineDrainMode;
            if (_lastSyncedAutomaticStepping == desiredAutomatic) return;

            Academy.Instance.AutomaticSteppingEnabled = desiredAutomatic;
            _lastSyncedAutomaticStepping = desiredAutomatic;
            SimLogger.Low($"[Orchestrator] Academy.AutomaticSteppingEnabled -> {desiredAutomatic} " +
                          $"(RLDecisionDrainMode={RLDecisionDrainMode}).");
        }

        /// <summary>
        /// Drains all CURRENTLY-READY decisions this tick through the live ML-Agents policy,
        /// manually forcing one Academy.EnvironmentStep() per decision instead of relying on
        /// the engine's automatic one-step-per-FixedUpdate cadence. This is the RL-path
        /// analogue of DrainHeuristicDecisions: same ready-set-bounded loop, but each
        /// iteration pays a real policy round-trip (CollectObservations -> policy ->
        /// OnActionReceived, synchronously, via EnvironmentStep) instead of an instant
        /// heuristic lookup. Requires automatic stepping to be off (see
        /// SyncAcademyManualStepping) so this loop is the only thing driving Academy.
        /// </summary>
        private void DrainRLDecisions()
        {
            const int guard = 1_000_000;   // paranoia; real count bounded by ready events
            int n = 0;
            while (n++ < guard)
            {
                DecisionRequest req = _decisions.FindNextDecision();
                if (req == null) break;

                CurrentDecision = req;
                IsWaitingForAction = true;   // Step() (via OnActionReceived) clears this
                OnDecisionRequired?.Invoke(CurrentDecision);   // flags the agent's pending request

                _envStepWatch.Start();
                Academy.Instance.EnvironmentStep();   // synchronous: send obs -> policy -> act
                _envStepWatch.Stop();
                _episodeEnvSteps++;

                if (IsWaitingForAction)
                {
                    // EnvironmentStep() returned without resolving the pending decision (e.g.
                    // no agent listening, or the communicator stalled). Fall back to the
                    // standard one-decision-per-tick path instead of spinning forever --
                    // the already-pending request is left in place and will resolve normally
                    // once automatic stepping comes back on next tick.
                    SimLogger.Error("[Orchestrator] DrainRLDecisions: EnvironmentStep() did not " +
                                     "resolve the pending decision. Disabling RLDecisionDrainMode " +
                                     "for the rest of this run.");
                    RLDecisionDrainMode = false;
                    break;
                }
            }

            if (n >= guard)
                SimLogger.Error("[Orchestrator] DrainRLDecisions hit guard — possible decision " +
                                "that doesn't change state. Investigate FindNextDecision.");
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
        /// arrival event's jobs (a single job, or a burst cluster when BurstArrivalsEnabled)
        /// and schedules the next arrival. Called every frame from Update.
        /// </summary>
        private void TickPoissonClock()
        {
            int cap = currentConfig.Stochastic?.DynamicJobCap ?? 0;
            // Spawn ALL arrival EVENTS whose scheduled time has passed this frame, not just one.
            while (SimTime >= _nextArrivalSimTime)
            {
                if (cap != 0 && _dynamicJobsSpawned >= cap) { _nextArrivalSimTime = float.MaxValue; break; }

                // A single event injects a burst of jobs at once (size 1 unless configured
                // otherwise) — all sharing this event's arrival timestamp.
                int burstSize = StochasticEventManager.Instance.SampleBurstSize();
                for (int i = 0; i < burstSize; i++)
                {
                    if (cap != 0 && _dynamicJobsSpawned >= cap) break;

                    FJSSPJobDefinition def = FJSSPJobGenerator.GenerateSingle(
                        _nextDynamicJobId++, currentConfig, cachedMachinesByType);
                    // Stamp the exact scheduled Poisson instant, not SimTime as observed by
                    // this tick — SimTime only overshoots _nextArrivalSimTime by up to one
                    // fixed tick, and using the scheduled time keeps ArrivalTime a pure
                    // function of the seeded arrival stream (deterministic, tick-count-
                    // independent) instead of tick-granularity-dependent.
                    def.ArrivalTime = _nextArrivalSimTime;
                    Jobs.AddDynamicJob(def, spawnVisuals: true);
                    _dynamicJobsSpawned++;
                    _lastDynamicArrivalSimTime = _nextArrivalSimTime;
                }

                bool moreExpected = cap == 0 || _dynamicJobsSpawned < cap;
                _nextArrivalSimTime = moreExpected
                    ? _nextArrivalSimTime + StochasticEventManager.Instance.SampleInterArrivalTime()
                    : float.MaxValue;
            }
        }

        /// <summary>
        /// Executes a single simulation step given an action index from the agent.
        /// Applies the dispatch or routing decision and returns the result. No reward is computed
        /// here — see <see cref="WriteRewardMetrics"/>.
        /// </summary>
        /// <param name="actionIndex">The index of the action to execute.</param>
        /// <returns>A StepResult containing the simulation status.</returns>
        public StepResult Step(int actionIndex)
        {
            IsWaitingForAction = false;

            if (CurrentDecision.Type == DecisionType.Routing)
                ExecuteRoutingDecision(actionIndex);
            else if (CurrentDecision.Type == DecisionType.Dispatch)
                ExecuteDispatchDecision(actionIndex);

            return new StepResult { Done = false, CurrentMakespan = SimTime };
        }

        /// <summary>
        /// Executes a routing decision by assigning the specified job to the selected machine
        /// and preparing AGV pickup if applicable.
        /// </summary>
        /// <param name="actionIndex">The action index encoding the machine selection.</param>
        private void ExecuteRoutingDecision(int actionIndex)
        {
            // Job half of the rule. When no rule was known at assembly time (agent, or heuristic via
            // Step), JobId is only the oldest routable job; apply the action's job-priority rule now so
            // routing uses both halves of the rule, as heuristic-drain mode already did. Before this, the
            // routed job was always the oldest one in these modes, for PDR sweeps and the agent alike.
            DecisionRequest req = CurrentDecision;
            if (!req.JobSelectedByRule && req.JobCandidateIds != null && req.JobCandidateIds.Length > 1)
            {
                int jobId = DispatchingEngine.SelectRoutingJob(actionIndex, new List<int>(req.JobCandidateIds), Jobs, SimTime);
                JobData picked = jobId >= 0 ? Jobs.Get(jobId) : null;
                if (picked != null && jobId != req.JobId && picked.State == JobState.NeedsRouting)
                {
                    DecisionRequest rebuilt = _decisions.BuildRoutingDecision(picked, countsAsNewDecision: false);
                    if (rebuilt.CandidateMachineIds.Length > 0)
                    {
                        rebuilt.JobCandidateIds = req.JobCandidateIds;
                        rebuilt.JobSelectedByRule = true;
                        CurrentDecision = rebuilt;
                    }
                }
            }

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
                CandidateStatC = string.Join("|", req.CandidateUtilization ?? Array.Empty<float>()),
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
          try
          {
            episodeActive = false;
            LogEpisodeTiming();
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
                    totalReward: 0.0,
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
                totalReward: 0.0,
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
            record.IoDocks = currentConfig.ioDocks;
            record.PreDispatchingMethod = currentConfig.preDispatchingMethod;

            // Deadlock watchdog outcome — see CheckForDeadlock
            record.DeadlockDetected = _deadlockDetected;
            record.DeadlockSimTime = _deadlockDetected ? _deadlockSimTime : -1.0;
            record.FirstStallSimTime = _firstStallSimTime;
            _collisions.Finish(SimTime);
            record.AgvCollisionEvents = _collisions.OverlapEvents;
            record.AgvCollisionPairSeconds = _collisions.OverlapPairSeconds;
            record.AgvClearanceEvents = _collisions.ClearanceEvents;
            record.AgvClearancePairSeconds = _collisions.ClearancePairSeconds;
            record.AgvParkingOverlapEvents = _collisions.ParkingOverlapEvents;
            record.AgvMinCentreDistance = _collisions.MinCentreDistance == float.MaxValue ? -1f : _collisions.MinCentreDistance;
            record.CollisionRecords = _collisions.Events;
            record.ReservationProtocol = currentConfig.reservationProtocol;
            LayoutSpec layoutSpec = currentConfig.Layout ?? LayoutSpec.Default;
            record.LayoutId = layoutSpec.Id;
            record.LayoutBelts = layoutSpec.DescribeBelts(layoutManager.LayoutRows);
            record.LayoutAisles = layoutSpec.AislesString;
            record.FloorWidth = layoutManager.FloorSize.x;
            record.FloorDepth = layoutManager.FloorSize.y;
            record.OrphanPreDispatchesReleased = _flags != null ? _flags.OrphanPreDispatchesReleased : 0;

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

            if (!string.IsNullOrEmpty(_decisionLogDir))
            {
                // Tag rows so they join to Python's decisions.csv (instance seed + seed-queue
                // position). Must run before OnEpisodeFinished: the agent's listener starts the
                // next episode synchronously, which overwrites the seed fields.
                record.Seed = _episodeSeed;
                record.InstanceName = $"{record.InstanceName}|seed_index={_episodeSeedIndex}";
                if (RLDecisionDrainMode) record.RuleName = "rl_policy";
                ResultsLogger.LogDecisions(record);
            }

            OnEpisodeFinished?.Invoke(record);
          }
          catch (Exception ex)
          {
            SimLogger.Error($"[Orchestrator] FinaliseEpisode threw: {ex}");
          }
        }

        /// <summary>
        /// Logs one [EpisodeTiming] line per episode: wall-clock duration split into time inside
        /// Academy.EnvironmentStep (observations + policy round-trip + action; drain mode only)
        /// versus everything else (simulation ticks), plus GC and live-object counts. Anything
        /// that trends upward across episodes while sim time and decisions stay flat is a leak.
        /// </summary>
        private void LogEpisodeTiming()
        {
            double wall = Time.realtimeSinceStartupAsDouble - _episodeWallStart;
            double envStep = _envStepWatch.Elapsed.TotalSeconds;
            int gameObjects = FindObjectsByType<Transform>(FindObjectsInactive.Include, FindObjectsSortMode.None).Length;
            int jobVisuals = FindObjectsByType<JobVisual>(FindObjectsInactive.Include, FindObjectsSortMode.None).Length;
            SimLogger.Low($"[EpisodeTiming] ep={_episodeIndex} wall={wall:F3}s envStep={envStep:F3}s " +
                          $"other={wall - envStep:F3}s ticks={_episodeTicks} envSteps={_episodeEnvSteps} " +
                          $"sim={SimTime:F0}s decisions={decisionCount} " +
                          $"gc0={GC.CollectionCount(0) - _episodeGcStart} " +
                          $"heapMB={GC.GetTotalMemory(false) / 1048576.0:F1} " +
                          $"gameObjects={gameObjects} jobVisuals={jobVisuals}");
        }

        /// <summary>
        /// Writes the current <see cref="RewardMetrics"/> snapshot for <see cref="RewardMetricsSensor"/>.
        /// The reward itself is computed in Python (env/rewards) from consecutive snapshots, so the
        /// total_reward reported in telemetry and results CSVs is always 0.
        /// </summary>
        public void WriteRewardMetrics(float[] buffer)
        {
            RewardMetrics.Fill(buffer, SimTime, episodeActive, decisionCount, Jobs,
                               layoutManager != null ? layoutManager.Machines : null,
                               agvPool != null ? agvPool.AllAGVs : null,
                               trafficZoneManager != null ? trafficZoneManager.Zones : null,
                               _tracker, _deadlockDetected, SimTime > MAX_EPISODE_SIM_SECONDS,
                               _episodeSeed, _episodeSeedIndex, _truncatedByTimeLimit);
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