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
using Assets.Scripts.Simulation.Types;
using System.IO;
using Unity.MLAgents;

namespace Assets.Scripts.Simulation
{

    /// @brief Central coordinator between the Unity physics simulation and the scheduling agent.
    ///
    /// @details Job location is tracked exclusively by JobManager. This bridge reacts to
    /// physics events, queues scheduling decisions, and delegates state changes through
    /// JobManager.TransitionJob — never through ConveyorBelt or PhysicalMachine directly.
    public class SimulationBridge : MonoBehaviour
    {
        public bool IsFactoryReady { get; private set; }
        public FJSSPConfig CurrentConfig => currentConfig;

        [Header("Lifecycle Events")]
        public UnityEvent OnFactorySpawned;

        private Dictionary<MachineType, List<int>> cachedMachinesByType;
        public string LastAppliedRule { get; private set; } = "Waiting...";

        private Queue<int> pendingDecisions = new Queue<int>();

        [Header("Scene References")]
        [SerializeField] private FactoryLayoutManager layoutManager;
        [SerializeField] private TrafficZoneManager trafficZoneManager;
        [SerializeField] private AGV.AGVPool agvPool;
        public JobManager JobManager;
        [SerializeField] private SchedulingAgent agent;
        private FJSSPConfig currentConfig;

        private long frameCount;
        private float timeScaleSum;
        public static SimulationBridge Instance;

        public float StartTime { get; private set; }

        [Header("Episode Configuration")]
        [SerializeField] private bool autoStartOnPlay = false;
        private Dictionary<int, int> routingJobSources = new Dictionary<int, int>();

        [Header("Events")]
        public UnityEvent<DecisionRequest> OnDecisionRequired;
        public UnityEvent<StepResult> OnStepCompleted;
        public UnityEvent<EpisodeResult> OnEpisodeFinished;

        [Header("Logging")]
        [SerializeField] private LogLevel logLevel = LogLevel.Low;

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

        public static int ActionCount => ActionToRule.Length;

        private bool episodeActive;
        private int decisionCount;
        private double totalReward;
        private double previousMakespan;
        private int[] perMachineDecisions;

        public int DecisionCount => decisionCount;
        public bool IsEpisodeActive => episodeActive;
        public bool IsDone => !episodeActive;

        public DecisionRequest CurrentDecision { get; private set; }
        public bool IsWaitingForAction { get; private set; }

        private int exitedJobCount = 0;
        public double SimTime => Time.time - StartTime;

        private Queue<int> pendingRoutingJobs = new Queue<int>();

        // ─────────────────────────────────────────────────────────
        //  Unity Lifecycle
        // ─────────────────────────────────────────────────────────

        private void Awake()
        {
            if (Instance != null)
            {
                Destroy(this);
                return;
            }
            Instance = this;
            string[] args = Environment.GetCommandLineArgs();
            for (int i = 0; i < args.Length - 1; i++)
            {
                if (args[i] == "-timescale" && float.TryParse(args[i + 1], out float ts))
                {
                    Time.timeScale = ts;
                    SimLogger.Low($"[SimBridge] Timescale set to {ts} from command line.");
                    break;
                }
            }

            string resultsDir = Application.isEditor
                ? Path.Combine(Application.dataPath, "..\\..", "Results")
                : Path.Combine(Application.dataPath, "..\\Results");

            Directory.CreateDirectory(resultsDir);
            ResultsLogger.OutputDirectory = resultsDir;
            SimLogger.ActiveLevel = logLevel;
            SimLogger.InitializeFileLogging();
        }

        private void Start()
        {
            string configPath = GetCommandLineArg("-config");
            if (!string.IsNullOrEmpty(configPath))
                LoadConfigFromFile(configPath);

            if (autoStartOnPlay)
            {
                SimLogger.Medium("[SimBridge] autoStartOnPlay=true — arming agent.");
                if (agent != null)
                    agent.IsArmed = true;
            }
        }

        private static string GetCommandLineArg(string key)
        {
            string[] args = Environment.GetCommandLineArgs();
            for (int i = 0; i < args.Length - 1; i++)
            {
                if (args[i] == key)
                    return args[i + 1];
            }
            return null;
        }

        private FJSSPConfig BuildTestConfig()
        {
            var layout = new MachineType[] {
                MachineType.Mill, MachineType.Lathe, MachineType.Weld,
                MachineType.Inspect, MachineType.Assemble
            };

            return new FJSSPConfig
            {
                JobCount = 2,
                MachinesPerType = 1,
                MachineTypeLayout = layout,
                MinProcTime = 5f,
                MaxProcTime = 10f,
                MinOpsPerJob = 5,
                MaxOpsPerJob = 5,
                MaxArrivalTime = 0f,
                Seed = 42
            };
        }

        private FJSSPConfig BuildDefaultConfig()
        {
            var layout = new MachineType[15];
            MachineType[] types = (MachineType[])Enum.GetValues(typeof(MachineType));
            for (int i = 0; i < 15; i++)
                layout[i] = types[i / 3];

            return new FJSSPConfig
            {
                Seed = 42,
                JobCount = 20,
                MachinesPerType = 3,
                MachineTypeLayout = layout,
                MinProcTime = 15f,
                MaxProcTime = 90f,
                MinOpsPerJob = 5,
                MaxOpsPerJob = 8,
                MaxArrivalTime = 0f
            };
        }

        // ─────────────────────────────────────────────────────────
        //  Episode Management
        // ─────────────────────────────────────────────────────────

        public bool LoadConfigFromFile(string path)
        {
            var config = ConfigLoader.LoadSingle(path);
            if (config == null)
            {
                SimLogger.Error($"[SimBridge] Failed to load config from: {path}");
                return false;
            }
            LoadConfig(config);
            return true;
        }

        public void LoadConfig(FJSSPConfig config)
        {
            currentConfig = config;
            IsFactoryReady = false;
            SimLogger.Medium($"[SimBridge] Config loaded: {config.Name} " +
                             $"({config.JobCount}J, {config.TotalMachines}M, seed={config.Seed})");
        }

        public void SpawnFactory()
        {
            if (currentConfig == null)
            {
                SimLogger.Error("[SimBridge] No config loaded. Call LoadConfig() first.");
                return;
            }

            if (IsFactoryReady || episodeActive)
            {
                StopEpisode();
                if (layoutManager != null) layoutManager.ClearFloor();
                if (JobManager != null) JobManager.Cleanup();
            }

            UnityEngine.Random.InitState(currentConfig.Seed);
            cachedMachinesByType = layoutManager.BuildFloor(currentConfig);

            if (trafficZoneManager != null)
                trafficZoneManager.BuildZoneGraph();

            if (agvPool != null)
                agvPool.InitializeFleet();

            IsFactoryReady = true;
            SimLogger.High($"[SimBridge] Factory spawned: {currentConfig.TotalMachines} machines");
            OnFactorySpawned?.Invoke();
        }

        public void StartSimulation()
        {
            if (!IsFactoryReady)
            {
                SimLogger.Error("[SimBridge] Factory not ready. Call SpawnFactory() first.");
                return;
            }
            if (episodeActive)
            {
                SimLogger.Error("[SimBridge] Episode already active.");
                return;
            }

            FJSSPJobDefinition[] jobDefs = FJSSPJobGenerator.Generate(currentConfig, cachedMachinesByType);

            if (JobManager != null)
                JobManager.Initialize(jobDefs, spawnVisuals: true);

            episodeActive = true;
            decisionCount = 0;
            totalReward = 0;
            previousMakespan = 0;
            perMachineDecisions = new int[layoutManager.MachineCount];
            IsWaitingForAction = false;
            pendingDecisions.Clear();
            pendingRoutingJobs.Clear();
            routingJobSources.Clear();
            exitedJobCount = 0;
            frameCount = 0;
            timeScaleSum = 0;

            StartTime = Time.time;
            SimLogger.High($"[SimBridge] Simulation started ({currentConfig.Name})");

            foreach (var tracker in JobManager.JobTrackers)
            {
                if (tracker.ArrivalTime <= 0f)
                    EnqueueRoutingDecision(tracker.JobId, -1, tracker.NextMachineType);
            }
        }

        public void StartEpisode()
        {
            if (currentConfig == null)
                currentConfig = BuildTestConfig();

            SpawnFactory();
            StartSimulation();
        }

        public void LoadAndSpawnFromFile(string path)
        {
            if (LoadConfigFromFile(path))
                SpawnFactory();
        }

        public void StartSimulationInteractive()
        {
            if (!IsFactoryReady)
            {
                SimLogger.Error("[SimBridge] Spawn the factory first.");
                return;
            }

            if (agent != null)
                agent.IsArmed = true;

            StartSimulation();
        }

        // ─────────────────────────────────────────────────────────
        //  Physics Event Listeners
        // ─────────────────────────────────────────────────────────

        /// @brief Called by AGVController.DoDropoff when a job arrives at a machine.
        ///        State transition already happened in JobManager.CompleteTransit.
        public void OnJobArrivedInQueue(int machineId, int jobId)
        {
            SimLogger.High($"[OnJobArrivedInQueue] job={jobId} at M{machineId} — checking dispatch needed");
            CheckIfDecisionNeeded(machineId);
        }

        /// @brief Called by PhysicalMachine when it finishes processing a job.
        public void OnMachineFinished(int machineId, int jobId)
        {
            SimLogger.High($"[OnMachineFinished] job={jobId} machine={machineId}");

            JobTracker tracker = JobManager.GetJobTracker(jobId);
            if (tracker == null)
            {
                SimLogger.Error($"[OnMachineFinished] tracker is NULL for job {jobId}!");
                return;
            }

            // Guard: if already complete and past all operations, don't re-process
            if (tracker.CurrentOperationIndex >= tracker.TotalOperations)
            {
                SimLogger.LogWarning($"[OnMachineFinished] job={jobId} already past all operations (opIdx={tracker.CurrentOperationIndex}/{tracker.TotalOperations}). Ignoring.");
                return;
            }

            JobManager.MarkOperationComplete(jobId, SimTime);

            SimLogger.High($"[OnMachineFinished] job={jobId} state={tracker.State} location={tracker.Location} " +
                           $"opIndex={tracker.CurrentOperationIndex}/{tracker.TotalOperations} " +
                           $"completedOps={tracker.CompletedOperations}");

            if (tracker.CompletedOperations >= tracker.TotalOperations)
            {
                SimLogger.High($"[OnMachineFinished] job={jobId} → DispatchToExit");
                DispatchToExit(jobId, machineId);
            }
            else if (tracker.Location == JobLocation.AwaitingTransport)
            {
                SimLogger.High($"[OnMachineFinished] job={jobId} → EnqueueRouting " +
                               $"nextType={tracker.NextMachineType}");
                EnqueueRoutingDecision(jobId, machineId, tracker.NextMachineType);
            }
            else
            {
                SimLogger.Error($"[OnMachineFinished] job={jobId} UNHANDLED location={tracker.Location} " +
                                $"— no routing or exit dispatched!");
            }

            CheckIfDecisionNeeded(machineId);
        }

        public void OnJobExited(int jobId)
        {
            exitedJobCount++;
            if (exitedJobCount >= JobManager.JobCount)
                FinaliseEpisode();
        }

        // ─────────────────────────────────────────────────────────
        //  Dispatch helpers
        // ─────────────────────────────────────────────────────────

        private void DispatchToExit(int jobId, int sourceMachineId)
        {
            if (agvPool == null) return;

            JobTracker tracker = JobManager.GetJobTracker(jobId);
            if (tracker != null)
                tracker.NextMachineId = -1; // -1 = exit

            // Transition to AwaitingPickup — AGV will pull this job when idle
            JobManager.TransitionJob(jobId, JobLocation.AwaitingPickup, sourceMachineId);

            // Nudge pool to check for idle AGVs (pull model)
            agvPool.TryAssignWork();
        }

        private void EnqueueRoutingDecision(int jobId, int sourceMachineId, MachineType requiredType)
        {
            if (!pendingRoutingJobs.Contains(jobId))
            {
                pendingRoutingJobs.Enqueue(jobId);
                routingJobSources[jobId] = sourceMachineId;
            }
        }

        /// @brief Dispatches an AGV for a job. Sets AwaitingPickup and calls pool.
        ///        The AGV resolves pickup/dropoff positions itself from JobManager.
        private void DispatchRealAGV(int jobId, int sourceMachineId, int targetMachineId)
        {
            PhysicalMachine targetMachine = layoutManager.GetMachine(targetMachineId);
            if (targetMachine == null)
            {
                SimLogger.Error($"[DispatchRealAGV] job={jobId} targetMachine {targetMachineId} is NULL!");
                return;
            }

            // Set where the job is going
            JobTracker tracker = JobManager.GetJobTracker(jobId);
            if (tracker != null)
                tracker.NextMachineId = targetMachineId;

            // Transition to AwaitingPickup at source location — AGV will pull
            JobManager.TransitionJob(jobId, JobLocation.AwaitingPickup, sourceMachineId);

            SimLogger.High($"[DispatchRealAGV] job={jobId} source=M{sourceMachineId} target=M{targetMachineId}");

            // Nudge pool to check for idle AGVs (pull model)
            agvPool.TryAssignWork();
        }

        /// @brief Checks if an idle machine has dispatchable jobs. Queries JobManager.
        private void CheckIfDecisionNeeded(int machineId)
        {
            PhysicalMachine machine = layoutManager.GetMachine(machineId);
            bool isIdle = machine != null && machine.IsIdle;
            bool hasJobs = JobManager.HasDispatchableJob(machineId);

            if (isIdle && hasJobs)
            {
                if (!pendingDecisions.Contains(machineId))
                {
                    SimLogger.Medium($"[CheckDecision] M{machineId} → enqueuing dispatch (idle={isIdle}, hasJobs={hasJobs})");
                    pendingDecisions.Enqueue(machineId);
                }
            }
            else
            {
                SimLogger.Medium($"[CheckDecision] M{machineId} → skipped (idle={isIdle}, hasJobs={hasJobs})");
            }
        }

        // ─────────────────────────────────────────────────────────
        //  Core Step
        // ─────────────────────────────────────────────────────────

        public StepResult Step(int actionIndex)
        {
            IsWaitingForAction = false;

            if (CurrentDecision.Type == DecisionType.Dispatch)
            {
                LastAppliedRule = ActionToRule[actionIndex].ToString();
                int chosenJobId = ApplyDispatchingRule(actionIndex, CurrentDecision.MachineId);

                if (chosenJobId < 0)
                {
                    SimLogger.LogWarning($"[Step] No dispatchable job at M{CurrentDecision.MachineId}.");
                }
                else
                {
                    float duration = JobManager.GetProcessingTime(chosenJobId, CurrentDecision.MachineId);
                    PhysicalMachine machine = layoutManager.GetMachine(CurrentDecision.MachineId);
                    JobManager.MarkOperationStarted(chosenJobId, SimTime);
                    machine.StartProcessing(chosenJobId, duration);
                }
            }
            else if (CurrentDecision.Type == DecisionType.Routing)
            {
                int chosenMachineId = ApplyMachineSelectionRule(actionIndex, CurrentDecision);

                int sourceMachineId = CurrentDecision.SourceMachineId;
                DispatchRealAGV(CurrentDecision.JobId, sourceMachineId, chosenMachineId);
            }

            float stepReward = CalculateReward();
            totalReward += stepReward;

            StepResult result = new StepResult
            {
                Reward = stepReward,
                Done = false,
                CurrentMakespan = SimTime
            };

            OnStepCompleted?.Invoke(result);
            return result;
        }

        private int ApplyMachineSelectionRule(int actionIndex, DecisionRequest req)
        {
            DispatchingRule rule = ActionToRule[actionIndex];
            int[] candidates = req.CandidateMachineIds;

            if (candidates.Length == 1) return candidates[0];

            return rule switch
            {
                DispatchingRule.SPT_SMPT or
                DispatchingRule.LPT_SMPT or
                DispatchingRule.SRT_SMPT =>
                    candidates[ArgMinIndex(req.CandidateJobTimes)],

                DispatchingRule.SPT_SRWT or
                DispatchingRule.SRT_SRWT or
                DispatchingRule.SDT_SRWT =>
                    candidates[ArgMinIndex(req.CandidateQueueLengths)],

                DispatchingRule.LPT_MMUR or
                DispatchingRule.LRT_MMUR =>
                    candidates[ArgMaxIndex(req.CandidateQueueLengths)],

                _ => candidates[0]
            };
        }

        private int ArgMinIndex(float[] values)
        {
            int best = 0;
            for (int i = 1; i < values.Length; i++)
                if (values[i] < values[best]) best = i;
            return best;
        }

        private int ArgMaxIndex(float[] values)
        {
            int best = 0;
            for (int i = 1; i < values.Length; i++)
                if (values[i] > values[best]) best = i;
            return best;
        }

        // ─────────────────────────────────────────────────────────
        //  Update Loop
        // ─────────────────────────────────────────────────────────

        private void Update()
        {
            frameCount++;
            timeScaleSum += Time.timeScale;
            if (!episodeActive) return;
            if (IsWaitingForAction) return;

            // Routing first — job is done, needs to move
            while (pendingRoutingJobs.Count > 0)
            {
                int jobId = pendingRoutingJobs.Dequeue();
                JobTracker tracker = JobManager.GetJobTracker(jobId);

                if (tracker == null) continue;

                bool validForRouting = tracker.Location == JobLocation.OnFactoryBelt ||
                                       tracker.Location == JobLocation.AwaitingTransport ||
                                       tracker.Location == JobLocation.PendingEntry;
                if (!validForRouting)
                {
                    SimLogger.Error($"[Update] job={jobId} SKIPPED routing — location={tracker.Location}");
                    continue;
                }

                int sourceMachineId = routingJobSources.TryGetValue(jobId, out int src) ? src : -1;
                routingJobSources.Remove(jobId);

                CurrentDecision = BuildRoutingDecisionRequest(jobId, tracker.NextMachineType, sourceMachineId);
                IsWaitingForAction = true;
                OnDecisionRequired?.Invoke(CurrentDecision);
                return;
            }

            // Then dispatch — idle machine with jobs waiting
            while (pendingDecisions.Count > 0)
            {
                int nextMachineId = pendingDecisions.Dequeue();
                PhysicalMachine machine = layoutManager.GetMachine(nextMachineId);

                bool isIdle = machine != null && machine.IsIdle;
                bool hasJobs = JobManager.HasDispatchableJob(nextMachineId);

                SimLogger.Medium($"[Update] Processing dispatch for M{nextMachineId}: idle={isIdle}, hasJobs={hasJobs}");

                if (isIdle && hasJobs)
                {
                    CurrentDecision = BuildDecisionRequest(nextMachineId);
                    IsWaitingForAction = true;
                    OnDecisionRequired?.Invoke(CurrentDecision);
                    return;
                }
            }

            // Safety scan: periodically check ALL machines for stuck dispatches.
            // This catches any case where a CheckIfDecisionNeeded was missed.
            if (frameCount % 60 == 0)
            {
                foreach (var machine in layoutManager.Machines)
                {
                    if (machine.IsIdle && JobManager.HasDispatchableJob(machine.MachineId))
                    {
                        if (!pendingDecisions.Contains(machine.MachineId))
                        {
                            SimLogger.LogWarning($"[SafetyScan] M{machine.MachineId} is idle with dispatchable jobs — re-enqueuing!");
                            pendingDecisions.Enqueue(machine.MachineId);
                        }
                    }
                }
            }
        }

        private DecisionRequest BuildRoutingDecisionRequest(int jobId, MachineType requiredType, int sourceMachineId)
        {
            List<int> candidates = new List<int>();
            foreach (var m in layoutManager.Machines)
                if (m.MachineType == requiredType)
                    candidates.Add(m.MachineId);

            float[] queueLengths = new float[candidates.Count];
            float[] jobTimes = new float[candidates.Count];

            for (int i = 0; i < candidates.Count; i++)
            {
                // Use JobManager for queue lengths (single source of truth)
                queueLengths[i] = JobManager.GetJobsInMachineQueue(candidates[i]).Count;
                jobTimes[i] = JobManager.GetProcessingTime(jobId, candidates[i]);
            }

            return new DecisionRequest
            {
                Type = DecisionType.Routing,
                SimTime = SimTime,
                SourceMachineId = sourceMachineId,
                DecisionIndex = decisionCount++,
                TotalJobs = JobManager.JobCount,
                CompletedJobs = CountCompletedJobs(),
                JobId = jobId,
                RequiredType = requiredType,
                CandidateMachineIds = candidates.ToArray(),
                CandidateQueueLengths = queueLengths,
                CandidateJobTimes = jobTimes,
            };
        }

        private int CountCompletedJobs()
        {
            int count = 0;
            foreach (var t in JobManager.JobTrackers)
                if (t.Location == JobLocation.Exited) count++;
            return count;
        }

        /// @brief Builds a dispatch decision using JobManager queries (not belt contents).
        private DecisionRequest BuildDecisionRequest(int machineId)
        {
            List<int> queue = JobManager.GetDispatchableJobs(machineId);
            int[] jobIds = queue.ToArray();
            double[] durations = new double[jobIds.Length];

            for (int i = 0; i < jobIds.Length; i++)
                durations[i] = JobManager.GetProcessingTime(jobIds[i], machineId);

            return new DecisionRequest
            {
                MachineId = machineId,
                SimTime = SimTime,
                QueuedJobIds = jobIds,
                QueuedDurations = durations,
                DecisionIndex = decisionCount++,
                TotalJobs = JobManager.JobCount,
                Type = DecisionType.Dispatch,
                CompletedJobs = CountCompletedJobs()
            };
        }

        // ─────────────────────────────────────────────────────────
        //  Reward
        // ─────────────────────────────────────────────────────────

        private float CalculateReward()
        {
            float currentSimTime = (float)SimTime;
            float delta = currentSimTime - (float)previousMakespan;
            previousMakespan = currentSimTime;

            int totalOps = 0;
            if (JobManager?.JobTrackers != null)
                foreach (var t in JobManager.JobTrackers)
                    totalOps += t.TotalOperations;
            return -delta / (Mathf.Max(totalOps, 1) * Time.timeScale);
        }

        // ─────────────────────────────────────────────────────────
        //  Dispatching rules — uses JobManager queries
        // ─────────────────────────────────────────────────────────

        /// @brief Selects the best job from JobManager's dispatchable list.
        private int ApplyDispatchingRule(int actionIndex, int machineId)
        {
            DispatchingRule rule = ActionToRule[actionIndex];
            List<int> queue = JobManager.GetDispatchableJobs(machineId);

            if (queue.Count == 0) return -1;
            if (queue.Count == 1) return queue[0];

            return rule switch
            {
                DispatchingRule.SPT_SMPT or
                DispatchingRule.SPT_SRWT =>
                    ArgMin(queue, jobId => GetCurrentOpTime(jobId, machineId)),

                DispatchingRule.LPT_MMUR or
                DispatchingRule.LPT_SMPT =>
                    ArgMax(queue, jobId => GetCurrentOpTime(jobId, machineId)),

                DispatchingRule.SRT_SRWT or
                DispatchingRule.SRT_SMPT =>
                    ArgMin(queue, jobId => GetRemainingWork(jobId)),

                DispatchingRule.LRT_MMUR =>
                    ArgMax(queue, jobId => GetRemainingWork(jobId)),

                DispatchingRule.SDT_SRWT =>
                    ArgMin(queue, jobId => GetTimeInSystem(jobId)),

                _ => queue[0]
            };
        }

        private float GetCurrentOpTime(int jobId, int machineId)
        {
            JobTracker t = JobManager.GetJobTracker(jobId);
            if (t == null) return float.MaxValue;
            if (t.CurrentOperationIndex < 0 || t.CurrentOperationIndex >= t.TotalOperations)
                return float.MaxValue;
            if (t.CurrentOperationIndex >= t.EligibleMachinesPerOp.Length)
                return float.MaxValue;

            var eligible = t.EligibleMachinesPerOp[t.CurrentOperationIndex];
            return eligible.TryGetValue(machineId, out float time) ? time : float.MaxValue;
        }

        private float GetRemainingWork(int jobId)
        {
            JobTracker t = JobManager.GetJobTracker(jobId);
            if (t == null) return 0f;

            float total = 0f;
            for (int o = t.CurrentOperationIndex; o < t.TotalOperations; o++)
            {
                float minTime = float.MaxValue;
                foreach (float procTime in t.EligibleMachinesPerOp[o].Values)
                    if (procTime < minTime) minTime = procTime;
                if (minTime < float.MaxValue) total += minTime;
            }
            return total;
        }

        private float GetTimeInSystem(int jobId)
        {
            JobTracker t = JobManager.GetJobTracker(jobId);
            if (t == null) return 0f;
            return (float)SimTime - t.ArrivalTime;
        }

        private int ArgMin(List<int> jobIds, Func<int, float> scorer)
        {
            int best = jobIds[0];
            float bestScore = float.MaxValue;
            foreach (int id in jobIds)
            {
                float score = scorer(id);
                if (score < bestScore) { bestScore = score; best = id; }
            }
            return best;
        }

        private int ArgMax(List<int> jobIds, Func<int, float> scorer)
        {
            int best = jobIds[0];
            float bestScore = float.MinValue;
            foreach (int id in jobIds)
            {
                float score = scorer(id);
                if (score > bestScore) { bestScore = score; best = id; }
            }
            return best;
        }

        // ─────────────────────────────────────────────────────────
        //  Episode finalization
        // ─────────────────────────────────────────────────────────

        private void FinaliseEpisode()
        {
            episodeActive = false;
            int totalOps = 0;
            foreach (var t in JobManager.JobTrackers) totalOps += t.TotalOperations;

            ResultsLogger.LogEpisode(
                ruleName: LastAppliedRule,
                seed: currentConfig.Seed,
                makespan: SimTime,
                jobCount: JobManager.JobCount,
                machineCount: layoutManager.MachineCount,
                totalOps: totalOps,
                decisionCount: decisionCount,
                totalReward: totalReward,
                timeScaleSum / frameCount
            );

            EpisodeResult result = new EpisodeResult
            {
                InstanceName = "unknown",
                RuleName = "agent",
                Makespan = SimTime,
                OptimalMakespan = 0,
                DecisionPoints = decisionCount,
                TotalReward = totalReward,
                PerMachineDecisions = perMachineDecisions
            };

            SimLogger.High($"[SimBridge] Episode complete: makespan={result.Makespan:F1}, decisions={result.DecisionPoints}");

            float expectedMinMakespan = 0f;
            foreach (var t in JobManager.JobTrackers)
            {
                float minJobTime = 0f;
                foreach (var op in t.EligibleMachinesPerOp)
                    minJobTime += op.Values.Min();
                expectedMinMakespan = Mathf.Max(expectedMinMakespan, minJobTime);
            }

            SimLogger.High($"[Validate] Makespan={SimTime:F1} | " +
                           $"TheoreticalMin={expectedMinMakespan:F1} | " +
                           $"Ratio={SimTime / expectedMinMakespan:F2}x | " +
                           $"TimeScale={Time.timeScale}");
            OnEpisodeFinished?.Invoke(result);

            var stats = Academy.Instance.StatsRecorder;
        }

        public void StopEpisode()
        {
            if (!episodeActive) return;

            episodeActive = false;
            IsWaitingForAction = false;
            pendingDecisions.Clear();
            pendingRoutingJobs.Clear();
            routingJobSources.Clear();
            exitedJobCount = 0;

            if (layoutManager != null) layoutManager.ClearFloor();
            if (JobManager != null) JobManager.Cleanup();

            if (agent != null) agent.EndEpisode();

            SimLogger.Low("[SimBridge] Episode stopped by user.");
        }

        public int GetRuleIndex(DispatchingRule rule) => Array.IndexOf(ActionToRule, rule);
    }
}