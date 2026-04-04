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

namespace Assets.Scripts.Simulation
{

    /// @brief Central coordinator between the Unity physics simulation and the scheduling agent.
    ///
    /// @details The bridge owns the episode lifecycle: loading a Taillard instance,
    /// spawning physical jobs and machines, reacting to physics events from
    /// @c PhysicalMachine, maintaining a queue of pending scheduling decisions,
    /// and surfacing one decision at a time to the @c SchedulingAgent via
    /// @c OnDecisionRequired. It also computes per-step rewards and fires
    /// @c OnEpisodeFinished when all jobs are complete.
    public class SimulationBridge : MonoBehaviour
    {
        /// @brief The dispatching rule last applied during @c Step().
        public string LastAppliedRule { get; private set; } = "Waiting...";

        private Queue<int> pendingDecisions = new Queue<int>();

        [Header("Scene References")]
        [SerializeField] private FactoryLayoutManager layoutManager;
        [SerializeField] private TrafficZoneManager trafficZoneManager;
        [SerializeField] private AGVPool agvPool;
        public JobManager JobManager;
        [SerializeField] private SchedulingAgent agent;
        private FJSSPConfig currentConfig;

        private long frameCount;
        private float timeScaleSum;
        /// @brief Singleton instance accessible from @c PhysicalMachine and @c JobManager.
        public static SimulationBridge Instance;

        /// @brief Wall-clock time when the current episode started.
        public float StartTime { get; private set; }

        [Header("Episode Configuration")]
        // [SerializeField] private TextAsset taillardJsonDefault;
        // public TextAsset TaillardJson { get; set; }
        [SerializeField] private bool autoStartOnPlay = false;
        private Dictionary<int, int> routingJobSources = new Dictionary<int, int>();

        [Header("Events")]
        public UnityEvent<DecisionRequest> OnDecisionRequired;
        public UnityEvent<StepResult> OnStepCompleted;
        public UnityEvent<EpisodeResult> OnEpisodeFinished;

        [Header("Logging")]
        [SerializeField] private LogLevel logLevel = LogLevel.Low;

        /// @brief Ordered mapping from discrete action index to @c DispatchingRule enum value.
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



        /// @brief Number of discrete actions available to the agent.
        public static int ActionCount => ActionToRule.Length;

        // private DESSimulator simulator;
        // private TaillardInstance currentInstance;
        private bool episodeActive;
        private int decisionCount;
        private double totalReward;
        private double previousMakespan;
        private int[] perMachineDecisions;

        public int DecisionCount => decisionCount;
        public bool IsEpisodeActive => episodeActive;
        public bool IsDone => !episodeActive;

        /// @brief The decision context the agent is currently expected to respond to.
        public DecisionRequest CurrentDecision { get; private set; }

        /// @brief True from the moment @c OnDecisionRequired fires until @c Step() is called.
        public bool IsWaitingForAction { get; private set; }

        private int exitedJobCount = 0;

        /// @brief Seconds elapsed since the episode started, derived from Unity wall time.
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
            // simulator = new DESSimulator();
            ResultsLogger.OutputDirectory = Path.Combine(Application.dataPath, "..\\..", "Results");
            Directory.CreateDirectory(ResultsLogger.OutputDirectory);
            SimLogger.ActiveLevel = logLevel;
        }

        private void Start()
        {
            if (autoStartOnPlay)
            {
                SimLogger.Medium("[Sim Bridge] Auto Starting Episode...");
                StartEpisode();
            }
        }
        private FJSSPConfig BuildTestConfig()
        {
            // 5 machines, 1 of each type, 2 jobs, 5 ops each (one per type)
            // MaxArrivalTime = 0 so both jobs start immediately
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
            // 15 machines, 3 of each type, laid out in type order
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
                MaxArrivalTime = 0f  // all jobs available at start for now
            };
        }

        // ─────────────────────────────────────────────────────────
        //  Episode Management
        // ─────────────────────────────────────────────────────────

        /// @brief Loads the configured Taillard instance, builds the factory floor,
        ///        spawns job visuals, and dispatches every job toward its first machine.
        public void StartEpisode()
        {
            //currentConfig = BuildDefaultConfig();  // temporary hardcoded config
            currentConfig = BuildTestConfig();
            UnityEngine.Random.InitState(currentConfig.Seed);
            var machinesByType = layoutManager.BuildFloor(currentConfig);
            exitedJobCount = 0;
            FJSSPJobDefinition[] jobDefs = FJSSPJobGenerator.Generate(currentConfig, machinesByType);

            if (trafficZoneManager != null)
            {
                trafficZoneManager.BuildZoneGraph();
            }
            if (JobManager != null)
            {
                JobManager.Initialize(jobDefs, spawnVisuals: true);
            }
            if (agvPool != null)
            {
                agvPool.InitializeFleet();
            }

            episodeActive = true;
            decisionCount = 0;
            totalReward = 0;
            previousMakespan = 0;
            perMachineDecisions = new int[layoutManager.MachineCount];
            IsWaitingForAction = false;
            pendingDecisions.Clear();

            StartTime = Time.time;

            SimLogger.High($"[SimBridge] Episode started");

            foreach (var tracker in JobManager.JobTrackers)
            {
                if (tracker.ArrivalTime <= 0f)
                    EnqueueRoutingDecision(tracker.JobId, -1, tracker.NextMachineType);
            }

        }

        // ─────────────────────────────────────────────────────────
        //  Physics Event Listeners
        // ─────────────────────────────────────────────────────────

        /// @brief Called by @c PhysicalMachine when a job enters its trigger zone.
        /// @param machineId  The machine the job arrived at.
        /// @param jobId      The job that physically arrived.
        public void OnJobArrivedInQueue(int machineId, int jobId)
        {
            JobManager.MarkJobArrivedAtMachine(jobId, machineId);
            CheckIfDecisionNeeded(machineId);
        }

        /// @brief Called by @c PhysicalMachine when it finishes processing a job.
        /// @details Updates job tracking, checks for episode completion, dispatches
        ///          an AGV to carry the job to its next machine, and checks whether
        ///          the now-idle machine has further jobs waiting.
        /// @param machineId  The machine that finished.
        /// @param jobId      The job that was completed.
        public void OnMachineFinished(int machineId, int jobId)
        {
            JobManager.MarkOperationComplete(jobId, SimTime);

            JobTracker tracker = JobManager.GetJobTracker(jobId);

            if (tracker != null && tracker.State == JobLifecycleState.Complete)
            {
                // Final delivery — send to outgoing belt, no routing decision needed
                PhysicalMachine sourceMachine = layoutManager.GetMachine(machineId);
                DispatchToExit(jobId, sourceMachine);
            }
            else if (tracker != null && tracker.State == JobLifecycleState.WaitingForTransport)
            {
                EnqueueRoutingDecision(jobId, machineId, tracker.NextMachineType);
            }

            // Check episode completion only after final AGV is dispatched
            // AreAllJobsComplete is checked in OnJobExited instead
            CheckIfDecisionNeeded(machineId);
        }
        public void OnJobExited(int jobId)
        {
            exitedJobCount++;
            if (exitedJobCount >= JobManager.JobCount)
                FinaliseEpisode();
        }
        private void DispatchToExit(int jobId, PhysicalMachine source)
        {
            if (agvPool == null) return;

            Vector3 pickupPos = source != null
                ? source.GetPickupPositionForJob(jobId)
                : Vector3.zero;

            Vector3 dropoffPos = layoutManager.OutgoingBeltPosition;

            JobManager.GetJobTracker(jobId)?.Visual?.SetState(JobLifecycleState.InTransit);
            agvPool.TryDispatch(jobId, pickupPos, dropoffPos, source, null);
        }
        private void EnqueueRoutingDecision(int jobId, int sourceMachineId, MachineType requiredType)
        {
            JobTracker tracker = JobManager.GetJobTracker(jobId);

            // Find all physical machines of the required type
            List<int> candidates = layoutManager.Machines
                .Where(m => m.MachineType == requiredType)
                .Select(m => m.MachineId)
                .ToList();

            float[] queueLengths = candidates
                .Select(id => (float)layoutManager.GetMachine(id).PhysicalQueue.Count)
                .ToArray();

            float[] jobTimes = candidates
                .Select(id => JobManager.GetProcessingTime(jobId, id))
                .ToArray();

            var req = new DecisionRequest
            {
                Type = DecisionType.Routing,
                SimTime = SimTime,
                DecisionIndex = decisionCount++,
                TotalJobs = JobManager.JobCount,
                CompletedJobs = JobManager.JobTrackers.Count(t => t.State == JobLifecycleState.Complete),
                JobId = jobId,
                RequiredType = requiredType,
                CandidateMachineIds = candidates.ToArray(),
                CandidateQueueLengths = queueLengths,
                CandidateJobTimes = jobTimes,
            };
            if (!pendingRoutingJobs.Contains(jobId))
            {
                pendingRoutingJobs.Enqueue(jobId);
                routingJobSources[jobId] = sourceMachineId;
            }

            // CurrentDecision = req;
            // IsWaitingForAction = true;
            // OnDecisionRequired?.Invoke(req);


        }

        /// @brief Enqueues @p machineId as a pending decision if it is idle and has jobs waiting.
        /// @param machineId  Machine to evaluate.
        private void CheckIfDecisionNeeded(int machineId)
        {
            PhysicalMachine machine = layoutManager.GetMachine(machineId);

            if (machine != null && machine.IsIdle && machine.PhysicalQueue.Count > 0)
            {
                if (!pendingDecisions.Contains(machineId))
                {
                    pendingDecisions.Enqueue(machineId);
                }
            }
        }

        // ─────────────────────────────────────────────────────────
        //  Core Step
        // ─────────────────────────────────────────────────────────

        /// @brief Applies the agent's chosen dispatching rule to @c CurrentDecision.
        ///
        /// @details Selects the best job from the machine's physical queue according
        /// to the rule identified by @p actionIndex, starts physical processing
        /// on that machine, and returns a @c StepResult carrying the reward.
        ///
        /// @param actionIndex  Index into @c ActionToRule; selects the dispatching rule.
        /// @return             A @c StepResult with reward and episode-done flag.
        public StepResult Step(int actionIndex)
        {
            IsWaitingForAction = false;

            if (CurrentDecision.Type == DecisionType.Dispatch)
            {
                // existing job-selection logic
                LastAppliedRule = ActionToRule[actionIndex].ToString();
                int chosenJobId = ApplyDispatchingRule(actionIndex, CurrentDecision.MachineId);
                float duration = JobManager.GetProcessingTime(chosenJobId, CurrentDecision.MachineId);

                PhysicalMachine machine = layoutManager.GetMachine(CurrentDecision.MachineId);
                machine.StartProcessing(chosenJobId, duration);
                JobManager.MarkOperationStarted(chosenJobId, SimTime);
            }
            else if (CurrentDecision.Type == DecisionType.Routing)
            {
                // machine-selection — pick from candidates using PDR machine-selection half
                int chosenMachineId = ApplyMachineSelectionRule(actionIndex, CurrentDecision);
                JobManager.GetJobTracker(CurrentDecision.JobId).NextMachineId = chosenMachineId;

                // Use SourceMachineId from the decision — tracker.CurrentMachineId is already -1 by now
                PhysicalMachine source = CurrentDecision.SourceMachineId >= 0
                    ? layoutManager.GetMachine(CurrentDecision.SourceMachineId)
                    : null;

                DispatchRealAGV(CurrentDecision.JobId, source, chosenMachineId);
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
        /// @brief Picks the best candidate machine using the machine-selection half of the PDR.
        /// SMPT — pick machine where this job's processing time is shortest
        /// SRWT — pick machine with least total queue work
        /// MMUR — pick most utilized machine (shortest idle time proxy: longest queue)
        private int ApplyMachineSelectionRule(int actionIndex, DecisionRequest req)
        {
            DispatchingRule rule = ActionToRule[actionIndex];
            int[] candidates = req.CandidateMachineIds;

            if (candidates.Length == 1) return candidates[0];

            return rule switch
            {
                // SMPT — minimize this job's proc time at destination
                DispatchingRule.SPT_SMPT or
                DispatchingRule.LPT_SMPT or
                DispatchingRule.SRT_SMPT =>
                    candidates[ArgMinIndex(req.CandidateJobTimes)],

                // SRWT — minimize queue backlog at destination
                DispatchingRule.SPT_SRWT or
                DispatchingRule.SRT_SRWT or
                DispatchingRule.SDT_SRWT =>
                    candidates[ArgMinIndex(req.CandidateQueueLengths)],

                // MMUR — maximize utilization, send to busiest machine
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

        /// @brief Drains the pending-decision queue one entry per frame while the
        ///        agent is free, firing @c OnDecisionRequired for the next machine.
        private void Update()
        {
            frameCount++;
            timeScaleSum += Time.timeScale;
            if (pendingRoutingJobs.Count > 0 || pendingDecisions.Count > 0)
                SimLogger.Medium($"[Bridge] Update: routing={pendingRoutingJobs.Count} dispatch={pendingDecisions.Count} waiting={IsWaitingForAction}");
            if (!episodeActive) return;
            if (IsWaitingForAction) return;

            // Routing first — job is already done, needs to move
            while (pendingRoutingJobs.Count > 0)
            {
                int jobId = pendingRoutingJobs.Dequeue();
                JobTracker tracker = JobManager.GetJobTracker(jobId);

                if (tracker == null) continue;

                bool validForRouting = tracker.State == JobLifecycleState.NotStarted ||
                                       tracker.State == JobLifecycleState.WaitingForTransport;
                if (!validForRouting) continue;

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

                if (machine != null && machine.IsIdle && machine.PhysicalQueue.Count > 0)
                {
                    CurrentDecision = BuildDecisionRequest(machine);
                    IsWaitingForAction = true;
                    OnDecisionRequired?.Invoke(CurrentDecision);
                    return;
                }
            }
        }
        private DecisionRequest BuildRoutingDecisionRequest(int jobId, MachineType requiredType, int sourceMachineId)
        {
            JobTracker tracker = JobManager.GetJobTracker(jobId);

            List<int> candidates = new List<int>();
            foreach (var m in layoutManager.Machines)
                if (m.MachineType == requiredType)
                    candidates.Add(m.MachineId);

            float[] queueLengths = new float[candidates.Count];
            float[] jobTimes = new float[candidates.Count];

            for (int i = 0; i < candidates.Count; i++)
            {
                queueLengths[i] = layoutManager.GetMachine(candidates[i]).PhysicalQueue.Count;
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
                if (t.State == JobLifecycleState.Complete) count++;
            return count;
        }

        // ─────────────────────────────────────────────────────────
        //  Helpers
        // ─────────────────────────────────────────────────────────

        /// @brief Constructs a @c DecisionRequest snapshot for the given machine.
        /// @param machine  The idle machine with pending jobs.
        /// @return         Fully populated @c DecisionRequest.
        private DecisionRequest BuildDecisionRequest(PhysicalMachine machine)
        {
            int[] jobIds = machine.PhysicalQueue.ToArray();
            double[] durations = new double[jobIds.Length];

            for (int i = 0; i < jobIds.Length; i++)
            {
                durations[i] = JobManager.GetProcessingTime(jobIds[i], machine.MachineId);
            }


            return new DecisionRequest
            {
                MachineId = machine.MachineId,
                SimTime = SimTime,
                QueuedJobIds = jobIds,
                QueuedDurations = durations,
                DecisionIndex = decisionCount++,
                TotalJobs = JobManager.JobCount,
                Type = DecisionType.Dispatch,
                CompletedJobs = JobManager.JobTrackers.Count(t => t.State == JobLifecycleState.Complete)
            };
        }


        /// @brief Negative normalised makespan delta.
        /// @details Normalized by total operations in the episode so reward magnitude
        ///          stays comparable across different problem sizes.
        private float CalculateReward()
        {
            float currentSimTime = (float)SimTime;
            float delta = currentSimTime - (float)previousMakespan;
            previousMakespan = currentSimTime;

            // Count total ops across all trackers — works for any J/M config
            int totalOps = 0;
            if (JobManager?.JobTrackers != null)
                foreach (var t in JobManager.JobTrackers)
                    totalOps += t.TotalOperations;

            return -delta / Mathf.Max(totalOps, 1);
        }

        /// @brief Selects the best job from a machine's queue using the job-selection
        ///        half of the composite PDR identified by <paramref name="actionIndex"/>.
        /// @details Machine-selection (SMPT/SRWT/MMUR) fires separately in the routing
        ///          decision flow after a job completes an operation.
        private int ApplyDispatchingRule(int actionIndex, int machineId)
        {
            DispatchingRule rule = ActionToRule[actionIndex];
            PhysicalMachine machine = layoutManager.GetMachine(machineId);
            List<int> queue = machine.PhysicalQueue;

            if (queue.Count == 0) return -1;
            if (queue.Count == 1) return queue[0];

            return rule switch
            {
                // ── Shortest Processing Time at this machine ──────────────────────
                DispatchingRule.SPT_SMPT or
                DispatchingRule.SPT_SRWT =>
                    ArgMin(queue, jobId => GetCurrentOpTime(jobId, machineId)),

                // ── Longest Processing Time at this machine ───────────────────────
                DispatchingRule.LPT_MMUR or
                DispatchingRule.LPT_SMPT =>
                    ArgMax(queue, jobId => GetCurrentOpTime(jobId, machineId)),

                // ── Shortest Remaining Work (sum of min times across future ops) ──
                DispatchingRule.SRT_SRWT or
                DispatchingRule.SRT_SMPT =>
                    ArgMin(queue, jobId => GetRemainingWork(jobId)),

                // ── Longest Remaining Work ────────────────────────────────────────
                DispatchingRule.LRT_MMUR =>
                    ArgMax(queue, jobId => GetRemainingWork(jobId)),

                // ── Shortest time in system (proxy for due date pressure) ─────────
                DispatchingRule.SDT_SRWT =>
                    ArgMin(queue, jobId => GetTimeInSystem(jobId)),

                _ => queue[0]
            };
        }

        /// @brief Processing time for a job's current operation at a specific machine.
        ///        Returns float.MaxValue if the machine isn't eligible (shouldn't happen
        ///        if routing decisions are correct, but guards against bad state).
        private float GetCurrentOpTime(int jobId, int machineId)
        {
            JobTracker t = JobManager.GetJobTracker(jobId);
            if (t == null) return float.MaxValue;

            var eligible = t.EligibleMachinesPerOp[t.CurrentOperationIndex];
            return eligible.TryGetValue(machineId, out float time) ? time : float.MaxValue;
        }

        /// @brief Sum of the minimum eligible processing times across all remaining ops.
        ///        "Remaining" means ops from CurrentOperationIndex onward.
        ///        Using the minimum eligible time per op gives a lower-bound on work left,
        ///        which is the standard SRT definition in FJSSP literature.
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

        /// @brief Seconds the job has been in the system since arrival.
        ///        Jobs that arrived earlier get priority under SDT.
        private float GetTimeInSystem(int jobId)
        {
            JobTracker t = JobManager.GetJobTracker(jobId);
            if (t == null) return 0f;
            return (float)SimTime - t.ArrivalTime;
        }

        // ── Generic min/max selectors ─────────────────────────────────────────────────

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

        /// @brief Marks the episode as finished, computes final statistics, and
        ///        broadcasts an @c EpisodeResult via @c OnEpisodeFinished.
        private void FinaliseEpisode()
        {
            episodeActive = false;
            int totalOps = 0;
            foreach (var t in JobManager.JobTrackers) totalOps += t.TotalOperations;

            ResultsLogger.LogEpisode(
                ruleName: LastAppliedRule,
                seed: currentConfig.Seed,      // store config as field: private FJSSPConfig currentConfig
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
            OnEpisodeFinished?.Invoke(result);
        }

        /// @brief Dispatches a real AGV to carry @p jobId from its current location
        ///        to the incoming staging area of the target machine.
        /// @param jobId            Job to transport.
        /// @param source           Machine the job is leaving (null for initial dispatch from spawn).
        /// @param targetMachineId  Destination machine index.
        private void DispatchRealAGV(int jobId, PhysicalMachine source, int targetMachineId)
        {
            if (agvPool == null) { SimLogger.Error("[SimBridge] AGVPool not assigned!"); return; }

            PhysicalMachine targetMachine = layoutManager.GetMachine(targetMachineId);
            if (targetMachine == null) return;

            // Reserve the slot NOW — one authoritative position flows through the whole chain
            Vector3 dropoffSlotPos = targetMachine.ReserveIncomingSlot(jobId);

            Vector3 pickupPos = source != null
                ? source.GetPickupPositionForJob(jobId)
                : JobManager.GetJobTracker(jobId)?.WorldPosition ?? Vector3.zero;

            if (pickupPos == Vector3.zero)
                SimLogger.Error($"[SimBridge] Job {jobId} has zero pickup position — check WorldPosition init");

            JobManager.GetJobTracker(jobId)?.Visual?.SetState(JobLifecycleState.WaitingForTransport);
            agvPool.TryDispatch(jobId, pickupPos, dropoffSlotPos, source, targetMachine);
        }

        /// @brief Immediately halts the active episode and tears down all scene objects.
        /// @details Safe to call mid-episode. Notifies the SchedulingAgent via
        ///          EndEpisode() so ML-Agents doesn't get stuck waiting for an action.
        public void StopEpisode()
        {
            if (!episodeActive) return;

            episodeActive = false;
            IsWaitingForAction = false;
            pendingDecisions.Clear();
            pendingRoutingJobs.Clear();
            routingJobSources.Clear();
            exitedJobCount = 0;
            //  TaillardJson = null;

            if (layoutManager != null) layoutManager.ClearFloor();
            if (JobManager != null) JobManager.Cleanup();

            if (agent != null) agent.EndEpisode();

            SimLogger.Low("[SimBridge] Episode stopped by user.");
        }

        public int GetRuleIndex(DispatchingRule rule) => Array.IndexOf(ActionToRule, rule);
    }
}