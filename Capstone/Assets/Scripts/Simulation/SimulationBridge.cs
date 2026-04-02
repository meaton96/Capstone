using System;
using System.Linq;
using UnityEngine;
using UnityEngine.Events;
using Assets.Scripts.Scheduling.Core;
using Assets.Scripts.Scheduling.Data;
using Newtonsoft.Json;
using System.Collections.Generic;
using Assets.Scripts.Logging;
using Assets.Scripts.Simulation.Machines;
using Assets.Scripts.Simulation.AGV;

namespace Assets.Scripts.Simulation
{
    /// @brief Snapshot of the state presented to the agent when a scheduling decision is needed.
    [Serializable]
    public struct DecisionRequest
    {
        /// @brief ID of the machine waiting for a job to be dispatched.
        public int MachineId;

        /// @brief Current simulation time in seconds when the decision was raised.
        public double SimTime;

        /// @brief Job IDs currently queued at the machine.
        public int[] QueuedJobIds;

        /// @brief Processing durations (in sim-seconds) corresponding to each queued job.
        public double[] QueuedDurations;

        /// @brief Sequential index of this decision point across the episode.
        public int DecisionIndex;

        /// @brief Total number of jobs in the current instance.
        public int TotalJobs;

        /// @brief Number of jobs that have completed all operations so far.
        public int CompletedJobs;
    }

    /// @brief Result returned by @c SimulationBridge.Step() after applying a dispatching rule.
    [Serializable]
    public struct StepResult
    {
        /// @brief Reward signal for the agent based on elapsed makespan delta.
        public float Reward;

        /// @brief True when the episode has ended (all jobs complete).
        public bool Done;

        /// @brief The next decision context, if one is immediately available.
        public DecisionRequest NextDecision;

        /// @brief Makespan at the time this step was resolved.
        public double CurrentMakespan;

        /// @brief Total operations completed across all jobs at this step.
        public int OperationsCompleted;
    }

    /// @brief Summary statistics produced when an episode ends.
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

        /// @brief Percentage deviation of achieved makespan from the known optimum.
        public double OptimalityGap => OptimalMakespan > 0
            ? (Makespan - OptimalMakespan) / OptimalMakespan * 100.0
            : 0;
    }

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
        [SerializeField] private AGVPool agvPool;
        public JobManager JobManager;
        [SerializeField] private SchedulingAgent agent;

        /// @brief Singleton instance accessible from @c PhysicalMachine and @c JobManager.
        public static SimulationBridge Instance;

        /// @brief Wall-clock time when the current episode started.
        public float StartTime { get; private set; }

        [Header("Episode Configuration")]
        [SerializeField] private TextAsset taillardJsonDefault;
        public TextAsset TaillardJson { get; set; }
        [SerializeField] private bool autoStartOnPlay = false;

        [Header("Events")]
        public UnityEvent<DecisionRequest> OnDecisionRequired;
        public UnityEvent<StepResult> OnStepCompleted;
        public UnityEvent<EpisodeResult> OnEpisodeFinished;

        [Header("Logging")]
        [SerializeField] private LogLevel logLevel = LogLevel.Low;

        /// @brief Ordered mapping from discrete action index to @c DispatchingRule enum value.
        private static readonly DispatchingRule[] ActionToRule = new DispatchingRule[]
        {
            DispatchingRule.SPT_SMPT, DispatchingRule.SPT_SRWT,
            DispatchingRule.LPT_MMUR, DispatchingRule.SRT_SRWT,
            DispatchingRule.LPT_SMPT, DispatchingRule.LRT_MMUR,
            DispatchingRule.SRT_SMPT, DispatchingRule.SDT_SRWT,
        };

        /// @brief Number of discrete actions available to the agent.
        public static int ActionCount => ActionToRule.Length;

        private DESSimulator simulator;
        private TaillardInstance currentInstance;
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

        /// @brief Seconds elapsed since the episode started, derived from Unity wall time.
        public double SimTime => Time.time - StartTime;

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
            simulator = new DESSimulator();
            SimLogger.ActiveLevel = logLevel;
        }

        private void Start()
        {
            if (autoStartOnPlay && TaillardJson != null)
            {
                SimLogger.Medium("[Sim Bridge] Auto Starting Episode...");
                StartEpisode();
            }
        }

        // ─────────────────────────────────────────────────────────
        //  Episode Management
        // ─────────────────────────────────────────────────────────

        /// @brief Loads the configured Taillard instance, builds the factory floor,
        ///        spawns job visuals, and dispatches every job toward its first machine.
        public void StartEpisode()
        {
            if (TaillardJson == null)
            {
                TaillardJson = taillardJsonDefault;
            }

            currentInstance = LoadInstance(TaillardJson);
            if (currentInstance == null) return;

            SimLogger.Medium($"[Sim Bridge] Loaded : {currentInstance.name}");
            SimLogger.Medium($"[Sim Bridge] Loaded : {currentInstance.MachineCount} machines");
            SimLogger.Medium($"[Sim Bridge] Loaded : {currentInstance.JobCount} jobs");

            simulator.LoadInstance(currentInstance);

            if (layoutManager != null)
            {
                layoutManager.BuildFloor(simulator);
            }
            if (JobManager != null)
            {
                JobManager.Initialize(currentInstance, spawnVisuals: true);
            }
            if (agvPool != null)
            {
                agvPool.InitializeFleet();
            }

            episodeActive = true;
            decisionCount = 0;
            totalReward = 0;
            previousMakespan = 0;
            perMachineDecisions = new int[simulator.Machines.Length];
            IsWaitingForAction = false;
            pendingDecisions.Clear();

            StartTime = Time.time;

            SimLogger.High($"[SimBridge] Episode started: {currentInstance.Name}");

            for (int i = 0; i < currentInstance.JobCount; i++)
            {
                int firstMachineId = currentInstance.machines_matrix[i][0];
                PhysicalMachine target = layoutManager.GetMachine(firstMachineId);
                Vector3 pickupPos = JobManager.GetJobTracker(i)?.WorldPosition ?? Vector3.zero;
                Vector3 dropoffSlotPos = target.ReserveIncomingSlot(i);

                agvPool.TryDispatchStaggered(i, pickupPos, dropoffSlotPos, null, target, i);
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

            if (JobManager.AreAllJobsComplete())
            {
                FinaliseEpisode();
                return;
            }

            JobTracker tracker = JobManager.GetJobTracker(jobId);
            if (tracker != null && tracker.NextMachineId != -1)
            {
                PhysicalMachine finishedMachine = layoutManager.GetMachine(machineId);
                DispatchRealAGV(jobId, finishedMachine, tracker.NextMachineId);
            }

            CheckIfDecisionNeeded(machineId);
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
            LastAppliedRule = ActionToRule[actionIndex].ToString();
            int chosenJobId = ApplyDispatchingRule(actionIndex, CurrentDecision.MachineId);
            float duration = GetDurationFromTaillardData(chosenJobId, CurrentDecision.MachineId);

            PhysicalMachine machine = layoutManager.GetMachine(CurrentDecision.MachineId);
            machine.StartProcessing(chosenJobId, duration);
            JobManager.MarkOperationStarted(chosenJobId, SimTime);

            float stepReward = CalculateReward();
            totalReward += stepReward;
            perMachineDecisions[CurrentDecision.MachineId]++;

            StepResult result = new StepResult
            {
                Reward = stepReward,
                Done = false,
                CurrentMakespan = SimTime
            };

            OnStepCompleted?.Invoke(result);
            return result;
        }

        // ─────────────────────────────────────────────────────────
        //  Update Loop
        // ─────────────────────────────────────────────────────────

        /// @brief Drains the pending-decision queue one entry per frame while the
        ///        agent is free, firing @c OnDecisionRequired for the next machine.
        private void Update()
        {
            if (!episodeActive) return;
            if (IsWaitingForAction) return;

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
                durations[i] = GetDurationFromTaillardData(jobIds[i], machine.MachineId);
            }

            return new DecisionRequest
            {
                MachineId = machine.MachineId,
                SimTime = SimTime,
                QueuedJobIds = jobIds,
                QueuedDurations = durations,
                DecisionIndex = decisionCount++,
                TotalJobs = currentInstance.JobCount,
                CompletedJobs = 0
            };
        }

        /// @brief Looks up the processing duration for @p jobId on @p machineId
        ///        using the Taillard instance's duration matrix.
        /// @param jobId      Job whose duration is queried.
        /// @param machineId  Machine performing the operation.
        /// @return           Duration in simulation seconds, or 0 if not found.
        private float GetDurationFromTaillardData(int jobId, int machineId)
        {
            if (currentInstance == null) return 0f;

            int[] machineSequence = currentInstance.machines_matrix[jobId];
            for (int opIndex = 0; opIndex < machineSequence.Length; opIndex++)
            {
                if (machineSequence[opIndex] == machineId)
                {
                    return currentInstance.duration_matrix[jobId][opIndex];
                }
            }
            return 0f;
        }

        /// @brief Computes a per-step reward as the negative normalised makespan delta.
        /// @return Negative float in the range (-1, 0].
        private float CalculateReward()
        {
            float currentSimTime = (float)SimTime;
            float delta = currentSimTime - (float)previousMakespan;
            previousMakespan = currentSimTime;

            int totalOps = currentInstance.JobCount * currentInstance.MachineCount;
            return -delta / Math.Max(totalOps, 1);
        }

        /// @brief Selects the best job from a machine's physical queue using the
        ///        dispatching rule identified by @p actionIndex.
        /// @param actionIndex  Index into @c ActionToRule.
        /// @param machineId    Machine whose queue is evaluated.
        /// @return             ID of the job that should be processed next.
        private int ApplyDispatchingRule(int actionIndex, int machineId)
        {
            DispatchingRule rule = ActionToRule[actionIndex];
            PhysicalMachine machine = layoutManager.GetMachine(machineId);

            int bestJobId = machine.PhysicalQueue[0];
            float shortestDuration = float.MaxValue;

            foreach (int jobId in machine.PhysicalQueue)
            {
                float duration = GetDurationFromTaillardData(jobId, machineId);

                if (rule == DispatchingRule.SPT_SMPT || rule == DispatchingRule.SPT_SRWT)
                {
                    if (duration < shortestDuration)
                    {
                        shortestDuration = duration;
                        bestJobId = jobId;
                    }
                }
            }

            return bestJobId;
        }

        /// @brief Marks the episode as finished, computes final statistics, and
        ///        broadcasts an @c EpisodeResult via @c OnEpisodeFinished.
        private void FinaliseEpisode()
        {
            episodeActive = false;
            EpisodeResult result = new EpisodeResult
            {
                InstanceName = currentInstance?.Name ?? "unknown",
                RuleName = "agent",
                Makespan = SimTime,
                OptimalMakespan = currentInstance?.metadata.optimum ?? 0,
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

            TaillardJson = null;

            if (layoutManager != null) layoutManager.ClearFloor();
            if (JobManager != null) JobManager.Cleanup();

            if (agent != null) agent.EndEpisode();

            SimLogger.Low("[SimBridge] Episode stopped by user.");
        }

        /// @brief Deserialises a @c TaillardInstance from a JSON @c TextAsset.
        /// @param json  Unity @c TextAsset containing the serialised instance.
        /// @return      Parsed instance, or @c null on failure.
        private TaillardInstance LoadInstance(TextAsset json)
        {
            try
            {
                return JsonConvert.DeserializeObject<TaillardInstance>(json.text);
            }
            catch (Exception ex)
            {
                SimLogger.Error($"[SimBridge] Parse error: {ex}");
                return null;
            }
        }
    }
}