using System;
using System.Linq;
using UnityEngine;
using UnityEngine.Events;
using Assets.Scripts.Scheduling.Core;
using Assets.Scripts.Scheduling.Data;
using Newtonsoft.Json;
using System.Collections.Generic;

namespace Assets.Scripts.Simulation
{
    [Serializable]
    public struct DecisionRequest
    {
        public int MachineId;
        public double SimTime;
        public int[] QueuedJobIds;
        public double[] QueuedDurations;
        public int DecisionIndex;
        public int TotalJobs;
        public int CompletedJobs;
    }

    [Serializable]
    public struct StepResult
    {
        public float Reward;
        public bool Done;
        public DecisionRequest NextDecision;
        public double CurrentMakespan;
        public int OperationsCompleted;
    }

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

        public double OptimalityGap => OptimalMakespan > 0
            ? (Makespan - OptimalMakespan) / OptimalMakespan * 100.0
            : 0;
    }

    public class SimulationBridge : MonoBehaviour
    {
        private Queue<int> pendingDecisions = new Queue<int>();
        [Header("Scene References")]
        [SerializeField] private FactoryLayoutManager layoutManager;
        public JobManager JobManager;

        public static SimulationBridge Instance;
        public float StartTime { get; private set; }

        [Header("Episode Configuration")]
        [SerializeField] private TextAsset taillardJson;
        public TextAsset TaillardJson { get; set; }
        [SerializeField] private bool autoStartOnPlay = true;

        [Header("Events")]
        public UnityEvent<DecisionRequest> OnDecisionRequired;
        public UnityEvent<StepResult> OnStepCompleted;
        public UnityEvent<EpisodeResult> OnEpisodeFinished;

        private static readonly DispatchingRule[] ActionToRule = new DispatchingRule[]
        {
            DispatchingRule.SPT_SMPT, DispatchingRule.SPT_SRWT,
            DispatchingRule.LPT_MMUR, DispatchingRule.SRT_SRWT,
            DispatchingRule.LPT_SMPT, DispatchingRule.LRT_MMUR,
            DispatchingRule.SRT_SMPT, DispatchingRule.SDT_SRWT,
        };

        public static int ActionCount => ActionToRule.Length;

        private DESSimulator simulator;
        private TaillardInstance currentInstance;
        private bool episodeActive;
        private int decisionCount;
        private double totalReward;
        private double previousMakespan;
        private int[] perMachineDecisions;

        public bool IsEpisodeActive => episodeActive;
        public bool IsDone => !episodeActive;
        public DecisionRequest CurrentDecision { get; private set; }

        // In the new system, we are waiting for action anytime the episode is active 
        // and we have broadcasted a DecisionRequest.
        public bool IsWaitingForAction { get; private set; }

        public double SimTime => Time.time - StartTime;

        private void Awake()
        {
            if (Instance != null)
            {
                Destroy(this);
                return;
            }
            Instance = this;
            simulator = new DESSimulator();
        }

        private void Start()
        {
            if (autoStartOnPlay && taillardJson != null)
            {
                StartEpisode();
            }
        }

        public void StartEpisode()
        {
            if (taillardJson == null) return;

            currentInstance = LoadInstance(taillardJson);
            if (currentInstance == null) return;

            simulator.LoadInstance(currentInstance);

            if (layoutManager != null)
            {
                layoutManager.BuildFloor(simulator);
            }
            if (JobManager != null)
            {
                JobManager.Initialize(currentInstance, spawnVisuals: true);
            }

            episodeActive = true;
            decisionCount = 0;
            totalReward = 0;
            previousMakespan = 0;
            perMachineDecisions = new int[simulator.Machines.Length];
            IsWaitingForAction = false;
            pendingDecisions.Clear();

            StartTime = Time.time;

            Debug.Log($"[SimBridge] Episode started: {currentInstance.Name}");
            for (int i = 0; i < currentInstance.JobCount; i++)
            {
                int firstMachineId = currentInstance.machines_matrix[i][0];
                DispatchGhostAGV(i, firstMachineId);
            }
        }

        // ─────────────────────────────────────────────────────────
        //  Physics Event Listeners
        // ─────────────────────────────────────────────────────────

        public void OnJobArrivedInQueue(int machineId, int jobId)
        {
            JobManager.MarkJobArrivedAtMachine(jobId, machineId);
            CheckIfDecisionNeeded(machineId);
        }

        public void OnMachineFinished(int machineId, int jobId)
        {
            // Update tracking
            JobManager.MarkOperationComplete(jobId, SimTime);


            // Check for Win State
            if (JobManager.AreAllJobsComplete())
            {
                FinaliseEpisode();
                return;
            }

            // TODO: In the future, this is where you will dispatch an AGV to pick up `jobId`
            JobTracker tracker = JobManager.GetJobTracker(jobId);
            if (tracker != null && tracker.NextMachineId != -1)
            {
                DispatchGhostAGV(jobId, tracker.NextMachineId);
            }
            // Check if the machine we just freed up has other jobs waiting
            CheckIfDecisionNeeded(machineId);
        }

        private void CheckIfDecisionNeeded(int machineId)
        {
            PhysicalMachine machine = layoutManager.GetMachine(machineId);

            if (machine != null && machine.IsIdle && machine.PhysicalQueue.Count > 0)
            {
                // Only add it if it's not already standing in line
                if (!pendingDecisions.Contains(machineId))
                {
                    pendingDecisions.Enqueue(machineId);
                }
            }
        }

        // ─────────────────────────────────────────────────────────
        //  The Core Step
        // ─────────────────────────────────────────────────────────

        public StepResult Step(int actionIndex)
        {
            IsWaitingForAction = false;
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
        //  Helpers
        // ─────────────────────────────────────────────────────────

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
                CompletedJobs = 0 // Update if you track global completion differently later
            };
        }

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

        private float CalculateReward()
        {
            float currentSimTime = (float)SimTime;
            float delta = currentSimTime - (float)previousMakespan;
            previousMakespan = currentSimTime;

            int totalOps = currentInstance.JobCount * currentInstance.MachineCount;
            return -delta / Math.Max(totalOps, 1);
        }

        private int ApplyDispatchingRule(int actionIndex, int machineId)
        {
            DispatchingRule rule = ActionToRule[actionIndex];
            PhysicalMachine machine = layoutManager.GetMachine(machineId);

            int bestJobId = machine.PhysicalQueue[0];
            float shortestDuration = float.MaxValue;

            foreach (int jobId in machine.PhysicalQueue)
            {
                float duration = GetDurationFromTaillardData(jobId, machineId);

                // Temp SPT Logic for testing compilation
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

            Debug.Log($"[SimBridge] Episode complete: makespan={result.Makespan:F1}, decisions={result.DecisionPoints}");
            OnEpisodeFinished?.Invoke(result);
        }
        private void Update()
        {
            if (!episodeActive) return;

            // If the ML-Agent is currently thinking about a decision, pause and wait.
            if (IsWaitingForAction) return;

            // If the Agent is free, and we have machines waiting for instructions...
            while (pendingDecisions.Count > 0)
            {
                int nextMachineId = pendingDecisions.Dequeue();
                PhysicalMachine machine = layoutManager.GetMachine(nextMachineId);

                // Double check the machine didn't already start processing somehow
                if (machine != null && machine.IsIdle && machine.PhysicalQueue.Count > 0)
                {
                    CurrentDecision = BuildDecisionRequest(machine);
                    IsWaitingForAction = true;

                    // Wake up the Agent!
                    OnDecisionRequired?.Invoke(CurrentDecision);

                    // Break out of the loop so we don't ask the Agent for 
                    // another decision until it finishes this one.
                    return;
                }
            }
        }

        private void DispatchGhostAGV(int jobId, int targetMachineId)
        {
            PhysicalMachine targetMachine = layoutManager.GetMachine(targetMachineId);
            JobTracker tracker = JobManager.GetJobTracker(jobId);

            if (targetMachine != null && tracker != null && tracker.Visual != null)
            {
                // 1. Visually set the job to "In Transit" (Blue color)
                tracker.Visual.SetState(JobLifecycleState.InTransit);

                // 2. Set the destination to the machine. 
                // The JobVisual's Update() loop will smoothly glide it across the floor!
                tracker.Visual.SetTargetPosition(targetMachine.transform.position);

                Debug.Log($"[Ghost AGV] Dispatched Job {jobId} to Machine {targetMachineId}");
            }
        }

        private TaillardInstance LoadInstance(TextAsset json)
        {
            try
            {
                return JsonConvert.DeserializeObject<TaillardInstance>(json.text);
            }
            catch (Exception ex)
            {
                Debug.LogError($"[SimBridge] Parse error: {ex}");
                return null;
            }
        }
    }
}