using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using Assets.Scripts.Scheduling.Core;

namespace Assets.Scripts.Simulation
{
    public enum JobLifecycleState
    {
        NotStarted,
        Queued,
        Processing,
        WaitingForTransport,
        InTransit,
        Complete,
    }

    [Serializable]
    public class JobTracker
    {
        public int JobId;
        public int TotalOperations;
        public JobLifecycleState State;
        public int CurrentOperationIndex;
        public int CompletedOperations;
        public Vector3 WorldPosition;
        public int CurrentMachineId;
        public int NextMachineId;
        public double StateEntryTime;
        public double TimeInCurrentState;
        public double TotalWaitTime;
        public double TotalTransitTime;
        public float OperationProgress;
        public float[] OperationStatuses;
        public int[] OperationMachineIds;
        public float[] OperationDurations;
        public bool PhysicallyAtMachine;
        public int IncomingQueueSlot;
        public JobVisual Visual;
    }

    public class JobManager : MonoBehaviour
    {
        [Header("References")]
        [SerializeField] private FactoryLayoutManager layoutManager;

        [Header("Visual")]
        [SerializeField] private GameObject jobVisualPrefab;
        [SerializeField] private float jobTokenHeight = 1.5f;
        [SerializeField] private float queueSpacing = 0.6f;

        [Header("Incoming Queue Layout")]
        [SerializeField] private Transform incomingQueueMarker;
        [SerializeField] private Vector3 incomingQueueOrigin = new Vector3(-5f, 0f, 0f);
        [SerializeField] private Vector3 queueRowDirection = Vector3.right;
        [SerializeField] private Vector3 queueColumnDirection = Vector3.forward;
        [SerializeField] private float queueGridSpacing = 1.0f;
        [SerializeField] private int queueRowSize = 10;

        [Header("Exit Area")]
        [SerializeField] private Transform exitAreaMarker;
        [SerializeField] private Vector3 exitAreaOrigin = new Vector3(-5f, 0f, 10f);

        [Header("Machine Queue")]
        [SerializeField] private Vector3 machineQueueDirection = Vector3.back;

        private JobTracker[] trackers;
        private DESSimulator simulator;
        private bool initialized;
        private Transform jobTokenParent;

        public JobTracker[] JobTrackers => trackers;
        public int JobCount => trackers?.Length ?? 0;
        public bool IsInitialized => initialized;

        public void Initialize(DESSimulator sim, bool spawnVisuals = true)
        {
            simulator = sim;
            int jobCount = sim.Jobs.Length;
            Cleanup();

            if (spawnVisuals)
            {
                var parentGo = new GameObject("_JobTokens");
                jobTokenParent = parentGo.transform;
            }

            trackers = new JobTracker[jobCount];

            for (int j = 0; j < jobCount; j++)
            {
                Job job = sim.Jobs[j];
                int opCount = job.Operations.Length;

                var tracker = new JobTracker
                {
                    JobId = j,
                    TotalOperations = opCount,
                    State = JobLifecycleState.NotStarted,
                    CurrentOperationIndex = 0,
                    CompletedOperations = 0,
                    WorldPosition = GetIncomingQueuePosition(j),
                    CurrentMachineId = -1,
                    NextMachineId = opCount > 0 ? job.Operations[0].MachineId : -1,
                    StateEntryTime = 0,
                    TimeInCurrentState = 0,
                    TotalWaitTime = 0,
                    TotalTransitTime = 0,
                    OperationProgress = 0f,
                    OperationStatuses = new float[opCount],
                    OperationMachineIds = new int[opCount],
                    OperationDurations = new float[opCount],
                    PhysicallyAtMachine = false,
                    IncomingQueueSlot = j,
                };

                for (int o = 0; o < opCount; o++)
                {
                    tracker.OperationMachineIds[o] = job.Operations[o].MachineId;
                    tracker.OperationDurations[o] = (float)job.Operations[o].Duration;
                    tracker.OperationStatuses[o] = 0f;
                }

                if (spawnVisuals && jobVisualPrefab != null)
                {
                    Vector3 spawnPos = GetIncomingQueuePosition(j);
                    GameObject tokenGo = Instantiate(jobVisualPrefab, spawnPos, Quaternion.identity, jobTokenParent);
                    tokenGo.name = $"Job_{j}";

                    JobVisual visual = tokenGo.GetComponent<JobVisual>();
                    if (visual == null) visual = tokenGo.AddComponent<JobVisual>();

                    visual.Initialize(j, opCount);
                    tracker.Visual = visual;
                }
                trackers[j] = tracker;
            }
            initialized = true;
        }

        public void SyncFromDES(double currentSimTime)
        {
            if (!initialized || simulator?.Jobs == null) return;

            for (int j = 0; j < trackers.Length; j++)
            {
                JobTracker tracker = trackers[j];
                Job job = simulator.Jobs[j];
                JobLifecycleState previousState = tracker.State;
                UpdateTrackerState(tracker, job, currentSimTime);

                if (tracker.State != previousState)
                {
                    double elapsed = currentSimTime - tracker.StateEntryTime;
                    if (previousState == JobLifecycleState.Queued)
                        tracker.TotalWaitTime += elapsed;
                    else if (previousState == JobLifecycleState.InTransit)
                        tracker.TotalTransitTime += elapsed;

                    tracker.StateEntryTime = currentSimTime;
                }

                tracker.TimeInCurrentState = currentSimTime - tracker.StateEntryTime;
                UpdateTrackerPosition(tracker);

                if (tracker.Visual != null)
                {
                    tracker.Visual.SetState(tracker.State);
                    tracker.Visual.SetTargetPosition(tracker.WorldPosition);
                    tracker.Visual.SetProgress(tracker.OperationProgress);
                }
            }
        }

        public void Cleanup()
        {
            if (jobTokenParent != null) Destroy(jobTokenParent.gameObject);
            trackers = null;
            initialized = false;
        }

        private void UpdateTrackerState(JobTracker tracker, Job job, double currentSimTime)
        {
            int completedOps = 0;
            int activeOpIndex = -1;
            Operation activeOp = null;

            for (int o = 0; o < job.Operations.Length; o++)
            {
                Operation op = job.Operations[o];
                if (op.EndTime > 0)
                {
                    completedOps++;
                    tracker.OperationStatuses[o] = 1.0f;
                }
                else if (op.StartTime > 0)
                {
                    activeOpIndex = o;
                    activeOp = op;
                    tracker.OperationStatuses[o] = 0.5f;
                }
                else
                {
                    tracker.OperationStatuses[o] = 0f;
                }
            }

            tracker.CompletedOperations = completedOps;

            if (completedOps == job.Operations.Length)
            {
                tracker.State = JobLifecycleState.Complete;
                tracker.CurrentOperationIndex = job.Operations.Length;
                tracker.CurrentMachineId = -1;
                tracker.NextMachineId = -1;
                tracker.OperationProgress = 0f;
                tracker.PhysicallyAtMachine = false;
            }
            else if (activeOp != null)
            {
                tracker.State = JobLifecycleState.Processing;
                tracker.CurrentOperationIndex = activeOpIndex;
                tracker.CurrentMachineId = activeOp.MachineId;
                tracker.OperationProgress = ComputeProgress(activeOp, currentSimTime);
                tracker.PhysicallyAtMachine = true;
                int nextIdx = activeOpIndex + 1;
                tracker.NextMachineId = nextIdx < job.Operations.Length ? job.Operations[nextIdx].MachineId : -1;
            }
            else if (completedOps > 0)
            {
                tracker.CurrentOperationIndex = completedOps;
                tracker.OperationProgress = 0f;
                int nextMachineId = job.Operations[completedOps].MachineId;
                Machine nextMachine = simulator.Machines[nextMachineId];
                bool inDESQueue = nextMachine.WaitingQueue.Any(op => op.JobId == tracker.JobId);

                if (inDESQueue)
                {
                    tracker.State = JobLifecycleState.Queued;
                    tracker.CurrentMachineId = nextMachineId;
                    tracker.PhysicallyAtMachine = true;
                }
                else
                {
                    tracker.State = JobLifecycleState.WaitingForTransport;
                    tracker.CurrentMachineId = -1;
                    tracker.PhysicallyAtMachine = false;
                }
                tracker.NextMachineId = nextMachineId;
            }
            else
            {
                int firstMachineId = job.Operations[0].MachineId;
                Machine firstMachine = simulator.Machines[firstMachineId];
                bool inFirstQueue = firstMachine.WaitingQueue.Any(op => op.JobId == tracker.JobId);

                if (inFirstQueue)
                {
                    tracker.State = JobLifecycleState.Queued;
                    tracker.CurrentMachineId = firstMachineId;
                    tracker.PhysicallyAtMachine = true;
                }
                else
                {
                    tracker.State = JobLifecycleState.NotStarted;
                    tracker.CurrentMachineId = -1;
                    tracker.PhysicallyAtMachine = false;
                }

                tracker.NextMachineId = firstMachineId;
                tracker.CurrentOperationIndex = 0;
                tracker.OperationProgress = 0f;
            }
        }

        private float ComputeProgress(Operation op, double currentSimTime)
        {
            if (op.Duration <= 0) return 1f;
            double elapsed = currentSimTime - op.StartTime;
            return Mathf.Clamp01((float)(elapsed / op.Duration));
        }

        private void UpdateTrackerPosition(JobTracker tracker)
        {
            switch (tracker.State)
            {
                case JobLifecycleState.NotStarted:
                case JobLifecycleState.WaitingForTransport:
                    tracker.WorldPosition = GetIncomingQueuePosition(tracker.IncomingQueueSlot);
                    break;

                case JobLifecycleState.Complete:
                    tracker.WorldPosition = GetExitAreaPosition(tracker.IncomingQueueSlot);
                    break;

                case JobLifecycleState.Processing:
                    if (tracker.CurrentMachineId >= 0 && layoutManager != null)
                    {
                        // FIXED: Uses GetMachine() to get the PhysicalMachine
                        PhysicalMachine pm = layoutManager.GetMachine(tracker.CurrentMachineId);
                        if (pm != null)
                        {
                            tracker.WorldPosition = pm.transform.position + Vector3.up * jobTokenHeight;
                        }
                    }
                    break;

                case JobLifecycleState.Queued:
                    if (tracker.CurrentMachineId >= 0 && layoutManager != null)
                    {
                        // FIXED: Uses GetMachine() to get the PhysicalMachine
                        PhysicalMachine pm = layoutManager.GetMachine(tracker.CurrentMachineId);
                        if (pm != null)
                        {
                            float offset = GetMachineQueueSlotOffset(tracker.CurrentMachineId, tracker.JobId);
                            tracker.WorldPosition = pm.transform.position +
                                Vector3.up * jobTokenHeight +
                                machineQueueDirection.normalized * (offset + queueSpacing);
                        }
                    }
                    else
                    {
                        tracker.WorldPosition = GetIncomingQueuePosition(tracker.IncomingQueueSlot);
                    }
                    break;

                case JobLifecycleState.InTransit:
                    tracker.WorldPosition = GetIncomingQueuePosition(tracker.IncomingQueueSlot);
                    break;
            }
        }

        private Vector3 GetIncomingQueuePosition(int slot)
        {
            Vector3 origin = incomingQueueMarker != null ? incomingQueueMarker.position : incomingQueueOrigin;
            int row = slot / Mathf.Max(queueRowSize, 1);
            int col = slot % Mathf.Max(queueRowSize, 1);
            Vector3 rowDir = queueRowDirection.normalized;
            Vector3 colDir = queueColumnDirection.normalized;
            return origin + rowDir * (col * queueGridSpacing) + colDir * (row * queueGridSpacing) + Vector3.up * jobTokenHeight;
        }

        private Vector3 GetExitAreaPosition(int slot)
        {
            Vector3 origin = exitAreaMarker != null ? exitAreaMarker.position : exitAreaOrigin;
            int row = slot / Mathf.Max(queueRowSize, 1);
            int col = slot % Mathf.Max(queueRowSize, 1);
            Vector3 rowDir = queueRowDirection.normalized;
            Vector3 colDir = queueColumnDirection.normalized;
            return origin + rowDir * (col * queueGridSpacing) + colDir * (row * queueGridSpacing) + Vector3.up * jobTokenHeight;
        }

        private float GetMachineQueueSlotOffset(int machineId, int jobId)
        {
            Machine machine = simulator.Machines[machineId];
            int slotIndex = 0;
            foreach (Operation op in machine.WaitingQueue)
            {
                if (op.JobId == jobId) break;
                slotIndex++;
            }
            return slotIndex * queueSpacing;
        }

        // ─────────────────────────────────────────────────────────
        //  Query API
        // ─────────────────────────────────────────────────────────

        public JobTracker GetJobTracker(int jobId)
        {
            if (trackers == null || jobId < 0 || jobId >= trackers.Length) return null;
            return trackers[jobId];
        }

        public Vector3 GetJobPosition(int jobId) => GetJobTracker(jobId)?.WorldPosition ?? Vector3.zero;
        public JobLifecycleState GetJobState(int jobId) => GetJobTracker(jobId)?.State ?? JobLifecycleState.NotStarted;
        public float GetOperationProgress(int jobId) => GetJobTracker(jobId)?.OperationProgress ?? 0f;
        public double GetWaitTime(int jobId) => GetJobTracker(jobId)?.TotalWaitTime ?? 0;
        public double GetTransitTime(int jobId) => GetJobTracker(jobId)?.TotalTransitTime ?? 0;

        public float[] GetSchedulingMatrixFlat(int numMachines)
        {
            if (!initialized) return null;
            int numJobs = trackers.Length;
            float[] matrix = new float[numJobs * numMachines * 3];
            float maxDuration = 1f;
            foreach (JobTracker t in trackers)
                foreach (float d in t.OperationDurations)
                    if (d > maxDuration) maxDuration = d;

            for (int j = 0; j < numJobs; j++)
            {
                JobTracker t = trackers[j];
                for (int o = 0; o < t.TotalOperations; o++)
                {
                    int m = t.OperationMachineIds[o];
                    if (m < 0 || m >= numMachines) continue;
                    int baseIdx = (j * numMachines + m) * 3;
                    matrix[baseIdx + 0] = 1.0f;
                    matrix[baseIdx + 1] = t.OperationDurations[o] / maxDuration;
                    matrix[baseIdx + 2] = t.OperationStatuses[o];
                }
            }
            return matrix;
        }

        public float[] GetJobPositionsFlat()
        {
            if (!initialized) return null;
            float[] positions = new float[trackers.Length * 2];
            for (int j = 0; j < trackers.Length; j++)
            {
                positions[j * 2 + 0] = trackers[j].WorldPosition.x;
                positions[j * 2 + 1] = trackers[j].WorldPosition.z;
            }
            return positions;
        }

        public float[] GetJobScalarsFlat(double currentSimTime)
        {
            if (!initialized) return null;
            float[] scalars = new float[trackers.Length * 4];
            double timeNorm = Math.Max(currentSimTime, 1.0);
            for (int j = 0; j < trackers.Length; j++)
            {
                JobTracker t = trackers[j];
                int idx = j * 4;
                scalars[idx + 0] = t.TotalOperations > 0 ? (float)t.CompletedOperations / t.TotalOperations : 0f;
                scalars[idx + 1] = t.OperationProgress;
                scalars[idx + 2] = (float)(t.TotalWaitTime / timeNorm);
                scalars[idx + 3] = StateToFloat(t.State);
            }
            return scalars;
        }

        private static float StateToFloat(JobLifecycleState state)
        {
            return state switch
            {
                JobLifecycleState.NotStarted => 0.0f,
                JobLifecycleState.Queued => 0.2f,
                JobLifecycleState.Processing => 0.4f,
                JobLifecycleState.WaitingForTransport => 0.6f,
                JobLifecycleState.InTransit => 0.8f,
                JobLifecycleState.Complete => 1.0f,
                _ => 0.0f,
            };
        }

        public void BeginTransit(int jobId, int destinationMachineId, double simTime)
        {
            JobTracker t = GetJobTracker(jobId);
            if (t == null) return;
            t.State = JobLifecycleState.InTransit;
            t.NextMachineId = destinationMachineId;
            t.StateEntryTime = simTime;
        }

        public void CompleteTransit(int jobId, int machineId, double simTime)
        {
            JobTracker t = GetJobTracker(jobId);
            if (t == null) return;
            double transitDuration = simTime - t.StateEntryTime;
            t.TotalTransitTime += transitDuration;
            t.State = JobLifecycleState.Queued;
            t.CurrentMachineId = machineId;
            t.StateEntryTime = simTime;
        }

        // ─────────────────────────────────────────────────────────
        //  NEW: Manual tracking for Unity Physics Clock updates
        // ─────────────────────────────────────────────────────────

        public void MarkOperationComplete(int jobId)
        {
            if (!initialized) return;

            JobTracker t = trackers[jobId];
            if (t.CurrentOperationIndex >= t.TotalOperations) return;

            t.CompletedOperations++;
            t.OperationStatuses[t.CurrentOperationIndex] = 1.0f; // Mark as done

            // Check if entire job is complete
            if (t.CompletedOperations >= t.TotalOperations)
            {
                t.State = JobLifecycleState.Complete;
                t.CurrentMachineId = -1;
                t.NextMachineId = -1;
                t.OperationProgress = 0f;
            }
            else
            {
                t.State = JobLifecycleState.WaitingForTransport;
                t.CurrentOperationIndex++;
                t.NextMachineId = t.OperationMachineIds[t.CurrentOperationIndex];
                t.CurrentMachineId = -1;
                t.OperationProgress = 0f;
            }

            // Immediately force a visual update
            if (t.Visual != null)
            {
                t.Visual.SetState(t.State);
            }
        }

        public bool AreAllJobsComplete()
        {
            if (!initialized) return false;

            foreach (var t in trackers)
            {
                if (t.State != JobLifecycleState.Complete) return false;
            }
            return true;
        }
    }
}