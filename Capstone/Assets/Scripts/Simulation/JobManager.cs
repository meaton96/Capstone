using System;
using System.Collections.Generic;
using UnityEngine;
using Assets.Scripts.Scheduling.Core;
using Assets.Scripts.Scheduling.Data;

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
        public double TotalWaitTime;
        public double TotalTransitTime;
        public float OperationProgress;
        public float[] OperationStatuses;
        public int[] OperationMachineIds;
        public float[] OperationDurations;
        public bool PhysicallyAtMachine;
        public int IncomingQueueSlot;
        public JobVisual Visual;
        public int TimeInCurrentState;
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

        private JobTracker[] trackers;
        private bool initialized;
        private Transform jobTokenParent;

        public JobTracker[] JobTrackers => trackers;
        public int JobCount => trackers?.Length ?? 0;
        public bool IsInitialized => initialized;

        // ─────────────────────────────────────────────────────────
        //  Initialization
        // ─────────────────────────────────────────────────────────

        public void Initialize(TaillardInstance instance, bool spawnVisuals = true)
        {
            int jobCount = instance.JobCount;
            Cleanup();

            if (spawnVisuals)
            {
                var parentGo = new GameObject("_JobTokens");
                jobTokenParent = parentGo.transform;
            }

            trackers = new JobTracker[jobCount];

            for (int j = 0; j < jobCount; j++)
            {
                int opCount = instance.MachineCount; // Taillard instances visit every machine

                var tracker = new JobTracker
                {
                    JobId = j,
                    TotalOperations = opCount,
                    State = JobLifecycleState.NotStarted,
                    CurrentOperationIndex = 0,
                    CompletedOperations = 0,
                    WorldPosition = GetIncomingQueuePosition(j),
                    CurrentMachineId = -1,
                    NextMachineId = instance.machines_matrix[j][0],
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
                    tracker.OperationMachineIds[o] = instance.machines_matrix[j][o];
                    tracker.OperationDurations[o] = (float)instance.duration_matrix[j][o];
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
            Debug.Log($"[JobManager] Initialized {jobCount} job trackers directly from JSON data.");
        }

        public void Cleanup()
        {
            if (jobTokenParent != null) Destroy(jobTokenParent.gameObject);
            trackers = null;
            initialized = false;
        }

        // ─────────────────────────────────────────────────────────
        //  Manual State Tracking (Driven by Physics)
        // ─────────────────────────────────────────────────────────

        public void MarkJobArrivedAtMachine(int jobId, int machineId)
        {
            JobTracker t = GetJobTracker(jobId);
            if (t == null) return;

            t.State = JobLifecycleState.Queued;
            t.CurrentMachineId = machineId;
            t.PhysicallyAtMachine = true;

            if (t.Visual != null) t.Visual.SetState(t.State);
        }

        public void MarkOperationStarted(int jobId, double simTime)
        {
            JobTracker t = GetJobTracker(jobId);
            if (t == null) return;

            t.State = JobLifecycleState.Processing;

            // Accurately log wait time from when it arrived until now
            t.TotalWaitTime += (simTime - t.StateEntryTime);
            t.StateEntryTime = simTime;

            t.OperationStatuses[t.CurrentOperationIndex] = 0.5f;

            if (t.Visual != null) t.Visual.SetState(t.State);
        }

        public void MarkOperationComplete(int jobId, double simTime)
        {
            if (!initialized) return;

            JobTracker t = trackers[jobId];
            if (t.CurrentOperationIndex >= t.TotalOperations) return;

            t.CompletedOperations++;
            t.OperationStatuses[t.CurrentOperationIndex] = 1.0f; // Mark as done
            t.StateEntryTime = simTime;

            // Check if entire job is complete
            if (t.CompletedOperations >= t.TotalOperations)
            {
                t.State = JobLifecycleState.Complete;
                t.CurrentMachineId = -1;
                t.NextMachineId = -1;
                t.OperationProgress = 0f;

                if (t.Visual != null)
                {
                    t.Visual.SetState(t.State);
                    t.Visual.SetTargetPosition(GetExitAreaPosition(t.IncomingQueueSlot));
                }
            }
            else
            {
                t.State = JobLifecycleState.WaitingForTransport;
                t.CurrentOperationIndex++;
                t.NextMachineId = t.OperationMachineIds[t.CurrentOperationIndex];
                t.CurrentMachineId = -1;
                t.OperationProgress = 0f;

                if (t.Visual != null) t.Visual.SetState(t.State);
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

        // ─────────────────────────────────────────────────────────
        //  Grid layout helpers
        // ─────────────────────────────────────────────────────────

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

        // ─────────────────────────────────────────────────────────
        //  Query API
        // ─────────────────────────────────────────────────────────

        public JobTracker GetJobTracker(int jobId)
        {
            if (trackers == null || jobId < 0 || jobId >= trackers.Length) return null;
            return trackers[jobId];
        }

        public float[] GetJobPositionsFlat()
        {
            if (!initialized) return null;
            float[] positions = new float[trackers.Length * 2];
            for (int j = 0; j < trackers.Length; j++)
            {
                // In physics mode, the Visual object holds the true physical location
                Vector3 pos = trackers[j].WorldPosition;
                if (trackers[j].Visual != null) pos = trackers[j].Visual.transform.position;

                positions[j * 2 + 0] = pos.x;
                positions[j * 2 + 1] = pos.z;
            }
            return positions;
        }

        // GetJobScalarsFlat and GetSchedulingMatrixFlat remain the same...
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
    }
}