using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Assets.Scripts.Scheduling.Core;
using Assets.Scripts.Scheduling.Data;
using Assets.Scripts.Simulation.FactoryLayout;
using Assets.Scripts.Logging;

namespace Assets.Scripts.Simulation.Jobs
{

    /// @brief Creates, tracks, and updates all jobs in a simulation episode.
    /// @details Authoritative record of job states. Transitions are driven by 
    /// SimulationBridge events to maintain synchronization with the scheduler.
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

        private Queue<int> pendingIncomingJobs = new Queue<int>();
        private JobTracker[] trackers;
        private bool initialized;
        private Transform jobTokenParent;

        public JobTracker[] JobTrackers => trackers;
        public int JobCount => trackers?.Length ?? 0;
        public bool IsInitialized => initialized;

        /// @brief Allocates trackers and optionally spawns visual tokens for every job.
        ///
        /// @details Maps the Taillard problem matrices into individual trackers. 
        /// Visuals are instantiated and deactivated until they are called to the incoming belt.
        ///
        /// @param instance The loaded Taillard problem instance.
        /// @param spawnVisuals When true, a JobVisual prefab is instantiated per job.
        ///
        /// @post trackers array is populated and initialized is set to true.
        public void Initialize(FJSSPJobDefinition[] jobDefs, bool spawnVisuals = true)
        {
            Cleanup();

            if (spawnVisuals)
            {
                var parentGo = new GameObject("_JobTokens");
                jobTokenParent = parentGo.transform;
            }

            trackers = new JobTracker[jobDefs.Length];

            for (int j = 0; j < jobDefs.Length; j++)
            {
                FJSSPJobDefinition def = jobDefs[j];
                int opCount = def.OperationSequence.Length;

                var tracker = new JobTracker
                {
                    JobId = def.JobId,
                    TotalOperations = opCount,
                    State = JobLifecycleState.NotStarted,
                    CurrentOperationIndex = 0,
                    CompletedOperations = 0,
                    WorldPosition = GetIncomingQueuePosition(j),
                    CurrentMachineId = -1,
                    NextMachineId = -1,           // unresolved — agent decides
                    NextMachineType = def.OperationSequence[0],
                    ArrivalTime = def.ArrivalTime,
                    StateEntryTime = 0,
                    TotalWaitTime = 0,
                    TotalTransitTime = 0,
                    OperationProgress = 0f,
                    OperationTypes = def.OperationSequence,
                    EligibleMachinesPerOp = def.EligibleMachinesPerOp,
                    OperationStatuses = new float[opCount],
                    PhysicallyAtMachine = false,
                    IncomingQueueSlot = j,
                };

                if (spawnVisuals && jobVisualPrefab != null)
                {
                    Vector3 spawnPos = GetIncomingQueuePosition(j);
                    GameObject tokenGo = Instantiate(jobVisualPrefab, spawnPos, Quaternion.identity, jobTokenParent);
                    tokenGo.name = $"Job_{j}";
                    tokenGo.SetActive(false);

                    JobVisual visual = tokenGo.GetComponent<JobVisual>();
                    if (visual == null) visual = tokenGo.AddComponent<JobVisual>();
                    visual.Initialize(j, opCount);
                    tracker.Visual = visual;
                }

                trackers[j] = tracker;
                pendingIncomingJobs.Enqueue(j);
            }

            initialized = true;
            Debug.Log($"[JobManager] Initialized {jobDefs.Length} FJSSP job trackers.");
        }

        /// @brief Logic loop managing the physical flow of jobs onto and off the factory floor.
        /// @details Drives the entrance of jobs into the factory via the IncomingBelt and 
        /// deactivates jobs that reach the final OutgoingBelt destination.
        private void Update()
        {
            if (!initialized || layoutManager == null) return;

            if (layoutManager.IncomingBelt != null)
            {
                while (pendingIncomingJobs.Count > 0 && !layoutManager.IncomingBelt.IsFull)
                {
                    int nextJobId = pendingIncomingJobs.Peek();
                    JobTracker tracker = trackers[nextJobId];

                    if (tracker.Visual != null) tracker.Visual.gameObject.SetActive(true);

                    if (layoutManager.IncomingBelt.TryEnqueue(nextJobId, tracker.Visual))
                    {
                        pendingIncomingJobs.Dequeue();
                    }
                }
            }

            if (layoutManager.OutgoingBelt != null && layoutManager.OutgoingBelt.Count > 0)
            {
                JobVisual frontVisual = layoutManager.OutgoingBelt.PeekFrontVisual();
                if (frontVisual != null)
                {
                    float dist = Vector3.Distance(frontVisual.transform.position, layoutManager.OutgoingBelt.OutputEndPosition);
                    if (dist < 0.05f)
                    {
                        var (jobId, vis) = layoutManager.OutgoingBelt.DequeueFront();
                        vis.gameObject.SetActive(false);
                        SimLogger.Low($"[JobManager] Job {jobId} removed at exit.");
                    }
                }
            }
        }

        /// @brief Wipes tracking data and destroys physical job tokens.
        /// @post initialized is false and trackers is null.
        public void Cleanup()
        {
            if (jobTokenParent != null) Destroy(jobTokenParent.gameObject);
            trackers = null;
            initialized = false;
        }

        /// @brief Updates state to @c Queued when a job arrives at its destination machine.
        /// @param jobId The arriving job.
        /// @param machineId The machine the job has reached.
        public void MarkJobArrivedAtMachine(int jobId, int machineId)
        {
            JobTracker t = GetJobTracker(jobId);
            if (t == null) return;

            t.State = JobLifecycleState.Queued;
            t.CurrentMachineId = machineId;
            t.PhysicallyAtMachine = true;
            if (t.Visual != null) t.Visual.SetState(t.State);
        }

        /// @brief Sets job to @c Processing and calculates the preceding wait duration.
        /// @param jobId The job starting its operation.
        /// @param simTime Current simulation time.
        public void MarkOperationStarted(int jobId, double simTime)
        {
            JobTracker t = GetJobTracker(jobId);
            if (t == null) return;

            t.State = JobLifecycleState.Processing;
            t.TotalWaitTime += (simTime - t.StateEntryTime);
            t.StateEntryTime = simTime;
            t.OperationStatuses[t.CurrentOperationIndex] = 0.5f;
            if (t.Visual != null) t.Visual.SetState(t.State);
        }

        /// @brief Assigns a job to an AGV and transitions state to @c InTransit.
        /// @param jobId The job being moved.
        /// @param destinationMachineId Target machine for the transfer.
        /// @param simTime Current simulation time.
        public void BeginTransit(int jobId, int destinationMachineId, double simTime)
        {
            JobTracker t = GetJobTracker(jobId);
            if (t == null) return;

            t.State = JobLifecycleState.InTransit;
            t.NextMachineId = destinationMachineId;
            t.TotalWaitTime += (simTime - t.StateEntryTime);
            t.StateEntryTime = simTime;
            if (t.Visual != null) t.Visual.SetState(t.State);
        }

        /// @brief Records the end of a transport leg and updates travel statistics.
        /// @param jobId The delivered job.
        /// @param machineId Target destination.
        /// @param simTime Current simulation time.
        public void CompleteTransit(int jobId, int machineId, double simTime)
        {
            JobTracker t = GetJobTracker(jobId);
            if (t == null) return;

            t.TotalTransitTime += (simTime - t.StateEntryTime);
            t.NextMachineId = machineId;
            t.StateEntryTime = simTime;
        }

        /// @brief Finalizes an operation and determines if the job is finished or needs more routing.
        ///
        /// @details If the job has completed all required machine operations, it is marked 
        /// as @c Complete and moved to the exit area. Otherwise, it is set to @c WaitingForTransport.
        ///
        /// @param jobId The finished job.
        /// @param simTime Current simulation time.
        ///
        /// @post CurrentOperationIndex is incremented if the job is not yet finished.
        public void MarkOperationComplete(int jobId, double simTime)
        {
            if (!initialized) return;
            JobTracker t = trackers[jobId];
            if (t.CurrentOperationIndex >= t.TotalOperations) return;

            t.CompletedOperations++;
            t.OperationStatuses[t.CurrentOperationIndex] = 1.0f;
            t.StateEntryTime = simTime;

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
                t.CurrentOperationIndex++;
                t.NextMachineId = -1;                   // unresolved
                t.NextMachineType = t.OperationTypes[t.CurrentOperationIndex];
                t.CurrentMachineId = -1;
                t.OperationProgress = 0f;
                t.State = JobLifecycleState.WaitingForTransport;
                if (t.Visual != null) t.Visual.SetState(t.State);
            }
        }
        public float GetProcessingTime(int jobId, int machineId)
        {
            JobTracker t = GetJobTracker(jobId);
            if (t == null) return 0f;
            int opIdx = t.CurrentOperationIndex;
            if (t.EligibleMachinesPerOp[opIdx].TryGetValue(machineId, out float time))
                return time;
            return 0f;
        }

        /// @brief Checks if the entire job set for the episode has reached completion.
        /// @return True if every job is in the Complete state.
        public bool AreAllJobsComplete()
        {
            if (!initialized) return false;
            foreach (var t in trackers) if (t.State != JobLifecycleState.Complete) return false;
            return true;
        }

        /// @brief Calculates the grid-based world position for the starting queue.
        /// @param slot The unique slot index for the job.
        private Vector3 GetIncomingQueuePosition(int slot)
        {
            Vector3 origin = layoutManager != null ? layoutManager.IncomingBeltPosition : incomingQueueOrigin;
            int row = slot / Mathf.Max(queueRowSize, 1);
            int col = slot % Mathf.Max(queueRowSize, 1);
            return origin + queueRowDirection.normalized * (col * queueGridSpacing) +
                   queueColumnDirection.normalized * (row * queueGridSpacing) + Vector3.up * jobTokenHeight;
        }

        /// @brief Calculates the world position for jobs that have finished all operations.
        private Vector3 GetExitAreaPosition(int slot)
        {
            Vector3 origin = layoutManager != null ? layoutManager.OutgoingBeltPosition : exitAreaOrigin;
            int row = slot / Mathf.Max(queueRowSize, 1);
            int col = slot % Mathf.Max(queueRowSize, 1);
            return origin + queueRowDirection.normalized * (col * queueGridSpacing) +
                   queueColumnDirection.normalized * (row * queueGridSpacing) + Vector3.up * jobTokenHeight;
        }

        /// @brief Safe accessor for individual job tracking data.
        /// @param jobId Zero-based job index.
        /// @return The JobTracker or null if out of range.
        public JobTracker GetJobTracker(int jobId)
        {
            if (trackers == null || jobId < 0 || jobId >= trackers.Length) return null;
            return trackers[jobId];
        }

        /// @brief Exports the world-space (X, Z) coordinates of all jobs for logging or AI observations.
        /// @return Flat float array: [j0_x, j0_z, j1_x, j1_z, ...].
        public float[] GetJobPositionsFlat()
        {
            if (!initialized) return null;
            float[] positions = new float[trackers.Length * 2];
            for (int j = 0; j < trackers.Length; j++)
            {
                Vector3 pos = trackers[j].Visual != null ? trackers[j].Visual.transform.position : trackers[j].WorldPosition;
                positions[j * 2 + 0] = pos.x;
                positions[j * 2 + 1] = pos.z;
            }
            return positions;
        }

        /// @brief Flattens job operation data into a normalized matrix for agent input.
        ///
        /// @details Each operation is represented by 3 values: presence flag, 
        /// normalized duration, and completion status (0 to 1).
        ///
        /// @param numMachines The machine count to bound the matrix columns.
        /// @return Flat array of size Jobs * Machines * 3.
        public float[] GetSchedulingMatrixFlat(int numMachines)
        {
            throw new NotImplementedException();
            // if (!initialized) return null;
            // int numJobs = trackers.Length;
            // float[] matrix = new float[numJobs * numMachines * 3];
            // float maxDuration = 1f;
            // foreach (JobTracker t in trackers)
            //     foreach (float d in t.OperationDurations) if (d > maxDuration) maxDuration = d;

            // for (int j = 0; j < numJobs; j++)
            // {
            //     JobTracker t = trackers[j];
            //     for (int o = 0; o < t.TotalOperations; o++)
            //     {
            //         int m = t.OperationMachineIds[o];
            //         if (m < 0 || m >= numMachines) continue;
            //         int baseIdx = (j * numMachines + m) * 3;
            //         matrix[baseIdx + 0] = 1.0f;
            //         matrix[baseIdx + 1] = t.OperationDurations[o] / maxDuration;
            //         matrix[baseIdx + 2] = t.OperationStatuses[o];
            //     }
            // }
            // return matrix;
        }

        /// @brief Exports job-level scalar features normalized by current simulation time.
        /// @param currentSimTime Used as a denominator for wait-time normalization.
        /// @return Flat float array: [compRatio, opProgress, waitNorm, stateEncoding, ...].
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