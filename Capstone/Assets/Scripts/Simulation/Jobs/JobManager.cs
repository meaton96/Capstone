using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Assets.Scripts.Simulation.FactoryLayout;
using Assets.Scripts.Logging;

namespace Assets.Scripts.Simulation.Jobs
{

    /// @brief Creates, tracks, and updates all jobs in a simulation episode.
    /// @details THE single source of truth for job state and location.
    ///          Every location change goes through TransitionJob().
    ///          AGVPool pulls from GetNextTransportJob() — no push queue.
    ///          Exit happens on state transition, not belt polling.
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

        // ─────────────────────────────────────────────────────────
        //  Initialization
        // ─────────────────────────────────────────────────────────

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
                    Location = JobLocation.PendingEntry,
                    LocationMachineId = -1,
                    AssignedAGVId = -1,
                    CurrentOperationIndex = 0,
                    CompletedOperations = 0,
                    WorldPosition = GetIncomingQueuePosition(j),
                    CurrentMachineId = -1,
                    NextMachineId = -1,
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

                // Validate array lengths match
                if (def.EligibleMachinesPerOp.Length != opCount)
                {
                    SimLogger.Error($"[JobManager] Job {j}: EligibleMachinesPerOp.Length ({def.EligibleMachinesPerOp.Length}) " +
                                    $"!= OperationSequence.Length ({opCount}). Generator bug!");
                }
            }

            initialized = true;
            SimLogger.Low($"[JobManager] Initialized {jobDefs.Length} FJSSP job trackers.");
        }

        // ─────────────────────────────────────────────────────────
        //  TransitionJob — THE single authority for location changes
        // ─────────────────────────────────────────────────────────

        public void TransitionJob(int jobId, JobLocation newLocation, int machineId = -1, int agvId = -1)
        {
            JobTracker t = GetJobTracker(jobId);
            if (t == null)
            {
                SimLogger.Error($"[TransitionJob] job={jobId} tracker is NULL!");
                return;
            }

            // Guard: don't transition a job that's already Exited
            if (t.Location == JobLocation.Exited && newLocation != JobLocation.Exited)
            {
                SimLogger.Error($"[TransitionJob] job={jobId} is already Exited — ignoring transition to {newLocation}");
                return;
            }

            JobLocation oldLocation = t.Location;
            int oldMachine = t.LocationMachineId;

            t.Location = newLocation;
            t.LocationMachineId = machineId;
            t.AssignedAGVId = agvId;

            // Update legacy State field for compatibility
            switch (newLocation)
            {
                case JobLocation.PendingEntry:
                case JobLocation.OnFactoryBelt:
                    t.State = JobLifecycleState.NotStarted;
                    break;
                case JobLocation.AwaitingPickup:
                    break; // keep current state
                case JobLocation.InTransit:
                case JobLocation.InTransitToExit:
                    t.State = JobLifecycleState.InTransit;
                    break;
                case JobLocation.InMachineQueue:
                    t.State = JobLifecycleState.Queued;
                    t.CurrentMachineId = machineId;
                    t.PhysicallyAtMachine = true;
                    break;
                case JobLocation.Processing:
                    t.State = JobLifecycleState.Processing;
                    t.CurrentMachineId = machineId;
                    break;
                case JobLocation.AwaitingTransport:
                    t.State = JobLifecycleState.WaitingForTransport;
                    break;
                case JobLocation.Exited:
                    t.State = JobLifecycleState.Complete;
                    break;
            }

            if (t.Visual != null)
                t.Visual.SetState(t.State);

            SimLogger.Medium($"[TransitionJob] job={jobId} {oldLocation}(M{oldMachine}) → {newLocation}(M{machineId}) agv={agvId}");
        }

        // ─────────────────────────────────────────────────────────
        //  Pull-model query for AGVPool
        // ─────────────────────────────────────────────────────────

        /// @brief Returns the next unclaimed job awaiting transport, or null if none.
        ///        AGVPool calls this instead of maintaining its own queue.
        ///        Skips jobs already claimed by an AGV (AssignedAGVId != -1).
        public JobTracker GetNextTransportJob()
        {
            if (!initialized) return null;
            JobTracker best = null;
            double bestTime = double.MaxValue;
            foreach (var t in trackers)
            {
                if (t.Location != JobLocation.AwaitingPickup) continue;
                if (t.AssignedAGVId >= 0) continue; // already claimed by an AGV
                if (t.StateEntryTime < bestTime)
                {
                    bestTime = t.StateEntryTime;
                    best = t;
                }
            }
            return best;
        }

        // ─────────────────────────────────────────────────────────
        //  Query methods — replace ConveyorBelt queries
        // ─────────────────────────────────────────────────────────

        public List<int> GetJobsInMachineQueue(int machineId)
        {
            var result = new List<int>();
            if (!initialized) return result;
            foreach (var t in trackers)
                if (t.Location == JobLocation.InMachineQueue && t.LocationMachineId == machineId)
                    result.Add(t.JobId);
            return result;
        }

        public List<int> GetJobsOnMachineOutgoing(int machineId)
        {
            var result = new List<int>();
            if (!initialized) return result;
            foreach (var t in trackers)
                if (t.Location == JobLocation.AwaitingTransport && t.LocationMachineId == machineId)
                    result.Add(t.JobId);
            return result;
        }

        public bool HasDispatchableJob(int machineId)
        {
            if (!initialized) return false;
            foreach (var t in trackers)
            {
                if (t.Location != JobLocation.InMachineQueue) continue;
                if (t.LocationMachineId != machineId) continue;
                if (t.CurrentOperationIndex < 0 || t.CurrentOperationIndex >= t.TotalOperations) continue;
                return true;
            }
            return false;
        }

        public List<int> GetDispatchableJobs(int machineId)
        {
            var result = new List<int>();
            if (!initialized) return result;
            foreach (var t in trackers)
            {
                if (t.Location != JobLocation.InMachineQueue) continue;
                if (t.LocationMachineId != machineId) continue;
                if (t.CurrentOperationIndex < 0 || t.CurrentOperationIndex >= t.TotalOperations) continue;
                result.Add(t.JobId);
            }
            return result;
        }

        public int GetTotalJobsAtMachine(int machineId)
        {
            int count = 0;
            if (!initialized) return count;
            foreach (var t in trackers)
            {
                if (t.LocationMachineId != machineId) continue;
                if (t.Location == JobLocation.InMachineQueue ||
                    t.Location == JobLocation.Processing ||
                    t.Location == JobLocation.AwaitingTransport)
                    count++;
            }
            return count;
        }

        // ─────────────────────────────────────────────────────────
        //  Operation lifecycle
        // ─────────────────────────────────────────────────────────

        public void MarkJobArrivedAtMachine(int jobId, int machineId)
        {
            TransitionJob(jobId, JobLocation.InMachineQueue, machineId);
        }

        public void MarkOperationStarted(int jobId, double simTime)
        {
            JobTracker t = GetJobTracker(jobId);
            if (t == null) return;

            TransitionJob(jobId, JobLocation.Processing, t.LocationMachineId);
            t.TotalWaitTime += (simTime - t.StateEntryTime);
            t.StateEntryTime = simTime;
            t.OperationStatuses[t.CurrentOperationIndex] = 0.5f;
        }

        public void MarkOperationComplete(int jobId, double simTime)
        {
            if (!initialized) return;
            JobTracker t = trackers[jobId];
            if (t.CurrentOperationIndex >= t.TotalOperations) return;

            t.CompletedOperations++;
            t.OperationStatuses[t.CurrentOperationIndex] = 1.0f;
            t.StateEntryTime = simTime;

            int currentMachine = t.LocationMachineId;

            if (t.CompletedOperations >= t.TotalOperations)
            {
                t.CurrentOperationIndex = t.TotalOperations; // prevent re-entry
                t.CurrentMachineId = -1;
                t.NextMachineId = -1;
                t.OperationProgress = 0f;
                TransitionJob(jobId, JobLocation.AwaitingTransport, currentMachine);
                t.State = JobLifecycleState.Complete;
                if (t.Visual != null) t.Visual.SetState(t.State);
            }
            else
            {
                t.CurrentOperationIndex++;
                t.NextMachineId = -1;
                t.NextMachineType = t.OperationTypes[t.CurrentOperationIndex];
                t.CurrentMachineId = -1;
                t.OperationProgress = 0f;
                TransitionJob(jobId, JobLocation.AwaitingTransport, currentMachine);
            }
        }

        /// @brief AGV picked up a job — transition to InTransit.
        public void BeginTransit(int jobId, int destinationMachineId, double simTime)
        {
            JobTracker t = GetJobTracker(jobId);
            if (t == null) return;

            t.NextMachineId = destinationMachineId;
            t.TotalWaitTime += (simTime - t.StateEntryTime);
            t.StateEntryTime = simTime;

            JobLocation dest = (destinationMachineId < 0)
                ? JobLocation.InTransitToExit
                : JobLocation.InTransit;

            TransitionJob(jobId, dest, -1);
            SimLogger.High($"[JobManager] begin transit of job {jobId} to machine {destinationMachineId}");
        }

        /// @brief AGV completed delivery.
        ///        For machine delivery: transitions to InMachineQueue.
        ///        For exit delivery: transitions DIRECTLY to Exited and fires OnJobExited.
        ///        The exit belt is visual-only — it does NOT drive state.
        public void CompleteTransit(int jobId, int machineId, double simTime)
        {
            JobTracker t = GetJobTracker(jobId);
            if (t == null) return;

            t.TotalTransitTime += (simTime - t.StateEntryTime);
            t.StateEntryTime = simTime;

            if (machineId < 0)
            {
                // EXIT: go straight to Exited — belt is just visual
                TransitionJob(jobId, JobLocation.Exited, -1);
                SimLogger.Low($"[JobManager] Job {jobId} exited factory.");
                SimulationBridge.Instance?.OnJobExited(jobId);
            }
            else
            {
                t.NextMachineId = machineId;
                TransitionJob(jobId, JobLocation.InMachineQueue, machineId);
            }
        }

        // ─────────────────────────────────────────────────────────
        //  Update — ONLY handles factory incoming belt visuals
        // ─────────────────────────────────────────────────────────

        /// @brief Feeds pending jobs onto the factory incoming belt (visual only).
        ///        Exit belt cleanup just deactivates visuals — no state changes.
        private void Update()
        {
            if (!initialized || layoutManager == null) return;

            // Feed pending jobs onto factory incoming belt
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
                        TransitionJob(nextJobId, JobLocation.OnFactoryBelt, -1);
                    }
                }
            }

            // Exit belt: just deactivate visuals that reach the end.
            // State already transitioned to Exited in CompleteTransit — this is visual cleanup only.
            if (layoutManager.OutgoingBelt != null && layoutManager.OutgoingBelt.Count > 0)
            {
                JobVisual frontVisual = layoutManager.OutgoingBelt.PeekFrontVisual();
                if (frontVisual != null)
                {
                    float dist = Vector3.Distance(frontVisual.transform.position, layoutManager.OutgoingBelt.OutputEndPosition);
                    if (dist < 0.05f)
                    {
                        var (jobId, vis) = layoutManager.OutgoingBelt.DequeueFront();
                        vis?.gameObject.SetActive(false);
                        // NO state transition here — CompleteTransit already handled it
                    }
                }
            }
        }

        // ─────────────────────────────────────────────────────────
        //  Accessors
        // ─────────────────────────────────────────────────────────

        public float GetProcessingTime(int jobId, int machineId)
        {
            JobTracker t = GetJobTracker(jobId);
            if (t == null) return 0f;
            int opIdx = t.CurrentOperationIndex;
            if (opIdx < 0 || opIdx >= t.TotalOperations) return 0f;
            if (opIdx >= t.EligibleMachinesPerOp.Length)
            {
                SimLogger.Error($"[GetProcessingTime] job={jobId} opIdx={opIdx} out of EligibleMachinesPerOp (len={t.EligibleMachinesPerOp.Length})");
                return 0f;
            }
            if (t.EligibleMachinesPerOp[opIdx].TryGetValue(machineId, out float time))
                return time;
            return 0f;
        }

        public bool AreAllJobsComplete()
        {
            if (!initialized) return false;
            foreach (var t in trackers)
                if (t.Location != JobLocation.Exited) return false;
            return true;
        }

        public JobTracker GetJobTracker(int jobId)
        {
            if (trackers == null || jobId < 0 || jobId >= trackers.Length) return null;
            return trackers[jobId];
        }

        public void Cleanup()
        {
            if (jobTokenParent != null) Destroy(jobTokenParent.gameObject);
            trackers = null;
            initialized = false;
        }

        // ─────────────────────────────────────────────────────────
        //  Position helpers
        // ─────────────────────────────────────────────────────────

        private Vector3 GetIncomingQueuePosition(int slot)
        {
            Vector3 origin = layoutManager != null ? layoutManager.IncomingBeltPosition : incomingQueueOrigin;
            int row = slot / Mathf.Max(queueRowSize, 1);
            int col = slot % Mathf.Max(queueRowSize, 1);
            return origin + queueRowDirection.normalized * (col * queueGridSpacing) +
                   queueColumnDirection.normalized * (row * queueGridSpacing) + Vector3.up * jobTokenHeight;
        }

        // ─────────────────────────────────────────────────────────
        //  Observation helpers (for ML agent)
        // ─────────────────────────────────────────────────────────

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

        public float[] GetSchedulingMatrixFlat(int numMachines)
        {
            throw new NotImplementedException();
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