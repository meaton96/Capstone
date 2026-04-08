using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Assets.Scripts.Simulation.FactoryLayout;
using Assets.Scripts.Logging;

namespace Assets.Scripts.Simulation.Jobs
{

    /// @brief Creates, tracks, and updates all jobs in a simulation episode.
    /// @details Authoritative record of job states AND locations. Every change to
    ///          where a job is must go through TransitionJob(). External code queries
    ///          GetJobsInMachineQueue() etc. instead of reading ConveyorBelt contents.
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
            }

            initialized = true;
            SimLogger.Low($"[JobManager] Initialized {jobDefs.Length} FJSSP job trackers.");
        }

        // ─────────────────────────────────────────────────────────
        //  TransitionJob — THE single authority for location changes
        // ─────────────────────────────────────────────────────────

        /// @brief Moves a job to a new location. This is the ONLY method that should
        ///        change Location/LocationMachineId/AssignedAGVId.
        /// @param jobId       The job to move.
        /// @param newLocation The destination location.
        /// @param machineId   Machine context (-1 if not applicable).
        /// @param agvId       AGV carrying the job (-1 if not applicable).
        public void TransitionJob(int jobId, JobLocation newLocation, int machineId = -1, int agvId = -1)
        {
            JobTracker t = GetJobTracker(jobId);
            if (t == null)
            {
                SimLogger.Error($"[TransitionJob] job={jobId} tracker is NULL!");
                return;
            }

            JobLocation oldLocation = t.Location;
            int oldMachine = t.LocationMachineId;

            t.Location = newLocation;
            t.LocationMachineId = machineId;
            t.AssignedAGVId = agvId;

            // Update the legacy State field to keep compatibility
            switch (newLocation)
            {
                case JobLocation.PendingEntry:
                case JobLocation.OnFactoryBelt:
                    t.State = JobLifecycleState.NotStarted;
                    break;
                case JobLocation.AwaitingPickup:
                    // Keep current state — could be NotStarted or WaitingForTransport
                    break;
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
                case JobLocation.OnExitBelt:
                case JobLocation.Exited:
                    t.State = JobLifecycleState.Complete;
                    break;
            }

            if (t.Visual != null)
                t.Visual.SetState(t.State);

            SimLogger.Medium($"[TransitionJob] job={jobId} {oldLocation}(M{oldMachine}) → {newLocation}(M{machineId}) agv={agvId}");
            // Verify assignment
            SimLogger.Medium($"[TransitionJob] VERIFY job={jobId}: Location={t.Location}, LocationMachineId={t.LocationMachineId}");
        }

        // ─────────────────────────────────────────────────────────
        //  Query methods — replace ConveyorBelt queries
        // ─────────────────────────────────────────────────────────

        /// @brief Returns job IDs queued at a machine's incoming area.
        ///        Replaces PhysicalMachine.IncomingQueue / PhysicalQueue.
        public List<int> GetJobsInMachineQueue(int machineId)
        {
            var result = new List<int>();
            if (!initialized) return result;
            foreach (var t in trackers)
                if (t.Location == JobLocation.InMachineQueue && t.LocationMachineId == machineId)
                    result.Add(t.JobId);
            return result;
        }

        /// @brief Returns job IDs on a machine's outgoing belt (awaiting transport).
        public List<int> GetJobsOnMachineOutgoing(int machineId)
        {
            var result = new List<int>();
            if (!initialized) return result;
            foreach (var t in trackers)
                if (t.Location == JobLocation.AwaitingTransport && t.LocationMachineId == machineId)
                    result.Add(t.JobId);
            return result;
        }

        /// @brief True if at least one dispatchable job is queued at this machine.
        public bool HasDispatchableJob(int machineId)
        {
            if (!initialized) { SimLogger.LogWarning("[HasDispatchableJob] NOT INITIALIZED"); return false; }
            foreach (var t in trackers)
            {
                if (t.Location != JobLocation.InMachineQueue) continue;
                if (t.LocationMachineId != machineId) continue;
                if (t.CurrentOperationIndex < 0 || t.CurrentOperationIndex >= t.EligibleMachinesPerOp.Length)
                {
                    SimLogger.LogWarning($"[HasDispatchableJob] job={t.JobId} at M{machineId} SKIPPED: opIndex={t.CurrentOperationIndex}, opsLength={t.EligibleMachinesPerOp.Length}");
                    continue;
                }
                return true;
            }

            // Dump all tracker states when returning false — this helps find the desync
            string dump = $"[HasDispatchableJob] M{machineId} → FALSE. All trackers: ";
            foreach (var t in trackers)
                dump += $"\n  job={t.JobId} loc={t.Location} locM={t.LocationMachineId} opIdx={t.CurrentOperationIndex}/{t.TotalOperations}";
            SimLogger.LogWarning(dump);

            return false;
        }

        /// @brief Returns dispatchable jobs at a machine (valid op index, correct location).
        public List<int> GetDispatchableJobs(int machineId)
        {
            var result = new List<int>();
            if (!initialized) return result;
            foreach (var t in trackers)
            {
                if (t.Location != JobLocation.InMachineQueue) continue;
                if (t.LocationMachineId != machineId) continue;
                if (t.CurrentOperationIndex < 0 || t.CurrentOperationIndex >= t.EligibleMachinesPerOp.Length) continue;
                result.Add(t.JobId);
            }
            return result;
        }

        /// @brief Count of jobs at a machine (incoming queue + processing + outgoing).
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
        //  Operation lifecycle (called by SimulationBridge)
        // ─────────────────────────────────────────────────────────

        /// @brief Called when an AGV drops a job at a machine.
        ///        Transitions to InMachineQueue.
        public void MarkJobArrivedAtMachine(int jobId, int machineId)
        {
            JobTracker t = GetJobTracker(jobId);
            if (t == null) return;
            TransitionJob(jobId, JobLocation.InMachineQueue, machineId);
        }

        /// @brief Called when a machine begins processing a job.
        public void MarkOperationStarted(int jobId, double simTime)
        {
            JobTracker t = GetJobTracker(jobId);
            if (t == null) return;

            TransitionJob(jobId, JobLocation.Processing, t.LocationMachineId);
            t.TotalWaitTime += (simTime - t.StateEntryTime);
            t.StateEntryTime = simTime;
            t.OperationStatuses[t.CurrentOperationIndex] = 0.5f;
        }

        /// @brief Called when a machine finishes processing. Advances operation index.
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
                // All operations done — job goes to outgoing belt awaiting exit transport
                t.CurrentMachineId = -1;
                t.NextMachineId = -1;
                t.OperationProgress = 0f;
                TransitionJob(jobId, JobLocation.AwaitingTransport, t.LocationMachineId);
                t.State = JobLifecycleState.Complete; // override: logically complete
                if (t.Visual != null) t.Visual.SetState(t.State);
            }
            else
            {
                t.CurrentOperationIndex++;
                t.NextMachineId = -1;
                t.NextMachineType = t.OperationTypes[t.CurrentOperationIndex];
                t.CurrentMachineId = -1;
                t.OperationProgress = 0f;
                TransitionJob(jobId, JobLocation.AwaitingTransport, t.LocationMachineId);
            }
        }

        /// @brief Called when an AGV picks up a job and begins carrying it.
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

        /// @brief Called when an AGV completes delivery of a job.
        public void CompleteTransit(int jobId, int machineId, double simTime)
        {
            JobTracker t = GetJobTracker(jobId);
            if (t == null) return;

            t.TotalTransitTime += (simTime - t.StateEntryTime);
            t.NextMachineId = machineId;
            t.StateEntryTime = simTime;

            if (machineId < 0)
                TransitionJob(jobId, JobLocation.OnExitBelt, -1);
            else
                TransitionJob(jobId, JobLocation.InMachineQueue, machineId);
        }

        // ─────────────────────────────────────────────────────────
        //  Factory belt management
        // ─────────────────────────────────────────────────────────

        /// @brief Drives jobs from PendingEntry onto the factory IncomingBelt,
        ///        and handles exit belt departures.
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

            // Handle exit belt departures
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
                        TransitionJob(jobId, JobLocation.Exited, -1);
                        SimLogger.Low($"[JobManager] Job {jobId} exited factory.");
                        SimulationBridge.Instance?.OnJobExited(jobId);
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
            if (opIdx < 0 || opIdx >= t.EligibleMachinesPerOp.Length) return 0f;
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

        private Vector3 GetExitAreaPosition(int slot)
        {
            Vector3 origin = layoutManager != null ? layoutManager.OutgoingBeltPosition : exitAreaOrigin;
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