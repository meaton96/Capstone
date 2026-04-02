using System;
using System.Collections.Generic;
using UnityEngine;
using Assets.Scripts.Scheduling.Core;
using Assets.Scripts.Scheduling.Data;
using Assets.Scripts.Simulation.FactoryLayout;
using Assets.Scripts.Logging;

namespace Assets.Scripts.Simulation
{
    /// @brief Lifecycle states a job passes through from creation to completion.
    public enum JobLifecycleState
    {
        NotStarted,
        Queued,
        Processing,
        WaitingForTransport,
        InTransit,
        Complete,
    }

    /// @brief Runtime tracking data for a single job across all of its operations.
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

    /// @brief Creates, tracks, and updates all jobs in a simulation episode.
    ///
    /// @details Initialises @c JobTracker instances from a @c TaillardInstance,
    /// optionally spawning @c JobVisual tokens in the scene. State transitions are
    /// driven by physics events forwarded from @c SimulationBridge rather than
    /// polled every frame, so this class acts as the authoritative record of each
    /// job's position in the schedule.
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

        /// @brief All job trackers for the current episode, indexed by job ID.
        public JobTracker[] JobTrackers => trackers;

        /// @brief Number of jobs in the current episode.
        public int JobCount => trackers?.Length ?? 0;

        /// @brief True after @c Initialize() completes successfully.
        public bool IsInitialized => initialized;

        // ─────────────────────────────────────────────────────────
        //  Initialization
        // ─────────────────────────────────────────────────────────

        /// @brief Allocates trackers and optionally spawns visual tokens for every job.
        /// @param instance      The loaded Taillard problem instance.
        /// @param spawnVisuals  When true, a @c JobVisual prefab is instantiated per job.
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
                int opCount = instance.MachineCount;

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
                pendingIncomingJobs.Enqueue(j);
            }
            initialized = true;
            Debug.Log($"[JobManager] Initialized {jobCount} job trackers directly from JSON data.");
        }

        private void Update()
        {
            if (!initialized || layoutManager == null) return;

            // 1. INCOMING BELT: Instantly fill any available space
            if (layoutManager.IncomingBelt != null)
            {
                // Using a while-loop means it will spawn 3 jobs instantly if capacity is 3
                while (pendingIncomingJobs.Count > 0 && !layoutManager.IncomingBelt.IsFull)
                {
                    int nextJobId = pendingIncomingJobs.Peek();
                    JobTracker tracker = trackers[nextJobId];

                    if (tracker.Visual != null)
                    {
                        tracker.Visual.gameObject.SetActive(true);
                    }

                    if (layoutManager.IncomingBelt.TryEnqueue(nextJobId, tracker.Visual))
                    {
                        pendingIncomingJobs.Dequeue();
                    }
                }
            }

            // 2. OUTGOING BELT: Reap completed jobs as they reach the end
            if (layoutManager.OutgoingBelt != null && layoutManager.OutgoingBelt.Count > 0)
            {
                JobVisual frontVisual = layoutManager.OutgoingBelt.PeekFrontVisual();
                if (frontVisual != null)
                {
                    // Check how close the visual is to the final output position
                    float dist = Vector3.Distance(frontVisual.transform.position, layoutManager.OutgoingBelt.OutputEndPosition);

                    // If it's close enough, remove it from the belt
                    if (dist < 0.05f)
                    {
                        var (jobId, vis) = layoutManager.OutgoingBelt.DequeueFront();

                        // Hide it (or you could use Destroy(vis.gameObject) if preferred)
                        vis.gameObject.SetActive(false);

                        SimLogger.Low($"[JobManager] Job {jobId} reached the end of the outgoing belt and was removed.");
                    }
                }
            }
        }

        /// @brief Destroys all visual tokens and resets the manager to an uninitialised state.
        public void Cleanup()
        {
            if (jobTokenParent != null) Destroy(jobTokenParent.gameObject);
            trackers = null;
            initialized = false;
        }

        // ─────────────────────────────────────────────────────────
        //  State Transitions
        // ─────────────────────────────────────────────────────────

        /// @brief Transitions a job to the @c Queued state when it physically
        ///        enters a machine's trigger zone.
        /// @param jobId      The arriving job.
        /// @param machineId  The machine the job has arrived at.
        public void MarkJobArrivedAtMachine(int jobId, int machineId)
        {
            JobTracker t = GetJobTracker(jobId);
            if (t == null) return;

            t.State = JobLifecycleState.Queued;
            t.CurrentMachineId = machineId;
            t.PhysicallyAtMachine = true;

            if (t.Visual != null) t.Visual.SetState(t.State);
        }

        /// @brief Transitions a job to @c Processing and accrues its wait time.
        /// @param jobId    The job starting its operation.
        /// @param simTime  Current simulation time used to compute wait duration.
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

        /// @brief Transitions a job to @c InTransit toward its next destination.
        /// @param jobId                 The job being transported.
        /// @param destinationMachineId  Machine the AGV is heading to.
        /// @param simTime               Current simulation time.
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

        /// @brief Records transit completion time. The @c Queued transition is
        ///        deferred to @c PhysicalMachine.OnTriggerEnter().
        /// @param jobId      The job that has been delivered.
        /// @param machineId  Machine the job was delivered to.
        /// @param simTime    Current simulation time.
        public void CompleteTransit(int jobId, int machineId, double simTime)
        {
            JobTracker t = GetJobTracker(jobId);
            if (t == null) return;

            double transitDuration = simTime - t.StateEntryTime;
            t.TotalTransitTime += transitDuration;
            t.NextMachineId = machineId;
            t.StateEntryTime = simTime;
        }

        /// @brief Marks the current operation as finished and advances the tracker.
        ///
        /// @details If all operations are complete the job transitions to @c Complete
        ///          and its visual moves to the exit area. Otherwise it transitions to
        ///          @c WaitingForTransport and its next machine ID is updated.
        ///
        /// @param jobId    The job whose operation just finished.
        /// @param simTime  Current simulation time.
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
                t.State = JobLifecycleState.WaitingForTransport;
                t.CurrentOperationIndex++;
                t.NextMachineId = t.OperationMachineIds[t.CurrentOperationIndex];
                t.CurrentMachineId = -1;
                t.OperationProgress = 0f;

                if (t.Visual != null) t.Visual.SetState(t.State);
            }
        }

        /// @brief Returns true when every job has reached the @c Complete state.
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
        //  Grid Layout Helpers
        // ─────────────────────────────────────────────────────────

        /// @brief Computes the world-space spawn position for a job in the incoming queue.
        /// @param slot  Job index used to determine row and column in the grid.
        /// @return      World position including the configured token height offset.
        private Vector3 GetIncomingQueuePosition(int slot)
        {
            // Use the LayoutManager's Incoming Belt as the origin
            Vector3 origin = layoutManager != null ? layoutManager.IncomingBeltPosition : incomingQueueOrigin;
            int row = slot / Mathf.Max(queueRowSize, 1);
            int col = slot % Mathf.Max(queueRowSize, 1);
            Vector3 rowDir = queueRowDirection.normalized;
            Vector3 colDir = queueColumnDirection.normalized;
            return origin + rowDir * (col * queueGridSpacing) + colDir * (row * queueGridSpacing) + Vector3.up * jobTokenHeight;
        }

        private Vector3 GetExitAreaPosition(int slot)
        {
            // Use the LayoutManager's Outgoing Belt as the origin
            Vector3 origin = layoutManager != null ? layoutManager.OutgoingBeltPosition : exitAreaOrigin;
            int row = slot / Mathf.Max(queueRowSize, 1);
            int col = slot % Mathf.Max(queueRowSize, 1);
            Vector3 rowDir = queueRowDirection.normalized;
            Vector3 colDir = queueColumnDirection.normalized;
            return origin + rowDir * (col * queueGridSpacing) + colDir * (row * queueGridSpacing) + Vector3.up * jobTokenHeight;
        }

        // ─────────────────────────────────────────────────────────
        //  Query API
        // ─────────────────────────────────────────────────────────

        /// @brief Returns the @c JobTracker for the given job ID, or @c null if invalid.
        /// @param jobId  Zero-based job index.
        public JobTracker GetJobTracker(int jobId)
        {
            if (trackers == null || jobId < 0 || jobId >= trackers.Length) return null;
            return trackers[jobId];
        }

        /// @brief Returns a flat array of (x, z) world positions for every job visual.
        /// @return Float array of length @c JobCount * 2, or @c null if uninitialised.
        public float[] GetJobPositionsFlat()
        {
            if (!initialized) return null;
            float[] positions = new float[trackers.Length * 2];
            for (int j = 0; j < trackers.Length; j++)
            {
                Vector3 pos = trackers[j].WorldPosition;
                if (trackers[j].Visual != null) pos = trackers[j].Visual.transform.position;

                positions[j * 2 + 0] = pos.x;
                positions[j * 2 + 1] = pos.z;
            }
            return positions;
        }

        /// @brief Builds a flat (jobs × machines × 3) scheduling matrix.
        ///
        /// @details Each cell stores three normalised floats: operation-exists flag,
        ///          normalised duration, and completion status.
        ///
        /// @param numMachines  Column count; should match the instance's machine count.
        /// @return             Float array of length @c JobCount * @p numMachines * 3.
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

        /// @brief Builds a flat array of per-job scalar observations for the agent.
        ///
        /// @details Each job contributes four values: completion ratio, operation
        ///          progress, normalised total wait time, and a float encoding of
        ///          the current lifecycle state.
        ///
        /// @param currentSimTime  Used to normalise wait-time values.
        /// @return                Float array of length @c JobCount * 4.
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

        /// @brief Maps a @c JobLifecycleState to a normalised float in [0, 1].
        /// @param state  The lifecycle state to convert.
        /// @return       Float value spaced evenly across the range.
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