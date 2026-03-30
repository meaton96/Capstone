using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using Assets.Scripts.Scheduling.Core;

namespace Assets.Scripts.Simulation
{
    /// @file JobManager.cs
    /// @brief Tracks per-job lifecycle, physical position, and state across the
    /// factory floor. Exposes an IoT-style sensor API that the ObservationBuilder
    /// can query identically to how it would query real RFID / MES systems.
    ///
    /// @details The JobManager does not own the scheduling logic — that stays in
    /// the DES. Instead, it *observes* the DES state each time the bridge advances
    /// and maintains a parallel tracking layer with:
    ///   - Physical world-space position for each job (for the spatial grid)
    ///   - Lifecycle state (for the scheduling matrix and event flags)
    ///   - Timing metadata (wait time, transit time, processing progress)
    ///
    /// @par IoT Sensor Equivalents
    /// | JobManager API              | Real-World Sensor           |
    /// | :-------------------------- | :-------------------------- |
    /// | GetJobPosition(id)          | UWB / RFID tag read         |
    /// | GetJobState(id)             | MES status register         |
    /// | GetOperationProgress(id)    | Machine PLC cycle counter   |
    /// | GetWaitTime(id)             | Queue timestamp diff        |
    /// | GetTransitTime(id)          | AGV fleet telemetry         |
    ///
    /// @par Integration Points
    /// - **SimulationBridge**: Calls @ref SyncFromDES after each step to refresh
    ///   all trackers from the current DES state.
    /// - **FactoryLayoutManager**: Provides machine world positions so jobs can
    ///   be placed at the correct location.
    /// - **AGVPool** (future): Will call @ref BeginTransit / @ref CompleteTransit
    ///   to update job positions during physical transport.
    /// - **ObservationBuilder** (future): Reads @ref JobTrackers, @ref GetSchedulingMatrixData,
    ///   and @ref GetSpatialLayerData to construct observation tensors.

    // ═════════════════════════════════════════════════════════════
    //  Supporting types
    // ═════════════════════════════════════════════════════════════

    /// @brief Lifecycle state of a job from the factory-floor perspective.
    ///
    /// @details These states map to what an RFID/MES system would report.
    /// The DES only knows Queued/Processing/Complete; we add WaitingForTransport
    /// and InTransit to model the physical reality between machines.
    public enum JobLifecycleState
    {
        /// @brief Job exists but hasn't entered the shop floor yet.
        /// First operation not yet queued on any machine.
        NotStarted,

        /// @brief Job is sitting in a machine's input queue, waiting to be processed.
        /// RFID equivalent: tag detected at machine station, no PLC start signal.
        Queued,

        /// @brief Job is actively being processed on a machine.
        /// RFID equivalent: tag at machine + PLC reporting active cycle.
        Processing,

        /// @brief Current operation is complete, job needs transport to next machine.
        /// In the current DES (no AGVs), this state is instantaneous.
        /// Once AGVPool exists, jobs will stay here until an AGV is dispatched.
        WaitingForTransport,

        /// @brief Job is physically moving between machines on an AGV.
        /// Not used until AGVPool is implemented — included now so the
        /// state machine doesn't need to change later.
        InTransit,

        /// @brief All operations complete. Job has exited the system.
        Complete,
    }

    /// @brief Per-job tracking data, refreshed each time the bridge advances.
    ///
    /// @details This is the "sensor reading" for one job. Everything an
    /// ObservationBuilder needs to populate the scheduling matrix row and
    /// the spatial grid cell for this job.
    [Serializable]
    public class JobTracker
    {
        // ── Identity ─────────────────────────────────────────
        /// @brief The job's index in the DES Jobs array.
        public int JobId;

        /// @brief Total number of operations in this job.
        public int TotalOperations;

        // ── Lifecycle ────────────────────────────────────────
        /// @brief Current lifecycle state (the "MES status register").
        public JobLifecycleState State;

        /// @brief Index of the operation currently active or next to be processed.
        /// Range: 0 to TotalOperations. Equals TotalOperations when complete.
        public int CurrentOperationIndex;

        /// @brief Number of operations fully completed.
        public int CompletedOperations;

        // ── Spatial ──────────────────────────────────────────
        /// @brief Current world-space position of this job on the factory floor.
        /// When Processing or Queued, this is the machine's position.
        /// When InTransit, this is the AGV's current position.
        /// When NotStarted, this is a staging area position.
        public Vector3 WorldPosition;

        /// @brief ID of the machine this job is currently at (or heading to).
        /// -1 if not associated with any machine (NotStarted, InTransit, Complete).
        public int CurrentMachineId;

        /// @brief ID of the machine this job needs to go to next.
        /// -1 if no remaining operations.
        public int NextMachineId;

        // ── Timing (IoT sensor equivalents) ──────────────────
        /// @brief Sim time when the job entered its current state.
        public double StateEntryTime;

        /// @brief How long the job has been in its current state.
        /// Updated by @ref JobManager.SyncFromDES each step.
        public double TimeInCurrentState;

        /// @brief Cumulative time this job has spent waiting in queues.
        /// IoT equivalent: sum of all queue-entry to processing-start deltas.
        public double TotalWaitTime;

        /// @brief Cumulative time this job has spent being transported.
        /// Always 0 until AGVPool is implemented.
        public double TotalTransitTime;

        /// @brief Progress through current operation: 0.0 (just started) to 1.0 (done).
        /// Only meaningful when State == Processing. 0 otherwise.
        public float OperationProgress;

        // ── Per-operation detail ─────────────────────────────
        /// @brief Status of each operation: 0 = not started, 0.5 = in progress, 1.0 = complete.
        /// Length == TotalOperations. Used to build the scheduling matrix.
        public float[] OperationStatuses;

        /// @brief Machine ID assigned to each operation (from the problem instance).
        /// Length == TotalOperations. Static after initialization.
        public int[] OperationMachineIds;

        /// @brief Processing duration of each operation.
        /// Length == TotalOperations. Static after initialization.
        public float[] OperationDurations;

        // ── Physical delivery ─────────────────────────────────
        /// @brief True once this job has been physically transported to its
        /// current machine. False while it's still in the incoming queue
        /// (even if the DES has logically queued it on a machine).
        ///
        /// Pre-AGV: set to true when the DES starts processing the operation.
        /// Post-AGV: set to true when the AGV completes delivery.
        ///
        /// This separates "DES logical queue" from "physical presence on floor."
        public bool PhysicallyAtMachine;

        /// @brief Slot index in the incoming queue grid. Assigned at init,
        /// reused when the job returns between operations.
        public int IncomingQueueSlot;

        // ── Visual ───────────────────────────────────────────
        /// @brief Reference to the visual token GameObject for this job.
        /// Null when visuals are disabled.
        public JobVisual Visual;
    }

    // ═════════════════════════════════════════════════════════════
    //  JobManager
    // ═════════════════════════════════════════════════════════════

    public class JobManager : MonoBehaviour
    {
        // ─────────────────────────────────────────────────────────
        //  Inspector
        // ─────────────────────────────────────────────────────────

        [Header("References")]
        [SerializeField] private FactoryLayoutManager layoutManager;

        [Header("Visual")]
        [Tooltip("Prefab for the per-job visual token. If null, no visuals are spawned.")]
        [SerializeField] private GameObject jobVisualPrefab;

        [Tooltip("Y offset above the machine surface for queued job tokens.")]
        [SerializeField] private float jobTokenHeight = 1.5f;

        [Tooltip("Spacing between queued job tokens at the same machine.")]
        [SerializeField] private float queueSpacing = 0.6f;

        [Header("Incoming Queue Layout")]
        [Tooltip("Optional transform marking the origin of the incoming queue area. " +
                 "If null, uses incomingQueueOrigin vector. Place this on the side of your factory floor.")]
        [SerializeField] private Transform incomingQueueMarker;

        [Tooltip("Fallback origin if no marker transform is assigned.")]
        [SerializeField] private Vector3 incomingQueueOrigin = new Vector3(-5f, 0f, 0f);

        [Tooltip("Direction the queue grid extends in (rows). Normalized at runtime.")]
        [SerializeField] private Vector3 queueRowDirection = Vector3.right;

        [Tooltip("Direction for additional columns if jobs exceed one row.")]
        [SerializeField] private Vector3 queueColumnDirection = Vector3.forward;

        [Tooltip("Spacing between job tokens in the queue grid.")]
        [SerializeField] private float queueGridSpacing = 1.0f;

        [Tooltip("Max jobs per row before wrapping to the next column.")]
        [SerializeField] private int queueRowSize = 10;

        [Header("Exit Area")]
        [Tooltip("Optional transform marking where completed jobs go.")]
        [SerializeField] private Transform exitAreaMarker;

        [Tooltip("Fallback position if no exit marker is assigned.")]
        [SerializeField] private Vector3 exitAreaOrigin = new Vector3(-5f, 0f, 10f);

        [Header("Machine Queue")]
        [Tooltip("Offset direction for jobs queued physically at a machine (e.g. lined up behind it).")]
        [SerializeField] private Vector3 machineQueueDirection = Vector3.back;

        // ─────────────────────────────────────────────────────────
        //  Runtime state
        // ─────────────────────────────────────────────────────────

        private JobTracker[] trackers;
        private DESSimulator simulator;
        private bool initialized;

        /// @brief Parent transform for all spawned job tokens. Created on init.
        private Transform jobTokenParent;

        // ─────────────────────────────────────────────────────────
        //  Public read-only accessors
        // ─────────────────────────────────────────────────────────

        /// @brief All job trackers for the current episode. Null before initialization.
        public JobTracker[] JobTrackers => trackers;

        /// @brief Number of jobs being tracked.
        public int JobCount => trackers?.Length ?? 0;

        /// @brief True after Initialize has been called for this episode.
        public bool IsInitialized => initialized;

        // ─────────────────────────────────────────────────────────
        //  Lifecycle API — called by SimulationBridge
        // ─────────────────────────────────────────────────────────

        /// @brief Initializes tracking for all jobs in the current DES instance.
        ///
        /// @details Called by SimulationBridge.StartEpisode after the DES is loaded.
        /// Creates a JobTracker for each job, populates static per-operation data
        /// (machine assignments, durations), and optionally spawns visual tokens.
        ///
        /// @param sim The active DES simulator (already loaded with an instance).
        /// @param spawnVisuals Whether to instantiate visual token GameObjects.
        public void Initialize(DESSimulator sim, bool spawnVisuals = true)
        {
            simulator = sim;
            int jobCount = sim.Jobs.Length;

            // Clean up previous episode
            Cleanup();

            // Create parent for visual tokens
            if (spawnVisuals)
            {
                var parentGo = new GameObject("_JobTokens");
                jobTokenParent = parentGo.transform;
            }

            // Build trackers
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

                // Populate static operation data
                for (int o = 0; o < opCount; o++)
                {
                    tracker.OperationMachineIds[o] = job.Operations[o].MachineId;
                    tracker.OperationDurations[o] = (float)job.Operations[o].Duration;
                    tracker.OperationStatuses[o] = 0f; // not started
                }

                // Spawn visual token
                if (spawnVisuals && jobVisualPrefab != null)
                {
                    Vector3 spawnPos = GetIncomingQueuePosition(j);
                    GameObject tokenGo = Instantiate(jobVisualPrefab, spawnPos, Quaternion.identity, jobTokenParent);
                    tokenGo.name = $"Job_{j}";

                    JobVisual visual = tokenGo.GetComponent<JobVisual>();
                    if (visual == null)
                        visual = tokenGo.AddComponent<JobVisual>();

                    visual.Initialize(j, opCount);
                    tracker.Visual = visual;
                }

                trackers[j] = tracker;
            }

            initialized = true;

            Debug.Log($"[JobManager] Initialized {jobCount} job trackers" +
                      $"{(spawnVisuals && jobVisualPrefab != null ? " with visuals" : "")}.");
        }

        /// @brief Syncs all job trackers to the current DES state.
        ///
        /// @details Called by SimulationBridge after each Step() or during
        /// AdvanceToNextDecision. Reads the DES Jobs array and updates each
        /// tracker's state, position, and timing.
        ///
        /// This is the core "sensor poll" — equivalent to reading all RFID
        /// tags and PLC registers in one scan cycle.
        ///
        /// @param currentSimTime The current simulation clock time.
        public void SyncFromDES(double currentSimTime)
        {
            if (!initialized || simulator?.Jobs == null) return;

            for (int j = 0; j < trackers.Length; j++)
            {
                JobTracker tracker = trackers[j];
                Job job = simulator.Jobs[j];

                // Determine state from DES operation data
                JobLifecycleState previousState = tracker.State;
                UpdateTrackerState(tracker, job, currentSimTime);

                // Update timing
                if (tracker.State != previousState)
                {
                    // State transition: accumulate time in previous state
                    double elapsed = currentSimTime - tracker.StateEntryTime;

                    if (previousState == JobLifecycleState.Queued)
                        tracker.TotalWaitTime += elapsed;
                    else if (previousState == JobLifecycleState.InTransit)
                        tracker.TotalTransitTime += elapsed;

                    tracker.StateEntryTime = currentSimTime;
                }

                tracker.TimeInCurrentState = currentSimTime - tracker.StateEntryTime;

                // Update world position
                UpdateTrackerPosition(tracker);

                // Update visual token
                if (tracker.Visual != null)
                {
                    tracker.Visual.SetState(tracker.State);
                    tracker.Visual.SetTargetPosition(tracker.WorldPosition);
                    tracker.Visual.SetProgress(tracker.OperationProgress);
                }
            }
        }

        /// @brief Destroys all visual tokens and resets trackers.
        public void Cleanup()
        {
            if (jobTokenParent != null)
            {
                Destroy(jobTokenParent.gameObject);
                jobTokenParent = null;
            }

            trackers = null;
            initialized = false;
        }

        // ─────────────────────────────────────────────────────────
        //  State derivation from DES
        // ─────────────────────────────────────────────────────────

        /// @brief Derives the lifecycle state of a job from its DES operations.
        ///
        /// @details Walks the operation array to determine the DES logical state,
        /// then layers on the physical delivery model:
        ///
        /// - The DES queues jobs on machines instantly (no transport time).
        /// - The JobManager adds a "physical delivery" concept: a job isn't
        ///   considered at a machine until either processing starts (pre-AGV)
        ///   or an AGV delivers it (post-AGV).
        /// - Jobs that the DES has queued but haven't been physically delivered
        ///   stay in the incoming queue visually.
        ///
        /// This separation is what makes the IoT swap possible: in the real
        /// world, an RFID reader doesn't see a job at a machine until it
        /// physically arrives there.
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

            // ── Determine lifecycle state ────────────────────

            if (completedOps == job.Operations.Length)
            {
                // All operations done
                tracker.State = JobLifecycleState.Complete;
                tracker.CurrentOperationIndex = job.Operations.Length;
                tracker.CurrentMachineId = -1;
                tracker.NextMachineId = -1;
                tracker.OperationProgress = 0f;
                tracker.PhysicallyAtMachine = false;
            }
            else if (activeOp != null)
            {
                // An operation is actively processing — job IS at the machine.
                tracker.State = JobLifecycleState.Processing;
                tracker.CurrentOperationIndex = activeOpIndex;
                tracker.CurrentMachineId = activeOp.MachineId;
                tracker.OperationProgress = ComputeProgress(activeOp, currentSimTime);
                tracker.PhysicallyAtMachine = true; // processing = definitely there

                int nextIdx = activeOpIndex + 1;
                tracker.NextMachineId = nextIdx < job.Operations.Length
                    ? job.Operations[nextIdx].MachineId
                    : -1;
            }
            else if (completedOps > 0)
            {
                // Some operations complete, none active.
                tracker.CurrentOperationIndex = completedOps;
                tracker.OperationProgress = 0f;

                int nextMachineId = job.Operations[completedOps].MachineId;
                Machine nextMachine = simulator.Machines[nextMachineId];
                bool inDESQueue = nextMachine.WaitingQueue.Any(op => op.JobId == tracker.JobId);

                // if (inDESQueue && tracker.PhysicallyAtMachine)
                if (inDESQueue)
                {
                    // DES has queued it AND it's been physically delivered
                    tracker.State = JobLifecycleState.Queued;
                    tracker.CurrentMachineId = nextMachineId;
                    tracker.PhysicallyAtMachine = true;
                }
                else
                {
                    // Needs transport to next machine.
                    // Pre-AGV: stays in incoming queue until DES starts processing.
                    // Post-AGV: AGV will call BeginTransit → CompleteTransit.
                    tracker.State = JobLifecycleState.WaitingForTransport;
                    tracker.CurrentMachineId = -1;
                    tracker.PhysicallyAtMachine = false;
                }

                tracker.NextMachineId = nextMachineId;
            }
            else
            {
                // No operations started yet.
                int firstMachineId = job.Operations[0].MachineId;
                Machine firstMachine = simulator.Machines[firstMachineId];

                // Check if the DES has assigned it to the first machine's queue
                bool inFirstQueue = firstMachine.WaitingQueue.Any(op => op.JobId == tracker.JobId);

                if (inFirstQueue)
                {
                    tracker.State = JobLifecycleState.Queued;
                    tracker.CurrentMachineId = firstMachineId;
                    tracker.PhysicallyAtMachine = true; // Send it to the machine!
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
            // else
            // {
            //     // No operations started yet.
            //     int firstMachineId = job.Operations[0].MachineId;

            //     // The DES may have already queued this job on the machine,
            //     // but physically it's still in the incoming queue area.
            //     // Pre-AGV: job stays in incoming queue until processing starts.
            //     // Post-AGV: AGV delivers it, then it's physically queued.

            //     if (tracker.PhysicallyAtMachine)
            //     {
            //         // Only true if an AGV (future) delivered it
            //         tracker.State = JobLifecycleState.Queued;
            //         tracker.CurrentMachineId = firstMachineId;
            //     }
            //     else
            //     {
            //         tracker.State = JobLifecycleState.NotStarted;
            //         tracker.CurrentMachineId = -1;
            //     }

            //     tracker.NextMachineId = firstMachineId;
            //     tracker.CurrentOperationIndex = 0;
            //     tracker.OperationProgress = 0f;
            // }
        }

        /// @brief Computes 0-1 progress for an active operation.
        private float ComputeProgress(Operation op, double currentSimTime)
        {
            if (op.Duration <= 0) return 1f;
            double elapsed = currentSimTime - op.StartTime;
            return Mathf.Clamp01((float)(elapsed / op.Duration));
        }

        /// @brief Sets the tracker's world position based on physical delivery state.
        ///
        /// @details The key rule: if PhysicallyAtMachine is false, the job goes
        /// to the incoming queue area (or exit area if complete). Only jobs that
        /// have been physically delivered appear at machines.
        ///
        /// Pre-AGV: jobs jump from incoming queue → machine when processing starts.
        /// Post-AGV: jobs move via AGV from incoming queue → machine.
        private void UpdateTrackerPosition(JobTracker tracker)
        {
            switch (tracker.State)
            {
                case JobLifecycleState.NotStarted:
                case JobLifecycleState.WaitingForTransport:
                    // Job is in the incoming queue area, waiting for transport.
                    tracker.WorldPosition = GetIncomingQueuePosition(tracker.IncomingQueueSlot);
                    break;

                case JobLifecycleState.Complete:
                    tracker.WorldPosition = GetExitAreaPosition(tracker.IncomingQueueSlot);
                    break;

                case JobLifecycleState.Processing:
                    // Job is actively on the machine — position at the machine.
                    if (tracker.CurrentMachineId >= 0 && layoutManager != null)
                    {
                        MachineVisual mv = layoutManager.GetMachineVisual(tracker.CurrentMachineId);
                        if (mv != null)
                        {
                            tracker.WorldPosition = mv.transform.position +
                                Vector3.up * jobTokenHeight;
                        }
                    }
                    break;

                case JobLifecycleState.Queued:
                    if (tracker.CurrentMachineId >= 0 && layoutManager != null)
                    {
                        MachineVisual mv = layoutManager.GetMachineVisual(tracker.CurrentMachineId);
                        if (mv != null)
                        {
                            float offset = GetMachineQueueSlotOffset(tracker.CurrentMachineId, tracker.JobId);
                            tracker.WorldPosition = mv.transform.position +
                                Vector3.up * jobTokenHeight +
                                machineQueueDirection.normalized * (offset + queueSpacing);
                        }
                    }
                    else
                    {
                        tracker.WorldPosition = GetIncomingQueuePosition(tracker.IncomingQueueSlot);
                    }
                    break;

                // case JobLifecycleState.Queued:
                //     // Job has been physically delivered and is waiting at the machine.
                //     if (tracker.PhysicallyAtMachine && tracker.CurrentMachineId >= 0 && layoutManager != null)
                //     {
                //         MachineVisual mv = layoutManager.GetMachineVisual(tracker.CurrentMachineId);
                //         if (mv != null)
                //         {
                //             float offset = GetMachineQueueSlotOffset(tracker.CurrentMachineId, tracker.JobId);
                //             tracker.WorldPosition = mv.transform.position +
                //                 Vector3.up * jobTokenHeight +
                //                 machineQueueDirection.normalized * (offset + queueSpacing);
                //         }
                //     }
                //     else
                //     {
                //         // DES says queued but not physically delivered — incoming area
                //         tracker.WorldPosition = GetIncomingQueuePosition(tracker.IncomingQueueSlot);
                //     }
                //     break;

                case JobLifecycleState.InTransit:
                    // Position is set by AGVPool.UpdateTransit() once implemented.
                    // For now, stay at incoming queue position (AGV hasn't arrived yet).
                    tracker.WorldPosition = GetIncomingQueuePosition(tracker.IncomingQueueSlot);
                    break;
            }
        }

        // ─────────────────────────────────────────────────────────
        //  Grid layout helpers
        // ─────────────────────────────────────────────────────────

        /// @brief Computes the world position for a slot in the incoming queue grid.
        ///
        /// @details Lays out jobs in a grid: rows along queueRowDirection,
        /// columns along queueColumnDirection. Wraps after queueRowSize.
        ///
        ///   Slot 0  Slot 1  Slot 2  ... Slot 9     ← row 0
        ///   Slot 10 Slot 11 Slot 12 ... Slot 19    ← row 1
        ///
        /// @param slot The slot index (typically == jobId).
        /// @returns World-space position for that slot.
        private Vector3 GetIncomingQueuePosition(int slot)
        {
            Vector3 origin = incomingQueueMarker != null
                ? incomingQueueMarker.position
                : incomingQueueOrigin;

            int row = slot / Mathf.Max(queueRowSize, 1);
            int col = slot % Mathf.Max(queueRowSize, 1);

            Vector3 rowDir = queueRowDirection.normalized;
            Vector3 colDir = queueColumnDirection.normalized;

            return origin
                + rowDir * (col * queueGridSpacing)
                + colDir * (row * queueGridSpacing)
                + Vector3.up * jobTokenHeight;
        }

        /// @brief Computes the world position for a slot in the exit area grid.
        /// Same grid layout as incoming queue, offset to the exit area.
        private Vector3 GetExitAreaPosition(int slot)
        {
            Vector3 origin = exitAreaMarker != null
                ? exitAreaMarker.position
                : exitAreaOrigin;

            int row = slot / Mathf.Max(queueRowSize, 1);
            int col = slot % Mathf.Max(queueRowSize, 1);

            Vector3 rowDir = queueRowDirection.normalized;
            Vector3 colDir = queueColumnDirection.normalized;

            return origin
                + rowDir * (col * queueGridSpacing)
                + colDir * (row * queueGridSpacing)
                + Vector3.up * jobTokenHeight;
        }

        /// @brief Computes the queue slot offset for a job at a given machine.
        /// Jobs further back in the machine's physical queue get offset further.
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
        //  Query API — for ObservationBuilder (IoT sensor reads)
        // ─────────────────────────────────────────────────────────

        /// @brief Returns the tracker for a specific job.
        /// IoT equivalent: full sensor readout for one tagged item.
        public JobTracker GetJobTracker(int jobId)
        {
            if (trackers == null || jobId < 0 || jobId >= trackers.Length)
                return null;
            return trackers[jobId];
        }

        /// @brief Returns the world position of a job.
        /// IoT equivalent: RFID / UWB localization read.
        public Vector3 GetJobPosition(int jobId)
        {
            JobTracker t = GetJobTracker(jobId);
            return t?.WorldPosition ?? Vector3.zero;
        }

        /// @brief Returns the lifecycle state of a job.
        /// IoT equivalent: MES status register query.
        public JobLifecycleState GetJobState(int jobId)
        {
            JobTracker t = GetJobTracker(jobId);
            return t?.State ?? JobLifecycleState.NotStarted;
        }

        /// @brief Returns the 0-1 progress of the current operation.
        /// IoT equivalent: PLC cycle progress / elapsed ratio.
        public float GetOperationProgress(int jobId)
        {
            JobTracker t = GetJobTracker(jobId);
            return t?.OperationProgress ?? 0f;
        }

        /// @brief Cumulative time this job has spent waiting in queues.
        /// IoT equivalent: sum of RFID timestamps at queue entry/exit.
        public double GetWaitTime(int jobId)
        {
            JobTracker t = GetJobTracker(jobId);
            return t?.TotalWaitTime ?? 0;
        }

        /// @brief Cumulative transit time (0 until AGVPool exists).
        public double GetTransitTime(int jobId)
        {
            JobTracker t = GetJobTracker(jobId);
            return t?.TotalTransitTime ?? 0;
        }

        // ─────────────────────────────────────────────────────────
        //  Bulk query API — for ObservationBuilder tensor construction
        // ─────────────────────────────────────────────────────────

        /// @brief Builds the scheduling matrix data for all jobs.
        ///
        /// @details Returns a flat array of shape [numJobs × numMachines × 3]:
        ///   Channel 0: Machine assignment (1.0 if operation o is on machine m, else 0)
        ///   Channel 1: Processing time (normalized by max duration)
        ///   Channel 2: Status (0 = not started, 0.5 = in progress, 1.0 = complete)
        ///
        /// The ObservationBuilder will reshape this into the (n × 2m × 3) image
        /// that feeds the CNN-SPPF encoder.
        ///
        /// @param numMachines Number of machines in the instance.
        /// @returns Flat float array, or null if not initialized.
        public float[] GetSchedulingMatrixFlat(int numMachines)
        {
            if (!initialized) return null;

            int numJobs = trackers.Length;
            float[] matrix = new float[numJobs * numMachines * 3];

            // Find max duration for normalization
            float maxDuration = 1f;
            foreach (JobTracker t in trackers)
            {
                foreach (float d in t.OperationDurations)
                {
                    if (d > maxDuration) maxDuration = d;
                }
            }

            for (int j = 0; j < numJobs; j++)
            {
                JobTracker t = trackers[j];

                for (int o = 0; o < t.TotalOperations; o++)
                {
                    int m = t.OperationMachineIds[o];
                    if (m < 0 || m >= numMachines) continue;

                    int baseIdx = (j * numMachines + m) * 3;

                    matrix[baseIdx + 0] = 1.0f;                              // assignment
                    matrix[baseIdx + 1] = t.OperationDurations[o] / maxDuration; // normalized duration
                    matrix[baseIdx + 2] = t.OperationStatuses[o];            // status
                }
            }

            return matrix;
        }

        /// @brief Returns all job positions as a flat array [x0, z0, x1, z1, ...].
        ///
        /// @details For rasterizing into the spatial occupancy grid's job layer.
        /// Y (height) is omitted since the grid is a top-down 2D projection.
        ///
        /// @returns Flat array of length numJobs × 2, or null if not initialized.
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

        /// @brief Returns per-job summary scalars for the global observation.
        ///
        /// @details Array of [numJobs × 4]:
        ///   [0] Normalized progress (completedOps / totalOps)
        ///   [1] Current operation progress (0-1)
        ///   [2] Normalized wait time (waitTime / simTime)
        ///   [3] State as float (0=NotStarted, 0.2=Queued, 0.4=Processing,
        ///       0.6=WaitingForTransport, 0.8=InTransit, 1.0=Complete)
        ///
        /// @param currentSimTime Current simulation clock, used to normalize wait times.
        public float[] GetJobScalarsFlat(double currentSimTime)
        {
            if (!initialized) return null;

            float[] scalars = new float[trackers.Length * 4];
            double timeNorm = Math.Max(currentSimTime, 1.0);

            for (int j = 0; j < trackers.Length; j++)
            {
                JobTracker t = trackers[j];
                int idx = j * 4;

                scalars[idx + 0] = t.TotalOperations > 0
                    ? (float)t.CompletedOperations / t.TotalOperations
                    : 0f;
                scalars[idx + 1] = t.OperationProgress;
                scalars[idx + 2] = (float)(t.TotalWaitTime / timeNorm);
                scalars[idx + 3] = StateToFloat(t.State);
            }

            return scalars;
        }

        /// @brief Encodes lifecycle state as a normalized float for the neural network.
        private static float StateToFloat(JobLifecycleState state)
        {
            switch (state)
            {
                case JobLifecycleState.NotStarted: return 0.0f;
                case JobLifecycleState.Queued: return 0.2f;
                case JobLifecycleState.Processing: return 0.4f;
                case JobLifecycleState.WaitingForTransport: return 0.6f;
                case JobLifecycleState.InTransit: return 0.8f;
                case JobLifecycleState.Complete: return 1.0f;
                default: return 0.0f;
            }
        }

        // ─────────────────────────────────────────────────────────
        //  Event API — for future AGVPool integration
        // ─────────────────────────────────────────────────────────

        /// @brief Called by AGVPool when an AGV picks up a job and begins transport.
        ///
        /// @param jobId The job being transported.
        /// @param destinationMachineId The machine the AGV is heading to.
        /// @param simTime Current simulation time.
        public void BeginTransit(int jobId, int destinationMachineId, double simTime)
        {
            JobTracker t = GetJobTracker(jobId);
            if (t == null) return;

            t.State = JobLifecycleState.InTransit;
            t.NextMachineId = destinationMachineId;
            t.StateEntryTime = simTime;

            Debug.Log($"[JobManager] Job {jobId}: transit to M{destinationMachineId}");
        }

        /// @brief Called by AGVPool when the AGV arrives and deposits the job.
        ///
        /// @param jobId The job that was delivered.
        /// @param machineId The machine where the job was deposited.
        /// @param simTime Current simulation time.
        public void CompleteTransit(int jobId, int machineId, double simTime)
        {
            JobTracker t = GetJobTracker(jobId);
            if (t == null) return;

            double transitDuration = simTime - t.StateEntryTime;
            t.TotalTransitTime += transitDuration;
            t.State = JobLifecycleState.Queued;
            t.CurrentMachineId = machineId;
            t.StateEntryTime = simTime;

            Debug.Log($"[JobManager] Job {jobId}: arrived at M{machineId} " +
                      $"(transit took {transitDuration:F1})");
        }

        // ─────────────────────────────────────────────────────────
        //  Diagnostics
        // ─────────────────────────────────────────────────────────

        /// @brief Logs a summary of all job states. Useful for debugging.
        public void LogSummary()
        {
            if (!initialized)
            {
                Debug.Log("[JobManager] Not initialized.");
                return;
            }

            int[] stateCounts = new int[Enum.GetValues(typeof(JobLifecycleState)).Length];
            foreach (JobTracker t in trackers)
                stateCounts[(int)t.State]++;

            Debug.Log($"[JobManager] {trackers.Length} jobs: " +
                      $"NotStarted={stateCounts[0]}, " +
                      $"Queued={stateCounts[1]}, " +
                      $"Processing={stateCounts[2]}, " +
                      $"WaitingForTransport={stateCounts[3]}, " +
                      $"InTransit={stateCounts[4]}, " +
                      $"Complete={stateCounts[5]}");
        }
    }
}