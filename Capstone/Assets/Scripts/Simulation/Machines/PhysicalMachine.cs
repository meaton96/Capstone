using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Assets.Scripts.Simulation.Jobs;
using Assets.Scripts.Logging;

namespace Assets.Scripts.Simulation.Machines
{
    /// @brief Physical anchor for a machine in the Unity scene.
    /// @details Manages real-time processing via coroutines and delegates visual updates
    /// to MachineVisual. Conveyor belts are VISUAL ONLY — job ownership is tracked
    /// exclusively by JobManager. Queue queries go through JobManager, not belt contents.
    [RequireComponent(typeof(Collider))]
    [RequireComponent(typeof(Rigidbody))]
    [RequireComponent(typeof(MachineVisual))]
    public class PhysicalMachine : MonoBehaviour
    {
        public int MachineId { get; private set; }
        public MachineType MachineType { get; private set; }
        public bool IsIdle { get; private set; } = true;
        public bool IsBlocked { get; private set; } = false;

        [Header("Primary Conveyor Belts")]
        [SerializeField] private ConveyorBelt incomingConveyor;
        [SerializeField] private ConveyorBelt outgoingConveyor;

        [Header("Secondary Conveyor Belts (double-sided machines)")]
        [SerializeField] private ConveyorBelt secondaryIncomingConveyor;
        [SerializeField] private ConveyorBelt secondaryOutgoingConveyor;

        private bool preferSecondaryInput;
        private bool preferSecondaryOutput;
        private MachineVisual visualLayer;

        // ─────────────────────────────────────────────────────────
        //  Queue queries — delegate to JobManager (single source of truth)
        // ─────────────────────────────────────────────────────────

        /// @brief Jobs logically queued at this machine's incoming area.
        ///        Reads from JobManager, NOT from conveyor belt contents.
        public List<int> IncomingQueue =>
            SimulationBridge.Instance?.JobManager?.GetJobsInMachineQueue(MachineId) ?? new List<int>();

        /// @brief Jobs logically on this machine's outgoing area.
        public List<int> OutgoingQueue =>
            SimulationBridge.Instance?.JobManager?.GetJobsOnMachineOutgoing(MachineId) ?? new List<int>();

        /// @brief Alias for IncomingQueue — what the scheduler sees.
        public List<int> PhysicalQueue => IncomingQueue;

        /// @brief Visual-only counts for the HUD labels.
        private int VisualIncomingCount => (incomingConveyor?.Count ?? 0) + (secondaryIncomingConveyor?.Count ?? 0);
        private int VisualOutgoingCount => (outgoingConveyor?.Count ?? 0) + (secondaryOutgoingConveyor?.Count ?? 0);

        // ─────────────────────────────────────────────────────────
        //  Initialization
        // ─────────────────────────────────────────────────────────

        public void Initialize(int id, MachineType type)
        {
            MachineId = id;
            MachineType = type;
            IsIdle = true;
            IsBlocked = false;
            preferSecondaryInput = false;
            preferSecondaryOutput = false;

            visualLayer = GetComponent<MachineVisual>();
            if (visualLayer != null)
                visualLayer.Initialise(id, type);

            ResetQueues();
        }

        public void ResetQueues()
        {
            incomingConveyor?.Clear();
            secondaryIncomingConveyor?.Clear();
            outgoingConveyor?.Clear();
            secondaryOutgoingConveyor?.Clear();
        }

        // ─────────────────────────────────────────────────────────
        //  AGV docking positions
        // ─────────────────────────────────────────────────────────

        /// @brief World position where an AGV should deliver a job.
        ///        Resolved fresh each time — never cached/stale.
        public Vector3 GetDropoffPosition(int jobId)
        {
            ConveyorBelt belt = PickNextIncomingBelt();
            if (belt != null)
                return belt.InputEndPosition;
            return transform.position + transform.TransformDirection(new Vector3(-2.5f, 0.5f, 0f));
        }

        /// @brief World position where an AGV should pick up a job.
        public Vector3 GetPickupPositionForJob(int jobId)
        {
            if (outgoingConveyor != null && outgoingConveyor.Contains(jobId))
                return outgoingConveyor.OutputEndPosition;
            if (secondaryOutgoingConveyor != null && secondaryOutgoingConveyor.Contains(jobId))
                return secondaryOutgoingConveyor.OutputEndPosition;

            // Fallback: job might not be on visual belt yet
            if (outgoingConveyor != null) return outgoingConveyor.OutputEndPosition;
            if (secondaryOutgoingConveyor != null) return secondaryOutgoingConveyor.OutputEndPosition;
            return transform.position + transform.TransformDirection(new Vector3(2.5f, 0.5f, 0f));
        }

        // ─────────────────────────────────────────────────────────
        //  Job reception (AGV delivers a job here)
        // ─────────────────────────────────────────────────────────

        /// @brief Places a job's visual on an incoming belt.
        /// @details The job is ALREADY logically at this machine (TransitionJob was
        ///          called by AGVController.DoDropoff). This method only handles the
        ///          visual placement on the conveyor belt. It CANNOT fail in a way
        ///          that loses the job — worst case the visual snaps to machine center.
        public void ReceiveJobVisual(int jobId, JobVisual visual)
        {
            ConveyorBelt belt = PickClosestIncoming(visual);

            if (belt != null && !belt.TryEnqueue(jobId, visual))
            {
                // Primary belt full, try secondary
                ConveyorBelt fallback = GetOtherIncoming(belt);
                if (fallback == null || !fallback.TryEnqueue(jobId, visual))
                {
                    // All belts full — snap visual to machine center as fallback
                    if (visual != null)
                    {
                        visual.SetOnConveyor(false);
                        visual.SnapToPosition(transform.position + Vector3.up * 0.5f);
                    }
                    SimLogger.LogWarning($"[PhysicalMachine M{MachineId}] Belt full for visual of Job {jobId} — snapped to center.");
                }
            }
            else if (belt == null)
            {
                // No conveyor at all — snap to machine
                if (visual != null)
                {
                    visual.SetOnConveyor(false);
                    visual.SnapToPosition(transform.position + Vector3.up * 0.5f);
                }
            }

            if (visual != null)
                visual.SetState(JobLifecycleState.Queued);

            visualLayer?.UpdateIncomingQueueLabel(VisualIncomingCount);
        }

        // ─────────────────────────────────────────────────────────
        //  Processing
        // ─────────────────────────────────────────────────────────

        /// @brief Starts processing a job. Removes it from the visual incoming belt.
        public void StartProcessing(int jobId, float realTimeDuration)
        {
            IsIdle = false;
            IsBlocked = false;

            // Remove from visual belt
            RemoveVisualFromAnyIncoming(jobId);

            JobTracker tracker = SimulationBridge.Instance.JobManager.GetJobTracker(jobId);
            JobVisual visual = tracker?.Visual;
            if (visual != null)
            {
                visual.SetOnConveyor(false);
                visual.SetState(JobLifecycleState.Processing);
                visual.SetTargetPosition(transform.position);
            }

            visualLayer?.BeginOperation(jobId, Time.time, realTimeDuration);
            visualLayer?.UpdateIncomingQueueLabel(VisualIncomingCount);
            SimLogger.High($"[Machine] {MachineId} began processing job {jobId} for {realTimeDuration} seconds.");
            StartCoroutine(ProcessJobRoutine(jobId, realTimeDuration));
        }

        /// @brief AGV has picked up a job from our outgoing belt — remove its visual.
        public void ReleaseVisualFromOutgoing(int jobId)
        {
            if (outgoingConveyor != null && outgoingConveyor.Contains(jobId))
                outgoingConveyor.RemoveJob(jobId);
            else if (secondaryOutgoingConveyor != null && secondaryOutgoingConveyor.Contains(jobId))
                secondaryOutgoingConveyor.RemoveJob(jobId);

            visualLayer?.UpdateOutgoingQueueLabel(VisualOutgoingCount);
        }

        // ─────────────────────────────────────────────────────────
        //  Processing coroutine
        // ─────────────────────────────────────────────────────────

        private IEnumerator ProcessJobRoutine(int jobId, float duration)
        {
            yield return null;

            float elapsed = 0f;
            while (elapsed < duration)
            {
                elapsed += Time.deltaTime;
                visualLayer?.UpdateProgress(elapsed / duration);
                yield return null;
            }

            JobTracker tracker = SimulationBridge.Instance.JobManager.GetJobTracker(jobId);
            JobVisual visual = tracker?.Visual;

            // Guard: if job was already completed or moved away, abort this coroutine
            if (tracker == null || tracker.CurrentOperationIndex >= tracker.TotalOperations)
            {
                SimLogger.LogWarning($"[ProcessJobRoutine] M{MachineId} job={jobId} — aborting stale coroutine " +
                                     $"(opIdx={tracker?.CurrentOperationIndex}/{tracker?.TotalOperations})");
                IsIdle = true;
                yield break;
            }

            // Wait if all outgoing belts are visually full
            if (AllOutgoingFull())
            {
                IsBlocked = true;
                visualLayer?.SetBlockedAfterProcessing(jobId);
                while (AllOutgoingFull()) yield return null;
                IsBlocked = false;
            }

            // Place on visual outgoing belt
            ConveyorBelt outBelt = PickNextOutgoingBelt();
            if (outBelt != null)
            {
                if (visual != null)
                    visual.SetState(JobLifecycleState.WaitingForTransport);
                outBelt.TryEnqueue(jobId, visual);
                visualLayer?.UpdateOutgoingQueueLabel(VisualOutgoingCount);
            }

            IsIdle = true;
            visualLayer?.CompleteOperation(jobId);
            SimulationBridge.Instance?.OnMachineFinished(MachineId, jobId);
        }

        // ─────────────────────────────────────────────────────────
        //  Belt visual helpers (internal)
        // ─────────────────────────────────────────────────────────

        private ConveyorBelt PickClosestIncoming(JobVisual visual)
        {
            bool hasA = incomingConveyor != null;
            bool hasB = secondaryIncomingConveyor != null;
            if (hasA && !hasB) return incomingConveyor;
            if (hasB && !hasA) return secondaryIncomingConveyor;
            if (!hasA && !hasB) return null;
            if (visual == null) return incomingConveyor;

            float distA = Vector3.Distance(visual.transform.position, incomingConveyor.InputEndPosition);
            float distB = Vector3.Distance(visual.transform.position, secondaryIncomingConveyor.InputEndPosition);
            return distA <= distB ? incomingConveyor : secondaryIncomingConveyor;
        }

        private ConveyorBelt GetOtherIncoming(ConveyorBelt belt)
        {
            if (belt == incomingConveyor) return secondaryIncomingConveyor;
            if (belt == secondaryIncomingConveyor) return incomingConveyor;
            return null;
        }

        private ConveyorBelt PickNextIncomingBelt()
        {
            ConveyorBelt a = preferSecondaryInput ? secondaryIncomingConveyor : incomingConveyor;
            ConveyorBelt b = preferSecondaryInput ? incomingConveyor : secondaryIncomingConveyor;
            preferSecondaryInput = !preferSecondaryInput;
            if (a != null && !a.IsFull) return a;
            if (b != null && !b.IsFull) return b;
            return a ?? b;
        }

        private ConveyorBelt PickNextOutgoingBelt()
        {
            ConveyorBelt a = preferSecondaryOutput ? secondaryOutgoingConveyor : outgoingConveyor;
            ConveyorBelt b = preferSecondaryOutput ? outgoingConveyor : secondaryOutgoingConveyor;
            preferSecondaryOutput = !preferSecondaryOutput;
            if (a != null && !a.IsFull) return a;
            if (b != null && !b.IsFull) return b;
            return a ?? b;
        }

        private void RemoveVisualFromAnyIncoming(int jobId)
        {
            if (incomingConveyor != null && incomingConveyor.Contains(jobId))
                incomingConveyor.RemoveJob(jobId);
            else if (secondaryIncomingConveyor != null && secondaryIncomingConveyor.Contains(jobId))
                secondaryIncomingConveyor.RemoveJob(jobId);
        }

        private bool AllOutgoingFull()
        {
            if (outgoingConveyor != null && !outgoingConveyor.IsFull) return false;
            if (secondaryOutgoingConveyor != null && !secondaryOutgoingConveyor.IsFull) return false;
            return true;
        }
    }
}