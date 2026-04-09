using UnityEngine;
using Assets.Scripts.Simulation.Machines;
using Assets.Scripts.Simulation.Jobs;

namespace Assets.Scripts.Simulation.Machines
{
    /// <summary>
    /// Physical machine in the scene. Counts down a processing timer and sets a flag.
    /// Does NOT call SimulationBridge. Does NOT touch JobManager/JobStore.
    /// The orchestrator reads FinishedFlag and drives all state transitions.
    ///
    /// Conveyor belts remain attached for visuals, but the orchestrator tells us
    /// when to add/remove visuals — we never decide on our own.
    /// </summary>
    [RequireComponent(typeof(Collider))]
    [RequireComponent(typeof(Rigidbody))]
    public class PhysicalMachine : MonoBehaviour
    {
        // ── Identity (set once by layout manager) ────────────────
        public int MachineId { get; private set; }
        public MachineType MachineType { get; private set; }

        // ── State (read by orchestrator) ─────────────────────────

        /// Machine has no active job.
        public bool IsIdle { get; private set; } = true;

        /// The job we just finished. Orchestrator reads this, then calls ClearFinished().
        public bool FinishedFlag { get; private set; }

        /// Which job is currently processing (or just finished).
        public int ActiveJobId { get; private set; } = -1;

        // ── Internals ────────────────────────────────────────────
        private float remainingTime;
        private float totalDuration;

        [Header("Conveyor Belts (visual only)")]
        [SerializeField] private ConveyorBelt incomingConveyor;
        [SerializeField] private ConveyorBelt outgoingConveyor;
        [SerializeField] private ConveyorBelt secondaryIncomingConveyor;
        [SerializeField] private ConveyorBelt secondaryOutgoingConveyor;



        private MachineVisual visualLayer;

        // ─────────────────────────────────────────────────────────
        //  Setup
        // ─────────────────────────────────────────────────────────

        public void Initialize(int id, MachineType type)
        {
            MachineId = id;
            MachineType = type;
            IsIdle = true;
            FinishedFlag = false;
            ActiveJobId = -1;

            visualLayer = GetComponent<MachineVisual>();
            visualLayer?.Initialise(id, type);
            ClearConveyors();
        }

        // ─────────────────────────────────────────────────────────
        //  Commands (called ONLY by orchestrator)
        // ─────────────────────────────────────────────────────────

        /// <summary>
        /// Orchestrator tells us to start processing. We just count down.
        /// </summary>
        public void StartJob(int jobId, float duration, JobVisual visual = null)
        {
            ActiveJobId = jobId;
            remainingTime = duration;
            totalDuration = duration;
            IsIdle = false;
            FinishedFlag = false;

            // Visual: remove from incoming belt, show at machine center
            RemoveFromAnyIncoming(jobId);

            if (visual != null)
            {
                visual.SetOnConveyor(false);
                visual.SnapToPosition(transform.position);
            }
            visualLayer.BeginOperation(jobId, Time.time, duration);
        }

        /// <summary>
        /// Orchestrator acknowledges the finished flag. Resets for next job.
        /// </summary>
        public void ClearFinished()
        {
            FinishedFlag = false;
            IsIdle = true;
            ActiveJobId = -1;
            visualLayer?.CompleteOperation(-1);
        }

        // ─────────────────────────────────────────────────────────
        //  Visual helpers (called by orchestrator for belt visuals)
        // ─────────────────────────────────────────────────────────

        /// Place a job visual on the incoming conveyor (AGV just delivered it).
        public void PlaceOnIncoming(int jobId, JobVisual visual)
        {
            ConveyorBelt belt = PickIncomingBelt();
            if (belt != null && visual != null)
            {
                belt.TryEnqueue(jobId, visual);
                visual.SetOnConveyor(true);
            }
        }

        /// Remove a job visual from the outgoing conveyor (AGV picking it up).
        public void RemoveFromOutgoing(int jobId)
        {
            if (outgoingConveyor != null && outgoingConveyor.Contains(jobId))
                outgoingConveyor.RemoveJob(jobId);
            else if (secondaryOutgoingConveyor != null && secondaryOutgoingConveyor.Contains(jobId))
                secondaryOutgoingConveyor.RemoveJob(jobId);
        }

        /// Place a finished job visual on the outgoing conveyor.
        public void PlaceOnOutgoing(int jobId, JobVisual visual)
        {
            ConveyorBelt belt = PickOutgoingBelt();
            if (belt != null && visual != null)
            {
                belt.TryEnqueue(jobId, visual);
                visual.SetOnConveyor(true);
            }
        }

        // ── AGV docking positions ────────────────────────────────

        public Vector3 GetDropoffPosition()
        {
            ConveyorBelt belt = PickIncomingBelt();
            if (belt != null) return belt.InputEndPosition;
            return transform.position + transform.TransformDirection(new Vector3(-2.5f, 0.5f, 0f));
        }

        public Vector3 GetPickupPosition()
        {
            if (outgoingConveyor != null) return outgoingConveyor.OutputEndPosition;
            if (secondaryOutgoingConveyor != null) return secondaryOutgoingConveyor.OutputEndPosition;
            return transform.position + transform.TransformDirection(new Vector3(2.5f, 0.5f, 0f));
        }

        // ─────────────────────────────────────────────────────────
        //  Update — ONLY counts down the timer
        // ─────────────────────────────────────────────────────────

        private void Update()
        {
            if (IsIdle || FinishedFlag) return;

            remainingTime -= Time.deltaTime;

            // Update visual progress bar
            // (we don't know total duration here, so visualLayer tracks it from BeginOperation)
            if (visualLayer != null && remainingTime > 0f)
                visualLayer.UpdateProgress(1f - (remainingTime / Mathf.Max(totalDuration, 0.001f)));
            if (remainingTime <= 0f)
            {
                FinishedFlag = true;
                // That's it. We do NOT call anything else.
                // The orchestrator will pick this up next frame.
            }
        }
        public void RefreshQueueLabels(int incomingCount, int outgoingCount)
        {
            visualLayer.UpdateIncomingQueueLabel(incomingCount);
            visualLayer.UpdateOutgoingQueueLabel(outgoingCount);
        }

        // ─────────────────────────────────────────────────────────
        //  Internal belt helpers
        // ─────────────────────────────────────────────────────────

        private void RemoveFromAnyIncoming(int jobId)
        {
            if (incomingConveyor != null && incomingConveyor.Contains(jobId))
                incomingConveyor.RemoveJob(jobId);
            else if (secondaryIncomingConveyor != null && secondaryIncomingConveyor.Contains(jobId))
                secondaryIncomingConveyor.RemoveJob(jobId);
        }

        private ConveyorBelt PickIncomingBelt()
        {
            if (incomingConveyor != null && !incomingConveyor.IsFull) return incomingConveyor;
            if (secondaryIncomingConveyor != null && !secondaryIncomingConveyor.IsFull) return secondaryIncomingConveyor;
            return incomingConveyor ?? secondaryIncomingConveyor;
        }

        private ConveyorBelt PickOutgoingBelt()
        {
            if (outgoingConveyor != null && !outgoingConveyor.IsFull) return outgoingConveyor;
            if (secondaryOutgoingConveyor != null && !secondaryOutgoingConveyor.IsFull) return secondaryOutgoingConveyor;
            return outgoingConveyor ?? secondaryOutgoingConveyor;
        }


        private void ClearConveyors()
        {
            incomingConveyor?.Clear();
            outgoingConveyor?.Clear();
            secondaryIncomingConveyor?.Clear();
            secondaryOutgoingConveyor?.Clear();
        }

        public void FullReset()
        {
            IsIdle = true;
            FinishedFlag = false;
            ActiveJobId = -1;
            remainingTime = 0f;
            ClearConveyors();
        }
    }
}