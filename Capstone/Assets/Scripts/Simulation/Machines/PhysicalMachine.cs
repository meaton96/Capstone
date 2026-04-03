using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Assets.Scripts.Scheduling.Core;

namespace Assets.Scripts.Simulation.Machines
{
    /// <summary>
    /// Physical anchor for a machine in the Unity scene.
    ///
    /// Manages real-time coroutine processing and delegates visual updates to
    /// <see cref="MachineVisual"/>. Supports up to two incoming and two outgoing
    /// <see cref="ConveyorBelt"/> components for double-sided machines.
    ///
    /// <para><b>Belt selection:</b>
    /// <list type="bullet">
    /// <item><see cref="ReserveIncomingSlot"/> alternates between incoming belts
    /// that have space, so successive jobs get distributed across both sides.</item>
    /// <item><see cref="ReceiveJob"/> uses proximity — it places the job on
    /// whichever incoming belt is closest to the visual's current position
    /// (matching the side the AGV physically delivered to).</item>
    /// <item>The processing coroutine alternates which outgoing belt it
    /// tries first when outputting a finished job.</item>
    /// </list></para>
    ///
    /// <para><b>Blocking behaviour:</b> when ALL outgoing conveyors are full
    /// after processing completes, the machine enters <c>IsBlocked</c> and
    /// holds the finished job until any belt has space.</para>
    ///
    /// <para>Edge-row machines with only one pair of belts work identically
    /// to before — the secondary references are simply left unassigned.</para>
    /// </summary>
    [RequireComponent(typeof(Collider))]
    [RequireComponent(typeof(Rigidbody))]
    [RequireComponent(typeof(MachineVisual))]
    public class PhysicalMachine : MonoBehaviour
    {
        // ─────────────────────────────────────────────────────────
        //  Identity
        // ─────────────────────────────────────────────────────────

        public int MachineId { get; private set; }
        public bool IsIdle { get; private set; } = true;

        /// <summary>
        /// True when processing is done but all outgoing belts are full.
        /// </summary>
        public bool IsBlocked { get; private set; } = false;

        // ─────────────────────────────────────────────────────────
        //  Conveyor References
        // ─────────────────────────────────────────────────────────

        [Header("Primary Conveyor Belts")]
        [Tooltip("Primary belt that feeds jobs INTO this machine (reverseFlow = true).")]
        [SerializeField] private ConveyorBelt incomingConveyor;

        [Tooltip("Primary belt that carries finished jobs OUT (reverseFlow = false).")]
        [SerializeField] private ConveyorBelt outgoingConveyor;

        [Header("Secondary Conveyor Belts (double-sided machines)")]
        [Tooltip("Secondary incoming belt on the opposite side. Leave empty for single-sided machines.")]
        [SerializeField] private ConveyorBelt secondaryIncomingConveyor;

        [Tooltip("Secondary outgoing belt on the opposite side. Leave empty for single-sided machines.")]
        [SerializeField] private ConveyorBelt secondaryOutgoingConveyor;

        // ─────────────────────────────────────────────────────────
        //  Alternation State
        // ─────────────────────────────────────────────────────────

        private bool preferSecondaryInput;
        private bool preferSecondaryOutput;

        // ─────────────────────────────────────────────────────────
        //  Queue Accessors (backward-compat, now aggregate)
        // ─────────────────────────────────────────────────────────

        public List<int> IncomingQueue
        {
            get
            {
                var ids = new List<int>();
                if (incomingConveyor != null)
                    ids.AddRange(incomingConveyor.GetJobIds());
                if (secondaryIncomingConveyor != null)
                    ids.AddRange(secondaryIncomingConveyor.GetJobIds());
                return ids;
            }
        }

        public List<int> OutgoingQueue
        {
            get
            {
                var ids = new List<int>();
                if (outgoingConveyor != null)
                    ids.AddRange(outgoingConveyor.GetJobIds());
                if (secondaryOutgoingConveyor != null)
                    ids.AddRange(secondaryOutgoingConveyor.GetJobIds());
                return ids;
            }
        }

        public List<int> PhysicalQueue => IncomingQueue;

        private int TotalIncomingCount =>
            (incomingConveyor?.Count ?? 0) +
            (secondaryIncomingConveyor?.Count ?? 0);

        private int TotalOutgoingCount =>
            (outgoingConveyor?.Count ?? 0) +
            (secondaryOutgoingConveyor?.Count ?? 0);

        private MachineVisual visualLayer;

        // ─────────────────────────────────────────────────────────
        //  Initialization
        // ─────────────────────────────────────────────────────────

        public void Initialize(int id, Machine coreMachineData)
        {
            MachineId = id;
            IsIdle = true;
            IsBlocked = false;
            preferSecondaryInput = false;
            preferSecondaryOutput = false;

            visualLayer = GetComponent<MachineVisual>();
            if (visualLayer != null)
                visualLayer.Initialise(id, coreMachineData);

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
        //  Slot Positions (for AGV dispatch)
        // ─────────────────────────────────────────────────────────

        /// <summary>
        /// Returns the input-end position of the next incoming belt with
        /// space, alternating between primary and secondary each call.
        /// The returned position determines which dock the AGV routes to.
        /// </summary>
        public Vector3 ReserveIncomingSlot(int jobId)
        {
            ConveyorBelt belt = PickNextIncomingBelt();
            if (belt != null)
                return belt.InputEndPosition;

            return transform.position
                + transform.TransformDirection(new Vector3(-2.5f, 0.5f, 0f));
        }

        /// <summary>
        /// Returns the output-end position of whichever outgoing belt
        /// currently holds <paramref name="jobId"/>. Falls back to the
        /// primary belt if the job isn't found (shouldn't happen).
        /// </summary>
        public Vector3 GetPickupPositionForJob(int jobId)
        {
            // Check which outgoing belt actually has this job.
            if (outgoingConveyor != null && outgoingConveyor.Contains(jobId))
                return outgoingConveyor.OutputEndPosition;

            if (secondaryOutgoingConveyor != null &&
                secondaryOutgoingConveyor.Contains(jobId))
                return secondaryOutgoingConveyor.OutputEndPosition;

            // Fallback — return any available output position.
            if (outgoingConveyor != null)
                return outgoingConveyor.OutputEndPosition;
            if (secondaryOutgoingConveyor != null)
                return secondaryOutgoingConveyor.OutputEndPosition;

            return transform.position
                + transform.TransformDirection(new Vector3(2.5f, 0.5f, 0f));
        }

        // ─────────────────────────────────────────────────────────
        //  Job Receive (AGV → Incoming Conveyor)
        // ─────────────────────────────────────────────────────────

        /// <summary>
        /// Places a job on the incoming conveyor closest to the visual's
        /// current position (matching the side the AGV delivered to).
        /// If that belt is full, tries the other side as fallback.
        /// </summary>
        public void ReceiveJob(int jobId, JobVisual visual)
        {
            ConveyorBelt belt = PickClosestIncoming(visual);

            if (belt == null)
            {
                Debug.LogError(
                    $"[PhysicalMachine M{MachineId}] No incoming conveyor wired!");
                return;
            }

            if (!belt.TryEnqueue(jobId, visual))
            {
                // Preferred side full — try the other.
                ConveyorBelt fallback = GetOtherIncoming(belt);
                if (fallback == null || !fallback.TryEnqueue(jobId, visual))
                {
                    Debug.LogWarning(
                        $"[PhysicalMachine M{MachineId}] All incoming conveyors " +
                        $"full — cannot accept Job {jobId}.");
                    return;
                }
            }

            if (visual != null)
                visual.SetState(JobLifecycleState.Queued);

            visualLayer?.UpdateIncomingQueueLabel(TotalIncomingCount);
            SimulationBridge.Instance?.OnJobArrivedInQueue(MachineId, jobId);
        }

        // ─────────────────────────────────────────────────────────
        //  Processing Start
        // ─────────────────────────────────────────────────────────

        public void StartProcessing(int jobId, float realTimeDuration)
        {
            IsIdle = false;
            IsBlocked = false;

            // Pull the job off whichever incoming belt holds it.
            RemoveFromAnyIncoming(jobId);

            // Move the token into the machine body.
            JobTracker tracker =
                SimulationBridge.Instance.JobManager.GetJobTracker(jobId);
            JobVisual visual = tracker?.Visual;
            if (visual != null)
            {
                visual.SetOnConveyor(false);
                visual.SetState(JobLifecycleState.Processing);
                visual.SetTargetPosition(transform.position);
            }

            visualLayer?.BeginOperation(jobId, Time.time, realTimeDuration);
            visualLayer?.UpdateIncomingQueueLabel(TotalIncomingCount);
            StartCoroutine(ProcessJobRoutine(jobId, realTimeDuration));
        }

        // ─────────────────────────────────────────────────────────
        //  Outgoing Release (AGV picks up)
        // ─────────────────────────────────────────────────────────

        /// <summary>
        /// Removes a job from whichever outgoing belt holds it.
        /// Called by the AGV when it picks up the job.
        /// </summary>
        public void ReleaseFromOutgoing(int jobId)
        {
            if (outgoingConveyor != null && outgoingConveyor.Contains(jobId))
                outgoingConveyor.RemoveJob(jobId);
            else if (secondaryOutgoingConveyor != null &&
                     secondaryOutgoingConveyor.Contains(jobId))
                secondaryOutgoingConveyor.RemoveJob(jobId);

            visualLayer?.UpdateOutgoingQueueLabel(TotalOutgoingCount);
        }

        // ─────────────────────────────────────────────────────────
        //  Processing Coroutine
        // ─────────────────────────────────────────────────────────

        private IEnumerator ProcessJobRoutine(int jobId, float duration)
        {
            // ── Phase 1: Active processing ──────────────────────
            float elapsed = 0f;
            while (elapsed < duration)
            {
                elapsed += Time.deltaTime;
                visualLayer?.UpdateProgress(elapsed / duration);
                yield return null;
            }

            // Processing is done — grab the visual.
            JobTracker tracker =
                SimulationBridge.Instance.JobManager.GetJobTracker(jobId);
            JobVisual visual = tracker?.Visual;

            // ── Phase 2: Blocked check ──────────────────────────
            // If ALL outgoing belts are full, wait for any slot to open.
            if (AllOutgoingFull())
            {
                IsBlocked = true;
                visualLayer?.SetBlockedAfterProcessing(jobId);

                while (AllOutgoingFull())
                    yield return null;

                IsBlocked = false;
            }

            // ── Phase 3: Output to an outgoing conveyor ─────────
            ConveyorBelt outBelt = PickNextOutgoingBelt();
            if (outBelt != null)
            {
                if (visual != null)
                    visual.SetState(JobLifecycleState.WaitingForTransport);

                outBelt.TryEnqueue(jobId, visual);
                visualLayer?.UpdateOutgoingQueueLabel(TotalOutgoingCount);
            }

            // ── Phase 4: Truly idle — notify the scheduler ──────
            IsIdle = true;
            visualLayer?.CompleteOperation(jobId);
            SimulationBridge.Instance?.OnMachineFinished(MachineId, jobId);
        }

        // ─────────────────────────────────────────────────────────
        //  Belt Selection Helpers
        // ─────────────────────────────────────────────────────────

        /// <summary>
        /// Returns the incoming belt closest to the visual's current
        /// world position, so the job lands on the side the AGV is at.
        /// Falls back to whichever belt exists if only one is wired.
        /// </summary>
        private ConveyorBelt PickClosestIncoming(JobVisual visual)
        {
            bool hasA = incomingConveyor != null;
            bool hasB = secondaryIncomingConveyor != null;

            if (hasA && !hasB) return incomingConveyor;
            if (hasB && !hasA) return secondaryIncomingConveyor;
            if (!hasA && !hasB) return null;

            // Both exist — use visual proximity to pick the right side.
            if (visual == null) return incomingConveyor;

            float distA = Vector3.Distance(
                visual.transform.position,
                incomingConveyor.InputEndPosition);
            float distB = Vector3.Distance(
                visual.transform.position,
                secondaryIncomingConveyor.InputEndPosition);

            return distA <= distB ? incomingConveyor : secondaryIncomingConveyor;
        }

        /// <summary>
        /// Returns the OTHER incoming belt (primary ↔ secondary).
        /// </summary>
        private ConveyorBelt GetOtherIncoming(ConveyorBelt belt)
        {
            if (belt == incomingConveyor)
                return secondaryIncomingConveyor;
            if (belt == secondaryIncomingConveyor)
                return incomingConveyor;
            return null;
        }

        /// <summary>
        /// Alternates between incoming belts that have space.
        /// Used by <see cref="ReserveIncomingSlot"/> so successive jobs
        /// get distributed across both sides.
        /// </summary>
        private ConveyorBelt PickNextIncomingBelt()
        {
            ConveyorBelt a = preferSecondaryInput
                ? secondaryIncomingConveyor : incomingConveyor;
            ConveyorBelt b = preferSecondaryInput
                ? incomingConveyor : secondaryIncomingConveyor;
            preferSecondaryInput = !preferSecondaryInput;

            if (a != null && !a.IsFull) return a;
            if (b != null && !b.IsFull) return b;
            return a ?? b;   // all full — return any available
        }

        /// <summary>
        /// Alternates between outgoing belts that have space.
        /// Used by the processing coroutine to distribute finished jobs.
        /// </summary>
        private ConveyorBelt PickNextOutgoingBelt()
        {
            ConveyorBelt a = preferSecondaryOutput
                ? secondaryOutgoingConveyor : outgoingConveyor;
            ConveyorBelt b = preferSecondaryOutput
                ? outgoingConveyor : secondaryOutgoingConveyor;
            preferSecondaryOutput = !preferSecondaryOutput;

            if (a != null && !a.IsFull) return a;
            if (b != null && !b.IsFull) return b;
            return a ?? b;
        }

        /// <summary>
        /// Removes a job from whichever incoming belt holds it.
        /// </summary>
        private void RemoveFromAnyIncoming(int jobId)
        {
            if (incomingConveyor != null && incomingConveyor.Contains(jobId))
                incomingConveyor.RemoveJob(jobId);
            else if (secondaryIncomingConveyor != null &&
                     secondaryIncomingConveyor.Contains(jobId))
                secondaryIncomingConveyor.RemoveJob(jobId);
        }

        /// <summary>
        /// True when every existing outgoing belt is full (or none exist).
        /// </summary>
        private bool AllOutgoingFull()
        {
            if (outgoingConveyor != null && !outgoingConveyor.IsFull)
                return false;
            if (secondaryOutgoingConveyor != null && !secondaryOutgoingConveyor.IsFull)
                return false;
            return true;
        }
    }
}