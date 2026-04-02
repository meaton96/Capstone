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
    /// <see cref="MachineVisual"/>. Uses two <see cref="ConveyorBelt"/> components
    /// for smooth incoming/outgoing queue visuals.
    ///
    /// <para><b>Blocking behaviour:</b> when the outgoing conveyor is full after
    /// processing completes, the machine enters <c>MachineState.Blocked</c> and
    /// holds the finished job until space opens. <c>IsIdle</c> remains false so
    /// the scheduler cannot assign new work. <c>OnMachineFinished</c> only fires
    /// once the job is successfully placed on the outgoing belt.</para>
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
        /// True when processing is done but the outgoing belt is full.
        /// Exposed so external systems (UI, analytics) can distinguish
        /// blocked from busy.
        /// </summary>
        public bool IsBlocked { get; private set; } = false;

        // ─────────────────────────────────────────────────────────
        //  Conveyor References
        // ─────────────────────────────────────────────────────────

        [Header("Conveyor Belts")]
        [Tooltip("Belt that feeds jobs INTO this machine. Set reverseFlow = true.")]
        [SerializeField] private ConveyorBelt incomingConveyor;

        [Tooltip("Belt that carries finished jobs OUT of this machine. reverseFlow = false.")]
        [SerializeField] private ConveyorBelt outgoingConveyor;

        // ─────────────────────────────────────────────────────────
        //  Backward-Compat Properties
        // ─────────────────────────────────────────────────────────

        public List<int> IncomingQueue => incomingConveyor != null
            ? incomingConveyor.GetJobIds()
            : new List<int>();

        public List<int> OutgoingQueue => outgoingConveyor != null
            ? outgoingConveyor.GetJobIds()
            : new List<int>();

        public List<int> PhysicalQueue => IncomingQueue;

        private MachineVisual visualLayer;

        // ─────────────────────────────────────────────────────────
        //  Initialization
        // ─────────────────────────────────────────────────────────

        public void Initialize(int id, Machine coreMachineData)
        {
            MachineId = id;
            IsIdle = true;
            IsBlocked = false;

            visualLayer = GetComponent<MachineVisual>();
            if (visualLayer != null)
                visualLayer.Initialise(id, coreMachineData);

            ResetQueues();
        }

        public void ResetQueues()
        {
            incomingConveyor?.Clear();
            outgoingConveyor?.Clear();
        }

        // ─────────────────────────────────────────────────────────
        //  Slot Positions (for AGV dispatch)
        // ─────────────────────────────────────────────────────────

        public Vector3 ReserveIncomingSlot(int jobId)
        {
            if (incomingConveyor != null)
                return incomingConveyor.InputEndPosition;

            return transform.position + transform.TransformDirection(new Vector3(-2.5f, 0.5f, 0f));
        }

        public Vector3 GetPickupPositionForJob(int jobId)
        {
            if (outgoingConveyor != null)
                return outgoingConveyor.OutputEndPosition;

            return transform.position + transform.TransformDirection(new Vector3(2.5f, 0.5f, 0f));
        }

        // ─────────────────────────────────────────────────────────
        //  Job Receive (AGV → Incoming Conveyor)
        // ─────────────────────────────────────────────────────────

        public void ReceiveJob(int jobId, JobVisual visual)
        {
            if (incomingConveyor == null)
            {
                Debug.LogError($"[PhysicalMachine M{MachineId}] No incoming conveyor wired!");
                return;
            }

            if (!incomingConveyor.TryEnqueue(jobId, visual))
            {
                Debug.LogWarning($"[PhysicalMachine M{MachineId}] Incoming conveyor full — " +
                                 $"cannot accept Job {jobId}.");
                return;
            }

            if (visual != null)
                visual.SetState(JobLifecycleState.Queued);

            visualLayer?.UpdateIncomingQueueLabel(incomingConveyor.Count);
            SimulationBridge.Instance?.OnJobArrivedInQueue(MachineId, jobId);
        }

        // ─────────────────────────────────────────────────────────
        //  Processing Start
        // ─────────────────────────────────────────────────────────

        public void StartProcessing(int jobId, float realTimeDuration)
        {
            IsIdle = false;
            IsBlocked = false;

            // Pull the job off the incoming conveyor
            incomingConveyor?.RemoveJob(jobId);

            // Move the token into the machine body
            JobTracker tracker = SimulationBridge.Instance.JobManager.GetJobTracker(jobId);
            JobVisual visual = tracker?.Visual;
            if (visual != null)
            {
                visual.SetOnConveyor(false);
                visual.SetState(JobLifecycleState.Processing);
                visual.SetTargetPosition(transform.position);
            }

            visualLayer?.BeginOperation(jobId, Time.time, realTimeDuration);
            visualLayer?.UpdateIncomingQueueLabel(incomingConveyor?.Count ?? 0);
            StartCoroutine(ProcessJobRoutine(jobId, realTimeDuration));
        }

        // ─────────────────────────────────────────────────────────
        //  Outgoing Release (AGV picks up)
        // ─────────────────────────────────────────────────────────

        public void ReleaseFromOutgoing(int jobId)
        {
            outgoingConveyor?.RemoveJob(jobId);
            visualLayer?.UpdateOutgoingQueueLabel(outgoingConveyor?.Count ?? 0);
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

            // Processing is done — grab the visual before trying to output.
            JobTracker tracker = SimulationBridge.Instance.JobManager.GetJobTracker(jobId);
            JobVisual visual = tracker?.Visual;

            // ── Phase 2: Blocked check ──────────────────────────
            // If outgoing belt is full, hold the job at the machine until
            // an AGV picks something up and frees a slot.
            if (outgoingConveyor != null && outgoingConveyor.IsFull)
            {
                IsBlocked = true;
                visualLayer?.SetBlockedAfterProcessing(jobId);

                // Poll each frame until space opens
                while (outgoingConveyor.IsFull)
                    yield return null;

                IsBlocked = false;
            }

            // ── Phase 3: Output to conveyor ─────────────────────
            if (outgoingConveyor != null)
            {
                if (visual != null)
                    visual.SetState(JobLifecycleState.WaitingForTransport);

                outgoingConveyor.TryEnqueue(jobId, visual);
                visualLayer?.UpdateOutgoingQueueLabel(outgoingConveyor.Count);
            }

            // ── Phase 4: Truly idle — notify the scheduler ──────
            IsIdle = true;
            visualLayer?.CompleteOperation(jobId);
            SimulationBridge.Instance?.OnMachineFinished(MachineId, jobId);
        }
    }
}