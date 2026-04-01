using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Assets.Scripts.Scheduling.Core;
using Assets.Scripts.Logging;

namespace Assets.Scripts.Simulation
{
    /// @brief Physical anchor for a machine in the Unity scene.
    ///
    /// @details Manages real-time coroutine processing and Unity Physics trigger
    /// events for job arrival and departure. Delegates visual updates to the
    /// attached @c MachineVisual component.
    [RequireComponent(typeof(Collider))]
    [RequireComponent(typeof(Rigidbody))]
    [RequireComponent(typeof(MachineVisual))]
    public class PhysicalMachine : MonoBehaviour
    {
        /// @brief Unique identifier matching the logical machine index.
        public int MachineId { get; private set; }

        /// @brief True when the machine is not currently processing a job.
        public bool IsIdle { get; private set; } = true;

        /// @brief Job IDs currently inside this machine's trigger zone.
        [Header("Queue Areas")]
        [Header("Queue Staging Areas")]
        [SerializeField] private Transform incomingStageRoot;  // child GO, offset to the left
        [SerializeField] private Transform outgoingStageRoot;  // child GO, offset to the right
        [SerializeField] private float slotSpacing = 0.75f;
        [SerializeField] private int maxSlots = 5;

        // Separate the existing PhysicalQueue into two explicit queues
        public List<int> IncomingQueue = new List<int>(); // waiting to be processed
        public List<int> OutgoingQueue = new List<int>(); // waiting for AGV pickup

        // Keep PhysicalQueue as a passthrough so SimulationBridge doesn't break yet
        public List<int> PhysicalQueue => IncomingQueue;

        private MachineVisual visualLayer;

        private Dictionary<int, Vector3> reservedIncomingSlots = new Dictionary<int, Vector3>();
        private int nextIncomingSlotIndex = 0;

        // ─────────────────────────────────────────────────────────
        //  Initialization
        // ─────────────────────────────────────────────────────────

        /// @brief Wires this physical machine to its logical data and visual layer.
        /// @param id         The machine index used throughout the simulation.
        /// @param coreMachineData  The logical @c Machine object carrying scheduling metadata.
        public void Initialize(int id, Machine coreMachineData)
        {
            MachineId = id;
            IsIdle = true;
            PhysicalQueue.Clear();

            visualLayer = GetComponent<MachineVisual>();
            if (visualLayer != null)
            {
                visualLayer.Initialise(id, coreMachineData);
            }
        }

        public Vector3 ReserveIncomingSlot(int jobId)
        {
            if (reservedIncomingSlots.TryGetValue(jobId, out Vector3 existing))
                return existing;

            Vector3 root = incomingStageRoot != null
                ? incomingStageRoot.position
                : transform.position + transform.TransformDirection(new Vector3(-2.5f, 0, 0));

            Vector3 pos = root + transform.right * (nextIncomingSlotIndex * slotSpacing)
                               + Vector3.up * 0.5f;

            reservedIncomingSlots[jobId] = pos;
            nextIncomingSlotIndex++;
            return pos;
        }

        public Vector3 GetIncomingSlotPosition(int index)
        {
            Vector3 root = incomingStageRoot != null
                ? incomingStageRoot.position
                : transform.position + transform.TransformDirection(new Vector3(2.5f, 0, 0));
            return root + transform.right * (index * slotSpacing) + Vector3.up * 0.5f;
        }

        public Vector3 GetOutgoingSlotPosition(int index)
        {
            Vector3 root = outgoingStageRoot != null
                ? outgoingStageRoot.position
                : transform.position + transform.TransformDirection(new Vector3(2.5f, 0, 0));
            return root + transform.right * (index * slotSpacing) + Vector3.up * 0.5f;
        }

        // AGV calls this to know exactly where to drive to for pickup
        public Vector3 GetPickupPositionForJob(int jobId)
        {
            int idx = OutgoingQueue.IndexOf(jobId);
            return idx >= 0 ? GetOutgoingSlotPosition(idx) : transform.position;
        }

        // AGV calls this on dropoff to know where to place the visual
        public Vector3 GetNextIncomingDropoffPosition()
            => GetIncomingSlotPosition(IncomingQueue.Count);

        public void ResetQueues()
        {
            IncomingQueue.Clear();
            OutgoingQueue.Clear();
            reservedIncomingSlots.Clear();
            nextIncomingSlotIndex = 0;
        }
        public void ReceiveJob(int jobId, JobVisual visual)
        {
            if (IncomingQueue.Contains(jobId)) return;
            IncomingQueue.Add(jobId);

            // Use the pre-reserved position — never recalculate
            if (visual != null && reservedIncomingSlots.TryGetValue(jobId, out Vector3 slotPos))
                visual.SnapToPosition(slotPos);

            visualLayer?.UpdateIncomingQueueLabel(IncomingQueue.Count);
            SimulationBridge.Instance?.OnJobArrivedInQueue(MachineId, jobId);
        }

        /// @brief Begins the coroutine timer that simulates job processing.
        /// @param jobId            The job to process.
        /// @param realTimeDuration Wall-clock seconds the operation should take.
        public void StartProcessing(int jobId, float realTimeDuration)
        {
            IsIdle = false;
            IncomingQueue.Remove(jobId);
            reservedIncomingSlots.Remove(jobId);

            // Zoop the job visual into the machine body
            JobTracker tracker = SimulationBridge.Instance.JobManager.GetJobTracker(jobId);
            if (tracker?.Visual != null)
            {
                tracker.Visual.SetState(JobLifecycleState.Processing);
                tracker.Visual.SetTargetPosition(transform.position + Vector3.up * 0.5f);
            }

            visualLayer?.BeginOperation(jobId, Time.time, realTimeDuration);
            visualLayer?.UpdateIncomingQueueLabel(IncomingQueue.Count);
            StartCoroutine(ProcessJobRoutine(jobId, realTimeDuration));
        }

        // Called by AGVController.HandlePickup() to formally release the job
        public void ReleaseFromOutgoing(int jobId)
        {
            OutgoingQueue.Remove(jobId);
            for (int i = 0; i < OutgoingQueue.Count; i++)
            {
                JobTracker t = SimulationBridge.Instance.JobManager.GetJobTracker(OutgoingQueue[i]);
                t?.Visual?.SnapToPosition(GetOutgoingSlotPosition(i)); // snap, not hop
            }
            visualLayer?.UpdateOutgoingQueueLabel(OutgoingQueue.Count);
        }

        // ─────────────────────────────────────────────────────────
        //  Processing
        // ─────────────────────────────────────────────────────────




        /// @brief Coroutine that advances a progress bar each frame and fires
        ///        @c SimulationBridge.OnMachineFinished when complete.
        /// @param jobId    The job being processed.
        /// @param duration Total wall-clock seconds for the operation.
        private IEnumerator ProcessJobRoutine(int jobId, float duration)
        {
            float elapsed = 0f;

            while (elapsed < duration)
            {
                elapsed += Time.deltaTime;

                visualLayer?.UpdateProgress(elapsed / duration);

                yield return null;
            }

            IsIdle = true;

            IsIdle = true;
            visualLayer?.CompleteOperation(jobId);

            // Move job visual to outgoing staging area
            JobTracker tracker = SimulationBridge.Instance.JobManager.GetJobTracker(jobId);
            OutgoingQueue.Add(jobId);
            int outSlot = OutgoingQueue.Count - 1;
            if (tracker?.Visual != null)
            {
                tracker.Visual.SnapToPosition(GetOutgoingSlotPosition(outSlot)); // not SetTargetPosition
                tracker.Visual.SetState(JobLifecycleState.WaitingForTransport);
            }

            SimulationBridge.Instance?.OnMachineFinished(MachineId, jobId);
        }
    }
}