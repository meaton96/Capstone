using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Assets.Scripts.Scheduling.Core;

namespace Assets.Scripts.Simulation.Machines
{
    /// @brief Physical anchor for a machine in the Unity scene.
    /// @details Manages real-time processing via coroutines and delegates visual updates 
    /// to MachineVisual. Supports multi-conveyor setups for double-sided machines, 
    /// handling load balancing between belts and proximity-based job reception.
    [RequireComponent(typeof(Collider))]
    [RequireComponent(typeof(Rigidbody))]
    [RequireComponent(typeof(MachineVisual))]
    public class PhysicalMachine : MonoBehaviour
    {
        public int MachineId { get; private set; }
        public bool IsIdle { get; private set; } = true;

        /// @brief True when processing is done but all outgoing belts are full.
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

        public List<int> IncomingQueue
        {
            get
            {
                var ids = new List<int>();
                if (incomingConveyor != null) ids.AddRange(incomingConveyor.GetJobIds());
                if (secondaryIncomingConveyor != null) ids.AddRange(secondaryIncomingConveyor.GetJobIds());
                return ids;
            }
        }

        public List<int> OutgoingQueue
        {
            get
            {
                var ids = new List<int>();
                if (outgoingConveyor != null) ids.AddRange(outgoingConveyor.GetJobIds());
                if (secondaryOutgoingConveyor != null) ids.AddRange(secondaryOutgoingConveyor.GetJobIds());
                return ids;
            }
        }

        public List<int> PhysicalQueue => IncomingQueue;

        private int TotalIncomingCount => (incomingConveyor?.Count ?? 0) + (secondaryIncomingConveyor?.Count ?? 0);
        private int TotalOutgoingCount => (outgoingConveyor?.Count ?? 0) + (secondaryOutgoingConveyor?.Count ?? 0);

        /// @brief Initializes the physical machine state and links visual components.
        /// @param id The unique machine index.
        /// @param coreMachineData The logical machine data from the simulation core.
        /// @post Machine is set to Idle, and all attached conveyors are cleared.
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

        /// @brief Clears all jobs from primary and secondary conveyors.
        public void ResetQueues()
        {
            incomingConveyor?.Clear();
            secondaryIncomingConveyor?.Clear();
            outgoingConveyor?.Clear();
            secondaryOutgoingConveyor?.Clear();
        }

        /// @brief Determines the world position where an AGV should deliver a job.
        /// @details Alternates between available incoming belts to balance the load across 
        /// both sides of double-sided machines.
        /// @param jobId The ID of the job to be delivered.
        /// @return Vector3 world position of the chosen conveyor's input end.
        public Vector3 ReserveIncomingSlot(int jobId)
        {
            ConveyorBelt belt = PickNextIncomingBelt();
            if (belt != null)
                return belt.InputEndPosition;

            return transform.position + transform.TransformDirection(new Vector3(-2.5f, 0.5f, 0f));
        }

        /// @brief Locates the specific world position where an AGV should pick up a job.
        /// @details Searches both outgoing conveyors to find which one currently holds the job.
        /// @param jobId The job requested for pickup.
        /// @return Vector3 world position of the conveyor's output end.
        public Vector3 GetPickupPositionForJob(int jobId)
        {
            if (outgoingConveyor != null && outgoingConveyor.Contains(jobId))
                return outgoingConveyor.OutputEndPosition;

            if (secondaryOutgoingConveyor != null && secondaryOutgoingConveyor.Contains(jobId))
                return secondaryOutgoingConveyor.OutputEndPosition;

            if (outgoingConveyor != null) return outgoingConveyor.OutputEndPosition;
            if (secondaryOutgoingConveyor != null) return secondaryOutgoingConveyor.OutputEndPosition;

            return transform.position + transform.TransformDirection(new Vector3(2.5f, 0.5f, 0f));
        }

        /// @brief Handshakes with an AGV to receive a job onto an incoming conveyor.
        /// @details Selects the belt physically closest to the job visual's position (the side 
        /// where the AGV is docking). Falls back to the opposite belt if the preferred one is full.
        /// @param jobId The job being delivered.
        /// @param visual The visual representation of the job.
        /// @post Job is enqueued on a belt and the visual state is updated to Queued.
        public void ReceiveJob(int jobId, JobVisual visual)
        {
            ConveyorBelt belt = PickClosestIncoming(visual);

            if (belt == null)
            {
                Debug.LogError($"[PhysicalMachine M{MachineId}] No incoming conveyor wired!");
                return;
            }

            if (!belt.TryEnqueue(jobId, visual))
            {
                ConveyorBelt fallback = GetOtherIncoming(belt);
                if (fallback == null || !fallback.TryEnqueue(jobId, visual))
                {
                    Debug.LogWarning($"[PhysicalMachine M{MachineId}] All incoming conveyors full — cannot accept Job {jobId}.");
                    return;
                }
            }

            if (visual != null)
                visual.SetState(JobLifecycleState.Queued);

            visualLayer?.UpdateIncomingQueueLabel(TotalIncomingCount);
            SimulationBridge.Instance?.OnJobArrivedInQueue(MachineId, jobId);
        }

        /// @brief Transitions a job from the queue into active processing.
        /// @details Removes the job from its conveyor, moves the visual into the machine body, 
        /// and initiates the timed process coroutine.
        /// @param jobId The job selected for processing.
        /// @param realTimeDuration Time in seconds the process will take.
        /// @pre Machine must be Idle.
        /// @post Machine state becomes Busy; IsIdle set to false.
        public void StartProcessing(int jobId, float realTimeDuration)
        {
            IsIdle = false;
            IsBlocked = false;

            RemoveFromAnyIncoming(jobId);

            JobTracker tracker = SimulationBridge.Instance.JobManager.GetJobTracker(jobId);
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

        /// @brief Finalizes the removal of a job from the machine's outgoing system.
        /// @details Triggered by an AGV collection event. Clears the job from the physical belt.
        /// @param jobId The job being removed.
        /// @post Outgoing queue count is updated in the visual layer.
        public void ReleaseFromOutgoing(int jobId)
        {
            if (outgoingConveyor != null && outgoingConveyor.Contains(jobId))
                outgoingConveyor.RemoveJob(jobId);
            else if (secondaryOutgoingConveyor != null && secondaryOutgoingConveyor.Contains(jobId))
                secondaryOutgoingConveyor.RemoveJob(jobId);

            visualLayer?.UpdateOutgoingQueueLabel(TotalOutgoingCount);
        }

        /// @brief Internal routine managing the processing timer and output logic.
        /// @details Handles the transition through processing, checks for output blockages, 
        /// and notifies the simulation bridge upon completion.
        /// @param jobId Job currently being processed.
        /// @param duration Processing time.
        private IEnumerator ProcessJobRoutine(int jobId, float duration)
        {
            float elapsed = 0f;
            while (elapsed < duration)
            {
                elapsed += Time.deltaTime;
                visualLayer?.UpdateProgress(elapsed / duration);
                yield return null;
            }

            JobTracker tracker = SimulationBridge.Instance.JobManager.GetJobTracker(jobId);
            JobVisual visual = tracker?.Visual;

            if (AllOutgoingFull())
            {
                IsBlocked = true;
                visualLayer?.SetBlockedAfterProcessing(jobId);
                while (AllOutgoingFull()) yield return null;
                IsBlocked = false;
            }

            ConveyorBelt outBelt = PickNextOutgoingBelt();
            if (outBelt != null)
            {
                if (visual != null)
                    visual.SetState(JobLifecycleState.WaitingForTransport);

                outBelt.TryEnqueue(jobId, visual);
                visualLayer?.UpdateOutgoingQueueLabel(TotalOutgoingCount);
            }

            IsIdle = true;
            visualLayer?.CompleteOperation(jobId);
            SimulationBridge.Instance?.OnMachineFinished(MachineId, jobId);
        }

        /// @brief Selects the incoming belt closest to the job's current world position.
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

        /// @brief Returns the opposite incoming conveyor for a given belt.
        private ConveyorBelt GetOtherIncoming(ConveyorBelt belt)
        {
            if (belt == incomingConveyor) return secondaryIncomingConveyor;
            if (belt == secondaryIncomingConveyor) return incomingConveyor;
            return null;
        }

        /// @brief Cycles through incoming belts to find one with available capacity.
        private ConveyorBelt PickNextIncomingBelt()
        {
            ConveyorBelt a = preferSecondaryInput ? secondaryIncomingConveyor : incomingConveyor;
            ConveyorBelt b = preferSecondaryInput ? incomingConveyor : secondaryIncomingConveyor;
            preferSecondaryInput = !preferSecondaryInput;

            if (a != null && !a.IsFull) return a;
            if (b != null && !b.IsFull) return b;
            return a ?? b;
        }

        /// @brief Cycles through outgoing belts to find one with available capacity.
        private ConveyorBelt PickNextOutgoingBelt()
        {
            ConveyorBelt a = preferSecondaryOutput ? secondaryOutgoingConveyor : outgoingConveyor;
            ConveyorBelt b = preferSecondaryOutput ? outgoingConveyor : secondaryOutgoingConveyor;
            preferSecondaryOutput = !preferSecondaryOutput;

            if (a != null && !a.IsFull) return a;
            if (b != null && !b.IsFull) return b;
            return a ?? b;
        }

        /// @brief Checks both conveyors for a job and removes it if found.
        private void RemoveFromAnyIncoming(int jobId)
        {
            if (incomingConveyor != null && incomingConveyor.Contains(jobId))
                incomingConveyor.RemoveJob(jobId);
            else if (secondaryIncomingConveyor != null && secondaryIncomingConveyor.Contains(jobId))
                secondaryIncomingConveyor.RemoveJob(jobId);
        }

        /// @brief Verifies if all assigned outgoing belts are currently at capacity.
        private bool AllOutgoingFull()
        {
            if (outgoingConveyor != null && !outgoingConveyor.IsFull) return false;
            if (secondaryOutgoingConveyor != null && !secondaryOutgoingConveyor.IsFull) return false;
            return true;
        }
    }
}