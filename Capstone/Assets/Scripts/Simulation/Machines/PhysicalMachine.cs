using UnityEngine;
using Assets.Scripts.Simulation.Machines;
using Assets.Scripts.Simulation.Jobs;

namespace Assets.Scripts.Simulation.Machines
{
    /// @brief Represents a physical processing unit within the factory simulation.
    ///
    /// @details Manages processing timers and state flags (@c FinishedFlag, @c AlmostDoneFlag). 
    /// This component is strictly passive; it does not initiate state transitions or 
    /// communicate with the orchestrator. Instead, the @c SimulationBridge polls 
    /// these flags to drive the factory lifecycle.
    [RequireComponent(typeof(Collider))]
    [RequireComponent(typeof(Rigidbody))]
    public class PhysicalMachine : MonoBehaviour
    {
        public int MachineId { get; private set; }
        public MachineType MachineType { get; private set; }

        public bool IsIdle { get; private set; } = true;

        public bool FinishedFlag { get; private set; }

        public int ActiveJobId { get; private set; } = -1;

        public bool AlmostDoneFlag { get; private set; }
        public int AlmostDoneJobId { get; private set; } = -1;

        private float remainingTime;
        private float totalDuration;
        private bool almostDoneFired;

        [Header("Conveyor Belts (visual only)")]
        [SerializeField] private ConveyorBelt incomingConveyor;
        [SerializeField] private ConveyorBelt outgoingConveyor;
        [SerializeField] private ConveyorBelt secondaryIncomingConveyor;
        [SerializeField] private ConveyorBelt secondaryOutgoingConveyor;

        private MachineVisual visualLayer;

        /// @brief Sets the machine identity and resets the visual layer.
        ///
        /// @param id Unique identifier for the machine instance.
        /// @param type The functional @c MachineType (e.g., Mill, Lathe).
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

        /// @brief Commences the processing of a specific job for a defined duration.
        ///
        /// @param jobId The identifier of the job being processed.
        /// @param duration The time in simulation seconds to complete the operation.
        /// @param visual The 3D representation of the job to be snapped to the machine.
        ///
        /// @details Resets internal timers and flags, removes the job from the 
        /// incoming conveyor, and triggers the machine's operational animations.
        public void StartJob(int jobId, float duration, JobVisual visual = null)
        {
            ActiveJobId = jobId;
            remainingTime = duration;
            totalDuration = duration;
            IsIdle = false;
            FinishedFlag = false;
            AlmostDoneFlag = false;
            AlmostDoneJobId = -1;
            almostDoneFired = false;

            RemoveFromAnyIncoming(jobId);

            if (visual != null)
            {
                visual.SetOnConveyor(false);
                visual.SnapToPosition(transform.position);
            }
            visualLayer.BeginOperation(jobId, Time.time, duration);
        }

        /// @brief Resets the finished status after the orchestrator acknowledges the completion.
        public void ClearFinished()
        {
            FinishedFlag = false;
            IsIdle = true;
            ActiveJobId = -1;
            visualLayer?.CompleteOperation(-1);
        }

        /// @brief Resets the pre-dispatch signaling flags.
        public void ClearAlmostDone()
        {
            AlmostDoneFlag = false;
            AlmostDoneJobId = -1;
        }

        /// @brief Places a job visual onto the most appropriate incoming conveyor belt.
        ///
        /// @param jobId ID of the job being delivered.
        /// @param visual The visual component to be queued.
        public void PlaceOnIncoming(int jobId, JobVisual visual)
        {
            ConveyorBelt belt = PickIncomingBelt();
            if (belt != null && visual != null)
            {
                belt.TryEnqueue(jobId, visual);
                visual.SetOnConveyor(true);
            }
        }

        /// @brief Removes a job from the outgoing belt systems.
        public void RemoveFromOutgoing(int jobId)
        {
            if (outgoingConveyor != null && outgoingConveyor.Contains(jobId))
                outgoingConveyor.RemoveJob(jobId);
            else if (secondaryOutgoingConveyor != null && secondaryOutgoingConveyor.Contains(jobId))
                secondaryOutgoingConveyor.RemoveJob(jobId);
        }

        /// @brief Transfers a finished job visual from the machine center to an outgoing belt.
        public void PlaceOnOutgoing(int jobId, JobVisual visual)
        {
            ConveyorBelt belt = PickOutgoingBelt();
            if (belt != null && visual != null)
            {
                belt.TryEnqueue(jobId, visual);
                visual.SetOnConveyor(true);
            }
        }

        /// @brief Returns the world position where AGVs should drop off jobs for this machine.
        public Vector3 GetDropoffPosition()
        {
            ConveyorBelt belt = PickIncomingBelt();
            if (belt != null) return belt.InputEndPosition;
            return transform.position + transform.TransformDirection(new Vector3(-2.5f, 0.5f, 0f));
        }

        /// @brief Returns the world position where AGVs should pick up jobs from this machine.
        public Vector3 GetPickupPosition()
        {
            if (outgoingConveyor != null) return outgoingConveyor.OutputEndPosition;
            if (secondaryOutgoingConveyor != null) return secondaryOutgoingConveyor.OutputEndPosition;
            return transform.position + transform.TransformDirection(new Vector3(2.5f, 0.5f, 0f));
        }

        /// @brief Updates the processing timer and fires flags based on remaining time.
        ///
        /// @details When @c remainingTime reaches the @c PreDispatchLeadTime defined in 
        /// the @c SimulationBridge, the @c AlmostDoneFlag is set to trigger 
        /// predictive AGV routing.
        private void Update()
        {
            if (IsIdle || FinishedFlag) return;

            remainingTime -= Time.deltaTime;

            if (visualLayer != null && remainingTime > 0f)
                visualLayer.UpdateProgress(1f - (remainingTime / Mathf.Max(totalDuration, 0.001f)));

            if (!almostDoneFired && remainingTime <= SimulationBridge.Instance.PreDispatchLeadTime)
            {
                almostDoneFired = true;
                AlmostDoneFlag = true;
                AlmostDoneJobId = ActiveJobId;
            }

            if (remainingTime <= 0f)
            {
                FinishedFlag = true;
            }
        }

        /// @brief Updates the numerical UI labels for the machine's current queue state.
        public void RefreshQueueLabels(int incomingCount, int outgoingCount)
        {
            visualLayer.UpdateIncomingQueueLabel(incomingCount);
            visualLayer.UpdateOutgoingQueueLabel(outgoingCount);
        }

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

        /// @brief Forces the machine into an idle state and clears all belt visuals.
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