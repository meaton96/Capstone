using UnityEngine;
using UnityEngine.AI;
using Assets.Scripts.Simulation.Machines;
using Assets.Scripts.Simulation.FactoryLayout;

namespace Assets.Scripts.Simulation.AGV
{
    /// <summary>
    /// Represents the current operational state of an AGV.
    /// </summary>
    public enum AGVState { Idle, HeadingToPickup, Transporting }

    /// <summary>
    /// Controls the navigation, job assignment, and lifecycle of a single Automated Guided Vehicle (AGV).
    /// </summary>
    [RequireComponent(typeof(NavMeshAgent))]
    public class AGVController : MonoBehaviour
    {
        /// <summary>
        /// Gets the unique identifier for this AGV.
        /// </summary>
        public int AgvId { get; private set; }

        /// <summary>
        /// Gets the current operational state of the AGV.
        /// </summary>
        public AGVState State { get; private set; } = AGVState.Idle;

        /// <summary>
        /// Gets the ID of the job currently assigned to this AGV. Returns -1 if no job is assigned.
        /// </summary>
        public int CurrentJobId { get; private set; } = -1;

        private NavMeshAgent agent;
        private JobVisual loadedJobVisual;
        private PhysicalMachine sourceMachine;
        private PhysicalMachine targetMachine;

        private System.Action onBecameIdle;
        private Vector3 lockedDropoffPos;

        /// <summary>
        /// Sets the callback to be invoked when the AGV transitions to an Idle state.
        /// </summary>
        /// <param name="callback">The action to invoke upon becoming idle.</param>
        public void SetIdleCallback(System.Action callback) => onBecameIdle = callback;

        /// <summary>
        /// Initializes the AGV with a specific ID and default component references.
        /// </summary>
        /// <param name="id">The unique integer identifier for this AGV.</param>
        public void Initialize(int id)
        {
            AgvId = id;
            agent = GetComponent<NavMeshAgent>();
            State = AGVState.Idle;
        }

        /// <summary>
        /// Assigns a delivery task to the AGV.
        /// </summary>
        /// <param name="jobId">The unique identifier of the job to transport.</param>
        /// <param name="pickupPosition">The world position of the outgoing conveyor's output end.</param>
        /// <param name="dropoffPosition">The world position of the incoming conveyor's input end.</param>
        /// <param name="source">The source machine of the job. Null if originating from the initial spawn area.</param>
        /// <param name="dropoff">The destination machine receiving the job.</param>
        public void Dispatch(int jobId, Vector3 pickupPosition, Vector3 dropoffPosition,
                             PhysicalMachine source, PhysicalMachine dropoff)
        {
            CurrentJobId = jobId;
            sourceMachine = source;
            targetMachine = dropoff;
            lockedDropoffPos = dropoffPosition;
            State = AGVState.HeadingToPickup;
            agent.SetDestination(pickupPosition);
        }

        private void Update()
        {
            if (State == AGVState.Idle) return;

            if (!agent.pathPending && agent.remainingDistance <= agent.stoppingDistance)
            {
                if (State == AGVState.HeadingToPickup)
                    HandlePickup();
                else if (State == AGVState.Transporting)
                    HandleDropoff();
            }
        }

        /// <summary>
        /// Handles job acquisition logic upon reaching the pickup destination.
        /// </summary>
        private void HandlePickup()
        {
            State = AGVState.Transporting;
            JobTracker tracker = SimulationBridge.Instance.JobManager.GetJobTracker(CurrentJobId);
            loadedJobVisual = tracker?.Visual;

            if (sourceMachine != null)
            {
                sourceMachine.ReleaseFromOutgoing(CurrentJobId);
            }
            else
            {
                // Null source means we are picking up from the main Incoming Belt
                var layoutManager = FindObjectOfType<FactoryLayoutManager>(); // Or pass this reference in Initialize
                layoutManager.IncomingBelt?.RemoveJob(CurrentJobId);
            }

            if (loadedJobVisual != null)
            {
                loadedJobVisual.AttachToCarrier(this.transform);
                loadedJobVisual.SetState(JobLifecycleState.InTransit);
            }

            agent.SetDestination(lockedDropoffPos);

            // Safety check for targetMachine being null (going to exit)
            int nextMachineId = targetMachine != null ? targetMachine.MachineId : -1;
            SimulationBridge.Instance.JobManager.BeginTransit(CurrentJobId, nextMachineId, Time.time);
        }

        private void HandleDropoff()
        {
            if (loadedJobVisual != null)
                loadedJobVisual.DetachFromCarrier(lockedDropoffPos);

            if (targetMachine != null)
            {
                targetMachine.ReceiveJob(CurrentJobId, loadedJobVisual);
            }
            else
            {
                // Null target means we are dropping off at the main Outgoing Belt

                FactoryLayoutManager.Instance.OutgoingBelt?.TryEnqueue(CurrentJobId, loadedJobVisual);
            }

            int currentMachineId = targetMachine != null ? targetMachine.MachineId : -1;
            SimulationBridge.Instance.JobManager.CompleteTransit(CurrentJobId, currentMachineId, Time.time);

            CurrentJobId = -1;
            loadedJobVisual = null;
            sourceMachine = null;
            targetMachine = null;
            lockedDropoffPos = Vector3.zero;
            State = AGVState.Idle;
            onBecameIdle?.Invoke();
        }
    }
}