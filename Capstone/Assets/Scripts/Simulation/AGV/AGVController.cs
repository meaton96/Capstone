using UnityEngine;
using UnityEngine.AI;
using Assets.Scripts.Simulation.Machines;

namespace Assets.Scripts.Simulation.AGV
{
    public enum AGVState { Idle, HeadingToPickup, Transporting }

    [RequireComponent(typeof(NavMeshAgent))]
    public class AGVController : MonoBehaviour
    {
        public int AgvId { get; private set; }
        public AGVState State { get; private set; } = AGVState.Idle;
        public int CurrentJobId { get; private set; } = -1;

        private NavMeshAgent agent;
        private JobVisual loadedJobVisual;
        private PhysicalMachine sourceMachine;
        private PhysicalMachine targetMachine;

        // Idle callback wired by AGVPool so it can drain its pending queue
        private System.Action onBecameIdle;
        private Vector3 lockedDropoffPos;
        public void SetIdleCallback(System.Action callback) => onBecameIdle = callback;

        // ─────────────────────────────────────────────────────────
        //  Initialization
        // ─────────────────────────────────────────────────────────

        public void Initialize(int id)
        {
            AgvId = id;
            agent = GetComponent<NavMeshAgent>();
            State = AGVState.Idle;
        }

        // ─────────────────────────────────────────────────────────
        //  Dispatch
        // ─────────────────────────────────────────────────────────

        /// <summary>
        /// Assigns a delivery task.
        /// <paramref name="pickupPosition"/> should be the outgoing conveyor's
        /// OutputEndPosition. <paramref name="dropoffPosition"/> should be the
        /// incoming conveyor's InputEndPosition. Source is null when the job is
        /// coming from the initial spawn area.
        /// </summary>
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

        // ─────────────────────────────────────────────────────────
        //  Update
        // ─────────────────────────────────────────────────────────

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

        // ─────────────────────────────────────────────────────────
        //  Pickup / Dropoff
        // ─────────────────────────────────────────────────────────

        private void HandlePickup()
        {
            State = AGVState.Transporting;
            JobTracker tracker = SimulationBridge.Instance.JobManager.GetJobTracker(CurrentJobId);
            loadedJobVisual = tracker?.Visual;

            // Release from outgoing conveyor FIRST so isOnConveyor is cleared,
            // then attach to AGV carrier which sets isCarried.
            sourceMachine?.ReleaseFromOutgoing(CurrentJobId);

            if (loadedJobVisual != null)
            {
                loadedJobVisual.AttachToCarrier(this.transform);
                loadedJobVisual.SetState(JobLifecycleState.InTransit);
            }

            agent.SetDestination(lockedDropoffPos);
            SimulationBridge.Instance.JobManager.BeginTransit(CurrentJobId, targetMachine.MachineId, Time.time);
        }

        private void HandleDropoff()
        {
            // Detach the visual from the AGV. We snap to the dropoff position
            // (the conveyor's input end). ReceiveJob will then hand ownership
            // to the conveyor belt, which smoothly slides the token forward.
            if (loadedJobVisual != null)
                loadedJobVisual.DetachFromCarrier(lockedDropoffPos);

            targetMachine?.ReceiveJob(CurrentJobId, loadedJobVisual);
            SimulationBridge.Instance.JobManager.CompleteTransit(CurrentJobId, targetMachine.MachineId, Time.time);

            // Clean up
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