using UnityEngine;
using UnityEngine.AI;

namespace Assets.Scripts.Simulation
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

        // ── Idle callback wired by AGVPool so it can drain its pending queue ──
        private System.Action onBecameIdle;
        private Vector3 lockedDropoffSlotPos;
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
        /// Assigns a delivery task. Source is null when the job is coming from
        /// the initial spawn area rather than a machine's outgoing queue.
        /// </summary>
        public void Dispatch(int jobId, Vector3 pickupPosition, Vector3 dropoffSlotPosition,
                     PhysicalMachine source, PhysicalMachine dropoff)
        {
            CurrentJobId = jobId;
            sourceMachine = source;
            targetMachine = dropoff;
            lockedDropoffSlotPos = dropoffSlotPosition; // already baked — never recalculate
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

            if (loadedJobVisual != null)
            {
                loadedJobVisual.AttachToCarrier(this.transform);
                loadedJobVisual.SetState(JobLifecycleState.InTransit);
            }

            sourceMachine?.ReleaseFromOutgoing(CurrentJobId);

            // Navigate to the staging position, NOT the machine body
            agent.SetDestination(lockedDropoffSlotPos);
            SimulationBridge.Instance.JobManager.BeginTransit(CurrentJobId, targetMachine.MachineId, Time.time);
        }

        private void HandleDropoff()
        {
            if (loadedJobVisual != null)
                loadedJobVisual.DetachFromCarrier(lockedDropoffSlotPos);

            targetMachine?.ReceiveJob(CurrentJobId, loadedJobVisual);
            SimulationBridge.Instance.JobManager.CompleteTransit(CurrentJobId, targetMachine.MachineId, Time.time);

            CurrentJobId = -1;
            loadedJobVisual = null;
            sourceMachine = null;
            targetMachine = null;
            lockedDropoffSlotPos = Vector3.zero;
            State = AGVState.Idle;
            onBecameIdle?.Invoke();
        }
    }
}