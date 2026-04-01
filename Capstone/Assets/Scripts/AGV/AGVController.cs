using UnityEngine;
using UnityEngine.AI;
using Assets.Scripts.Simulation;

namespace Assets.Scripts.AGV
{
    public enum AGVState { Idle, HeadingToPickup, Transporting }

    [RequireComponent(typeof(NavMeshAgent))]
    public class AGVController : MonoBehaviour
    {
        public int AgvId { get; private set; }
        public AGVState State { get; private set; } = AGVState.Idle;
        public int CurrentJobId { get; private set; } = -1;

        private NavMeshAgent agent;
        private Transform dropoffTarget;
        private JobVisual loadedJobVisual;

        public void Initialize(int id)
        {
            AgvId = id;
            agent = GetComponent<NavMeshAgent>();
            State = AGVState.Idle;
        }

        /// <summary>
        /// Called by the AGVPool to assign a delivery task to this robot.
        /// </summary>
        public void Dispatch(int jobId, Transform pickupLocation, Transform dropoffLocation)
        {
            CurrentJobId = jobId;
            dropoffTarget = dropoffLocation;
            State = AGVState.HeadingToPickup;

            // Tell the NavMesh to start driving to the machine where the job is currently sitting
            agent.SetDestination(pickupLocation.position);

            Debug.Log($"[AGV {AgvId}] Dispatched to pick up Job {jobId}.");
        }

        private void Update()
        {
            if (State == AGVState.Idle) return;

            // Check if we have reached our current destination
            if (!agent.pathPending && agent.remainingDistance <= agent.stoppingDistance)
            {
                if (State == AGVState.HeadingToPickup)
                {
                    HandlePickup();
                }
                else if (State == AGVState.Transporting)
                {
                    HandleDropoff();
                }
            }
        }

        private void HandlePickup()
        {
            State = AGVState.Transporting;

            // 1. Find the visual orb for this job
            JobTracker tracker = SimulationBridge.Instance.JobManager.GetJobTracker(CurrentJobId);
            loadedJobVisual = tracker.Visual;

            if (loadedJobVisual != null)
            {
                // 2. Physically attach the orb to the AGV!
                loadedJobVisual.transform.SetParent(this.transform);
                loadedJobVisual.SetTargetPosition(this.transform.position + Vector3.up * 0.5f); // Sit on top of AGV
                loadedJobVisual.SetState(JobLifecycleState.InTransit);
            }

            // 3. Set the NavMesh destination to the NEXT machine
            agent.SetDestination(dropoffTarget.position);

            PhysicalMachine targetMachine = dropoffTarget.GetComponent<PhysicalMachine>();
            int targetId = targetMachine != null ? targetMachine.MachineId : -1;

            // 4. Update the tracking stats
            SimulationBridge.Instance.JobManager.BeginTransit(CurrentJobId, targetId, Time.time);

            Debug.Log($"[AGV {AgvId}] Picked up Job {CurrentJobId}. Heading to Machine {targetId}.");
        }

        private void HandleDropoff()
        {
            if (loadedJobVisual != null)
            {
                // Detach the orb so it falls into the Machine's trigger collider
                loadedJobVisual.transform.SetParent(null);
            }

            Debug.Log($"[AGV {AgvId}] Dropped off Job {CurrentJobId}. Returning to Idle.");

            // Assuming your dropoffTarget is the PhysicalMachine transform:
            PhysicalMachine targetMachine = dropoffTarget.GetComponent<PhysicalMachine>();
            int targetId = targetMachine != null ? targetMachine.MachineId : -1;

            // Log the end of the transit time
            SimulationBridge.Instance.JobManager.CompleteTransit(CurrentJobId, targetId, Time.time);

            CurrentJobId = -1;
            loadedJobVisual = null;
            dropoffTarget = null;
            State = AGVState.Idle;
        }
    }
}