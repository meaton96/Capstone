using UnityEngine;
using UnityEngine.AI;
using Assets.Scripts.Simulation.Machines;
using Assets.Scripts.Simulation.FactoryLayout;

namespace Assets.Scripts.Simulation.AGV
{
    /// <summary>
    /// Represents the current operational state of an AGV.
    /// </summary>
    public enum AGVState
    {
        Idle,
        HeadingToPickup,
        AligningForPickup,
        Transporting,
        AligningForDropoff
    }

    /// <summary>
    /// Controls the navigation, job assignment, and lifecycle of a single Automated Guided Vehicle (AGV).
    ///
    /// <para><b>Movement model:</b> The AGV uses a turn-then-move pattern.
    /// When the angle to the next path corner exceeds <see cref="pathTurnThreshold"/>,
    /// the AGV stops and rotates in place. Once aligned, it drives straight.
    /// This mimics real factory AGVs that cannot steer while moving.</para>
    ///
    /// <para><b>Handshake:</b> On reaching a conveyor the AGV enters an alignment
    /// state, rotating to face the target position before executing the
    /// pickup or dropoff. The <see cref="arrivalDistance"/> field controls how
    /// far the AGV body stops from the conveyor end, letting you account for
    /// the front shelf geometry.</para>
    /// </summary>
    [RequireComponent(typeof(NavMeshAgent))]
    public class AGVController : MonoBehaviour
    {
        // ─────────────────────────────────────────────────────────
        //  Inspector
        // ─────────────────────────────────────────────────────────

        [Header("References")]
        [SerializeField] private Transform carryPos;

        [Header("Turn-Then-Move")]
        [Tooltip("Degrees per second the AGV rotates when turning in place.")]
        [SerializeField] private float turnSpeed = 180f;

        [Tooltip("Angle (degrees) to the next path corner above which the AGV " +
                 "stops and rotates in place instead of driving forward.")]
        [SerializeField] private float pathTurnThreshold = 10f;

        [Header("Conveyor Handshake")]
        [Tooltip("Distance from the conveyor end at which the AGV considers " +
                 "itself arrived and begins the alignment phase. Increase this " +
                 "if the front shelf collides with the conveyor before the " +
                 "agent thinks it has arrived.")]
        [SerializeField] private float arrivalDistance = 1.2f;

        [Tooltip("Angle (degrees) below which the AGV is considered aligned " +
                 "with the conveyor and can execute pickup/dropoff.")]
        [SerializeField] private float alignmentThreshold = 3f;

        // ─────────────────────────────────────────────────────────
        //  Public Properties
        // ─────────────────────────────────────────────────────────

        /// <summary>Gets the unique identifier for this AGV.</summary>
        public int AgvId { get; private set; }

        /// <summary>Gets the current operational state of the AGV.</summary>
        public AGVState State { get; private set; } = AGVState.Idle;

        /// <summary>
        /// Gets the ID of the job currently assigned to this AGV.
        /// Returns -1 if no job is assigned.
        /// </summary>
        public int CurrentJobId { get; private set; } = -1;

        // ─────────────────────────────────────────────────────────
        //  Private State
        // ─────────────────────────────────────────────────────────

        private NavMeshAgent agent;
        private JobVisual loadedJobVisual;
        private PhysicalMachine sourceMachine;
        private PhysicalMachine targetMachine;

        private System.Action onBecameIdle;
        private Vector3 lockedPickupPos;
        private Vector3 lockedDropoffPos;

        // ─────────────────────────────────────────────────────────
        //  Initialization
        // ─────────────────────────────────────────────────────────

        /// <summary>
        /// Sets the callback invoked when the AGV transitions to Idle.
        /// </summary>
        public void SetIdleCallback(System.Action callback) => onBecameIdle = callback;

        /// <summary>
        /// Initializes the AGV with a specific ID and configures the NavMeshAgent
        /// for manual rotation control (turn-then-move).
        /// </summary>
        public void Initialize(int id)
        {
            AgvId = id;
            agent = GetComponent<NavMeshAgent>();

            // We handle all rotation ourselves so the AGV
            // only turns in place, never while driving forward.
            agent.updateRotation = false;
            agent.angularSpeed = 0f;

            // Match the agent's built-in stopping distance to our
            // arrival distance so it decelerates naturally.
            agent.stoppingDistance = arrivalDistance;

            State = AGVState.Idle;
        }

        // ─────────────────────────────────────────────────────────
        //  Dispatch
        // ─────────────────────────────────────────────────────────

        /// <summary>
        /// Assigns a delivery task to the AGV.
        /// </summary>
        public void Dispatch(int jobId, Vector3 pickupPosition, Vector3 dropoffPosition,
                             PhysicalMachine source, PhysicalMachine dropoff)
        {
            CurrentJobId = jobId;
            sourceMachine = source;
            targetMachine = dropoff;
            lockedPickupPos = pickupPosition;
            lockedDropoffPos = dropoffPosition;

            State = AGVState.HeadingToPickup;
            agent.isStopped = false;
            agent.SetDestination(pickupPosition);
        }

        // ─────────────────────────────────────────────────────────
        //  Update — State Machine
        // ─────────────────────────────────────────────────────────

        private void Update()
        {
            switch (State)
            {
                case AGVState.Idle:
                    return;

                case AGVState.HeadingToPickup:
                    SteerAlongPath();
                    if (HasArrived())
                    {
                        agent.isStopped = true;
                        State = AGVState.AligningForPickup;
                    }
                    break;

                case AGVState.AligningForPickup:
                    if (RotateTowardPoint(lockedPickupPos))
                        HandlePickup();
                    break;

                case AGVState.Transporting:
                    SteerAlongPath();
                    if (HasArrived())
                    {
                        agent.isStopped = true;
                        State = AGVState.AligningForDropoff;
                    }
                    break;

                case AGVState.AligningForDropoff:
                    if (RotateTowardPoint(lockedDropoffPos))
                        HandleDropoff();
                    break;
            }
        }

        // ─────────────────────────────────────────────────────────
        //  Turn-Then-Move
        // ─────────────────────────────────────────────────────────

        /// <summary>
        /// Enforces turn-then-move along the NavMesh path.
        /// If the angle to the next path corner exceeds the threshold the
        /// AGV halts and rotates in place. Once aligned it resumes driving.
        /// </summary>
        private void SteerAlongPath()
        {
            if (agent.pathPending || !agent.hasPath) return;

            Vector3 toSteer = agent.steeringTarget - transform.position;
            toSteer.y = 0f;

            if (toSteer.sqrMagnitude < 0.001f) return;

            float angleToTarget = Vector3.Angle(transform.forward, toSteer);

            if (angleToTarget > pathTurnThreshold)
            {
                // Misaligned — stop driving and rotate in place.
                agent.isStopped = true;
                RotateTowardDirection(toSteer.normalized);
            }
            else
            {
                // Aligned — drive forward, applying gentle correction.
                agent.isStopped = false;
                RotateTowardDirection(toSteer.normalized);
            }
        }

        /// <summary>
        /// Returns true when the agent has finished computing a path and
        /// its remaining distance is within <see cref="arrivalDistance"/>.
        /// </summary>
        private bool HasArrived()
        {
            return !agent.pathPending
                && agent.remainingDistance <= arrivalDistance;
        }

        // ─────────────────────────────────────────────────────────
        //  Rotation Helpers
        // ─────────────────────────────────────────────────────────

        /// <summary>
        /// Rotates the AGV in place toward a world position.
        /// Returns <c>true</c> once the angle is within <see cref="alignmentThreshold"/>.
        /// </summary>
        private bool RotateTowardPoint(Vector3 worldTarget)
        {
            Vector3 dir = worldTarget - transform.position;
            dir.y = 0f;

            if (dir.sqrMagnitude < 0.001f) return true;

            float remaining = Vector3.Angle(transform.forward, dir);
            if (remaining <= alignmentThreshold) return true;

            RotateTowardDirection(dir.normalized);
            return false;
        }

        /// <summary>
        /// Smoothly rotates the AGV toward a flat direction vector at
        /// <see cref="turnSpeed"/> degrees per second.
        /// </summary>
        private void RotateTowardDirection(Vector3 flatDir)
        {
            Quaternion goal = Quaternion.LookRotation(flatDir, Vector3.up);
            transform.rotation = Quaternion.RotateTowards(
                transform.rotation, goal, turnSpeed * Time.deltaTime);
        }

        // ─────────────────────────────────────────────────────────
        //  Pickup / Dropoff Logic
        // ─────────────────────────────────────────────────────────

        /// <summary>
        /// Executes after the AGV has arrived at and aligned with the pickup
        /// conveyor. Grabs the job visual and begins navigating to the dropoff.
        /// </summary>
        private void HandlePickup()
        {
            State = AGVState.Transporting;

            JobTracker tracker =
                SimulationBridge.Instance.JobManager.GetJobTracker(CurrentJobId);
            loadedJobVisual = tracker?.Visual;

            if (sourceMachine != null)
            {
                sourceMachine.ReleaseFromOutgoing(CurrentJobId);
            }
            else
            {
                // Null source — picking up from the main Incoming Belt.
                FactoryLayoutManager.Instance.IncomingBelt?.RemoveJob(CurrentJobId);
            }

            if (loadedJobVisual != null)
            {
                loadedJobVisual.AttachToCarrier(carryPos);
                loadedJobVisual.SetState(JobLifecycleState.InTransit);
            }

            agent.isStopped = false;
            agent.SetDestination(lockedDropoffPos);

            int nextMachineId = targetMachine != null ? targetMachine.MachineId : -1;
            SimulationBridge.Instance.JobManager
                .BeginTransit(CurrentJobId, nextMachineId, Time.time);
        }

        /// <summary>
        /// Executes after the AGV has arrived at and aligned with the dropoff
        /// conveyor. Hands the job to the target machine and resets to Idle.
        /// </summary>
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
                // Null target — dropping off at the main Outgoing Belt.
                FactoryLayoutManager.Instance.OutgoingBelt?
                    .TryEnqueue(CurrentJobId, loadedJobVisual);
            }

            int currentMachineId = targetMachine != null ? targetMachine.MachineId : -1;
            SimulationBridge.Instance.JobManager
                .CompleteTransit(CurrentJobId, currentMachineId, Time.time);

            // Reset
            CurrentJobId = -1;
            loadedJobVisual = null;
            sourceMachine = null;
            targetMachine = null;
            lockedPickupPos = Vector3.zero;
            lockedDropoffPos = Vector3.zero;
            State = AGVState.Idle;
            onBecameIdle?.Invoke();
        }
    }
}