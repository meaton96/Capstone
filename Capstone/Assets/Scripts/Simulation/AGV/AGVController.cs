using UnityEngine;
using UnityEngine.AI;
using System.Collections.Generic;
using Assets.Scripts.Simulation.Machines;
using Assets.Scripts.Simulation.FactoryLayout;
using Assets.Scripts.Logging;
using Assets.Scripts.Simulation.Jobs;

namespace Assets.Scripts.Simulation.AGV
{
    public enum AGVState
    {
        Idle,
        NavigatingToPickup,
        AligningForPickup,
        ExecutingPickup,
        NavigatingToDropoff,
        AligningForDropoff,
        ExecutingDropoff,
        WaitingForZone
    }

    /// @brief Controls navigation, job assignment, and lifecycle of an AGV.
    /// @details Implements a Turn-Then-Move model. Routes are planned via the TrafficZoneManager 
    /// using BFS. Deadlocks are prevented through a reserve-ahead zone system.
    [RequireComponent(typeof(NavMeshAgent))]
    public class AGVController : MonoBehaviour
    {
        [SerializeField] private Transform carryPos;
        [SerializeField] private float handshakeDuration = 1.5f;
        [SerializeField] private float moveSpeed = 3.5f;
        [SerializeField] private float turnSpeed = 180f;
        [SerializeField] private float pathTurnThreshold = 10f;
        [SerializeField] private float waypointArrivalDist = 0.4f;
        [SerializeField] private float dockArrivalDist = 0.3f;
        [SerializeField] private float alignmentThreshold = 3f;
        [SerializeField] private float reservationRetryInterval = 0.25f;
        [SerializeField] private float groundOffset = 0.5f;

        public int AgvId { get; private set; }
        public AGVState State { get; private set; } = AGVState.Idle;
        public int CurrentJobId { get; private set; } = -1;
        public int CurrentZoneId => currentZoneId;

        private NavMeshAgent agent;
        private TrafficZoneManager trafficMgr;
        private JobVisual loadedJobVisual;
        private PhysicalMachine sourceMachine;
        private PhysicalMachine targetMachine;
        private System.Action onBecameIdle;

        private readonly List<int> currentRoute = new List<int>();
        private int routeIndex;
        private int currentZoneId = -1;
        private int previousZoneId = -1;

        private int pickupZoneId = -1;
        private int dropoffZoneId = -1;
        private DockPoint pickupDock;
        private DockPoint dropoffDock;

        private Vector3 currentWaypoint;

        private AGVState stateBeforeWait;
        private int pendingZoneId = -1;
        private float nextRetryTime;

        /// @brief Assigns a callback to be fired when the AGV returns to an Idle state.
        /// @param callback The action to execute.
        public void SetIdleCallback(System.Action callback) => onBecameIdle = callback;

        /// @brief Initializes the AGV controller and anchors it to the traffic grid.
        /// @details Parks the NavMeshAgent to use transform-based movement and reserves the starting zone.
        /// @param id The unique identifier for this AGV.
        /// @post Agent is stopped, ground offset is applied, and the initial zone is reserved.
        public void Initialize(int id)
        {
            AgvId = id;
            agent = GetComponent<NavMeshAgent>();
            agent.updatePosition = false;
            agent.updateRotation = false;
            agent.isStopped = true;

            trafficMgr = FactoryLayoutManager.Instance.GetComponent<TrafficZoneManager>();

            Vector3 pos = transform.position;
            pos.y = groundOffset;
            transform.position = pos;

            currentZoneId = FindZoneAtSelf();
            if (currentZoneId >= 0)
                trafficMgr.TryReserve(currentZoneId, AgvId);

            State = AGVState.Idle;
        }

        /// @brief Assigns a pickup and delivery task to the AGV.
        /// @details Resolves the nearest valid docks for the source and target machines, 
        /// plans a BFS route to the pickup location, and initiates navigation.
        /// @param jobId ID of the job to transport.
        /// @param pickupPosition Target world position for pickup.
        /// @param dropoffPosition Target world position for dropoff.
        /// @param source The machine providing the job (null for incoming belt).
        /// @param dropoff The machine receiving the job (null for outgoing belt).
        /// @pre AGV should be in an Idle state.
        /// @post State changes to NavigatingToPickup if a route is found; otherwise, returns to Idle.
        public void Dispatch(int jobId, Vector3 pickupPosition, Vector3 dropoffPosition, PhysicalMachine source, PhysicalMachine dropoff)
        {
            CurrentJobId = jobId;
            sourceMachine = source;
            targetMachine = dropoff;
            targetPickupPos = pickupPosition;
            targetDropoffPos = dropoffPosition;

            if (currentZoneId < 0)
            {
                currentZoneId = FindZoneAtSelf();
                if (currentZoneId >= 0)
                    trafficMgr.TryReserve(currentZoneId, AgvId);
            }

            if (source != null)
                (pickupZoneId, pickupDock) = FindDockForMachine(source.MachineId, currentZoneId, targetPickupPos);
            else
                (pickupZoneId, pickupDock) = FindSpecialDock(TrafficZoneManager.IncomingBeltId);

            if (dropoff != null)
                (dropoffZoneId, dropoffDock) = FindDockForMachine(dropoff.MachineId, pickupZoneId, targetDropoffPos);
            else
                (dropoffZoneId, dropoffDock) = FindSpecialDock(TrafficZoneManager.OutgoingBeltId);

            if (!PlanRoute(currentZoneId, pickupZoneId))
            {
                SimLogger.Error($"[AGV {AgvId}] No route to pickup zone {pickupZoneId}.");
                ResetToIdle();
                return;
            }

            State = AGVState.NavigatingToPickup;
            BeginNextWaypoint();
        }

        private Vector3 targetPickupPos;
        private Vector3 targetDropoffPos;

        /// @brief Standard Unity update loop driving the state machine.
        private void Update()
        {
            switch (State)
            {
                case AGVState.Idle: return;
                case AGVState.NavigatingToPickup:
                case AGVState.NavigatingToDropoff:
                    UpdateNavigation();
                    break;
                case AGVState.AligningForPickup:
                    if (AlignToDock(pickupDock))
                    {
                        State = AGVState.ExecutingPickup;
                        handshakeTimer = handshakeDuration;
                    }
                    break;
                case AGVState.ExecutingPickup:
                    handshakeTimer -= Time.deltaTime;
                    if (handshakeTimer <= 0f) ExecutePickup();
                    break;
                case AGVState.AligningForDropoff:
                    if (AlignToDock(dropoffDock))
                    {
                        State = AGVState.ExecutingDropoff;
                        handshakeTimer = handshakeDuration;
                    }
                    break;
                case AGVState.ExecutingDropoff:
                    handshakeTimer -= Time.deltaTime;
                    if (handshakeTimer <= 0f) ExecuteDropoff();
                    break;
                case AGVState.WaitingForZone:
                    UpdateWaiting();
                    break;
            }
            agent.nextPosition = transform.position;
        }

        private float handshakeTimer;

        /// @brief Advances movement toward the current waypoint and handles zone transitions.
        /// @details Monitors distance to waypoints. Upon arrival, it triggers zone reservation for 
        /// the next hop or transitions the AGV into alignment mode if at the final destination.
        private void UpdateNavigation()
        {
            float dist = FlatDistance(transform.position, currentWaypoint);
            bool pastRoute = (routeIndex >= currentRoute.Count);
            float threshold = pastRoute ? dockArrivalDist : waypointArrivalDist;

            if (dist <= threshold)
            {
                if (pastRoute)
                {
                    State = (State == AGVState.NavigatingToPickup) ? AGVState.AligningForPickup : AGVState.AligningForDropoff;
                    return;
                }

                OnEnteredZone(currentRoute[routeIndex]);
                routeIndex++;
                BeginNextWaypoint();

                if (State != AGVState.WaitingForZone)
                    MoveToward(currentWaypoint);

                return;
            }

            MoveToward(currentWaypoint);
        }

        /// @brief Sets the next target position in the route and attempts to reserve the zone.
        /// @details If the next zone in the route is occupied, the AGV enters a waiting state. 
        /// If the route is finished, the waypoint is set to the dock's approach position.
        /// @post State may change to WaitingForZone if reservation fails.
        private void BeginNextWaypoint()
        {
            if (routeIndex < currentRoute.Count)
            {
                int nextZoneId = currentRoute[routeIndex];

                if (nextZoneId == currentZoneId)
                {
                    routeIndex++;
                    BeginNextWaypoint();
                    return;
                }

                if (!trafficMgr.TryReserve(nextZoneId, AgvId))
                {
                    EnterWaitState(nextZoneId);
                    return;
                }

                TrafficZone zone = trafficMgr.GetZone(nextZoneId);
                currentWaypoint = FlatY(zone.Centre);
            }
            else
            {
                DockPoint dock = (State == AGVState.NavigatingToPickup) ? pickupDock : dropoffDock;
                currentWaypoint = FlatY(dock.ApproachPosition);
            }
        }

        /// @brief Updates internal zone tracking and releases the previous zone.
        /// @param newZoneId The ID of the zone the AGV has physically reached.
        /// @post The previous zone is freed in the TrafficZoneManager.
        private void OnEnteredZone(int newZoneId)
        {
            if (previousZoneId >= 0 && previousZoneId != newZoneId)
                trafficMgr.Release(previousZoneId, AgvId);

            previousZoneId = currentZoneId;
            currentZoneId = newZoneId;
        }

        /// @brief Suspends navigation and schedules a reservation retry.
        /// @param blockedZoneId The zone ID that is currently at capacity.
        /// @post State becomes WaitingForZone.
        private void EnterWaitState(int blockedZoneId)
        {
            stateBeforeWait = State;
            pendingZoneId = blockedZoneId;
            nextRetryTime = Time.time + reservationRetryInterval;
            State = AGVState.WaitingForZone;
        }

        /// @brief Periodically checks if a pending zone reservation has become available.
        /// @post Resumes previous navigation state if reservation is granted.
        private void UpdateWaiting()
        {
            if (Time.time < nextRetryTime) return;

            if (trafficMgr.TryReserve(pendingZoneId, AgvId))
            {
                State = stateBeforeWait;
                TrafficZone zone = trafficMgr.GetZone(pendingZoneId);
                currentWaypoint = FlatY(zone.Centre);
                pendingZoneId = -1;
                return;
            }

            nextRetryTime = Time.time + reservationRetryInterval;
        }

        /// @brief Moves the AGV using the Turn-Then-Move constraint.
        /// @details Rotates in place if the angle to the target exceeds pathTurnThreshold. 
        /// Once aligned, moves linearly toward the target position.
        /// @param target The world-space destination.
        private void MoveToward(Vector3 target)
        {
            Vector3 dir = target - transform.position;
            dir.y = 0f;
            if (dir.sqrMagnitude < 0.001f) return;

            float angle = Vector3.Angle(transform.forward, dir);

            if (angle > pathTurnThreshold)
            {
                RotateToward(dir.normalized);
            }
            else
            {
                transform.rotation = Quaternion.LookRotation(dir.normalized, Vector3.up);
                transform.position = Vector3.MoveTowards(transform.position, target, moveSpeed * Time.deltaTime);
            }
        }

        /// @brief Rotates the transform toward a specific direction.
        /// @param flatDir Normalized direction vector on the XZ plane.
        private void RotateToward(Vector3 flatDir)
        {
            Quaternion goal = Quaternion.LookRotation(flatDir, Vector3.up);
            transform.rotation = Quaternion.RotateTowards(transform.rotation, goal, turnSpeed * Time.deltaTime);
        }

        /// @brief Aligns the AGV heading with the dock's required handshake direction.
        /// @param dock The DockPoint to align with.
        /// @return True if the AGV is within the alignment threshold; false otherwise.
        private bool AlignToDock(DockPoint dock)
        {
            Vector3 desired = dock.FacingDirection;
            desired.y = 0f;
            if (desired.sqrMagnitude < 0.001f) return true;

            float remaining = Vector3.Angle(transform.forward, desired);
            if (remaining <= alignmentThreshold) return true;

            RotateToward(desired.normalized);
            return false;
        }

        /// @brief Transfers a job from a machine/belt to the AGV.
        /// @details Parent the job visual to the AGV and initiates the transit phase in the simulation. 
        /// Re-resolves the dropoff path based on the current location.
        /// @pre Handshake timer must have expired at the pickup dock.
        /// @post State changes to NavigatingToDropoff.
        private void ExecutePickup()
        {
            JobTracker tracker = SimulationBridge.Instance.JobManager.GetJobTracker(CurrentJobId);
            loadedJobVisual = tracker?.Visual;

            if (sourceMachine != null)
                sourceMachine.ReleaseFromOutgoing(CurrentJobId);
            else
                FactoryLayoutManager.Instance.IncomingBelt?.RemoveJob(CurrentJobId);

            if (loadedJobVisual != null)
            {
                loadedJobVisual.AttachToCarrier(carryPos);
                loadedJobVisual.SetState(JobLifecycleState.InTransit);
            }

            int nextMachineId = targetMachine != null ? targetMachine.MachineId : -1;
            SimulationBridge.Instance.JobManager.BeginTransit(CurrentJobId, nextMachineId, Time.time);

            if (targetMachine != null)
                (dropoffZoneId, dropoffDock) = FindDockForMachine(targetMachine.MachineId, currentZoneId, targetDropoffPos);
            else
                (dropoffZoneId, dropoffDock) = FindSpecialDock(TrafficZoneManager.OutgoingBeltId);

            if (!PlanRoute(currentZoneId, dropoffZoneId))
            {
                ResetToIdle();
                return;
            }

            State = AGVState.NavigatingToDropoff;
            BeginNextWaypoint();
        }

        /// @brief Transfers the job from the AGV to the target machine/belt.
        /// @details Detaches the visual, notifies the receiving machine, and marks the transit as complete.
        /// @pre Handshake timer must have expired at the dropoff dock.
        /// @post AGV returns to Idle and invokes the idle callback.
        private void ExecuteDropoff()
        {
            if (loadedJobVisual != null)
                loadedJobVisual.DetachFromCarrier(dropoffDock.HandshakePosition);

            if (targetMachine != null)
                targetMachine.ReceiveJob(CurrentJobId, loadedJobVisual);
            else
                FactoryLayoutManager.Instance.OutgoingBelt?.TryEnqueue(CurrentJobId, loadedJobVisual);

            int machineId = targetMachine != null ? targetMachine.MachineId : -1;
            SimulationBridge.Instance.JobManager.CompleteTransit(CurrentJobId, machineId, Time.time);

            ResetToIdle();
        }

        /// @brief Calculates a zone-level path between two zones.
        /// @param fromZone Starting zone ID.
        /// @param toZone Destination zone ID.
        /// @return True if a valid path exists; false otherwise.
        private bool PlanRoute(int fromZone, int toZone)
        {
            currentRoute.Clear();
            routeIndex = 0;
            if (fromZone < 0 || toZone < 0) return false;

            List<int> route = trafficMgr.GetRoute(fromZone, toZone);
            if (route == null || route.Count == 0) return false;

            currentRoute.AddRange(route);
            return true;
        }

        /// @brief Identifies the zone currently containing the AGV's transform.
        /// @return Zone ID, or -1 if outside the traffic grid.
        private int FindZoneAtSelf()
        {
            TrafficZone z = trafficMgr.GetZoneAtPosition(transform.position);
            return z?.ZoneId ?? -1;
        }

        /// @brief Searches for the dock point that physically matches the target conveyor.
        /// @param machineId ID of the target machine.
        /// @param fromZoneId Origin zone for context.
        /// @param targetConveyorPos The world position of the conveyor end.
        /// @return A tuple containing the resolved Zone ID and DockPoint.
        private (int zoneId, DockPoint dock) FindDockForMachine(int machineId, int fromZoneId, Vector3 targetConveyorPos)
        {
            List<int> candidates = trafficMgr.GetZonesForMachine(machineId);
            int bestZone = -1;
            DockPoint bestDock = default;
            float closestDist = float.MaxValue;

            foreach (int zId in candidates)
            {
                if (!trafficMgr.TryGetDockPoint(zId, machineId, out DockPoint d)) continue;
                float dist = Vector3.Distance(d.HandshakePosition, targetConveyorPos);
                if (dist < closestDist)
                {
                    closestDist = dist;
                    bestZone = zId;
                    bestDock = d;
                }
            }
            return (bestZone, bestDock);
        }

        /// @brief Finds a dock assigned to special infrastructure like belts.
        /// @param specialId The ID of the special zone/object.
        /// @return A tuple containing the Zone ID and DockPoint.
        private (int zoneId, DockPoint dock) FindSpecialDock(int specialId)
        {
            foreach (TrafficZone zone in trafficMgr.Zones)
            {
                if (zone.DockPoints.TryGetValue(specialId, out DockPoint d))
                    return (zone.ZoneId, d);
            }
            return (-1, default);
        }

        /// @brief Clears current job data and releases reserved zones.
        /// @post State is Idle, and external listeners are notified.
        private void ResetToIdle()
        {
            if (previousZoneId >= 0 && previousZoneId != currentZoneId)
                trafficMgr.Release(previousZoneId, AgvId);

            previousZoneId = -1;
            CurrentJobId = -1;
            loadedJobVisual = null;
            sourceMachine = null;
            targetMachine = null;
            currentRoute.Clear();
            routeIndex = 0;
            pickupZoneId = -1;
            dropoffZoneId = -1;
            pendingZoneId = -1;

            State = AGVState.Idle;
            onBecameIdle?.Invoke();
        }

        /// @brief Calculates 2D distance on the XZ plane.
        private static float FlatDistance(Vector3 a, Vector3 b)
        {
            float dx = a.x - b.x;
            float dz = a.z - b.z;
            return Mathf.Sqrt(dx * dx + dz * dz);
        }

        /// @brief Projects a position onto the AGV's driving height.
        private Vector3 FlatY(Vector3 pos)
        {
            pos.y = groundOffset;
            return pos;
        }

        private void OnDrawGizmosSelected()
        {
            if (currentRoute == null || currentRoute.Count == 0 || trafficMgr == null) return;

            Gizmos.color = (State == AGVState.NavigatingToPickup || State == AGVState.AligningForPickup)
                    ? new Color(0.3f, 1f, 0.3f, 0.7f)
                    : new Color(1f, 0.6f, 0.2f, 0.7f);

            Vector3 prev = transform.position;
            for (int i = routeIndex; i < currentRoute.Count; i++)
            {
                TrafficZone z = trafficMgr.GetZone(currentRoute[i]);
                if (z == null) continue;
                Gizmos.DrawLine(prev, z.Centre);
                Gizmos.DrawWireSphere(z.Centre, 0.25f);
                prev = z.Centre;
            }

            Gizmos.color = Color.cyan;
            Gizmos.DrawWireSphere(currentWaypoint, 0.2f);
            Gizmos.DrawLine(transform.position, currentWaypoint);
        }
    }
}