using UnityEngine;
using UnityEngine.AI;
using System.Collections.Generic;
using Assets.Scripts.Simulation.Machines;
using Assets.Scripts.Simulation.FactoryLayout;
using Assets.Scripts.Logging;

namespace Assets.Scripts.Simulation.AGV
{
    /// <summary>
    /// Represents the current operational state of an AGV.
    /// </summary>
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

    /// <summary>
    /// Controls the navigation, job assignment, and lifecycle of a single
    /// Automated Guided Vehicle (AGV).
    ///
    /// <para><b>Movement model — Turn-Then-Move:</b></para>
    /// The AGV rotates in place until aligned with the next waypoint, then
    /// drives straight. There is no simultaneous steering — just like a real
    /// factory AGV. All movement is transform-based (NavMeshAgent is parked).
    ///
    /// <para><b>Routing — Zone Graph:</b></para>
    /// Instead of letting NavMeshAgent find the shortest geometric path
    /// (which cuts diagonals and clips machines), the AGV plans a route
    /// through the <see cref="TrafficZoneManager"/> zone graph using BFS.
    /// Waypoints are zone centres, so the AGV naturally stays centred in
    /// aisles and follows one-way traffic flow.
    ///
    /// <para><b>Deadlock Prevention — Reserve-Ahead:</b></para>
    /// Before entering the next zone the AGV must reserve it via
    /// <see cref="TrafficZoneManager.TryReserve"/>. If the zone is full
    /// (capacity 1 for row aisles) the AGV waits and retries. On arrival
    /// at the new zone the previous zone is released. Since zones enforce
    /// one-way flow and capacity limits, head-on collisions are impossible.
    ///
    /// <para><b>Conveyor Handshake — DockPoints:</b></para>
    /// At the pickup/dropoff zone the AGV navigates to the DockPoint's
    /// <see cref="DockPoint.ApproachPosition"/>, then rotates to face
    /// <see cref="DockPoint.FacingDirection"/> before executing the
    /// job exchange. This guarantees the AGV's shelf is oriented toward
    /// the conveyor regardless of arrival direction.
    /// </summary>
    [RequireComponent(typeof(NavMeshAgent))]
    public class AGVController : MonoBehaviour
    {
        // ─────────────────────────────────────────────────────────
        //  Inspector
        // ─────────────────────────────────────────────────────────

        [Header("References")]
        [SerializeField] private Transform carryPos;

        [Header("Handshake")]
        [Tooltip("Time in seconds the AGV waits to simulate transferring a job.")]
        [SerializeField] private float handshakeDuration = 1.5f;

        private float handshakeTimer;
        private Vector3 targetPickupPos;
        private Vector3 targetDropoffPos;

        [Header("Movement")]
        [Tooltip("Forward driving speed in world units/sec.")]
        [SerializeField] private float moveSpeed = 3.5f;

        [Tooltip("Rotation speed in degrees/sec when turning in place.")]
        [SerializeField] private float turnSpeed = 180f;

        [Tooltip("Angle (degrees) to next waypoint above which the AGV stops " +
                 "and rotates in place instead of driving forward.")]
        [SerializeField] private float pathTurnThreshold = 10f;

        [Header("Waypoint Arrival")]
        [Tooltip("Distance from a zone centre at which the AGV considers " +
                 "itself arrived and advances to the next zone.")]
        [SerializeField] private float waypointArrivalDist = 0.4f;

        [Tooltip("Distance from the dock approach point at which the AGV " +
                 "switches to alignment. Increase if the shelf clips the conveyor.")]
        [SerializeField] private float dockArrivalDist = 0.3f;

        [Tooltip("Angle (degrees) considered 'aligned' for dock facing.")]
        [SerializeField] private float alignmentThreshold = 3f;

        [Header("Zone Reservation")]
        [Tooltip("Seconds between reservation retry attempts when blocked.")]
        [SerializeField] private float reservationRetryInterval = 0.25f;

        [Header("Positioning")]
        [Tooltip("Height above the floor plane for the AGV's transform. " +
                 "Set to half the AGV's body height so it sits on the surface.")]
        [SerializeField] private float groundOffset = 0.5f;

        // ─────────────────────────────────────────────────────────
        //  Public Properties
        // ─────────────────────────────────────────────────────────

        /// <summary>Unique identifier for this AGV.</summary>
        public int AgvId { get; private set; }

        /// <summary>Current operational state.</summary>
        public AGVState State { get; private set; } = AGVState.Idle;

        /// <summary>ID of the job being carried, or -1 if none.</summary>
        public int CurrentJobId { get; private set; } = -1;

        /// <summary>Zone the AGV currently occupies.</summary>
        public int CurrentZoneId => currentZoneId;

        // ─────────────────────────────────────────────────────────
        //  Private State
        // ─────────────────────────────────────────────────────────

        private NavMeshAgent agent;              // parked; kept for component compat
        private TrafficZoneManager trafficMgr;
        private JobVisual loadedJobVisual;
        private PhysicalMachine sourceMachine;
        private PhysicalMachine targetMachine;
        private System.Action onBecameIdle;

        // ── Zone navigation ──
        private readonly List<int> currentRoute = new List<int>();
        private int routeIndex;
        private int currentZoneId = -1;
        private int previousZoneId = -1;

        // ── Dock targets ──
        private int pickupZoneId = -1;
        private int dropoffZoneId = -1;
        private DockPoint pickupDock;
        private DockPoint dropoffDock;

        // ── Active waypoint ──
        private Vector3 currentWaypoint;

        // ── Wait / retry ──
        private AGVState stateBeforeWait;
        private int pendingZoneId = -1;
        private float nextRetryTime;

        // ─────────────────────────────────────────────────────────
        //  Initialisation
        // ─────────────────────────────────────────────────────────

        public void SetIdleCallback(System.Action callback) => onBecameIdle = callback;

        /// <summary>
        /// Initializes the AGV. Call after the factory floor and zone graph
        /// have been built.
        /// </summary>
        public void Initialize(int id)
        {
            AgvId = id;

            // Park the NavMeshAgent — all movement is via transform.
            agent = GetComponent<NavMeshAgent>();
            agent.updatePosition = false;
            agent.updateRotation = false;
            agent.isStopped = true;

            trafficMgr = FactoryLayoutManager.Instance
                             .GetComponent<TrafficZoneManager>();

            // Ensure the AGV sits above the floor, not embedded in it.
            Vector3 pos = transform.position;
            pos.y = groundOffset;
            transform.position = pos;

            // Reserve whatever zone we're standing in.
            currentZoneId = FindZoneAtSelf();
            if (currentZoneId >= 0)
                trafficMgr.TryReserve(currentZoneId, AgvId);

            State = AGVState.Idle;
        }

        // ─────────────────────────────────────────────────────────
        //  Dispatch
        // ─────────────────────────────────────────────────────────

        /// <summary>
        /// Assigns a delivery task to the AGV.
        /// </summary>
        /// <param name="jobId">Job to transport.</param>
        /// <param name="pickupPosition">World pos of the outgoing conveyor
        ///     output end (used to match the nearest dock).</param>
        /// <param name="dropoffPosition">World pos of the incoming conveyor
        ///     input end (used to match the nearest dock).</param>
        /// <param name="source">Source machine (null = main incoming belt).</param>
        /// <param name="dropoff">Target machine (null = main outgoing belt).</param>
        public void Dispatch(int jobId, Vector3 pickupPosition,
                             Vector3 dropoffPosition,
                             PhysicalMachine source, PhysicalMachine dropoff)
        {
            CurrentJobId = jobId;
            sourceMachine = source;
            targetMachine = dropoff;

            targetPickupPos = pickupPosition;
            targetDropoffPos = dropoffPosition;

            // ── Ensure we know our starting zone ──
            if (currentZoneId < 0)
            {
                currentZoneId = FindZoneAtSelf();
                if (currentZoneId >= 0)
                    trafficMgr.TryReserve(currentZoneId, AgvId);
            }

            // ── Resolve pickup zone & dock ──
            if (source != null)
                (pickupZoneId, pickupDock) = FindDockForMachine(source.MachineId, currentZoneId, targetPickupPos);
            else
                (pickupZoneId, pickupDock) = FindSpecialDock(TrafficZoneManager.IncomingBeltId);

            // ── Resolve dropoff zone & dock ──
            if (dropoff != null)
                (dropoffZoneId, dropoffDock) = FindDockForMachine(dropoff.MachineId, pickupZoneId, targetDropoffPos);
            else
                (dropoffZoneId, dropoffDock) = FindSpecialDock(TrafficZoneManager.OutgoingBeltId);

            // ── Plan route to pickup ──
            if (!PlanRoute(currentZoneId, pickupZoneId))
            {
                SimLogger.Error(
                    $"[AGV {AgvId}] No route from zone {currentZoneId} " +
                    $"to pickup zone {pickupZoneId}. Aborting dispatch.");
                ResetToIdle();
                return;
            }

            State = AGVState.NavigatingToPickup;
            BeginNextWaypoint();
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

            // Keep the NavMeshAgent transform in sync so its gizmos
            // and any external queries still reflect the real position.
            agent.nextPosition = transform.position;
        }

        // ─────────────────────────────────────────────────────────
        //  Navigation — Zone-to-Zone
        // ─────────────────────────────────────────────────────────

        /// <summary>
        /// Drives the AGV toward <see cref="currentWaypoint"/> using
        /// turn-then-move. When a waypoint is reached, advances to the
        /// next zone or transitions to the docking alignment state.
        /// </summary>
        private void UpdateNavigation()
        {
            float dist = FlatDistance(transform.position, currentWaypoint);

            // ── Are we at the final dock approach? ──
            bool pastRoute = (routeIndex >= currentRoute.Count);
            float threshold = pastRoute ? dockArrivalDist : waypointArrivalDist;

            if (dist <= threshold)
            {
                if (pastRoute)
                {
                    // Arrived at dock approach — begin alignment.
                    State = (State == AGVState.NavigatingToPickup)
                        ? AGVState.AligningForPickup
                        : AGVState.AligningForDropoff;
                    return;
                }

                // Arrived at an intermediate zone — advance.
                OnEnteredZone(currentRoute[routeIndex]);
                routeIndex++;
                BeginNextWaypoint();

                // If we got a new waypoint (not waiting for a zone),
                // start moving toward it THIS frame so there's no
                // visible stop-start hitch at each zone centre.
                if (State != AGVState.WaitingForZone)
                    MoveToward(currentWaypoint);

                return;
            }

            // ── Drive toward waypoint ──
            MoveToward(currentWaypoint);
        }

        /// <summary>
        /// Determines the next waypoint. If more zones remain in the route,
        /// the waypoint is the next zone's centre (reservation required).
        /// If the route is exhausted, the waypoint is the dock approach pos.
        /// </summary>
        private void BeginNextWaypoint()
        {
            if (routeIndex < currentRoute.Count)
            {
                int nextZoneId = currentRoute[routeIndex];

                // If we're already standing in this zone (start of route),
                // skip it — we don't need to walk to our own centre.
                if (nextZoneId == currentZoneId)
                {
                    routeIndex++;
                    BeginNextWaypoint();   // tail-recurse to next zone
                    return;
                }

                // ── Reserve the zone we're about to enter ──
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
                // Route consumed — final target is the dock approach position.
                DockPoint dock = (State == AGVState.NavigatingToPickup)
                    ? pickupDock
                    : dropoffDock;
                currentWaypoint = FlatY(dock.ApproachPosition);
            }
        }

        /// <summary>
        /// Book-keeping when the AGV physically reaches a new zone centre.
        /// Releases the zone we just left.
        /// </summary>
        private void OnEnteredZone(int newZoneId)
        {
            if (previousZoneId >= 0 && previousZoneId != newZoneId)
                trafficMgr.Release(previousZoneId, AgvId);

            previousZoneId = currentZoneId;
            currentZoneId = newZoneId;

            SimLogger.Low(
                $"[AGV {AgvId}] Entered zone " +
                $"{trafficMgr.GetZone(newZoneId)?.Name ?? newZoneId.ToString()}");
        }

        // ─────────────────────────────────────────────────────────
        //  Waiting For Zone
        // ─────────────────────────────────────────────────────────

        private void EnterWaitState(int blockedZoneId)
        {
            stateBeforeWait = State;
            pendingZoneId = blockedZoneId;
            nextRetryTime = Time.time + reservationRetryInterval;
            State = AGVState.WaitingForZone;

            SimLogger.Low(
                $"[AGV {AgvId}] Waiting for zone " +
                $"{trafficMgr.GetZone(blockedZoneId)?.Name}");
        }

        private void UpdateWaiting()
        {
            if (Time.time < nextRetryTime) return;

            if (trafficMgr.TryReserve(pendingZoneId, AgvId))
            {
                // Reservation granted — resume.
                State = stateBeforeWait;
                TrafficZone zone = trafficMgr.GetZone(pendingZoneId);
                currentWaypoint = FlatY(zone.Centre);
                pendingZoneId = -1;
                return;
            }

            // Still blocked.
            nextRetryTime = Time.time + reservationRetryInterval;
        }

        // ─────────────────────────────────────────────────────────
        //  Turn-Then-Move
        // ─────────────────────────────────────────────────────────

        /// <summary>
        /// Drives the AGV toward <paramref name="target"/> using a strict
        /// turn-then-move pattern. The AGV never translates and rotates
        /// simultaneously (above the threshold).
        /// </summary>
        private void MoveToward(Vector3 target)
        {
            Vector3 dir = target - transform.position;
            dir.y = 0f;
            if (dir.sqrMagnitude < 0.001f) return;

            float angle = Vector3.Angle(transform.forward, dir);

            if (angle > pathTurnThreshold)
            {
                // Misaligned — rotate in place only, zero translation.
                RotateToward(dir.normalized);
            }
            else
            {
                // Aligned — snap heading to exact target direction so
                // the AGV drives in a perfectly straight line with no arc.
                // The snap is sub-threshold so it's visually imperceptible.
                transform.rotation = Quaternion.LookRotation(dir.normalized, Vector3.up);
                transform.position = Vector3.MoveTowards(
                    transform.position, target, moveSpeed * Time.deltaTime);
            }
        }

        /// <summary>
        /// Smoothly rotates the AGV toward a flat direction vector at
        /// <see cref="turnSpeed"/> degrees per second.
        /// </summary>
        private void RotateToward(Vector3 flatDir)
        {
            Quaternion goal = Quaternion.LookRotation(flatDir, Vector3.up);
            transform.rotation = Quaternion.RotateTowards(
                transform.rotation, goal, turnSpeed * Time.deltaTime);
        }

        // ─────────────────────────────────────────────────────────
        //  Dock Alignment
        // ─────────────────────────────────────────────────────────

        /// <summary>
        /// Rotates in place to face the dock's required direction.
        /// Returns true once the angle is within <see cref="alignmentThreshold"/>.
        /// </summary>
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

        // ─────────────────────────────────────────────────────────
        //  Pickup / Dropoff
        // ─────────────────────────────────────────────────────────

        /// <summary>
        /// Grabs the job from the source conveyor and begins navigating
        /// toward the dropoff zone.
        /// </summary>
        private void ExecutePickup()
        {
            JobTracker tracker =
                SimulationBridge.Instance.JobManager.GetJobTracker(CurrentJobId);
            loadedJobVisual = tracker?.Visual;

            if (sourceMachine != null)
                sourceMachine.ReleaseFromOutgoing(CurrentJobId);
            else
                FactoryLayoutManager.Instance.IncomingBelt?
                    .RemoveJob(CurrentJobId);

            if (loadedJobVisual != null)
            {
                loadedJobVisual.AttachToCarrier(carryPos);
                loadedJobVisual.SetState(JobLifecycleState.InTransit);
            }

            int nextMachineId = targetMachine != null
                ? targetMachine.MachineId : -1;
            SimulationBridge.Instance.JobManager
                .BeginTransit(CurrentJobId, nextMachineId, Time.time);

            // ── Re-resolve dropoff dock from our actual position ──
            if (targetMachine != null)
                (dropoffZoneId, dropoffDock) =
                    FindDockForMachine(targetMachine.MachineId, currentZoneId, targetDropoffPos);
            else
                (dropoffZoneId, dropoffDock) =
                    FindSpecialDock(TrafficZoneManager.OutgoingBeltId);

            // ── Plan route to dropoff ──
            if (!PlanRoute(currentZoneId, dropoffZoneId))
            {
                SimLogger.Error(
                    $"[AGV {AgvId}] No route to dropoff zone {dropoffZoneId}.");
                ResetToIdle();
                return;
            }

            State = AGVState.NavigatingToDropoff;
            BeginNextWaypoint();
        }

        /// <summary>
        /// Hands the job to the target conveyor and resets to Idle.
        /// </summary>
        private void ExecuteDropoff()
        {
            if (loadedJobVisual != null)
                loadedJobVisual.DetachFromCarrier(dropoffDock.HandshakePosition);

            if (targetMachine != null)
                targetMachine.ReceiveJob(CurrentJobId, loadedJobVisual);
            else
                FactoryLayoutManager.Instance.OutgoingBelt?
                    .TryEnqueue(CurrentJobId, loadedJobVisual);

            int machineId = targetMachine != null
                ? targetMachine.MachineId : -1;
            SimulationBridge.Instance.JobManager
                .CompleteTransit(CurrentJobId, machineId, Time.time);

            ResetToIdle();
        }

        // ─────────────────────────────────────────────────────────
        //  Route Planning
        // ─────────────────────────────────────────────────────────

        /// <summary>
        /// Plans a zone-level route via BFS downstream links.
        /// </summary>
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

        // ─────────────────────────────────────────────────────────
        //  Zone & Dock Lookup
        // ─────────────────────────────────────────────────────────

        /// <summary>Returns the zone the AGV is standing in.</summary>
        private int FindZoneAtSelf()
        {
            TrafficZone z = trafficMgr.GetZoneAtPosition(transform.position);
            return z?.ZoneId ?? -1;
        }

        /// <summary>
        /// Finds the zone and DockPoint for <paramref name="machineId"/> by matching it 
        /// to the physical location of the assigned conveyor belt.
        /// </summary>
        private (int zoneId, DockPoint dock) FindDockForMachine(
            int machineId, int fromZoneId, Vector3 targetConveyorPos)
        {
            List<int> candidates = trafficMgr.GetZonesForMachine(machineId);
            int bestZone = -1;
            DockPoint bestDock = default;
            float closestDist = float.MaxValue;

            foreach (int zId in candidates)
            {
                if (!trafficMgr.TryGetDockPoint(zId, machineId, out DockPoint d))
                    continue;

                // Ignore routing hops to find the destination. The AGV MUST go to the 
                // dock that aligns physically with the conveyor belt target.
                float dist = Vector3.Distance(d.HandshakePosition, targetConveyorPos);

                if (dist < closestDist)
                {
                    closestDist = dist;
                    bestZone = zId;
                    bestDock = d;
                }
            }

            if (bestZone < 0)
                SimLogger.Error(
                    $"[AGV {AgvId}] No reachable dock for machine {machineId} " +
                    $"near conveyor position {targetConveyorPos}");

            return (bestZone, bestDock);
        }
        /// <summary>
        /// Finds the zone and DockPoint for a special infrastructure ID
        /// (incoming belt, outgoing belt, or parking area).
        /// </summary>
        private (int zoneId, DockPoint dock) FindSpecialDock(int specialId)
        {
            foreach (TrafficZone zone in trafficMgr.Zones)
            {
                if (zone.DockPoints.TryGetValue(specialId, out DockPoint d))
                    return (zone.ZoneId, d);
            }

            SimLogger.Error(
                $"[AGV {AgvId}] No dock found for special ID {specialId}");
            return (-1, default);
        }

        // ─────────────────────────────────────────────────────────
        //  Reset
        // ─────────────────────────────────────────────────────────

        private void ResetToIdle()
        {
            // Release every zone except the one we're standing in.
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

        // ─────────────────────────────────────────────────────────
        //  Utility
        // ─────────────────────────────────────────────────────────

        /// <summary>Flat XZ distance, ignoring vertical.</summary>
        private static float FlatDistance(Vector3 a, Vector3 b)
        {
            float dx = a.x - b.x;
            float dz = a.z - b.z;
            return Mathf.Sqrt(dx * dx + dz * dz);
        }

        /// <summary>
        /// Copies <paramref name="pos"/> but overwrites Y with
        /// <see cref="groundOffset"/> so the AGV rides above the floor.
        /// </summary>
        private Vector3 FlatY(Vector3 pos)
        {
            pos.y = groundOffset;
            return pos;
        }

        // ─────────────────────────────────────────────────────────
        //  Editor Gizmos
        // ─────────────────────────────────────────────────────────

        private void OnDrawGizmosSelected()
        {
            if (currentRoute == null || currentRoute.Count == 0) return;
            if (trafficMgr == null) return;

            // Draw the remaining route as a coloured path.
            Gizmos.color =
                (State == AGVState.NavigatingToPickup ||
                 State == AGVState.AligningForPickup)
                    ? new Color(0.3f, 1f, 0.3f, 0.7f)    // green = to pickup
                    : new Color(1f, 0.6f, 0.2f, 0.7f);    // orange = to dropoff

            Vector3 prev = transform.position;
            for (int i = routeIndex; i < currentRoute.Count; i++)
            {
                TrafficZone z = trafficMgr.GetZone(currentRoute[i]);
                if (z == null) continue;
                Gizmos.DrawLine(prev, z.Centre);
                Gizmos.DrawWireSphere(z.Centre, 0.25f);
                prev = z.Centre;
            }

            // Active waypoint
            Gizmos.color = Color.cyan;
            Gizmos.DrawWireSphere(currentWaypoint, 0.2f);
            Gizmos.DrawLine(transform.position, currentWaypoint);
        }
    }
}