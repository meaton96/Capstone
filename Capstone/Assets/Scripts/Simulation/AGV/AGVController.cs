using UnityEngine;
using UnityEngine.AI;
using System.Collections.Generic;
using Assets.Scripts.Simulation.Machines;
using Assets.Scripts.Simulation.FactoryLayout;
using Assets.Scripts.Logging;
using Assets.Scripts.Simulation.Jobs;
using TMPro;

namespace Assets.Scripts.Simulation.AGV
{
    /// @brief Defines the operational states of an AGV during its lifecycle.
    public enum AGVState
    {
        Idle,
        MovingToPickup,
        MovingToDropoff,
        ReturningToParking,
        MovingToPrePickup,
    }

    /// @brief Controls navigation and physical movement of a single Automated Guided Vehicle (AGV).
    ///
    /// @details Acts as a state-driven controller that interfaces with the @c NavMeshAgent 
    /// and @c TrafficZoneManager. This class follows an "orchestrator-flag" pattern: it 
    /// manages physical movement and sets status flags (@c PickedUpFlag, @c DeliveredFlag) 
    /// for a central supervisor to process, rather than triggering job state transitions directly.
    [RequireComponent(typeof(NavMeshAgent))]
    public class AGVController : MonoBehaviour
    {
        [Header("Settings")]
        [SerializeField] private Transform carryPos;
        [SerializeField] private float handshakeDuration = 1.5f;
        [SerializeField] private float moveSpeed = 3.5f;
        [SerializeField] private float turnSpeed = 180f;
        [SerializeField] private float pathTurnThreshold = 10f;
        [SerializeField] private float waypointArrivalDist = 0.4f;
        [SerializeField] private float dockArrivalDist = 0.3f;
        [SerializeField] private float alignmentThreshold = 3f;
        [SerializeField] private float reservationRetryInterval = 0.05f;
        [SerializeField] private float groundOffset = 0.5f;

        [Header("Debug Label")]
        [SerializeField] private TextMeshProUGUI statusLabel;

        public int AgvId { get; private set; }
        public AGVState State { get; private set; } = AGVState.Idle;
        public int CurrentJobId { get; private set; } = -1;
        public int CurrentZoneId => currentZoneId;
        public bool IsIdle => State == AGVState.Idle;
        public int PreDispatchedJobId { get; private set; } = -1;
        public bool IsPreDispatched => State == AGVState.MovingToPrePickup;

        public bool PickedUpFlag { get; private set; }
        public bool DeliveredFlag { get; private set; }
        public int DeliveredJobId { get; private set; } = -1;
        public int DeliveredMachineId { get; private set; } = -1;

        /// @brief Resets all completion flags and delivery metadata.
        ///
        /// @details Should be called by the orchestrator immediately after consuming 
        /// the status of a @c PickedUpFlag or @c DeliveredFlag.
        public void ClearFlags()
        {
            PickedUpFlag = false;
            DeliveredFlag = false;
            DeliveredJobId = -1;
            DeliveredMachineId = -1;
        }

        private NavMeshAgent navAgent;
        private TrafficZoneManager trafficMgr;
        private System.Action onBecameIdle;

        private PhysicalMachine sourceMachine;
        private PhysicalMachine targetMachine;
        private Vector3 targetPickupPos;
        private Vector3 targetDropoffPos;
        private JobVisual loadedJobVisual;

        private int pickupZoneId = -1;
        private int dropoffZoneId = -1;
        private DockPoint pickupDock;
        private DockPoint dropoffDock;
        private DockPoint parkingDock;
        private int parkingZoneId = -1;

        private readonly List<int> currentRoute = new List<int>();
        private int routeIndex;
        private Vector3 currentWaypoint;

        private int currentZoneId = -1;
        private int previousZoneId = -1;

        private bool waitingForZone;
        private int pendingZoneId = -1;
        private float nextRetryTime;

        private float pickupTimer;
        private float dropoffTimer;
        private bool atPickupDock;
        private bool atDropoffDock;

        public void SetIdleCallback(System.Action callback) => onBecameIdle = callback;

        /// @brief Sets up the AGV identity and initializes navigation components.
        ///
        /// @param id The unique integer identifier for this AGV instance.
        ///
        /// @details Disables automatic @c NavMeshAgent updates to allow for custom 
        /// state-based movement and reserves the starting traffic zone.
        public void Initialize(int id)
        {
            AgvId = id;
            navAgent = GetComponent<NavMeshAgent>();
            navAgent.updatePosition = false;
            navAgent.updateRotation = false;
            navAgent.isStopped = true;

            trafficMgr = FactoryLayoutManager.Instance.GetComponent<TrafficZoneManager>();

            Vector3 pos = transform.position;
            pos.y = groundOffset;
            transform.position = pos;

            currentZoneId = FindZoneAtSelf();
            if (currentZoneId >= 0)
                trafficMgr.TryReserve(currentZoneId, AgvId);

            State = AGVState.Idle;
            ClearFlags();
        }

        /// @brief Terminates the active route and releases future traffic zone reservations.
        private void CancelCurrentRoute()
        {
            if (!waitingForZone && routeIndex < currentRoute.Count)
            {
                int aheadZoneId = currentRoute[routeIndex];
                if (aheadZoneId != currentZoneId)
                    trafficMgr.Release(aheadZoneId, AgvId);
            }

            currentRoute.Clear();
            routeIndex = 0;
            waitingForZone = false;
            pendingZoneId = -1;
            parkingZoneId = -1;
        }

        /// @brief Commands the AGV to move to a pickup zone before a job is finished.
        ///
        /// @param jobId The ID of the job being anticipated.
        /// @param pickupPos The world position of the source conveyor/dock.
        /// @param source The machine instance the AGV is heading toward.
        ///
        /// @details Transitions the AGV to @c MovingToPrePickup. It will wait 
        /// at the designated dock until @c FinalizePreDispatch is called.
        public void PreDispatch(int jobId, Vector3 pickupPos, PhysicalMachine source)
        {
            //Debug.Log("Pre Dispatch");
            if (State != AGVState.Idle && State != AGVState.ReturningToParking)
            {
                SimLogger.Error($"[AGV {AgvId}] PreDispatch while unavailable (state={State}).");
                return;
            }

            if (State == AGVState.ReturningToParking)
            {
                CancelCurrentRoute();
            }

            PreDispatchedJobId = jobId;
            sourceMachine = source;
            targetPickupPos = pickupPos;
            targetMachine = null;
            loadedJobVisual = null;

            atPickupDock = false;
            atDropoffDock = false;
            pickupTimer = handshakeDuration;
            dropoffTimer = handshakeDuration;
            waitingForZone = false;
            pendingZoneId = -1;
            PickedUpFlag = false;

            if (currentZoneId < 0)
            {
                currentZoneId = FindZoneAtSelf();
                if (currentZoneId >= 0) trafficMgr.TryReserve(currentZoneId, AgvId);
            }

            if (source != null)
                (pickupZoneId, pickupDock) = FindDockForMachine(source.MachineId, currentZoneId, pickupPos);
            else
                (pickupZoneId, pickupDock) = FindSpecialDock(TrafficZoneManager.IncomingBeltId);

            if (pickupZoneId < 0 || !PlanRoute(currentZoneId, pickupZoneId))
            {
                SimLogger.Error($"[AGV {AgvId}] Cannot pre-dispatch to machine for job {jobId}.");
                PreDispatchedJobId = -1;
                return;
            }

            State = AGVState.MovingToPrePickup;
            BeginNextWaypoint();
            SimLogger.High($"[AGV {AgvId}] Pre-dispatched for job {jobId} — heading to pickup zone.");
        }

        /// @brief Upgrades a pre-dispatched AGV to a full pickup and delivery task.
        ///
        /// @details If the AGV is already waiting at the dock, it begins the 
        /// @c handshakeDuration immediately. Otherwise, it stores the target data 
        /// and proceeds with the pickup on arrival.
        public void FinalizePreDispatch(int jobId, Vector3 dropoffPos,
                                         PhysicalMachine target, JobVisual visual)
        {
            if (PreDispatchedJobId != jobId)
            {
                SimLogger.Error($"[AGV {AgvId}] FinalizePreDispatch job mismatch (expected {PreDispatchedJobId}, got {jobId}).");
                return;
            }

            CurrentJobId = jobId;
            targetMachine = target;
            targetDropoffPos = dropoffPos;
            loadedJobVisual = visual;
            PreDispatchedJobId = -1;

            State = AGVState.MovingToPickup;
            if (atPickupDock)
            {
                pickupTimer = handshakeDuration;
                SimLogger.High($"[AGV {AgvId}] Finalized pre-dispatch for job {jobId} — starting handshake.");
            }
            else
            {
                SimLogger.High($"[AGV {AgvId}] Finalized pre-dispatch for job {jobId} — en-route.");
            }
        }

        /// @brief Assigns a complete pickup and dropoff task to an idle AGV.
        public void Dispatch(int jobId, Vector3 pickupPos, Vector3 dropoffPos,
                             PhysicalMachine source, PhysicalMachine target,
                             JobVisual visual)
        {
            //Debug.Log("Dispatch");
            if (State != AGVState.Idle && State != AGVState.ReturningToParking)
            {
                SimLogger.Error($"[AGV {AgvId}] Dispatch while busy (state={State}).");
                return;
            }

            if (State == AGVState.ReturningToParking)
            {
                CancelCurrentRoute();
                SimLogger.High($"[AGV {AgvId}] Redirected from parking to job {jobId}.");
            }

            CurrentJobId = jobId;
            loadedJobVisual = visual;
            sourceMachine = source;
            targetMachine = target;
            targetPickupPos = pickupPos;
            targetDropoffPos = dropoffPos;

            atPickupDock = false;
            atDropoffDock = false;
            pickupTimer = handshakeDuration;
            dropoffTimer = handshakeDuration;
            waitingForZone = false;
            pendingZoneId = -1;
            PickedUpFlag = false;

            if (currentZoneId < 0)
            {
                currentZoneId = FindZoneAtSelf();
                if (currentZoneId >= 0) trafficMgr.TryReserve(currentZoneId, AgvId);
            }

            if (sourceMachine != null)
                (pickupZoneId, pickupDock) = FindDockForMachine(sourceMachine.MachineId, currentZoneId, pickupPos);
            else
                (pickupZoneId, pickupDock) = FindSpecialDock(TrafficZoneManager.IncomingBeltId);

            if (pickupZoneId < 0 || !PlanRoute(currentZoneId, pickupZoneId))
            {
                SimLogger.Error($"[AGV {AgvId}] Cannot reach pickup for job {jobId}.");
                FullReset();
                return;
            }

            State = AGVState.MovingToPickup;
            SimLogger.High($"[AGV] {AgvId} dispatched to pickup job {CurrentJobId} from machine {(targetMachine != null ? targetMachine.MachineId : -1)}");
            BeginNextWaypoint();
        }

        private void FixedUpdate()
        {
            switch (State)
            {
                case AGVState.MovingToPickup:
                    UpdateMovement();
                    if (!waitingForZone && ReachedDock(pickupDock))
                    {
                        if (!atPickupDock) { atPickupDock = true; AlignToDock(pickupDock); }
                        if (IsFacingDock(pickupDock))
                        {
                            pickupTimer -= Time.fixedDeltaTime;
                            if (pickupTimer <= 0f) DoPickup();
                        }
                        else AlignToDock(pickupDock);
                    }
                    break;

                case AGVState.MovingToDropoff:
                    UpdateMovement();
                    if (!waitingForZone && ReachedDock(dropoffDock))
                    {
                        if (!atDropoffDock) { atDropoffDock = true; AlignToDock(dropoffDock); }
                        if (IsFacingDock(dropoffDock))
                        {
                            dropoffTimer -= Time.fixedDeltaTime;
                            if (dropoffTimer <= 0f) DoDropoff();
                        }
                        else AlignToDock(dropoffDock);
                    }
                    break;

                case AGVState.ReturningToParking:
                    UpdateMovement();
                    if (!waitingForZone && ReachedParking()) ArriveAtParking();
                    break;

                case AGVState.MovingToPrePickup:
                    UpdateMovement();
                    if (!waitingForZone && ReachedDock(pickupDock))
                    {
                        if (!atPickupDock)
                        {
                            atPickupDock = true;
                            AlignToDock(pickupDock);
                            SimLogger.High($"[AGV {AgvId}] At pre-pickup dock for job {PreDispatchedJobId}.");
                        }
                    }
                    break;
            }

            navAgent.nextPosition = transform.position;
            UpdateStatusLabel();
        }

        /// @brief Executes the physical loading of a job and plans the route to the target.
        private void DoPickup()
        {
            if (sourceMachine != null)
                sourceMachine.RemoveFromOutgoing(CurrentJobId);
            else
                FactoryLayoutManager.Instance.IncomingBelt?.RemoveJob(CurrentJobId);

            if (loadedJobVisual != null)
                loadedJobVisual.AttachToCarrier(carryPos);

            if (targetMachine != null)
                (dropoffZoneId, dropoffDock) = FindDockForMachine(targetMachine.MachineId, currentZoneId, targetDropoffPos);
            else
                (dropoffZoneId, dropoffDock) = FindSpecialDock(TrafficZoneManager.OutgoingBeltId);

            if (dropoffZoneId < 0 || !PlanRoute(currentZoneId, dropoffZoneId))
            {
                SimLogger.Error($"[AGV {AgvId}] Cannot reach dropoff for job {CurrentJobId}.");
                FullReset();
                return;
            }

            PickedUpFlag = true;
            State = AGVState.MovingToDropoff;
            atPickupDock = false;
            dropoffTimer = handshakeDuration;
            BeginNextWaypoint();
        }

        /// @brief Unloads the job at the destination and initiates the return to parking.
        private void DoDropoff()
        {
            if (loadedJobVisual != null)
                loadedJobVisual.DetachFromCarrier(dropoffDock.HandshakePosition);

            if (targetMachine != null)
                targetMachine.PlaceOnIncoming(CurrentJobId, loadedJobVisual);
            else
                FactoryLayoutManager.Instance.OutgoingBelt?.TryEnqueue(CurrentJobId, loadedJobVisual);

            DeliveredFlag = true;
            DeliveredJobId = CurrentJobId;
            DeliveredMachineId = targetMachine != null ? targetMachine.MachineId : -1;

            CurrentJobId = -1;
            loadedJobVisual = null;
            sourceMachine = null;
            targetMachine = null;
            pickupZoneId = -1;
            dropoffZoneId = -1;
            atDropoffDock = false;

            State = AGVState.ReturningToParking;
            BeginParkingRoute();
        }

        /// @brief Calculates the path back to the AGV's assigned parking station.
        private void BeginParkingRoute()
        {
            Vector3 parkPos = AGVPool.Instance.GetParkingPosition(AgvId);
            TrafficZone parkZone = trafficMgr.GetZoneAtPosition(parkPos);
            parkingZoneId = parkZone?.ZoneId ?? -1;

            if (parkZone != null)
            {
                parkingDock = new DockPoint
                {
                    HandshakePosition = parkPos,
                    ApproachPosition = parkPos,
                    FacingDirection = Vector3.forward
                };
            }

            if (parkingZoneId < 0 || !PlanRoute(currentZoneId, parkingZoneId))
            {
                SimLogger.Error($"[AGV {AgvId}] No route to parking — resetting in place.");
                ArriveAtParking();
                return;
            }

            BeginNextWaypoint();
        }

        private bool ReachedParking()
        {
            Vector3 parkPos = AGVPool.Instance.GetParkingPosition(AgvId);
            return FlatDistance(transform.position, parkPos) <= waypointArrivalDist;
        }

        /// @brief Finalizes state and releases all zone reservations once parked.
        private void ArriveAtParking()
        {
            if (previousZoneId >= 0) { trafficMgr.Release(previousZoneId, AgvId); previousZoneId = -1; }
            if (currentZoneId >= 0) { trafficMgr.Release(currentZoneId, AgvId); currentZoneId = -1; }
            currentRoute.Clear();
            routeIndex = 0;
            waitingForZone = false;
            pendingZoneId = -1;
            parkingZoneId = -1;
            State = AGVState.Idle;
            onBecameIdle?.Invoke();
        }

        /// @brief Forces the AGV to abort its current job and return home.
        ///
        /// @details Used exclusively for error recovery if a route calculation fails 
        /// or a machine becomes unreachable mid-transit.
        private void FullReset()
        {
            SimLogger.Error($"[AGV {AgvId}] FullReset for job {CurrentJobId}.");
            CurrentJobId = -1;
            loadedJobVisual = null;
            sourceMachine = null;
            targetMachine = null;
            ClearFlags();

            if (previousZoneId >= 0) { trafficMgr.Release(previousZoneId, AgvId); previousZoneId = -1; }

            Vector3 parkPos = AGVPool.Instance.GetParkingPosition(AgvId);
            TrafficZone parkZone = trafficMgr.GetZoneAtPosition(parkPos);
            parkingZoneId = parkZone?.ZoneId ?? -1;

            if (parkingZoneId >= 0 && PlanRoute(currentZoneId, parkingZoneId))
            {
                State = AGVState.ReturningToParking;
                BeginNextWaypoint();
            }
            else
            {
                State = AGVState.Idle;
                onBecameIdle?.Invoke();
            }
        }

        public void SetCarryVisual(JobVisual visual) => loadedJobVisual = visual;

        /// @brief Updates the AGV's position toward the current waypoint.
        private void UpdateMovement()
        {
            if (waitingForZone)
            {
                TryResumeFromWait();
                return;
            }

            float dist = FlatDistance(transform.position, currentWaypoint);
            bool pastRoute = routeIndex >= currentRoute.Count;
            float threshold = pastRoute ? dockArrivalDist : waypointArrivalDist;

            if (dist <= threshold)
            {
                if (pastRoute) return;

                OnEnteredZone(currentRoute[routeIndex]);
                routeIndex++;
                BeginNextWaypoint();
            }
            else
            {
                MoveToward(currentWaypoint);
            }
        }

        /// @brief Attempts to acquire the next traffic zone in the planned route.
        ///
        /// @details If the next zone is occupied, the AGV sets @c waitingForZone 
        /// to true and stays at its current position until the zone clears.
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
                    waitingForZone = true;
                    pendingZoneId = nextZoneId;
                    nextRetryTime = Time.fixedTime + reservationRetryInterval;
                    return;
                }

                TrafficZone zone = trafficMgr.GetZone(nextZoneId);
                currentWaypoint = FlatY(zone.Centre);
            }
            else
            {
                DockPoint finalDock = State switch
                {
                    AGVState.MovingToPickup => pickupDock,
                    AGVState.MovingToDropoff => dropoffDock,
                    AGVState.ReturningToParking => parkingDock,
                    _ => pickupDock
                };
                currentWaypoint = FlatY(finalDock.ApproachPosition);
            }
        }

        private void TryResumeFromWait()
        {
            if (Time.fixedTime < nextRetryTime) return;

            if (trafficMgr.TryReserve(pendingZoneId, AgvId))
            {
                TrafficZone zone = trafficMgr.GetZone(pendingZoneId);
                currentWaypoint = FlatY(zone.Centre);
                waitingForZone = false;
                pendingZoneId = -1;
            }
            else
            {
                nextRetryTime = Time.fixedTime + reservationRetryInterval;
            }
        }

        private void OnEnteredZone(int newZoneId)
        {
            if (previousZoneId >= 0 && previousZoneId != newZoneId)
                trafficMgr.Release(previousZoneId, AgvId);

            previousZoneId = currentZoneId;
            currentZoneId = newZoneId;
        }

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

        private void MoveToward(Vector3 target)
        {
            Vector3 dir = target - transform.position;
            dir.y = 0f;
            if (dir.sqrMagnitude < 0.001f) return;

            float angle = Vector3.Angle(transform.forward, dir);
            if (angle > pathTurnThreshold)
                RotateToward(dir.normalized);
            else
            {
                transform.rotation = Quaternion.LookRotation(dir.normalized, Vector3.up);
                transform.position = Vector3.MoveTowards(transform.position, target, moveSpeed * Time.fixedDeltaTime);
            }
        }

        private void RotateToward(Vector3 flatDir)
        {
            Quaternion goal = Quaternion.LookRotation(flatDir, Vector3.up);
            transform.rotation = Quaternion.RotateTowards(transform.rotation, goal, turnSpeed * Time.fixedDeltaTime);
        }

        private void AlignToDock(DockPoint dock)
        {
            Vector3 desired = dock.FacingDirection;
            desired.y = 0f;
            if (desired.sqrMagnitude > 0.001f)
                RotateToward(desired.normalized);
        }

        private bool IsFacingDock(DockPoint dock)
        {
            Vector3 desired = dock.FacingDirection;
            desired.y = 0f;
            if (desired.sqrMagnitude < 0.001f) return true;
            return Vector3.Angle(transform.forward, desired) <= alignmentThreshold;
        }

        private bool ReachedDock(DockPoint dock)
        {
            return FlatDistance(transform.position, dock.ApproachPosition) <= dockArrivalDist;
        }

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

        private (int zoneId, DockPoint dock) FindSpecialDock(int specialId)
        {
            foreach (TrafficZone zone in trafficMgr.Zones)
            {
                if (zone.DockPoints.TryGetValue(specialId, out DockPoint d))
                    return (zone.ZoneId, d);
            }
            return (-1, default);
        }

        private int FindZoneAtSelf()
        {
            TrafficZone z = trafficMgr.GetZoneAtPosition(transform.position);
            return z?.ZoneId ?? -1;
        }

        private static float FlatDistance(Vector3 a, Vector3 b)
        {
            float dx = a.x - b.x;
            float dz = a.z - b.z;
            return Mathf.Sqrt(dx * dx + dz * dz);
        }

        private Vector3 FlatY(Vector3 pos)
        {
            pos.y = groundOffset;
            return pos;
        }

        private void UpdateStatusLabel()
        {
            if (statusLabel == null) return;

            string jobStr = CurrentJobId >= 0 ? $"J{CurrentJobId}" : "-";
            string target = targetMachine != null ? $"M{targetMachine.MachineId}" : "belt";
            string waitStr = waitingForZone ? $" [BLOCKED z{pendingZoneId}]" : "";

            statusLabel.text = State switch
            {
                AGVState.Idle => $"AGV{AgvId} [Idle]\nwaiting",
                AGVState.MovingToPickup => $"AGV{AgvId} [Pickup]{waitStr}\n {jobStr}",
                AGVState.MovingToDropoff => $"AGV{AgvId} [Dropoff]{waitStr}\n{jobStr}  {target}",
                AGVState.ReturningToParking => $"AGV{AgvId} [Parking]{waitStr}\n",
                AGVState.MovingToPrePickup => atPickupDock
                                               ? $"AGV{AgvId} [PreWait]{waitStr}\nJ{PreDispatchedJobId}"
                                               : $"AGV{AgvId} [PreRoute]{waitStr}\nJ{PreDispatchedJobId}",
                _ => $"AGV{AgvId}"
            };

            statusLabel.color = State switch
            {
                AGVState.Idle => Color.white,
                AGVState.ReturningToParking => Color.cyan,
                AGVState.MovingToPrePickup => Color.green,
                _ when waitingForZone => Color.red,
                _ => Color.yellow
            };
        }

        private void OnDrawGizmosSelected()
        {
            if (currentRoute == null || currentRoute.Count == 0 || trafficMgr == null) return;

            Gizmos.color = State == AGVState.MovingToPickup
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