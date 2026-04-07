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
    public enum AGVState
    {
        Idle,
        MovingToPickup,
        MovingToDropoff,
        ReturningToParking,
    }

    /// @brief Controls navigation, job assignment, and lifecycle of a single AGV.
    ///
    /// @details Implements a 4-state machine: Idle → MovingToPickup → MovingToDropoff
    /// → ReturningToParking → Idle. All state transitions happen exclusively inside
    /// FixedUpdate(). Worker methods (DoPickup, DoDropoff) never touch State directly.
    ///
    /// Zone reservation is handled as a blocking-flag on the current movement state,
    /// not as a separate state, eliminating the re-entrancy bugs that arise from
    /// WaitingForZone interrupting pickup/dropoff execution.
    [RequireComponent(typeof(NavMeshAgent))]
    public class AGVController : MonoBehaviour
    {
        // ─────────────────────────────────────────────────────────
        //  Inspector
        // ─────────────────────────────────────────────────────────

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

        // ─────────────────────────────────────────────────────────
        //  Public read-only state
        // ─────────────────────────────────────────────────────────

        public int AgvId { get; private set; }
        public AGVState State { get; private set; } = AGVState.Idle;
        public int CurrentJobId { get; private set; } = -1;
        public int CurrentZoneId => currentZoneId;

        // ─────────────────────────────────────────────────────────
        //  Private fields
        // ─────────────────────────────────────────────────────────

        private NavMeshAgent agent;
        private TrafficZoneManager trafficMgr;
        private System.Action onBecameIdle;

        // Job context — set once per Dispatch(), cleared in FullReset()
        private PhysicalMachine sourceMachine;
        private PhysicalMachine targetMachine;
        private Vector3 targetPickupPos;
        private Vector3 targetDropoffPos;
        private JobVisual loadedJobVisual;

        // Dock / zone targets
        private int pickupZoneId = -1;
        private int dropoffZoneId = -1;
        private DockPoint pickupDock;
        private DockPoint dropoffDock;
        private DockPoint parkingDock;
        private int parkingZoneId = -1;

        // Route
        private readonly List<int> currentRoute = new List<int>();
        private int routeIndex;
        private Vector3 currentWaypoint;

        // Zone tracking
        private int currentZoneId = -1;
        private int previousZoneId = -1;

        // Zone-wait (flag, not a state)
        private bool waitingForZone;
        private int pendingZoneId = -1;
        private float nextRetryTime;

        // Handshake timers — counted down inside FixedUpdate once at dock
        private float pickupTimer;
        private float dropoffTimer;
        private bool atPickupDock;
        private bool atDropoffDock;

        // ─────────────────────────────────────────────────────────
        //  Initialisation
        // ─────────────────────────────────────────────────────────

        /// @brief Registers the idle callback used by AGVPool to drain the request queue.
        public void SetIdleCallback(System.Action callback) => onBecameIdle = callback;

        /// @brief Initialises the AGV, anchors it to the NavMesh, and reserves its
        ///        starting zone.
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

        // ─────────────────────────────────────────────────────────
        //  Dispatch (called by AGVPool)
        // ─────────────────────────────────────────────────────────

        /// @brief Assigns a pickup-and-deliver task to this AGV.
        ///
        /// @pre  AGV must be in Idle state.
        /// @post State becomes MovingToPickup if a route exists; otherwise stays Idle.
        public void Dispatch(int jobId, Vector3 pickupPosition, Vector3 dropoffPosition,
                             PhysicalMachine source, PhysicalMachine dropoff)
        {
            if (State != AGVState.Idle)
            {
                SimLogger.Error($"[AGV {AgvId}] Dispatch called while not Idle (state={State}). Ignoring.");
                return;
            }

            CurrentJobId = jobId;
            sourceMachine = source;
            targetMachine = dropoff;
            targetPickupPos = pickupPosition;
            targetDropoffPos = dropoffPosition;

            loadedJobVisual = null;
            atPickupDock = false;
            atDropoffDock = false;
            pickupTimer = handshakeDuration;
            dropoffTimer = handshakeDuration;
            waitingForZone = false;
            pendingZoneId = -1;

            // Ensure we have a valid zone
            if (currentZoneId < 0)
            {
                currentZoneId = FindZoneAtSelf();
                if (currentZoneId >= 0)
                    trafficMgr.TryReserve(currentZoneId, AgvId);
            }

            // Resolve pickup dock
            if (source != null)
                (pickupZoneId, pickupDock) = FindDockForMachine(source.MachineId, currentZoneId, targetPickupPos);
            else
                (pickupZoneId, pickupDock) = FindSpecialDock(TrafficZoneManager.IncomingBeltId);

            if (pickupZoneId < 0)
            {
                SimLogger.Error($"[AGV {AgvId}] Could not resolve pickup dock for job {jobId}.");
                FullReset();
                return;
            }

            if (!PlanRoute(currentZoneId, pickupZoneId))
            {
                SimLogger.Error($"[AGV {AgvId}] No route to pickup zone {pickupZoneId} for job {jobId}.");
                FullReset();
                return;
            }

            State = AGVState.MovingToPickup;
            BeginNextWaypoint();
        }

        // ─────────────────────────────────────────────────────────
        //  FixedUpdate — THE only place State is written
        // ─────────────────────────────────────────────────────────

        private void FixedUpdate()
        {
            switch (State)
            {
                case AGVState.Idle:
                    break;

                case AGVState.MovingToPickup:
                    UpdateMovement();

                    if (!waitingForZone && ReachedDock(pickupDock))
                    {
                        if (!atPickupDock)
                        {
                            atPickupDock = true;
                            AlignToDock(pickupDock); // begin alignment on arrival frame
                        }

                        if (IsFacingDock(pickupDock))
                        {
                            pickupTimer -= Time.fixedDeltaTime;
                            if (pickupTimer <= 0f)
                            {
                                if (DoPickup())
                                {
                                    State = AGVState.MovingToDropoff;
                                    atPickupDock = false;
                                }
                                else
                                {
                                    SimLogger.Error($"[AGV {AgvId}] DoPickup failed for job {CurrentJobId}.");
                                    FullReset();
                                }
                            }
                        }
                        else
                        {
                            AlignToDock(pickupDock);
                        }
                    }
                    break;

                case AGVState.MovingToDropoff:
                    UpdateMovement();

                    if (!waitingForZone && ReachedDock(dropoffDock))
                    {
                        if (!atDropoffDock)
                        {
                            atDropoffDock = true;
                            AlignToDock(dropoffDock);
                        }

                        if (IsFacingDock(dropoffDock))
                        {
                            dropoffTimer -= Time.fixedDeltaTime;
                            if (dropoffTimer <= 0f)
                            {
                                if (DoDropoff())
                                {
                                    State = AGVState.ReturningToParking;
                                    atDropoffDock = false;
                                    BeginParkingRoute();
                                }
                                else
                                {
                                    SimLogger.Error($"[AGV {AgvId}] DoDropoff failed for job {CurrentJobId}.");
                                    FullReset();
                                }
                            }
                        }
                        else
                        {
                            AlignToDock(dropoffDock);
                        }
                    }
                    break;

                case AGVState.ReturningToParking:
                    UpdateMovement();

                    if (!waitingForZone && ReachedParking())
                    {
                        ArriveAtParking();  // sets State = Idle, fires idle callback
                    }
                    break;
            }

            agent.nextPosition = transform.position;
            UpdateStatusLabel();
        }

        // ─────────────────────────────────────────────────────────
        //  Worker methods — never write State
        // ─────────────────────────────────────────────────────────

        /// @brief Picks up the job from its source. Returns true on success.
        private bool DoPickup()
        {
            if (CurrentJobId < 0)
            {
                SimLogger.Error($"[AGV {AgvId}] DoPickup: CurrentJobId is -1.");
                return false;
            }

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

            // Resolve dropoff dock now that we know current zone
            if (targetMachine != null)
                (dropoffZoneId, dropoffDock) = FindDockForMachine(targetMachine.MachineId, currentZoneId, targetDropoffPos);
            else
                (dropoffZoneId, dropoffDock) = FindSpecialDock(TrafficZoneManager.OutgoingBeltId);

            if (dropoffZoneId < 0)
            {
                SimLogger.Error($"[AGV {AgvId}] DoPickup: could not resolve dropoff dock.");
                return false;
            }

            if (!PlanRoute(currentZoneId, dropoffZoneId))
            {
                SimLogger.Error($"[AGV {AgvId}] DoPickup: no route to dropoff zone {dropoffZoneId}.");
                return false;
            }

            int nextMachineId = targetMachine != null ? targetMachine.MachineId : -1;
            SimulationBridge.Instance.JobManager.BeginTransit(CurrentJobId, nextMachineId, Time.time);
            SimLogger.High($"[AGV {AgvId}] Executed pickup of job {CurrentJobId}");

            dropoffTimer = handshakeDuration;
            BeginNextWaypoint();
            return true;
        }

        /// @brief Drops off the job at its destination. Returns true on success.
        private bool DoDropoff()
        {
            if (CurrentJobId < 0)
            {
                SimLogger.Error($"[AGV {AgvId}] DoDropoff: CurrentJobId is -1.");
                return false;
            }

            if (loadedJobVisual != null)
                loadedJobVisual.DetachFromCarrier(dropoffDock.HandshakePosition);

            if (targetMachine != null)
                targetMachine.ReceiveJob(CurrentJobId, loadedJobVisual);
            else
                FactoryLayoutManager.Instance.OutgoingBelt?.TryEnqueue(CurrentJobId, loadedJobVisual);

            int machineId = targetMachine != null ? targetMachine.MachineId : -1;
            SimulationBridge.Instance.JobManager.CompleteTransit(CurrentJobId, machineId, Time.time);
            SimLogger.High($"[AGV {AgvId}] Executed dropoff of job {CurrentJobId}");

            // Clear job state immediately so nothing can reuse it
            int completedJob = CurrentJobId;
            CurrentJobId = -1;
            loadedJobVisual = null;
            sourceMachine = null;
            targetMachine = null;
            pickupZoneId = -1;
            dropoffZoneId = -1;

            return true;
        }

        // ─────────────────────────────────────────────────────────
        //  Parking
        // ─────────────────────────────────────────────────────────

        private void BeginParkingRoute()
        {
            Vector3 parkPos = AGVPool.Instance.GetParkingPosition(AgvId);
            TrafficZone parkZone = trafficMgr.GetZoneAtPosition(parkPos);
            parkingZoneId = parkZone?.ZoneId ?? -1;

            if (parkZone != null)
            {
                // Build a synthetic dock at the parking position
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

            SimLogger.High($"[AGV {AgvId}] Returning to parking at {parkPos}.");
            BeginNextWaypoint();
        }

        private bool ReachedParking()
        {
            Vector3 parkPos = AGVPool.Instance.GetParkingPosition(AgvId);
            return FlatDistance(transform.position, parkPos) <= waypointArrivalDist;
        }

        /// @brief Called when the AGV physically reaches its parking spot.
        ///        Releases all zones, sets Idle, and fires the pool callback.
        private void ArriveAtParking()
        {
            if (previousZoneId >= 0)
            {
                trafficMgr.Release(previousZoneId, AgvId);
                previousZoneId = -1;
            }
            // Keep currentZoneId reserved so we have a valid starting zone for next Dispatch

            currentRoute.Clear();
            routeIndex = 0;
            waitingForZone = false;
            pendingZoneId = -1;
            parkingZoneId = -1;

            State = AGVState.Idle;
            SimLogger.High($"[AGV {AgvId}] Parked — zones released.");
            onBecameIdle?.Invoke();
        }

        // ─────────────────────────────────────────────────────────
        //  Navigation helpers
        // ─────────────────────────────────────────────────────────

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
                if (pastRoute) return; // at final dock, let FixedUpdate handle it

                OnEnteredZone(currentRoute[routeIndex]);
                routeIndex++;
                BeginNextWaypoint();
            }
            else
            {
                MoveToward(currentWaypoint);
            }
        }

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
                // Choose final dock based on current state
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

        // ─────────────────────────────────────────────────────────
        //  Movement primitives
        // ─────────────────────────────────────────────────────────

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

        // ─────────────────────────────────────────────────────────
        //  Dock resolution
        // ─────────────────────────────────────────────────────────

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

        // ─────────────────────────────────────────────────────────
        //  Full reset (error path only)
        // ─────────────────────────────────────────────────────────

        /// @brief Emergency reset used only when route planning fails.
        ///        The AGV returns to parking so it doesn't block the floor.
        private void FullReset()
        {
            SimLogger.Error($"[AGV {AgvId}] FullReset triggered — clearing job {CurrentJobId}.");

            CurrentJobId = -1;
            loadedJobVisual = null;
            sourceMachine = null;
            targetMachine = null;
            pickupZoneId = -1;
            dropoffZoneId = -1;
            atPickupDock = false;
            atDropoffDock = false;
            waitingForZone = false;
            pendingZoneId = -1;
            currentRoute.Clear();
            routeIndex = 0;

            if (previousZoneId >= 0)
            {
                trafficMgr.Release(previousZoneId, AgvId);
                previousZoneId = -1;
            }

            // Try to get home; if that fails too, just go idle in place
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

        // ─────────────────────────────────────────────────────────
        //  Utility
        // ─────────────────────────────────────────────────────────

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

        // ─────────────────────────────────────────────────────────
        //  HUD label
        // ─────────────────────────────────────────────────────────

        private void UpdateStatusLabel()
        {
            if (statusLabel == null) return;

            string jobStr = CurrentJobId >= 0 ? $"J{CurrentJobId}" : "–";
            string target = targetMachine != null ? $"M{targetMachine.MachineId}" : "belt";
            string waitStr = waitingForZone ? $" [BLOCKED z{pendingZoneId}]" : "";

            statusLabel.text = State switch
            {
                AGVState.Idle => $"AGV{AgvId} [Idle]\nwaiting",
                AGVState.MovingToPickup => $"AGV{AgvId} [Pickup]{waitStr}\n→ {jobStr}",
                AGVState.MovingToDropoff => $"AGV{AgvId} [Dropoff]{waitStr}\n{jobStr} → {target}",
                AGVState.ReturningToParking => $"AGV{AgvId} [Parking]{waitStr}\n↩",
                _ => $"AGV{AgvId}"
            };

            statusLabel.color = State switch
            {
                AGVState.Idle => Color.white,
                AGVState.ReturningToParking => Color.cyan,
                _ when waitingForZone => Color.red,
                _ => Color.yellow
            };
        }

        // ─────────────────────────────────────────────────────────
        //  Gizmos
        // ─────────────────────────────────────────────────────────

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