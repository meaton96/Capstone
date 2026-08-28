using UnityEngine;
using UnityEngine.AI;
using System.Collections.Generic;
using Assets.Scripts.Simulation.Machines;
using Assets.Scripts.Simulation.FactoryLayout;
using Assets.Scripts.Simulation.Logging;
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

        /// <summary>
        /// If a single zone reservation attempt keeps failing for longer than this (sim-seconds,
        /// same clock as SimTime — see Time.fixedTime), it's treated as a circular-wait deadlock
        /// (TrafficZoneManager.TryReserve has no backoff/priority/timeout of its own, so a true
        /// cycle never self-resolves) rather than ordinary congestion. Set well below
        /// FactoryOrchestrator.DEADLOCK_STALL_SECONDS (3000s) so per-AGV self-recovery gets many
        /// chances to break a cycle before the system-wide watchdog would otherwise kill the
        /// episode outright. See HandleZoneStall.
        /// </summary>
        [SerializeField] private float zoneStallTimeoutSeconds = 180f;

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

        /// <summary>
        /// Set when this AGV self-recovers from a suspected traffic-zone deadlock (see
        /// HandleZoneStall). Read by FlagHarvester.HarvestStalledAGVs, which owns returning
        /// StalledJobId's JobData to a re-dispatchable state — mirrors the PickedUpFlag/
        /// DeliveredFlag pattern so AGVController never touches JobData directly.
        /// </summary>
        public bool StalledFlag { get; private set; }
        public int StalledJobId { get; private set; } = -1;

        /// <summary>
        /// Transit duration (sim-seconds) of the most recently completed pickup→dropoff trip.
        /// Read by FlagHarvester when processing DeliveredFlag to stamp JobData.OperationTravelTimes.
        /// </summary>
        public float LastTripDuration { get; private set; }

        /// @brief Resets all completion flags and delivery metadata.
        public void ClearFlags()
        {
            PickedUpFlag = false;
            DeliveredFlag = false;
            DeliveredJobId = -1;
            DeliveredMachineId = -1;
            StalledFlag = false;
            StalledJobId = -1;
        }
        /// <summary>
        /// Zeros all per-episode statistics. Called from Initialize() and from
        /// FactoryOrchestrator.StartEpisode() when the factory is reused.
        /// </summary>
        public void ResetEpisodeStats()
        {
            _statTimeIdle = 0.0;
            _statTimeWaitingRoute = 0.0;
            _statTimeTraveling = 0.0;
            _statTimeLoading = 0.0;
            _statTimeUnloading = 0.0;
            _statTotalPathLength = 0.0;
            _statRerouteCount = 0;
            _statStallRecoveryCount = 0;
            _statTotalTrips = 0;
            _statTripAccumulator = 0.0;
            _statCurrentTripStart = 0.0;
            _blockStartTime = -1f;
        }

        /// <summary>
        /// Builds an AGVRecord snapshot from accumulated stats at episode end.
        /// Pass the episode makespan so derived fractions are well-defined.
        /// </summary>
        public Assets.Scripts.Simulation.Types.AGVRecord GetRecord(double makespan)
        {
            double meanTrip = _statTotalTrips > 0
                ? _statTripAccumulator / _statTotalTrips
                : 0.0;

            return new Assets.Scripts.Simulation.Types.AGVRecord
            {
                AgvId = AgvId,
                TotalTrips = _statTotalTrips,
                MeanTripDuration = meanTrip,
                TimeIdle = _statTimeIdle,
                TimeWaitingRoute = _statTimeWaitingRoute,
                TimeTraveling = _statTimeTraveling,
                TimeLoading = _statTimeLoading,
                TimeUnloading = _statTimeUnloading,
                TotalPathLength = _statTotalPathLength,
                RerouteCount = _statRerouteCount,
                StallRecoveryCount = _statStallRecoveryCount,
            };
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

        // ── Per-episode statistics ────────────────────────────────────────────────
        // Accumulated in FixedUpdate; reset by ResetEpisodeStats() each episode.
        // Collected by FactoryOrchestrator.FinaliseEpisode() via GetRecord().

        private double _statTimeIdle;
        private double _statTimeWaitingRoute;  // blocked waiting for zone clearance
        private double _statTimeTraveling;
        private double _statTimeLoading;       // handshake timer at pickup dock
        private double _statTimeUnloading;     // handshake timer at dropoff dock
        private double _statTotalPathLength;   // cumulative NavMesh distance
        private int _statRerouteCount;      // RedirectDropoff calls
        private int _statStallRecoveryCount; // HandleZoneStall calls (suspected deadlock self-recoveries)
        private int _statTotalTrips;        // complete pickup→dropoff cycles
        private double _statTripAccumulator;   // sum of completed trip durations
        private double _statCurrentTripStart;  // fixedTime when current trip began (set in DoPickup)
        private float _blockStartTime = -1f;  // fixedTime when zone blocking began


        /// @brief Sets up the AGV identity and initializes navigation components.
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
            ResetEpisodeStats();
        }
        /// @brief Aborts an in-progress dropoff when no redirect target is available.
        /// Detaches the job visual at the AGV's current position and returns to Idle
        /// so the job can be re-dispatched by the scheduler.
        public void AbortTransit()
        {
            if (State != AGVState.MovingToDropoff) return;

            if (loadedJobVisual != null)
                loadedJobVisual.DetachFromCarrier(transform.position);

            SimLogger.Low($"[AGV {AgvId}] AbortTransit — job {CurrentJobId} released at {transform.position}.");

            CurrentJobId = -1;
            loadedJobVisual = null;
            sourceMachine = null;
            targetMachine = null;

            CancelCurrentRoute();
            State = AGVState.ReturningToParking;
            BeginParkingRoute();
        }
        /// @brief Cancels an in-progress pickup route when the destination machine fails
        /// before the job has been loaded. AGV releases its zone reservations and
        /// returns to parking. The job will be re-dispatched by the scheduler.
        public void CancelPickup()
        {
            if (State != AGVState.MovingToPickup)
            {
                SimLogger.Error($"[AGV {AgvId}] CancelPickup called in wrong state ({State}).");
                return;
            }

            SimLogger.Low($"[AGV {AgvId}] CancelPickup — abandoning pickup for job {CurrentJobId}.");

            CurrentJobId = -1;
            loadedJobVisual = null;
            sourceMachine = null;
            targetMachine = null;

            CancelCurrentRoute();
            State = AGVState.ReturningToParking;
            BeginParkingRoute();
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
        /// @brief Redirects an AGV that is already carrying a job to a new dropoff machine.
        /// Called when the original destination machine fails mid-transit.
        /// Safe to call from MovingToDropoff state only.
        public void RedirectDropoff(Vector3 newDropoffPos, PhysicalMachine newTarget, JobVisual visual)
        {
            if (State != AGVState.MovingToDropoff)
            {
                SimLogger.Error($"[AGV {AgvId}] RedirectDropoff called in wrong state ({State}).");
                return;
            }

            // Cancel the current route and any pending zone reservation
            CancelCurrentRoute();
            _statRerouteCount++;

            targetMachine = newTarget;
            targetDropoffPos = newDropoffPos;
            loadedJobVisual = visual ?? loadedJobVisual;

            atDropoffDock = false;
            dropoffTimer = handshakeDuration;

            (dropoffZoneId, dropoffDock) = newTarget != null
                ? FindDockForMachine(newTarget.MachineId, currentZoneId, newDropoffPos)
                : FindSpecialDock(TrafficZoneManager.OutgoingBeltId);

            if (dropoffZoneId < 0 || !PlanRoute(currentZoneId, dropoffZoneId))
            {
                SimLogger.Error($"[AGV {AgvId}] RedirectDropoff: no route to new target for job {CurrentJobId}. FullReset.");
                FullReset();
                return;
            }

            BeginNextWaypoint();
            SimLogger.High($"[AGV {AgvId}] Redirected job {CurrentJobId} dropoff → machine {newTarget?.MachineId ?? -1}.");
        }

        /// @brief Commands the AGV to move to a pickup zone before a job is finished.
        public void PreDispatch(int jobId, Vector3 pickupPos, PhysicalMachine source)
        {
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

        /// @brief Primary state machine loop executing logic based on @c AGVState.
        private void FixedUpdate()
        {
            float dt = Time.fixedDeltaTime;

            switch (State)
            {
                case AGVState.Idle:
                    _statTimeIdle += dt;    // ← NEW
                    break;

                case AGVState.MovingToPickup:
                    // Time accounting: waiting-for-zone takes priority, then loading, then traveling
                    if (waitingForZone)
                        _statTimeWaitingRoute += dt;    // ← NEW
                    else if (atPickupDock && pickupTimer > 0f)
                        _statTimeLoading += dt;          // ← NEW
                    else
                        _statTimeTraveling += dt;        // ← NEW

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
                    if (waitingForZone)
                        _statTimeWaitingRoute += dt;    // ← NEW
                    else if (atDropoffDock && dropoffTimer > 0f)
                        _statTimeUnloading += dt;        // ← NEW
                    else
                        _statTimeTraveling += dt;        // ← NEW

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
                    if (waitingForZone)
                        _statTimeWaitingRoute += dt;    // ← NEW
                    else
                        _statTimeTraveling += dt;        // ← NEW

                    UpdateMovement();
                    if (!waitingForZone && ReachedParking()) ArriveAtParking();
                    break;

                case AGVState.MovingToPrePickup:
                    if (waitingForZone)
                        _statTimeWaitingRoute += dt;     // ← NEW
                    else if (atPickupDock)
                        _statTimeIdle += dt;             // ← NEW — waiting for job to finish
                    else
                        _statTimeTraveling += dt;        // ← NEW

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


        /// @brief Executes physical loading from a machine or belt and initiates movement to dropoff.
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
            _statTotalTrips++;
            _statCurrentTripStart = Time.fixedTime;     // ← NEW

            State = AGVState.MovingToDropoff;
            atPickupDock = false;
            dropoffTimer = handshakeDuration;
            BeginNextWaypoint();
        }

        /// @brief Physical unloading logic that updates delivery flags and routes the AGV to parking.
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
            if (_statCurrentTripStart > 0)
            {
                LastTripDuration = (float)(Time.fixedTime - _statCurrentTripStart);
                _statTripAccumulator += LastTripDuration;
                _statCurrentTripStart = 0;
            }


            State = AGVState.ReturningToParking;
            BeginParkingRoute();
        }

        /// @brief Determines the return path to the assigned @c AGVPool parking station.
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

        /// @brief Checks if the AGV has physically arrived at its parking coordinate.
        private bool ReachedParking()
        {
            Vector3 parkPos = AGVPool.Instance.GetParkingPosition(AgvId);
            return FlatDistance(transform.position, parkPos) <= waypointArrivalDist;
        }

        /// @brief Cleanup function to release final zone reservations and return to @c Idle state.
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

        /// @brief Emergency recovery method to clear active job data and attempt a return to home.
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

        /// @brief Handles incremental movement toward the active waypoint or traffic zone center.
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

        /// @brief Logic for attempting to reserve and navigate into the next @c TrafficZone in the route.
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
                    _blockStartTime = Time.fixedTime;
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

        /// @brief Polls the @c TrafficZoneManager for a previously blocked zone reservation.
        private void TryResumeFromWait()
        {
            if (Time.fixedTime < nextRetryTime) return;

            if (trafficMgr.TryReserve(pendingZoneId, AgvId))
            {
                // Report block duration to zone manager for congestion logging  ← NEW
                if (_blockStartTime >= 0f)
                {
                    trafficMgr.RecordBlockTime(pendingZoneId, Time.fixedTime - _blockStartTime);
                    _blockStartTime = -1f;
                }

                TrafficZone zone = trafficMgr.GetZone(pendingZoneId);
                currentWaypoint = FlatY(zone.Centre);
                waitingForZone = false;
                pendingZoneId = -1;
            }

            else
            {
                nextRetryTime = Time.fixedTime + reservationRetryInterval;
                if (_blockStartTime >= 0f && Time.fixedTime - _blockStartTime > zoneStallTimeoutSeconds)
                    HandleZoneStall();
            }
        }

        /// @brief Recovery for an AGV that has failed to acquire the same zone for longer than
        /// zoneStallTimeoutSeconds — TrafficZoneManager.TryReserve has no backoff, priority, or
        /// timeout, so a genuine circular-wait deadlock never self-resolves; this is the
        /// signature of one. Releases any job in progress for redispatch (StalledFlag, so
        /// JobData only ever changes through FlagHarvester, never directly from here), then
        /// hands off to RetreatFromStall for the physical recovery.
        ///
        /// NOTE: this does NOT call CancelPickup/AbortTransit — those clean up job state but
        /// then immediately call BeginParkingRoute() from the SAME currentZoneId this AGV is
        /// stuck in, which just re-enters the same jam wanting a different destination. A
        /// stalled AGV was blocked trying to reserve the NEXT zone; it never lost its CURRENT
        /// one, so re-planning alone changes nothing. RetreatFromStall is what actually frees
        /// the zone.
        private void HandleZoneStall()
        {
            SimLogger.Error($"[AGV {AgvId}] Zone {pendingZoneId} reservation stalled past " +
                             $"{zoneStallTimeoutSeconds:F0}s (state={State}) — likely circular-wait " +
                             $"deadlock. Releasing job and retreating.");
            _blockStartTime = -1f;
            _statStallRecoveryCount++;

            switch (State)
            {
                case AGVState.MovingToPickup:
                    // Not yet attached to the carrier — nothing to detach physically.
                    StalledFlag = true;
                    StalledJobId = CurrentJobId;
                    CurrentJobId = -1;
                    loadedJobVisual = null;
                    sourceMachine = null;
                    targetMachine = null;
                    break;

                case AGVState.MovingToDropoff:
                    if (loadedJobVisual != null)
                        loadedJobVisual.DetachFromCarrier(transform.position);
                    StalledFlag = true;
                    StalledJobId = CurrentJobId;
                    CurrentJobId = -1;
                    loadedJobVisual = null;
                    sourceMachine = null;
                    targetMachine = null;
                    break;

                case AGVState.MovingToPrePickup:
                    // Job hasn't been picked up yet — it's still Processing at its source
                    // machine, untouched by this abandonment. Only the pre-dispatch claim
                    // needs releasing so a fresh AGV can be pre-dispatched normally later.
                    StalledFlag = true;
                    StalledJobId = PreDispatchedJobId;
                    PreDispatchedJobId = -1;
                    break;

                case AGVState.ReturningToParking:
                    break;  // no job at stake — just needs to physically get unstuck
            }

            RetreatFromStall();
        }

        /// @brief Physically frees the zone this AGV currently occupies. Backs it up into the
        /// zone it just came from if that's free — the only thing that actually removes it
        /// from a wait-for cycle, since it still holds its current reservation throughout the
        /// whole stall. If the previous zone is ALSO occupied (a bumper-to-bumper queue with no
        /// free slot in any direction), force-releases every reservation this AGV holds and
        /// snaps it directly to parking as a last resort — accepting a one-time fidelity
        /// compromise to guarantee the deadlock actually breaks rather than just relocating.
        private void RetreatFromStall()
        {
            currentRoute.Clear();
            routeIndex = 0;
            waitingForZone = false;
            pendingZoneId = -1;
            parkingZoneId = -1;

            if (previousZoneId >= 0 && previousZoneId != currentZoneId &&
                trafficMgr.TryReserve(previousZoneId, AgvId))
            {
                trafficMgr.Release(currentZoneId, AgvId);
                currentZoneId = previousZoneId;
                previousZoneId = -1;
                currentWaypoint = FlatY(trafficMgr.GetZone(currentZoneId).Centre);

                State = AGVState.ReturningToParking;
                BeginParkingRoute();
                return;
            }

            SimLogger.Error($"[AGV {AgvId}] Cannot retreat — previous zone also occupied. " +
                             $"Forcing release and snapping to parking.");
            trafficMgr.ReleaseAll(AgvId);
            currentZoneId = -1;
            previousZoneId = -1;
            Vector3 parkPos = AGVPool.Instance.GetParkingPosition(AgvId);
            Vector3 pos = parkPos; pos.y = groundOffset;
            transform.position = pos;
            ArriveAtParking();
        }

        /// @brief Manages the transition of reservations when crossing zone boundaries.
        private void OnEnteredZone(int newZoneId)
        {
            if (previousZoneId >= 0 && previousZoneId != newZoneId)
                trafficMgr.Release(previousZoneId, AgvId);

            previousZoneId = currentZoneId;
            currentZoneId = newZoneId;
        }

        /// @brief Requests a list of zone IDs from the @c TrafficZoneManager to form a navigation path.
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

        /// @brief Translates and rotates the AGV toward a world-space coordinate.
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
                float step = moveSpeed * Time.fixedDeltaTime;
                transform.position = Vector3.MoveTowards(transform.position, target, step);
                _statTotalPathLength += step;    // ← NEW
            }

        }

        /// @brief Helper to rotate the AGV to face a specific direction vector.
        private void RotateToward(Vector3 flatDir)
        {
            Quaternion goal = Quaternion.LookRotation(flatDir, Vector3.up);
            transform.rotation = Quaternion.RotateTowards(transform.rotation, goal, turnSpeed * Time.fixedDeltaTime);
        }

        /// @brief Aligns the AGV's forward vector to the required @c FacingDirection of a dock.
        private void AlignToDock(DockPoint dock)
        {
            Vector3 desired = dock.FacingDirection;
            desired.y = 0f;
            if (desired.sqrMagnitude > 0.001f)
                RotateToward(desired.normalized);
        }

        /// @brief Checks if the AGV's current orientation is within the @c alignmentThreshold.
        private bool IsFacingDock(DockPoint dock)
        {
            Vector3 desired = dock.FacingDirection;
            desired.y = 0f;
            if (desired.sqrMagnitude < 0.001f) return true;
            return Vector3.Angle(transform.forward, desired) <= alignmentThreshold;
        }

        /// @brief Utility to check if the AGV is within arrival distance of a dock's approach point.
        private bool ReachedDock(DockPoint dock)
        {
            return FlatDistance(transform.position, dock.ApproachPosition) <= dockArrivalDist;
        }

        /// @brief Searches available traffic zones to find the optimal dock for a specific machine ID.
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

        /// @brief Finds specific static docks (e.g., incoming/outgoing factory belts).
        private (int zoneId, DockPoint dock) FindSpecialDock(int specialId)
        {
            foreach (TrafficZone zone in trafficMgr.Zones)
            {
                if (zone.DockPoints.TryGetValue(specialId, out DockPoint d))
                    return (zone.ZoneId, d);
            }
            return (-1, default);
        }

        /// @brief Queries the @c TrafficZoneManager for the zone ID at the AGV's current position.
        private int FindZoneAtSelf()
        {
            TrafficZone z = trafficMgr.GetZoneAtPosition(transform.position);
            return z?.ZoneId ?? -1;
        }

        /// @brief Calculates the Euclidean distance between two points on the XZ plane.
        private static float FlatDistance(Vector3 a, Vector3 b)
        {
            float dx = a.x - b.x;
            float dz = a.z - b.z;
            return Mathf.Sqrt(dx * dx + dz * dz);
        }

        /// @brief Returns a copy of a position vector snapped to the AGV's @c groundOffset.
        private Vector3 FlatY(Vector3 pos)
        {
            pos.y = groundOffset;
            return pos;
        }

        /// @brief Updates the TextMeshPro debug overlay with state and navigation info.
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

        /// @brief Visualizes the planned path and current target in the Unity Editor.
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