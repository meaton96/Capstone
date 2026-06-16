using System.Collections.Generic;
using UnityEngine;
using Assets.Scripts.Simulation.FactoryLayout;
using Assets.Scripts.Simulation.Logging;
using Assets.Scripts.Simulation.Types;
using Assets.Scripts.Simulation.Machines;

namespace Assets.Scripts.Simulation.AGV
{
    /// @brief Manages the lifecycle and retrieval of the AGV fleet.
    ///
    /// @details Serves as a centralized container and factory. The @c SimulationBridge
    /// orchestrator queries this pool to identify and dispatch available units based
    /// on their current operational state.
    public class AGVPool : MonoBehaviour
    {
        public static AGVPool Instance;
        private Vector3[] parkingPositions;
        [SerializeField] private AGVController agvPrefab;
        [SerializeField] private FactoryLayoutManager layoutManager;

        private List<AGVController> fleet = new List<AGVController>();

        public IReadOnlyList<AGVController> AllAGVs => fleet;
        [SerializeField] private float parkingSlotSpacing = 2f;
        private void Awake()
        {
            Instance = this;
        }

        /// @brief Destroys the existing fleet and instantiates new AGV units.
        ///
        /// @param fleetSize The number of AGV units to spawn, driven by @c FJSSPConfig.AGVCount.
        ///
        /// @details Calculates parking positions based on the @c layoutManager coordinates
        /// and spawns units in a linear arrangement. Each unit is initialized with
        /// a unique ID corresponding to its index in the fleet.
        public void InitializeFleet(FJSSPConfig config)
        {
            int fleetSize = config.AGVCount;
            foreach (var agv in fleet) Destroy(agv.gameObject);
            fleet.Clear();

            parkingPositions = new Vector3[fleetSize];

            // "multiple" needs real per-aisle alcoves; the south-fallback (RowAisleIndex < 0)
            // is treated as single so AGVs centre on the one pool.
            bool multiple = layoutManager != null
                && layoutManager.ActiveParkingMethod == ParkingMethod.Multiple
                && layoutManager.ParkingAreas != null
                && layoutManager.ParkingAreas.Count > 0
                && layoutManager.ParkingAreas[0].RowAisleIndex >= 0;

            if (multiple)
                AssignMultipleParkingPositions(fleetSize);
            else
                AssignSingleParkingPositions(fleetSize);

            for (int i = 0; i < fleetSize; i++)
            {
                AGVController newAgv = Instantiate(agvPrefab, parkingPositions[i], Quaternion.identity, this.transform);
                newAgv.gameObject.name = $"AGV_{i}";
                newAgv.Initialize(i);
                fleet.Add(newAgv);
            }

            SimLogger.Medium($"[AGVPool] Spawned fleet of {fleetSize} AGVs ({(multiple ? "multiple" : "single")} parking).");
        }
        /// @brief Original behaviour: AGVs line up along X, centred on the single parking pool.
        private void AssignSingleParkingPositions(int fleetSize)
        {
            Vector3 baseParkingPos = layoutManager != null ? layoutManager.AGVParkingPosition : Vector3.zero;
            float span = (fleetSize - 1) * parkingSlotSpacing;
            Vector3 startPos = baseParkingPos - new Vector3(span / 2f, 0f, 0f);

            for (int i = 0; i < fleetSize; i++)
                parkingPositions[i] = startPos + new Vector3(i * parkingSlotSpacing, 0f, 0f);
        }

        /// @brief Spreads AGVs across the per-aisle alcoves round-robin (AGV i -> zone i % numZones).
        ///        Slot 0 sits at the alcove centre (the dock); additional AGVs in the same alcove
        ///        line up toward the outside (away from floor centre), keeping the entry side clear.
        ///        Outward spacing is compressed if needed so every slot stays inside its zone box.
        private void AssignMultipleParkingPositions(int fleetSize)
        {
            var areas = layoutManager.ParkingAreas;
            int numZones = areas.Count;

            // How many AGVs land in each zone, so we can size the lineup per zone.
            int[] countPerZone = new int[numZones];
            for (int i = 0; i < fleetSize; i++)
                countPerZone[i % numZones]++;

            int[] slotInZone = new int[numZones];

            float margin = 0.5f;  // keep AGVs off the zone-box edge so GetZoneAtPosition still resolves
            float usableOutward = Mathf.Max(0f, layoutManager.ParkingAlcoveDepth / 2f - margin);

            for (int i = 0; i < fleetSize; i++)
            {
                int z = i % numZones;
                ParkingArea area = areas[z];
                int slot = slotInZone[z]++;
                int count = countPerZone[z];

                // Left alcove extends in -X, right alcove in +X (both away from floor centre).
                Vector3 outward = area.IsLeftSide ? Vector3.left : Vector3.right;

                float spacing = parkingSlotSpacing;
                if (count > 1 && (count - 1) * spacing > usableOutward)
                    spacing = usableOutward / (count - 1);   // overflow won't fit at full spacing -> compress

                Vector3 pos = area.Position + outward * (slot * spacing);
                pos.y = area.Position.y;
                parkingPositions[i] = pos;

                SimLogger.Low($"[AGVPool] AGV {i} -> alcove {z} (aisle {area.RowAisleIndex}, " +
                            $"{(area.IsLeftSide ? "L" : "R")}), slot {slot}/{count - 1}.");
            }
        }

        /// @brief Destroys all AGVs in the fleet and clears the pool.
        public void ClearFleet()
        {
            foreach (var agv in fleet)
            {
                if (agv != null)
                    Destroy(agv.gameObject);
            }
            fleet.Clear();
        }

        /// @brief Retrieves the designated world-space parking coordinate for a specific AGV.
        ///
        /// @param agvId The unique identifier of the AGV.
        /// @return The @c Vector3 position for the unit's parking station, or @c Vector3.zero if invalid.
        public Vector3 GetParkingPosition(int agvId)
        {
            if (parkingPositions != null && agvId < parkingPositions.Length)
                return parkingPositions[agvId];
            return Vector3.zero;
        }

        /// @brief Locates the first unit that is currently in a strictly @c Idle state.
        ///
        /// @return An @c AGVController instance if an idle unit exists; otherwise, @c null.
        public AGVController GetIdleAGV()
        {
            foreach (var agv in fleet)
                if (agv.IsIdle) return agv;
            return null;
        }

        /// @brief Identifies the best candidate for a new task dispatch.
        ///
        /// @details Performs a two-pass search: first for units already at their
        /// parking stations (@c Idle), and second for units currently @c ReturningToParking
        /// that can be redirected mid-route to optimize travel time.
        public AGVController GetAvailableAGV()
        {
            foreach (var agv in fleet)
                if (agv.IsIdle) return agv;

            foreach (var agv in fleet)
                if (agv.State == AGVState.ReturningToParking) return agv;

            return null;
        }
        // @brief Selects the available AGV with the fewest one-way zone hops to the pickup.
        /// @details Prefers Idle units; falls back to the nearest ReturningToParking unit
        ///          (redirectable mid-route) when none are idle. Uses zone-graph hop count
        ///          rather than Euclidean distance — one-way aisles and aisle-exit parking
        ///          make straight-line distance a poor proxy for actual travel cost.
        /// @param pickupMachine The source machine (null = incoming belt).
        /// @param pickupPos     Reserved for future tie-breaking; not used for routing.
        public AGVController GetNearestAvailableAGV(PhysicalMachine pickupMachine, Vector3 pickupPos)
        {
            if (TrafficZoneManager.Instance == null) return GetAvailableAGV();   // no zone graph → legacy behaviour

            // A machine can dock from more than one aisle; seed the BFS with all its zones.
            List<int> pickupZones;
            if (pickupMachine != null)
                pickupZones = TrafficZoneManager.Instance.GetZonesForMachine(pickupMachine.MachineId);
            else
            {
                int beltZone = TrafficZoneManager.Instance.GetZoneIdForDock(TrafficZoneManager.IncomingBeltId);
                pickupZones = beltZone >= 0 ? new List<int> { beltZone } : new List<int>();
            }
            if (pickupZones == null || pickupZones.Count == 0) return GetAvailableAGV();

            Dictionary<int, int> hops = TrafficZoneManager.Instance.GetHopDistancesToNearest(pickupZones);

            AGVController bestIdle = null; int bestIdleDist = int.MaxValue;
            AGVController bestReturn = null; int bestReturnDist = int.MaxValue;

            foreach (var agv in fleet)
            {
                bool idle = agv.IsIdle;
                bool returning = agv.State == AGVState.ReturningToParking;
                if (!idle && !returning) continue;

                int zone = ResolveZone(agv);
                int d = (zone >= 0 && hops.TryGetValue(zone, out int hop)) ? hop : int.MaxValue;

                if (idle)
                {
                    if (bestIdle == null || d < bestIdleDist) { bestIdle = agv; bestIdleDist = d; }
                }
                else
                {
                    if (bestReturn == null || d < bestReturnDist) { bestReturn = agv; bestReturnDist = d; }
                }
            }

            return bestIdle ?? bestReturn;
        }

        /// @brief Resolves an AGV's current zone, falling back to a spatial lookup for idle
        ///        units, which release their zone reservation on arrival at parking.
        private int ResolveZone(AGVController agv)
        {
            if (agv.CurrentZoneId >= 0) return agv.CurrentZoneId;
            TrafficZone z = TrafficZoneManager.Instance.GetZoneAtPosition(agv.transform.position);
            return z?.ZoneId ?? -1;
        }

        /// @brief Returns the AGV unit assigned to a specific job ID via pre-dispatch.
        ///
        /// @param jobId The identifier of the job to check.
        /// @return The assigned @c AGVController if found; otherwise, @c null.
        public AGVController GetPreDispatchedAGV(int jobId)
        {
            foreach (var agv in fleet)
                if (agv.IsPreDispatched && agv.PreDispatchedJobId == jobId)
                    return agv;
            return null;
        }
    }
}