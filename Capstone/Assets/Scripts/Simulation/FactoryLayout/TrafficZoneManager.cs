using System;
using System.Collections.Generic;
using UnityEngine;
using Assets.Scripts.Logging;

namespace Assets.Scripts.Simulation.FactoryLayout
{
    // ─────────────────────────────────────────────────────────
    //  Enums & Data Structures
    // ─────────────────────────────────────────────────────────

    /// <summary>
    /// Categorises the aisle type a zone belongs to.
    /// </summary>
    public enum AisleType
    {
        /// <summary>Narrow aisle between two machine rows.</summary>
        RowAisle,
        /// <summary>Wide top or bottom peripheral spine aisle.</summary>
        SpineAisle,
        /// <summary>Left or right vertical connector aisle.</summary>
        VerticalAisle
    }

    /// <summary>
    /// One-way flow direction of a zone's parent aisle.
    /// </summary>
    public enum FlowDirection
    {
        East,       // +X
        West,       // -X
        North,      // +Z
        South       // -Z
    }

    /// <summary>
    /// A single reservable zone within the traffic network.
    /// Zones are the atomic units of space that AGVs must reserve
    /// before entering.
    /// </summary>
    [Serializable]
    public class TrafficZone
    {
        public int ZoneId;
        public string Name;
        public AisleType AisleType;
        public FlowDirection Flow;

        /// <summary>World-space centre of this zone.</summary>
        public Vector3 Centre;

        /// <summary>World-space bounds of this zone (width along aisle × depth).</summary>
        public Vector3 Size;

        /// <summary>Maximum AGVs allowed in this zone simultaneously.</summary>
        public int Capacity = 1;

        /// <summary>IDs of zones reachable from this zone following flow direction.</summary>
        public List<int> Downstream = new List<int>();

        /// <summary>IDs of zones that flow into this zone.</summary>
        public List<int> Upstream = new List<int>();

        /// <summary>
        /// Machine docking points accessible from this zone.
        /// Key = machineId, Value = (approachPos, dockPos, facingDir).
        /// </summary>
        public Dictionary<int, DockPoint> DockPoints = new Dictionary<int, DockPoint>();

        // ── Runtime ──
        /// <summary>Set of AGV IDs currently occupying or reserved into this zone.</summary>
        [NonSerialized] public HashSet<int> OccupantAgvIds = new HashSet<int>();

        public bool IsFull => OccupantAgvIds.Count >= Capacity;
        public bool IsEmpty => OccupantAgvIds.Count == 0;
    }

    /// <summary>
    /// Describes where an AGV must position itself to interact
    /// with a machine's conveyor from within an aisle zone.
    /// </summary>
    [Serializable]
    public struct DockPoint
    {
        /// <summary>Position where the AGV should navigate to before docking.</summary>
        public Vector3 ApproachPosition;

        /// <summary>Position the AGV's carry point must reach for handshake.</summary>
        public Vector3 HandshakePosition;

        /// <summary>Direction the AGV must face during docking (toward conveyor).</summary>
        public Vector3 FacingDirection;

        /// <summary>Whether this dock point is for pickup (outgoing) or dropoff (incoming).</summary>
        public bool IsPickup;
    }

    // ─────────────────────────────────────────────────────────
    //  Traffic Zone Manager
    // ─────────────────────────────────────────────────────────

    /// <summary>
    /// Manages the zone-based traffic control network for the factory floor.
    ///
    /// <para><b>Zone layout (matches FactoryLayoutManager):</b></para>
    /// The floor is divided into reservable zones. Each narrow row aisle
    /// is split into segments (one per machine column gap). Spine and
    /// vertical aisles are split at intersection points.
    ///
    /// <para><b>Reservation protocol:</b></para>
    /// 1. AGV requests its next zone via <see cref="TryReserve"/>.
    /// 2. If the zone has capacity, the AGV is added and may enter.
    /// 3. If full, the AGV must wait (the caller decides behaviour).
    /// 4. When the AGV leaves a zone, it calls <see cref="Release"/>.
    ///
    /// <para><b>One-way enforcement:</b></para>
    /// Zones expose <see cref="TrafficZone.Downstream"/> connections.
    /// The AGV pathfinder should only request zones along the downstream
    /// chain. The manager does NOT enforce direction — it trusts the
    /// pathfinder — but provides <see cref="GetRoute"/> for convenience.
    /// </summary>
    [RequireComponent(typeof(FactoryLayoutManager))]
    public class TrafficZoneManager : MonoBehaviour
    {
        // ─────────────────────────────────────────────────────────
        //  Inspector
        // ─────────────────────────────────────────────────────────

        private FactoryLayoutManager layoutManager;

        [Header("Debug")]
        [Tooltip("Draw zone bounds and occupancy in the Scene view.")]
        [SerializeField] private bool drawGizmos = true;

        [Tooltip("Draw zone labels in the Scene view.")]
        [SerializeField] private bool drawLabels = true;

        // ─────────────────────────────────────────────────────────
        //  Runtime State
        // ─────────────────────────────────────────────────────────

        private readonly List<TrafficZone> zones = new List<TrafficZone>();
        private readonly Dictionary<int, TrafficZone> zoneById = new Dictionary<int, TrafficZone>();
        private int nextZoneId;

        /// <summary>Maps machineId → list of zone IDs adjacent to that machine.</summary>
        private readonly Dictionary<int, List<int>> machineToZones = new Dictionary<int, List<int>>();

        // ─────────────────────────────────────────────────────────
        //  Public Accessors
        // ─────────────────────────────────────────────────────────

        public IReadOnlyList<TrafficZone> Zones => zones;

        public TrafficZone GetZone(int zoneId) =>
            zoneById.TryGetValue(zoneId, out var z) ? z : null;

        // ─────────────────────────────────────────────────────────
        //  Initialisation
        // ─────────────────────────────────────────────────────────

        /// <summary>
        /// Builds the full zone graph from the current factory layout.
        /// Call this after <see cref="FactoryLayoutManager.BuildFloor"/>.
        /// </summary>
        public void BuildZoneGraph()
        {
            zones.Clear();
            zoneById.Clear();
            machineToZones.Clear();
            nextZoneId = 0;

            if (layoutManager == null)
            {
                SimLogger.Error("[TrafficZones] No FactoryLayoutManager assigned.");
                return;
            }

            int rows = layoutManager.LayoutRows;
            int cols = layoutManager.LayoutCols;

            if (rows == 0 || cols == 0)
            {
                SimLogger.Error("[TrafficZones] Layout not built yet.");
                return;
            }

            // Phase 1: Create row aisle zones
            var rowAisleZones = BuildRowAisleZones(rows, cols);

            // Phase 2: Create spine aisle zones
            var topSpineZones = BuildSpineZones(true, cols);
            var botSpineZones = BuildSpineZones(false, cols);

            // Phase 3: Create vertical aisle zones
            var leftVertZones = BuildVerticalZones(true, rows);
            var rightVertZones = BuildVerticalZones(false, rows);

            // Phase 4: Connect zones into a directed graph
            ConnectZoneGraph(rowAisleZones, topSpineZones, botSpineZones,
                             leftVertZones, rightVertZones, rows, cols);

            // Phase 5: Register dock points for each machine
            RegisterDockPoints(rowAisleZones, topSpineZones, botSpineZones,
                               rows, cols);

            SimLogger.Medium($"[TrafficZones] Built zone graph: " +
                $"{zones.Count} zones, {rows - 1} row aisles, " +
                $"2 spines, 2 verticals.");
        }

        // ─────────────────────────────────────────────────────────
        //  Zone Construction
        // ─────────────────────────────────────────────────────────

        /// <summary>
        /// Creates zones for each row aisle, split into segments at
        /// each machine column boundary.
        /// Returns [aisleIndex][segmentIndex] = zoneId.
        /// </summary>
        private int[][] BuildRowAisleZones(int rows, int cols)
        {
            int numAisles = rows - 1;
            int[][] result = new int[numAisles][];

            for (int a = 0; a < numAisles; a++)
            {
                Vector3 aisleCentre = layoutManager.GetRowAisleCentre(a);
                Vector3 flowDir = layoutManager.GetRowAisleDirection(a);
                FlowDirection flow = flowDir.x > 0
                    ? FlowDirection.East : FlowDirection.West;

                int numSegments = cols; // one segment per machine column
                result[a] = new int[numSegments];

                float segWidth = layoutManager.MachineSpacingX;
                float halfTotalWidth = ((cols - 1) * segWidth) / 2f;

                for (int s = 0; s < numSegments; s++)
                {
                    float segCentreX = -halfTotalWidth + s * segWidth;

                    var zone = new TrafficZone
                    {
                        ZoneId = nextZoneId++,
                        Name = $"RowAisle{a}_Seg{s}",
                        AisleType = AisleType.RowAisle,
                        Flow = flow,
                        Centre = new Vector3(
                            aisleCentre.x + segCentreX,
                            aisleCentre.y,
                            aisleCentre.z),
                        Size = new Vector3(
                            segWidth,
                            0.1f,
                            layoutManager.RowAisleWidth),
                        Capacity = 1
                    };

                    RegisterZone(zone);
                    result[a][s] = zone.ZoneId;
                }
            }

            return result;
        }

        /// <summary>
        /// Creates zones for a spine aisle (top or bottom), split into
        /// segments aligned with machine columns.
        /// </summary>
        private int[] BuildSpineZones(bool isTop, int cols)
        {
            int numSegments = cols + 1; // extra segments at the ends for vertical connections
            int[] result = new int[numSegments];

            float z = isTop
                ? layoutManager.GetTopSpineZ()
                : layoutManager.GetBottomSpineZ();

            Vector3 floorCentre = layoutManager.transform.position;

            FlowDirection flow = isTop ? FlowDirection.East : FlowDirection.West;
            float segWidth = layoutManager.MachineSpacingX;
            float halfTotalWidth = ((cols - 1) * segWidth) / 2f;

            // Left end segment (in vertical aisle zone)
            float leftEdge = -halfTotalWidth - layoutManager.MachineDepth / 2f
                             - layoutManager.VerticalAisleWidth / 2f;

            for (int s = 0; s < numSegments; s++)
            {
                float segCentreX;
                float width;

                if (s == 0)
                {
                    // Left end
                    segCentreX = leftEdge;
                    width = layoutManager.VerticalAisleWidth;
                }
                else if (s == numSegments - 1)
                {
                    // Right end
                    segCentreX = halfTotalWidth + layoutManager.MachineDepth / 2f
                                 + layoutManager.VerticalAisleWidth / 2f;
                    width = layoutManager.VerticalAisleWidth;
                }
                else
                {
                    segCentreX = -halfTotalWidth + (s - 1) * segWidth;
                    width = segWidth;
                }

                var zone = new TrafficZone
                {
                    ZoneId = nextZoneId++,
                    Name = $"{(isTop ? "TopSpine" : "BotSpine")}_Seg{s}",
                    AisleType = AisleType.SpineAisle,
                    Flow = flow,
                    Centre = new Vector3(
                        floorCentre.x + segCentreX,
                        0.01f,
                        floorCentre.z + z),
                    Size = new Vector3(
                        width,
                        0.1f,
                        layoutManager.SpineAisleWidth),
                    Capacity = 2 // spine is wider, allow 2 AGVs
                };

                RegisterZone(zone);
                result[s] = zone.ZoneId;
            }

            return result;
        }

        void Awake()
        {
            layoutManager = GetComponent<FactoryLayoutManager>();
        }

        /// <summary>
        /// Creates zones for a vertical aisle (left or right), one per
        /// row aisle intersection.
        /// </summary>
        private int[] BuildVerticalZones(bool isLeft, int rows)
        {
            int numRowAisles = rows - 1;
            int numSegments = numRowAisles + 2; // +2 for spine connections
            int[] result = new int[numSegments];

            float halfMachineAreaW = ((layoutManager.LayoutCols - 1)
                                     * layoutManager.MachineSpacingX) / 2f
                                     + layoutManager.MachineDepth / 2f;

            float x = isLeft
                ? -(halfMachineAreaW + layoutManager.VerticalAisleWidth / 2f)
                : (halfMachineAreaW + layoutManager.VerticalAisleWidth / 2f);

            FlowDirection flow = isLeft ? FlowDirection.North : FlowDirection.South;

            Vector3 floorCentre = layoutManager.transform.position;

            for (int s = 0; s < numSegments; s++)
            {
                float z;
                float height;
                string name;

                if (s == 0)
                {
                    z = layoutManager.GetTopSpineZ();
                    height = layoutManager.SpineAisleWidth;
                    name = "TopConn";
                }
                else if (s == numSegments - 1)
                {
                    z = layoutManager.GetBottomSpineZ();
                    height = layoutManager.SpineAisleWidth;
                    name = "BotConn";
                }
                else
                {
                    Vector3 aisleCentre = layoutManager.GetRowAisleCentre(s - 1);
                    z = aisleCentre.z - floorCentre.z;
                    height = layoutManager.RowAisleWidth;
                    name = $"Row{s - 1}";
                }

                var zone = new TrafficZone
                {
                    ZoneId = nextZoneId++,
                    Name = $"{(isLeft ? "LeftVert" : "RightVert")}_{name}",
                    AisleType = AisleType.VerticalAisle,
                    Flow = flow,
                    Centre = new Vector3(
                        floorCentre.x + x,
                        0.01f,
                        floorCentre.z + z),
                    Size = new Vector3(
                        layoutManager.VerticalAisleWidth,
                        0.1f,
                        height),
                    Capacity = 1
                };

                RegisterZone(zone);
                result[s] = zone.ZoneId;
            }

            return result;
        }

        private void RegisterZone(TrafficZone zone)
        {
            zones.Add(zone);
            zoneById[zone.ZoneId] = zone;
        }

        // ─────────────────────────────────────────────────────────
        //  Graph Connectivity
        // ─────────────────────────────────────────────────────────

        /// <summary>
        /// Wires up the downstream/upstream links between all zones
        /// to form the one-way circulation loop.
        /// </summary>
        private void ConnectZoneGraph(
            int[][] rowAisles, int[] topSpine, int[] botSpine,
            int[] leftVert, int[] rightVert,
            int rows, int cols)
        {
            // Row aisle internal chains
            for (int a = 0; a < rowAisles.Length; a++)
            {
                bool eastbound = (a % 2 == 0);
                int[] segs = rowAisles[a];

                if (eastbound)
                {
                    for (int s = 0; s < segs.Length - 1; s++)
                        LinkZones(segs[s], segs[s + 1]);
                }
                else
                {
                    for (int s = segs.Length - 1; s > 0; s--)
                        LinkZones(segs[s], segs[s - 1]);
                }
            }

            // Spine internal chains
            // Top spine: east (0 -> Length)
            for (int s = 0; s < topSpine.Length - 1; s++)
                LinkZones(topSpine[s], topSpine[s + 1]);

            // Bottom spine: west (Length -> 0)
            for (int s = botSpine.Length - 1; s > 0; s--)
                LinkZones(botSpine[s], botSpine[s - 1]);

            // Vertical internal chains
            // Left: north (bottom -> top) -> (Length -> 0)
            for (int s = leftVert.Length - 1; s > 0; s--)
                LinkZones(leftVert[s], leftVert[s - 1]);

            // Right: south (top -> bottom) -> (0 -> Length)
            for (int s = 0; s < rightVert.Length - 1; s++)
                LinkZones(rightVert[s], rightVert[s + 1]);

            // Cross-connections: vertical ↔ row aisles
            for (int a = 0; a < rowAisles.Length; a++)
            {
                bool eastbound = (a % 2 == 0);
                int[] segs = rowAisles[a];
                int leftVertIdx = a + 1;   // +1 because index 0 is top spine conn
                int rightVertIdx = a + 1;

                if (eastbound)
                {
                    // Left vert -> row aisle first segment
                    LinkZones(leftVert[leftVertIdx], segs[0]);
                    // Row aisle last segment -> right vert
                    LinkZones(segs[segs.Length - 1], rightVert[rightVertIdx]);
                }
                else
                {
                    // Right vert -> row aisle last segment (westbound entry)
                    LinkZones(rightVert[rightVertIdx], segs[segs.Length - 1]);
                    // Row aisle first segment -> left vert
                    LinkZones(segs[0], leftVert[leftVertIdx]);
                }
            }

            // Cross-connections: spine ↔ vertical (The 4 corners)

            // Top-Left: Left vert (top) flows into Top spine (left end)
            LinkZones(leftVert[0], topSpine[0]);

            // Top-Right: Top spine (right end) flows into Right vert (top)
            LinkZones(topSpine[topSpine.Length - 1], rightVert[0]);

            // Bottom-Right: Right vert (bottom) flows into Bottom spine (right end)
            LinkZones(rightVert[rightVert.Length - 1], botSpine[botSpine.Length - 1]);

            // Bottom-Left: Bottom spine (left end) flows into Left vert (bottom)
            LinkZones(botSpine[0], leftVert[leftVert.Length - 1]);
        }



        private void LinkZones(int fromId, int toId)
        {
            if (!zoneById.ContainsKey(fromId) || !zoneById.ContainsKey(toId))
                return;

            TrafficZone from = zoneById[fromId];
            TrafficZone to = zoneById[toId];

            if (!from.Downstream.Contains(toId))
                from.Downstream.Add(toId);
            if (!to.Upstream.Contains(fromId))
                to.Upstream.Add(fromId);
        }

        // ─────────────────────────────────────────────────────────
        //  Dock Points
        // ─────────────────────────────────────────────────────────

        /// <summary>
        /// For each machine, registers dock points (approach position,
        /// handshake position, facing direction) in the adjacent aisle
        /// zones.
        /// </summary>
        private void RegisterDockPoints(
            int[][] rowAisles, int[] topSpine, int[] botSpine,
            int rows, int cols)
        {
            float standoff = 1.5f; // distance from conveyor end to approach point
            Vector3 floorCentre = layoutManager.transform.position;

            for (int i = 0; i < layoutManager.MachineCount; i++)
            {
                int row = i / cols;
                int col = i % cols;
                Vector3 machinePos = layoutManager.Machines[i].transform.position;

                // South-facing dock (into the aisle below this machine row)
                int southAisleIdx = row; // aisle 0 is between row 0 and row 1
                if (southAisleIdx < rowAisles.Length && col < rowAisles[southAisleIdx].Length)
                {
                    int zoneId = rowAisles[southAisleIdx][col];
                    TrafficZone zone = zoneById[zoneId];

                    Vector3 conveyorEnd = machinePos - Vector3.forward *
                        (layoutManager.MachineDepth / 2f + layoutManager.ConveyorReach);
                    Vector3 approach = conveyorEnd - Vector3.forward * standoff;

                    zone.DockPoints[i] = new DockPoint
                    {
                        ApproachPosition = approach,
                        HandshakePosition = conveyorEnd,
                        FacingDirection = Vector3.forward, // face north toward machine
                        IsPickup = false // outgoing conveyor → pickup for AGV
                    };

                    if (!machineToZones.ContainsKey(i))
                        machineToZones[i] = new List<int>();
                    machineToZones[i].Add(zoneId);
                }

                // North-facing dock (into the aisle above this machine row)
                int northAisleIdx = row - 1;
                if (northAisleIdx >= 0 && col < rowAisles[northAisleIdx].Length)
                {
                    int zoneId = rowAisles[northAisleIdx][col];
                    TrafficZone zone = zoneById[zoneId];

                    Vector3 conveyorEnd = machinePos + Vector3.forward *
                        (layoutManager.MachineDepth / 2f + layoutManager.ConveyorReach);
                    Vector3 approach = conveyorEnd + Vector3.forward * standoff;

                    zone.DockPoints[i] = new DockPoint
                    {
                        ApproachPosition = approach,
                        HandshakePosition = conveyorEnd,
                        FacingDirection = -Vector3.forward, // face south toward machine
                        IsPickup = true // incoming conveyor → dropoff for AGV
                    };

                    if (!machineToZones.ContainsKey(i))
                        machineToZones[i] = new List<int>();
                    machineToZones[i].Add(zoneId);
                }

                // Top row machines: dock into top spine for north-side conveyor
                if (row == 0)
                {
                    int spineSegIdx = col + 1; // +1 for left-end segment
                    if (spineSegIdx < topSpine.Length)
                    {
                        int zoneId = topSpine[spineSegIdx];
                        TrafficZone zone = zoneById[zoneId];

                        Vector3 conveyorEnd = machinePos + Vector3.forward *
                            (layoutManager.MachineDepth / 2f + layoutManager.ConveyorReach);
                        Vector3 approach = conveyorEnd + Vector3.forward * standoff;

                        zone.DockPoints[i] = new DockPoint
                        {
                            ApproachPosition = approach,
                            HandshakePosition = conveyorEnd,
                            FacingDirection = -Vector3.forward,
                            IsPickup = true
                        };

                        if (!machineToZones.ContainsKey(i))
                            machineToZones[i] = new List<int>();
                        machineToZones[i].Add(zoneId);
                    }
                }

                // Bottom row machines: dock into bottom spine
                if (row == layoutManager.LayoutRows - 1)
                {
                    int spineSegIdx = col + 1;
                    if (spineSegIdx < botSpine.Length)
                    {
                        int zoneId = botSpine[spineSegIdx];
                        TrafficZone zone = zoneById[zoneId];

                        Vector3 conveyorEnd = machinePos - Vector3.forward *
                            (layoutManager.MachineDepth / 2f + layoutManager.ConveyorReach);
                        Vector3 approach = conveyorEnd - Vector3.forward * standoff;

                        zone.DockPoints[i] = new DockPoint
                        {
                            ApproachPosition = approach,
                            HandshakePosition = conveyorEnd,
                            FacingDirection = Vector3.forward,
                            IsPickup = false
                        };

                        if (!machineToZones.ContainsKey(i))
                            machineToZones[i] = new List<int>();
                        machineToZones[i].Add(zoneId);
                    }
                }
            }
        }

        // ─────────────────────────────────────────────────────────
        //  Reservation API
        // ─────────────────────────────────────────────────────────

        /// <summary>
        /// Attempts to reserve a zone for the given AGV.
        /// </summary>
        /// <returns>True if the reservation was granted.</returns>
        public bool TryReserve(int zoneId, int agvId)
        {
            if (!zoneById.TryGetValue(zoneId, out TrafficZone zone))
            {
                SimLogger.Error($"[TrafficZones] Unknown zone {zoneId}");
                return false;
            }

            if (zone.OccupantAgvIds.Contains(agvId))
                return true; // already in this zone

            if (zone.IsFull)
            {
                SimLogger.Low($"[TrafficZones] Zone {zone.Name} full — " +
                    $"AGV {agvId} denied. Occupants: " +
                    $"{string.Join(",", zone.OccupantAgvIds)}");
                return false;
            }

            zone.OccupantAgvIds.Add(agvId);
            SimLogger.Low($"[TrafficZones] AGV {agvId} reserved zone " +
                $"{zone.Name} ({zone.OccupantAgvIds.Count}/{zone.Capacity})");
            return true;
        }

        /// <summary>
        /// Releases the AGV's reservation on a zone.
        /// </summary>
        public void Release(int zoneId, int agvId)
        {
            if (!zoneById.TryGetValue(zoneId, out TrafficZone zone))
                return;

            if (zone.OccupantAgvIds.Remove(agvId))
            {
                SimLogger.Low($"[TrafficZones] AGV {agvId} released zone " +
                    $"{zone.Name} ({zone.OccupantAgvIds.Count}/{zone.Capacity})");
            }
        }

        /// <summary>
        /// Releases the AGV from ALL zones. Useful when an AGV
        /// finishes a task or encounters an error.
        /// </summary>
        public void ReleaseAll(int agvId)
        {
            foreach (var zone in zones)
                zone.OccupantAgvIds.Remove(agvId);
        }

        // ─────────────────────────────────────────────────────────
        //  Query API
        // ─────────────────────────────────────────────────────────

        /// <summary>
        /// Returns the zone whose bounds contain the given world position,
        /// or null if the position is outside all zones.
        /// </summary>
        public TrafficZone GetZoneAtPosition(Vector3 worldPos)
        {
            foreach (var zone in zones)
            {
                Vector3 half = zone.Size / 2f;
                Vector3 d = worldPos - zone.Centre;
                if (Mathf.Abs(d.x) <= half.x && Mathf.Abs(d.z) <= half.z)
                    return zone;
            }
            return null;
        }

        /// <summary>
        /// Returns the dock point for a specific machine from a specific zone.
        /// </summary>
        public bool TryGetDockPoint(int zoneId, int machineId, out DockPoint dock)
        {
            dock = default;
            if (!zoneById.TryGetValue(zoneId, out TrafficZone zone))
                return false;
            return zone.DockPoints.TryGetValue(machineId, out dock);
        }

        /// <summary>
        /// Returns all zone IDs adjacent to the given machine.
        /// </summary>
        public List<int> GetZonesForMachine(int machineId)
        {
            return machineToZones.TryGetValue(machineId, out var list)
                ? list : new List<int>();
        }

        /// <summary>
        /// Simple BFS to find a zone-level route from one zone to another,
        /// following only downstream links.
        /// Returns the list of zone IDs (inclusive of start and end), or
        /// empty if no path exists.
        /// </summary>
        public List<int> GetRoute(int fromZoneId, int toZoneId)
        {
            if (fromZoneId == toZoneId)
                return new List<int> { fromZoneId };

            var visited = new HashSet<int>();
            var parent = new Dictionary<int, int>();
            var queue = new Queue<int>();

            queue.Enqueue(fromZoneId);
            visited.Add(fromZoneId);

            while (queue.Count > 0)
            {
                int current = queue.Dequeue();
                TrafficZone zone = zoneById[current];

                foreach (int next in zone.Downstream)
                {
                    if (visited.Contains(next)) continue;
                    visited.Add(next);
                    parent[next] = current;

                    if (next == toZoneId)
                    {
                        // Reconstruct path
                        var path = new List<int>();
                        int node = toZoneId;
                        while (node != fromZoneId)
                        {
                            path.Add(node);
                            node = parent[node];
                        }
                        path.Add(fromZoneId);
                        path.Reverse();
                        return path;
                    }

                    queue.Enqueue(next);
                }
            }

            return new List<int>(); // no path found
        }

        // ─────────────────────────────────────────────────────────
        //  Editor Gizmos
        // ─────────────────────────────────────────────────────────

        private void OnDrawGizmos()
        {
            if (!drawGizmos || zones.Count == 0) return;

            foreach (var zone in zones)
            {
                // Color by type and occupancy
                var col = zone.AisleType switch
                {
                    AisleType.SpineAisle => new Color(0.1f, 0.7f, 0.5f, 0.15f),
                    AisleType.VerticalAisle => new Color(0.2f, 0.4f, 0.9f, 0.15f),
                    _ => new Color(0.9f, 0.7f, 0.2f, 0.12f),
                };

                // Red tint when occupied
                if (!zone.IsEmpty)
                {
                    col = Color.Lerp(col, Color.red, 0.4f);
                    col.a = 0.3f;
                }

                Gizmos.color = col;
                Gizmos.DrawCube(zone.Centre, zone.Size);

                // Outline
                col.a = 0.5f;
                Gizmos.color = col;
                Gizmos.DrawWireCube(zone.Centre, zone.Size);

                // Flow arrow
                Vector3 flowVec = zone.Flow switch
                {
                    FlowDirection.East => Vector3.right,
                    FlowDirection.West => Vector3.left,
                    FlowDirection.North => Vector3.forward,
                    FlowDirection.South => Vector3.back,
                    _ => Vector3.zero
                };

                Gizmos.color = new Color(1f, 1f, 0f, 0.4f);
                Vector3 arrowStart = zone.Centre - 0.3f * zone.Size.x * flowVec;
                Vector3 arrowEnd = zone.Centre + 0.3f * zone.Size.x * flowVec;
                Gizmos.DrawLine(arrowStart, arrowEnd);

                // Dock points
                foreach (var kvp in zone.DockPoints)
                {
                    DockPoint dp = kvp.Value;
                    Gizmos.color = dp.IsPickup
                        ? new Color(1f, 0.3f, 0.3f, 0.6f)
                        : new Color(0.3f, 1f, 0.3f, 0.6f);
                    Gizmos.DrawWireSphere(dp.ApproachPosition, 0.2f);
                    Gizmos.DrawLine(dp.ApproachPosition, dp.HandshakePosition);
                    Gizmos.DrawWireSphere(dp.HandshakePosition, 0.15f);
                }

#if UNITY_EDITOR
                if (drawLabels)
                {
                    string label = zone.IsEmpty
                        ? zone.Name
                        : $"{zone.Name} [{string.Join(",", zone.OccupantAgvIds)}]";
                    UnityEditor.Handles.Label(
                        zone.Centre + Vector3.up * 0.5f, label);
                }
#endif
            }
        }
    }
}