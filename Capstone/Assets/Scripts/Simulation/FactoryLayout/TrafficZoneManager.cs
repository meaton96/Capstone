using System;
using System.Collections.Generic;
using UnityEngine;
using Assets.Scripts.Logging;

namespace Assets.Scripts.Simulation.FactoryLayout
{
    /// @brief Categorises the aisle type a zone belongs to.
    public enum AisleType
    {
        RowAisle,
        SpineAisle,
        VerticalAisle
    }

    /// @brief One-way flow direction of a zone's parent aisle.
    public enum FlowDirection
    {
        East, West, North, South
    }

    /// @brief A single reservable zone within the traffic network.
    [Serializable]
    public class TrafficZone
    {
        public int ZoneId;
        public string Name;
        public AisleType AisleType;
        public FlowDirection Flow;
        public Vector3 Centre;
        public Vector3 Size;
        public int Capacity = 1;
        public List<int> Downstream = new List<int>();
        public List<int> Upstream = new List<int>();
        public Dictionary<int, DockPoint> DockPoints = new Dictionary<int, DockPoint>();

        [NonSerialized] public HashSet<int> OccupantAgvIds = new HashSet<int>();

        public bool IsFull => OccupantAgvIds.Count >= Capacity;
        public bool IsEmpty => OccupantAgvIds.Count == 0;
    }

    /// @brief Describes positioning for AGV-conveyor interaction.
    [Serializable]
    public struct DockPoint
    {
        public Vector3 ApproachPosition;
        public Vector3 HandshakePosition;
        public Vector3 FacingDirection;
        public bool IsPickup;
    }

    /// @brief Manages the zone-based traffic control network for the factory floor.
    /// @details Divides the floor into reservable segments to manage one-way flow and prevent deadlocks.
    [RequireComponent(typeof(FactoryLayoutManager))]
    public class TrafficZoneManager : MonoBehaviour
    {
        private FactoryLayoutManager layoutManager;

        [Header("Debug")]
        [SerializeField] private bool drawGizmos = true;
        [SerializeField] private bool drawLabels = true;

        public const int IncomingBeltId = -1;
        public const int OutgoingBeltId = -2;
        public const int ParkingAreaId = -3;

        private readonly List<TrafficZone> zones = new List<TrafficZone>();
        private readonly Dictionary<int, TrafficZone> zoneById = new Dictionary<int, TrafficZone>();
        private int nextZoneId;
        private readonly Dictionary<int, List<int>> machineToZones = new Dictionary<int, List<int>>();

        public IReadOnlyList<TrafficZone> Zones => zones;

        /// @brief Retrieves a zone by its unique ID.
        public TrafficZone GetZone(int zoneId) => zoneById.TryGetValue(zoneId, out var z) ? z : null;

        void Awake()
        {
            layoutManager = GetComponent<FactoryLayoutManager>();
        }

        /// @brief Constructs the topological traffic network from the current factory layout.
        /// @details Orchestrates the creation of row, spine, and vertical zones, establishes 
        /// directed links, and registers dock points for all machines and I/O belts.
        /// @pre FactoryLayoutManager must have already built the physical floor.
        /// @post The @ref zones list and lookup dictionaries are fully populated.
        public void BuildZoneGraph()
        {
            zones.Clear();
            zoneById.Clear();
            machineToZones.Clear();
            nextZoneId = 0;

            if (layoutManager == null || layoutManager.LayoutRows == 0)
            {
                SimLogger.Error("[TrafficZones] Layout manager missing or layout not built.");
                return;
            }

            int rows = layoutManager.LayoutRows;
            int cols = layoutManager.LayoutCols;

            int[][] rowAisleZones = BuildRowAisleZones(rows, cols);
            int[] topSpineZones = BuildSpineZones(true, cols);
            int[] botSpineZones = BuildSpineZones(false, cols);
            int[] leftVertZones = BuildVerticalZones(true, rows);
            int[] rightVertZones = BuildVerticalZones(false, rows);

            ConnectZoneGraph(rowAisleZones, topSpineZones, botSpineZones, leftVertZones, rightVertZones, rows, cols);
            RegisterDockPoints(rowAisleZones, topSpineZones, botSpineZones, rows, cols);

            SimLogger.Medium($"[TrafficZones] Built zone graph: {zones.Count} zones.");
        }

        /// @brief Segments row aisles into discrete zones based on machine columns.
        /// @param rows Number of machine rows.
        /// @param cols Number of machine columns.
        /// @return A 2D array mapping [aisleIndex][segmentIndex] to zone IDs.
        private int[][] BuildRowAisleZones(int rows, int cols)
        {
            int numAisles = rows - 1;
            int[][] result = new int[numAisles][];

            for (int a = 0; a < numAisles; a++)
            {
                Vector3 aisleCentre = layoutManager.GetRowAisleCentre(a);
                Vector3 flowDir = layoutManager.GetRowAisleDirection(a);
                FlowDirection flow = flowDir.x > 0 ? FlowDirection.East : FlowDirection.West;

                result[a] = new int[cols];
                float segWidth = layoutManager.MachineSpacingX;
                float halfTotalWidth = ((cols - 1) * segWidth) / 2f;

                for (int s = 0; s < cols; s++)
                {
                    var zone = new TrafficZone
                    {
                        ZoneId = nextZoneId++,
                        Name = $"RowAisle{a}_Seg{s}",
                        AisleType = AisleType.RowAisle,
                        Flow = flow,
                        Centre = new Vector3(aisleCentre.x + (-halfTotalWidth + s * segWidth), aisleCentre.y, aisleCentre.z),
                        Size = new Vector3(segWidth, 0.1f, layoutManager.RowAisleWidth),
                        Capacity = 1
                    };
                    RegisterZone(zone);
                    result[a][s] = zone.ZoneId;
                }
            }
            return result;
        }

        /// @brief Segments spine aisles (top/bottom peripheral) into zones.
        /// @param isTop True if building the top spine, false for bottom.
        /// @param cols Number of machine columns.
        /// @return An array of zone IDs for the spine.
        private int[] BuildSpineZones(bool isTop, int cols)
        {
            int numSegments = cols + 1;
            int[] result = new int[numSegments];
            float z = isTop ? layoutManager.GetTopSpineZ() : layoutManager.GetBottomSpineZ();
            Vector3 floorCentre = layoutManager.transform.position;
            FlowDirection flow = isTop ? FlowDirection.East : FlowDirection.West;
            float segWidth = layoutManager.MachineSpacingX;
            float halfTotalWidth = ((cols - 1) * segWidth) / 2f;
            float leftEdge = -halfTotalWidth - layoutManager.MachineDepth / 2f - layoutManager.VerticalAisleWidth / 2f;

            for (int s = 0; s < numSegments; s++)
            {
                float segCentreX, width;
                if (s == 0) { segCentreX = leftEdge; width = layoutManager.VerticalAisleWidth; }
                else if (s == numSegments - 1) { segCentreX = -leftEdge; width = layoutManager.VerticalAisleWidth; }
                else { segCentreX = -halfTotalWidth + (s - 1) * segWidth; width = segWidth; }

                var zone = new TrafficZone
                {
                    ZoneId = nextZoneId++,
                    Name = $"{(isTop ? "TopSpine" : "BotSpine")}_Seg{s}",
                    AisleType = AisleType.SpineAisle,
                    Flow = flow,
                    Centre = new Vector3(floorCentre.x + segCentreX, 0.01f, floorCentre.z + z),
                    Size = new Vector3(width, 0.1f, layoutManager.SpineAisleWidth),
                    Capacity = 2
                };
                RegisterZone(zone);
                result[s] = zone.ZoneId;
            }
            return result;
        }

        /// @brief Segments vertical connector aisles (left/right) into zones.
        /// @param isLeft True for the left aisle, false for right.
        /// @param rows Number of machine rows.
        /// @return An array of zone IDs for the vertical aisle.
        private int[] BuildVerticalZones(bool isLeft, int rows)
        {
            int numRowAisles = rows - 1;
            int numSegments = numRowAisles + 2;
            int[] result = new int[numSegments];
            float halfMachineAreaW = ((layoutManager.LayoutCols - 1) * layoutManager.MachineSpacingX) / 2f + layoutManager.MachineDepth / 2f;
            float x = isLeft ? -(halfMachineAreaW + layoutManager.VerticalAisleWidth / 2f) : (halfMachineAreaW + layoutManager.VerticalAisleWidth / 2f);
            FlowDirection flow = isLeft ? FlowDirection.North : FlowDirection.South;
            Vector3 floorCentre = layoutManager.transform.position;

            for (int s = 0; s < numSegments; s++)
            {
                float z, height; string name;
                if (s == 0) { z = layoutManager.GetTopSpineZ(); height = layoutManager.SpineAisleWidth; name = "TopConn"; }
                else if (s == numSegments - 1) { z = layoutManager.GetBottomSpineZ(); height = layoutManager.SpineAisleWidth; name = "BotConn"; }
                else { Vector3 aisleCentre = layoutManager.GetRowAisleCentre(s - 1); z = aisleCentre.z - floorCentre.z; height = layoutManager.RowAisleWidth; name = $"Row{s - 1}"; }

                var zone = new TrafficZone
                {
                    ZoneId = nextZoneId++,
                    Name = $"{(isLeft ? "LeftVert" : "RightVert")}_{name}",
                    AisleType = AisleType.VerticalAisle,
                    Flow = flow,
                    Centre = new Vector3(floorCentre.x + x, 0.01f, floorCentre.z + z),
                    Size = new Vector3(layoutManager.VerticalAisleWidth, 0.1f, height),
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

        /// @brief Wires up downstream and upstream links between all zones to form a circulation loop.
        /// @details Connects internal chains for rows, spines, and vertical aisles, then bridges 
        /// intersections and corners based on restricted flow directions.
        /// @post Every zone has populated Downstream/Upstream lists forming a directed graph.
        private void ConnectZoneGraph(int[][] rowAisles, int[] topSpine, int[] botSpine, int[] leftVert, int[] rightVert, int rows, int cols)
        {
            for (int a = 0; a < rowAisles.Length; a++)
            {
                bool eastbound = (a % 2 == 0);
                int[] segs = rowAisles[a];
                if (eastbound) for (int s = 0; s < segs.Length - 1; s++) LinkZones(segs[s], segs[s + 1]);
                else for (int s = segs.Length - 1; s > 0; s--) LinkZones(segs[s], segs[s - 1]);
            }

            for (int s = 0; s < topSpine.Length - 1; s++) LinkZones(topSpine[s], topSpine[s + 1]);
            for (int s = botSpine.Length - 1; s > 0; s--) LinkZones(botSpine[s], botSpine[s - 1]);
            for (int s = leftVert.Length - 1; s > 0; s--) LinkZones(leftVert[s], leftVert[s - 1]);
            for (int s = 0; s < rightVert.Length - 1; s++) LinkZones(rightVert[s], rightVert[s + 1]);

            for (int a = 0; a < rowAisles.Length; a++)
            {
                bool eastbound = (a % 2 == 0);
                int[] segs = rowAisles[a];
                int vIdx = a + 1;
                if (eastbound) { LinkZones(leftVert[vIdx], segs[0]); LinkZones(segs[segs.Length - 1], rightVert[vIdx]); }
                else { LinkZones(rightVert[vIdx], segs[segs.Length - 1]); LinkZones(segs[0], leftVert[vIdx]); }
            }

            LinkZones(leftVert[0], topSpine[0]);
            LinkZones(topSpine[topSpine.Length - 1], rightVert[0]);
            LinkZones(rightVert[rightVert.Length - 1], botSpine[botSpine.Length - 1]);
            LinkZones(botSpine[0], leftVert[leftVert.Length - 1]);
        }

        private void LinkZones(int fromId, int toId)
        {
            if (!zoneById.TryGetValue(fromId, out var from) || !zoneById.TryGetValue(toId, out var to)) return;
            if (!from.Downstream.Contains(toId)) from.Downstream.Add(toId);
            if (!to.Upstream.Contains(fromId)) to.Upstream.Add(fromId);
        }

        /// @brief Calculates and registers AGV interaction points for machines and belts.
        /// @details Resolves the handshake and approach positions for every physical machine 
        /// and maps them to the nearest reservable traffic zone.
        /// @post TrafficZone.DockPoints dictionaries are populated for relevant zones.
        private void RegisterDockPoints(int[][] rowAisles, int[] topSpine, int[] botSpine, int rows, int cols)
        {
            if (topSpine.Length > 0 && layoutManager.IncomingBelt != null)
            {
                TrafficZone inZone = zoneById[topSpine[0]];
                Vector3 handshake = layoutManager.IncomingBelt.OutputEndPosition;
                inZone.DockPoints[IncomingBeltId] = new DockPoint { ApproachPosition = handshake - Vector3.forward * 1.5f, HandshakePosition = handshake, FacingDirection = Vector3.forward, IsPickup = true };
            }

            if (botSpine.Length > 0 && layoutManager.OutgoingBelt != null)
            {
                TrafficZone outZone = zoneById[botSpine[botSpine.Length - 1]];
                Vector3 handshake = layoutManager.OutgoingBelt.InputEndPosition;
                outZone.DockPoints[OutgoingBeltId] = new DockPoint { ApproachPosition = handshake + Vector3.forward * 1.5f, HandshakePosition = handshake, FacingDirection = -Vector3.forward, IsPickup = false };
            }

            if (botSpine.Length > 0)
            {
                TrafficZone parkZone = zoneById[botSpine[0]];
                parkZone.Capacity = 10;
                parkZone.DockPoints[ParkingAreaId] = new DockPoint { ApproachPosition = layoutManager.AGVParkingPosition, HandshakePosition = layoutManager.AGVParkingPosition, FacingDirection = -Vector3.left, IsPickup = false };
            }

            float standoff = 1.5f;
            for (int i = 0; i < layoutManager.MachineCount; i++)
            {
                int row = i / cols; int col = i % cols;
                Vector3 machinePos = layoutManager.Machines[i].transform.position;

                if (row < rowAisles.Length)
                {
                    int zId = rowAisles[row][col];
                    Vector3 conveyorEnd = machinePos - Vector3.forward * (layoutManager.MachineDepth / 2f + layoutManager.ConveyorReach);
                    zoneById[zId].DockPoints[i] = new DockPoint { ApproachPosition = conveyorEnd - Vector3.forward * standoff, HandshakePosition = conveyorEnd, FacingDirection = Vector3.forward, IsPickup = false };
                    if (!machineToZones.ContainsKey(i)) machineToZones[i] = new List<int>();
                    machineToZones[i].Add(zId);
                }

                if (row > 0 || row == 0) // Check north aisles
                {
                    int zId = -1;
                    if (row > 0) zId = rowAisles[row - 1][col];
                    else if (row == 0) zId = topSpine[col + 1];

                    if (zId != -1)
                    {
                        Vector3 conveyorEnd = machinePos + Vector3.forward * (layoutManager.MachineDepth / 2f + layoutManager.ConveyorReach);
                        zoneById[zId].DockPoints[i] = new DockPoint { ApproachPosition = conveyorEnd + Vector3.forward * standoff, HandshakePosition = conveyorEnd, FacingDirection = -Vector3.forward, IsPickup = true };
                        if (!machineToZones.ContainsKey(i)) machineToZones[i] = new List<int>();
                        machineToZones[i].Add(zId);
                    }
                }
            }
        }

        /// @brief Attempts to secure a spot in a zone for an AGV.
        /// @param zoneId The ID of the zone to enter.
        /// @param agvId The ID of the AGV requesting entry.
        /// @return True if capacity is available or AGV is already registered; false if full.
        /// @post If true, the AGV ID is added to the zone's occupant set.
        public bool TryReserve(int zoneId, int agvId)
        {
            if (!zoneById.TryGetValue(zoneId, out TrafficZone zone)) return false;
            if (zone.OccupantAgvIds.Contains(agvId)) return true;
            if (zone.IsFull) return false;

            zone.OccupantAgvIds.Add(agvId);
            return true;
        }

        /// @brief Releases an AGV's reservation on a specific zone.
        /// @post The occupant count for the zone is decremented.
        public void Release(int zoneId, int agvId)
        {
            if (zoneById.TryGetValue(zoneId, out TrafficZone zone))
                zone.OccupantAgvIds.Remove(agvId);
        }

        /// @brief Forcefully removes an AGV from all zones it may be occupying.
        public void ReleaseAll(int agvId)
        {
            foreach (var zone in zones) zone.OccupantAgvIds.Remove(agvId);
        }

        /// @brief Finds the zone containing the specified world position.
        public TrafficZone GetZoneAtPosition(Vector3 worldPos)
        {
            foreach (var zone in zones)
            {
                Vector3 half = zone.Size / 2f; Vector3 d = worldPos - zone.Centre;
                if (Mathf.Abs(d.x) <= half.x && Mathf.Abs(d.z) <= half.z) return zone;
            }
            return null;
        }

        /// @brief Returns the dock configuration for a machine within a specific zone.
        public bool TryGetDockPoint(int zoneId, int machineId, out DockPoint dock)
        {
            dock = default;
            return zoneById.TryGetValue(zoneId, out TrafficZone zone) && zone.DockPoints.TryGetValue(machineId, out dock);
        }

        /// @brief Returns all zones that have interaction points for a specific machine.
        public List<int> GetZonesForMachine(int machineId)
        {
            return machineToZones.TryGetValue(machineId, out var list) ? list : new List<int>();
        }

        /// @brief Calculates a zone-level path using BFS following restricted flow.
        /// @param fromZoneId Starting zone ID.
        /// @param toZoneId Destination zone ID.
        /// @return A list of zone IDs representing the route; empty if no valid path exists.
        public List<int> GetRoute(int fromZoneId, int toZoneId)
        {
            if (fromZoneId == toZoneId) return new List<int> { fromZoneId };
            var visited = new HashSet<int>(); var parent = new Dictionary<int, int>(); var queue = new Queue<int>();
            queue.Enqueue(fromZoneId); visited.Add(fromZoneId);

            while (queue.Count > 0)
            {
                int current = queue.Dequeue();
                foreach (int next in zoneById[current].Downstream)
                {
                    if (visited.Contains(next)) continue;
                    visited.Add(next); parent[next] = current;
                    if (next == toZoneId)
                    {
                        var path = new List<int>(); int node = toZoneId;
                        while (node != fromZoneId) { path.Add(node); node = parent[node]; }
                        path.Add(fromZoneId); path.Reverse(); return path;
                    }
                    queue.Enqueue(next);
                }
            }
            return new List<int>();
        }

        private void OnDrawGizmos()
        {
            if (!drawGizmos || zones.Count == 0) return;
            foreach (var zone in zones)
            {
                var col = zone.AisleType switch
                {
                    AisleType.SpineAisle => new Color(0.1f, 0.7f, 0.5f, 0.15f),
                    AisleType.VerticalAisle => new Color(0.2f, 0.4f, 0.9f, 0.15f),
                    _ => new Color(0.9f, 0.7f, 0.2f, 0.12f),
                };
                if (!zone.IsEmpty) { col = Color.Lerp(col, Color.red, 0.4f); col.a = 0.3f; }
                Gizmos.color = col; Gizmos.DrawCube(zone.Centre, zone.Size);
                col.a = 0.5f; Gizmos.color = col; Gizmos.DrawWireCube(zone.Centre, zone.Size);

                Vector3 flowVec = zone.Flow switch
                {
                    FlowDirection.East => Vector3.right,
                    FlowDirection.West => Vector3.left,
                    FlowDirection.North => Vector3.forward,
                    FlowDirection.South => Vector3.back,
                    _ => Vector3.zero
                };
                Gizmos.color = new Color(1f, 1f, 0f, 0.4f);
                Gizmos.DrawLine(zone.Centre - 0.3f * zone.Size.x * flowVec, zone.Centre + 0.3f * zone.Size.x * flowVec);

                foreach (var dp in zone.DockPoints.Values)
                {
                    Gizmos.color = dp.IsPickup ? new Color(1f, 0.3f, 0.3f, 0.6f) : new Color(0.3f, 1f, 0.3f, 0.6f);
                    Gizmos.DrawWireSphere(dp.ApproachPosition, 0.2f);
                    Gizmos.DrawLine(dp.ApproachPosition, dp.HandshakePosition);
                    Gizmos.DrawWireSphere(dp.HandshakePosition, 0.15f);
                }

#if UNITY_EDITOR
                if (drawLabels)
                {
                    string label = zone.IsEmpty ? zone.Name : $"{zone.Name} [{string.Join(",", zone.OccupantAgvIds)}]";
                    UnityEditor.Handles.Label(zone.Centre + Vector3.up * 0.5f, label);
                }
#endif
            }
        }
    }
}