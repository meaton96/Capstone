using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using Assets.Scripts.Simulation.Logging;
using Assets.Scripts.Simulation.Types;

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
        /// True for the parking lane's lane and bay zones (ParkingMethod.Lane). Routes may enter this
        /// set only to park or from inside it (see TrafficZoneManager.GetRoute).
        public bool IsParkingLane;
        public List<int> Downstream = new List<int>();
        public List<int> Upstream = new List<int>();
        public Dictionary<int, DockPoint> DockPoints = new Dictionary<int, DockPoint>();

        [NonSerialized] public HashSet<int> OccupantAgvIds = new HashSet<int>();

        /// True if a floor position is inside this zone's box (same test as TrafficZoneManager.GetZoneAtPosition).
        public bool Contains(Vector3 p)
        {
            Vector3 half = Size / 2f; Vector3 d = p - Centre;
            return Mathf.Abs(d.x) <= half.x && Mathf.Abs(d.z) <= half.z;
        }

        public bool IsFull => OccupantAgvIds.Count >= Capacity;
        public bool IsEmpty => OccupantAgvIds.Count == 0;
        [NonSerialized] public int TraversalCount;   // successful TryReserve entries
        [NonSerialized] public int BlockEvents;       // failed TryReserve calls (zone full)
        [NonSerialized] public float TotalBlockTime;    // cumulative wait time reported by AGVs


    }

    /// @brief Describes positioning for AGV-conveyor interaction.
    [Serializable]
    public class DockPoint
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
        // Zones a pickup from each machine is served from. Same list as machineToZones except in passthrough
        // layouts, where only the output-side dock counts (the input-side dock is for dropoffs).
        private readonly Dictionary<int, List<int>> machinePickupZones = new Dictionary<int, List<int>>();

        public IReadOnlyList<TrafficZone> Zones => zones;

        /// @brief Retrieves a zone by its unique ID.
        public TrafficZone GetZone(int zoneId) => zoneById.TryGetValue(zoneId, out var z) ? z : null;

        private readonly List<int> parkingZoneIds = new List<int>();
        public IReadOnlyList<int> ParkingZoneIds => parkingZoneIds;
        /// True when parking is a real lane with one reserved bay per AGV (ParkingMethod.Lane): idle
        /// AGVs keep their bay reserved and parking is not an abstraction the collision monitor skips.
        public bool ParkingIsBayed => layoutManager != null && layoutManager.ActiveParkingMethod == ParkingMethod.Lane;

        public static TrafficZoneManager Instance;


        void Awake()
        {
            Instance = this;
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
            machinePickupZones.Clear();
            parkingZoneIds.Clear();
            nextZoneId = 0;

            if (layoutManager == null || layoutManager.LayoutRows == 0)
            {
                SimLogger.Error("[TrafficZones] Layout manager missing or layout not built.");
                return;
            }

            int rows = layoutManager.LayoutRows;
            int cols = layoutManager.LayoutCols;
            bool twoWay = layoutManager.ActiveLayout != null && layoutManager.ActiveLayout.Aisles == Types.AisleTopology.TwoWay;

            // One-way: each row aisle is one lane in the aisle's own direction (GetRowAisleDirection), and it
            // is both the "north lane" and the "south lane" below. Two-way (F-J): two stacked one-way lanes,
            // north lane westbound and south lane eastbound in every aisle. There are NO lane-change links:
            // an AGV reverses by leaving its lane onto the one-way vertical at the aisle end and turning into
            // the other lane at that same junction (or a later one). A direct Fwd<->Rev link is a two-zone
            // cycle, which two AGVs can deadlock on — the failure behind every earlier two-way attempt
            // (docs/LAYOUT_CONFIGURATION_SCOPE.md section 18). The perimeter stays one-way in every layout.
            int[][] rowLaneN, rowLaneS;
            if (!twoWay) rowLaneN = rowLaneS = BuildRowAisleZones(rows, cols);
            else
            {
                float off = layoutManager.RowLaneOffset;
                rowLaneN = BuildRowAisleZones(rows, cols, off, "_N", eastbound: false);
                rowLaneS = BuildRowAisleZones(rows, cols, -off, "_S", eastbound: true);
            }
            // Verticals first: each spine's two corner zones ARE the vertical connectors' TopConn /
            // BotConn zones (same patch of floor), so they must exist before the spines are built.
            var (leftVertZones, leftChain) = BuildVerticalZones(true, rows);
            var (rightVertZones, rightChain) = BuildVerticalZones(false, rows);
            int[] topSpineZones = BuildSpineZones(true, cols, leftVertZones[0], rightVertZones[0]);
            int[] botSpineZones = BuildSpineZones(false, cols,
                                                  leftVertZones[leftVertZones.Length - 1],
                                                  rightVertZones[rightVertZones.Length - 1]);

            ConnectZoneGraph(rowLaneN, rowLaneS, twoWay, topSpineZones, botSpineZones, leftVertZones, rightVertZones, leftChain, rightChain);
            RegisterDockPoints(rowLaneN, rowLaneS, topSpineZones, botSpineZones, rows, cols, leftVertZones, rightVertZones);
            if (layoutManager.ActiveIoDocks != IoDockMethod.Corner) BuildIoSidings(leftChain, rightChain, topSpineZones);
            SimLogger.Medium($"[TrafficZones] Built zone graph: {zones.Count} zones{(twoWay ? " (two-way row aisles)" : "")}.");
            CheckZoneGraph();
        }

        /// @brief Static sanity check of the built graph, logged once per build.
        /// @details (1) Strong connectivity: every zone reaches every other, so no dock or parking bay is
        /// unreachable (would have caught the first two-way attempt's disconnected reverse lanes without a sim
        /// run). (2) Every machine dock's approach point lies inside the zone that reserves it — otherwise an
        /// AGV drives into floor another zone owns (the first two-way attempt registered each dock on both
        /// lanes, so the far lane's AGV crossed the near lane to reach it). (3) Girth: the shortest directed
        /// cycle over the shared Capacity-1 zones (parking lane/bays excluded — bays are per-AGV). A wait-for
        /// deadlock needs every zone of some cycle held, and a blocked AGV holds at most 2 zones under
        /// holdPrevious (1 under releasePrevious), so no fleet smaller than ceil(girth/2) can deadlock.
        private void CheckZoneGraph()
        {
            if (zones.Count == 0) return;

            int Reach(bool forward)
            {
                var seen = new HashSet<int> { zones[0].ZoneId }; var q = new Queue<int>(seen);
                while (q.Count > 0)
                {
                    var z = zoneById[q.Dequeue()];
                    foreach (int n in forward ? z.Downstream : z.Upstream)
                        if (seen.Add(n)) q.Enqueue(n);
                }
                return seen.Count;
            }
            bool stronglyConnected = Reach(true) == zones.Count && Reach(false) == zones.Count;

            int misplacedDocks = 0;
            foreach (var z in zones)
                foreach (var kv in z.DockPoints)
                    if (kv.Key >= 0 && !z.Contains(kv.Value.ApproachPosition))
                    {
                        misplacedDocks++;
                        SimLogger.Error($"[TrafficZones] Dock for machine {kv.Key} approaches outside its zone {z.Name}.");
                    }

            int girth = int.MaxValue;
            foreach (var s in zones)
            {
                if (s.IsParkingLane || s.Capacity > 1) continue;
                var dist = new Dictionary<int, int> { [s.ZoneId] = 0 }; var q = new Queue<int>(); q.Enqueue(s.ZoneId);
                while (q.Count > 0)
                {
                    int u = q.Dequeue();
                    if (dist[u] + 1 >= girth) break;
                    foreach (int v in zoneById[u].Downstream)
                    {
                        var vz = zoneById[v];
                        if (vz.IsParkingLane || vz.Capacity > 1) continue;
                        if (v == s.ZoneId) { girth = Math.Min(girth, dist[u] + 1); continue; }
                        if (dist.ContainsKey(v)) continue;
                        dist[v] = dist[u] + 1; q.Enqueue(v);
                    }
                }
            }

            string msg = $"[TrafficZones] Graph check: strongly connected={stronglyConnected}, misplaced docks={misplacedDocks}, " +
                         (girth == int.MaxValue ? "no cycle" : $"girth={girth} (no deadlock below {(girth + 1) / 2} AGVs under holdPrevious).");
            if (!stronglyConnected || misplacedDocks > 0) SimLogger.Error(msg); else SimLogger.Medium(msg);
        }

        /// @brief Segments row aisles into discrete zones: one dock zone per machine column plus
        /// one transit zone between each pair, so AGVs can queue along a row at closer spacing
        /// without raising per-zone Capacity above 1 (AGVController has no inter-AGV avoidance —
        /// Capacity=1 with a dedicated Centre per zone is what keeps two AGVs from converging on
        /// the same point).
        /// @param rows Number of machine rows.
        /// @param cols Number of machine columns.
        /// @param zOffset Lane centre offset from the aisle centreline (0 one-way, +-RowLaneOffset two-way).
        /// @param suffix Appended to every zone name ("_N"/"_S" for two-way lanes).
        /// @param eastbound Lane direction label; null = the aisle's one-way direction. Graph edges come
        /// from ConnectZoneGraph, which must be given the same direction.
        /// @return A 2D array mapping [aisleIndex][segmentIndex] to zone IDs — even indices are
        /// dock zones (index/2 = machine column), odd indices are the transit zones between them.
        private int[][] BuildRowAisleZones(int rows, int cols, float zOffset = 0f, string suffix = "", bool? eastbound = null)
        {
            int numAisles = rows - 1;
            int[][] result = new int[numAisles][];

            for (int a = 0; a < numAisles; a++)
            {
                Vector3 aisleCentre = layoutManager.GetRowAisleCentre(a);
                bool east = eastbound ?? layoutManager.GetRowAisleDirection(a).x > 0;
                FlowDirection flow = east ? FlowDirection.East : FlowDirection.West;

                int numSegs = 2 * cols - 1;
                result[a] = new int[numSegs];
                float segWidth = layoutManager.MachineSpacingX;
                float subWidth = segWidth / 2f;
                float halfTotalWidth = ((cols - 1) * segWidth) / 2f;

                for (int s = 0; s < numSegs; s++)
                {
                    bool isDock = s % 2 == 0;
                    int machineCol = s / 2;
                    float centreX = -halfTotalWidth + machineCol * segWidth + (isDock ? 0f : subWidth);

                    var zone = new TrafficZone
                    {
                        ZoneId = nextZoneId++,
                        Name = (isDock ? $"RowAisle{a}_Dock{machineCol}" : $"RowAisle{a}_Transit{machineCol}") + suffix,
                        AisleType = AisleType.RowAisle,
                        Flow = flow,
                        Centre = new Vector3(aisleCentre.x + centreX, aisleCentre.y, aisleCentre.z + zOffset),
                        Size = new Vector3(subWidth, 0.1f, layoutManager.RowLaneWidth),
                        Capacity = 1
                    };
                    RegisterZone(zone);
                    result[a][s] = zone.ZoneId;
                }
            }
            return result;
        }

        /// <summary>Index of the spine zone that hosts machine column <paramref name="col"/>'s pickup dock.</summary>
        private static int SpineDockIndex(int col) => 1 + 2 * col;

        /// @brief Builds a spine as [cornerL, Dock0, Transit0, ..., DockN-1, cornerR].
        /// Dock/transit zones sit at exactly the row aisles' x positions and are Capacity=1, so each
        /// has its own centre (3 units apart). The two corner entries are NOT new zones: they are the
        /// vertical connectors' TopConn/BotConn zones, which occupy the identical patch of floor.
        /// Building a separate corner zone there gave two Capacity-1 zones over one spot, so
        /// reservations could not keep two AGVs apart (AGVCollisionMonitor: ~96% of floor overlaps
        /// on the original layout). The corner-to-dock spacing is 2.5 units, above the 2.36 clearance.
        private int[] BuildSpineZones(bool isTop, int cols, int cornerLeftZoneId, int cornerRightZoneId)
        {
            int numDockTransit = 2 * cols - 1;
            int[] result = new int[numDockTransit + 2];
            float z = isTop ? layoutManager.GetTopSpineZ() : layoutManager.GetBottomSpineZ();
            Vector3 floorCentre = layoutManager.transform.position;
            FlowDirection flow = isTop ? FlowDirection.East : FlowDirection.West;
            string side = isTop ? "TopSpine" : "BotSpine";
            float segWidth = layoutManager.MachineSpacingX;
            float subWidth = segWidth / 2f;
            float halfTotalWidth = ((cols - 1) * segWidth) / 2f;

            result[0] = cornerLeftZoneId;
            result[result.Length - 1] = cornerRightZoneId;

            for (int s = 1; s < result.Length - 1; s++)
            {
                int k = s - 1;                       // 0-based within the dock/transit run
                bool isDock = k % 2 == 0;
                float centreX = -halfTotalWidth + k * subWidth;

                var zone = new TrafficZone
                {
                    ZoneId = nextZoneId++,
                    Name = $"{side}_{(isDock ? $"Dock{k / 2}" : $"Transit{k / 2}")}",
                    AisleType = AisleType.SpineAisle,
                    Flow = flow,
                    Centre = new Vector3(floorCentre.x + centreX, 0.01f, floorCentre.z + z),
                    Size = new Vector3(subWidth, 0.1f, layoutManager.SpineAisleWidth),
                    Capacity = 1
                };
                RegisterZone(zone);
                result[s] = zone.ZoneId;
            }
            return result;
        }

        /// <summary>Minimum centre-to-centre spacing of vertical zones: just above 2 x the 1.18 NavMesh clearance.</summary>
        public const float MinVerticalPitch = 2.4f;

        /// @brief Segments a vertical connector aisle (left/right) into Capacity=1 zones.
        /// @details The junction zones (TopConn, Row0..RowN-1, BotConn) sit where a row aisle or spine
        /// meets the connector and keep their old names and positions, so every row-aisle / spine /
        /// parking link is unchanged. Between consecutive junctions this inserts evenly spaced
        /// intermediate zones (pitch >= MinVerticalPitch) so the stretch beside each machine row is
        /// covered and reservable. Previously the connector was one zone per junction: ~7.5 units
        /// apart with 3-4.5 unit uncovered stretches between them (47 % of the connector length),
        /// so an AGV that reached one junction held the zone behind it for a further ~15 units of
        /// travel and the follower could not start until then.
        /// @return junctions: zone ids of TopConn, Row0.., BotConn (index = old array layout).
        ///         chain: every zone top to bottom (junctions plus intermediates), for linking.
        private (int[] junctions, int[] chain) BuildVerticalZones(bool isLeft, int rows)
        {
            int numRowAisles = rows - 1;
            int numJunctions = numRowAisles + 2;
            int[] junctions = new int[numJunctions];
            var chain = new List<int>();
            float halfMachineAreaW = ((layoutManager.LayoutCols - 1) * layoutManager.MachineSpacingX) / 2f + layoutManager.MachineDepth / 2f;
            float x = isLeft ? -(halfMachineAreaW + layoutManager.VerticalAisleWidth / 2f) : (halfMachineAreaW + layoutManager.VerticalAisleWidth / 2f);
            FlowDirection flow = isLeft ? FlowDirection.North : FlowDirection.South;
            Vector3 floorCentre = layoutManager.transform.position;
            string side = isLeft ? "LeftVert" : "RightVert";

            float[] zs = new float[numJunctions];
            float[] heights = new float[numJunctions];
            string[] names = new string[numJunctions];
            for (int s = 0; s < numJunctions; s++)
            {
                if (s == 0) { zs[s] = layoutManager.GetTopSpineZ(); heights[s] = layoutManager.SpineAisleWidth; names[s] = "TopConn"; }
                else if (s == numJunctions - 1) { zs[s] = layoutManager.GetBottomSpineZ(); heights[s] = layoutManager.SpineAisleWidth; names[s] = "BotConn"; }
                else { zs[s] = layoutManager.GetRowAisleCentre(s - 1).z - floorCentre.z; heights[s] = layoutManager.RowAisleWidth; names[s] = $"Row{s - 1}"; }
            }

            for (int s = 0; s < numJunctions; s++)
            {
                junctions[s] = AddVerticalZone($"{side}_{names[s]}", x, zs[s], heights[s], flow, floorCentre);
                chain.Add(junctions[s]);

                if (s == numJunctions - 1) break;
                float gap = zs[s] - zs[s + 1];
                int segments = Mathf.Max(1, Mathf.FloorToInt(gap / MinVerticalPitch));
                float pitch = gap / segments;
                for (int k = 1; k < segments; k++)
                    chain.Add(AddVerticalZone($"{side}_Gap{s}_{k}", x, zs[s] - k * pitch, pitch, flow, floorCentre));
            }
            return (junctions, chain.ToArray());
        }

        private int AddVerticalZone(string name, float x, float z, float height, FlowDirection flow, Vector3 floorCentre)
        {
            var zone = new TrafficZone
            {
                ZoneId = nextZoneId++,
                Name = name,
                AisleType = AisleType.VerticalAisle,
                Flow = flow,
                Centre = new Vector3(floorCentre.x + x, 0.01f, floorCentre.z + z),
                Size = new Vector3(layoutManager.VerticalAisleWidth, 0.1f, height),
                Capacity = 1
            };
            RegisterZone(zone);
            return zone.ZoneId;
        }

        private void RegisterZone(TrafficZone zone)
        {
            zones.Add(zone);
            zoneById[zone.ZoneId] = zone;
        }

        /// @brief Wires up downstream and upstream links between all zones to form a circulation loop.
        /// @details Connects internal chains for rows, spines, and vertical aisles, then bridges 
        /// intersections and corners based on restricted flow directions.
        /// Two-way: both row lanes are wired the same way, each into the vertical junction at its upstream
        /// end and out onto the one at its downstream end, so each Row junction becomes a 2-in/2-out
        /// intersection (straight on the vertical, or turn into the lane leading away from it).
        /// @post Every zone has populated Downstream/Upstream lists forming a directed graph.
        private void ConnectZoneGraph(int[][] rowLaneN, int[][] rowLaneS, bool twoWay, int[] topSpine, int[] botSpine,
                                       int[] leftVert, int[] rightVert, int[] leftChain, int[] rightChain)
        {
            // Link order matters: Downstream order is GetRoute's BFS tie-break, so one-way keeps its original
            // order (lane chains, then perimeter chains, then junction joins) to stay byte-identical.
            for (int a = 0; a < rowLaneN.Length; a++) WireRowLanes(a, rowLaneN, rowLaneS, twoWay, leftVert, rightVert, junctions: false);

            for (int s = 0; s < topSpine.Length - 1; s++) LinkZones(topSpine[s], topSpine[s + 1]);
            for (int s = botSpine.Length - 1; s > 0; s--) LinkZones(botSpine[s], botSpine[s - 1]);
            // Chains include the intermediate zones; left flows north (bottom -> top), right flows south.
            for (int s = leftChain.Length - 1; s > 0; s--) LinkZones(leftChain[s], leftChain[s - 1]);
            for (int s = 0; s < rightChain.Length - 1; s++) LinkZones(rightChain[s], rightChain[s + 1]);

            for (int a = 0; a < rowLaneN.Length; a++) WireRowLanes(a, rowLaneN, rowLaneS, twoWay, leftVert, rightVert, junctions: true);

            LinkZones(leftVert[0], topSpine[0]);
            LinkZones(topSpine[topSpine.Length - 1], rightVert[0]);
            LinkZones(rightVert[rightVert.Length - 1], botSpine[botSpine.Length - 1]);
            LinkZones(botSpine[0], leftVert[leftVert.Length - 1]);
        }

        /// @brief Wires aisle a's lane(s): junctions=false chains each lane in its direction; junctions=true
        /// joins each lane to the vertical junction at its upstream (in) and downstream (out) end.
        private void WireRowLanes(int a, int[][] rowLaneN, int[][] rowLaneS, bool twoWay, int[] leftVert, int[] rightVert, bool junctions)
        {
            var lanes = twoWay
                ? new[] { (segs: rowLaneN[a], east: false), (segs: rowLaneS[a], east: true) }
                : new[] { (segs: rowLaneN[a], east: layoutManager.GetRowAisleDirection(a).x > 0f) };
            int left = leftVert[a + 1], right = rightVert[a + 1];
            foreach (var (segs, east) in lanes)
            {
                int last = segs.Length - 1;
                if (!junctions)
                {
                    if (east) for (int s = 0; s < last; s++) LinkZones(segs[s], segs[s + 1]);
                    else for (int s = last; s > 0; s--) LinkZones(segs[s], segs[s - 1]);
                }
                else if (east) { LinkZones(left, segs[0]); LinkZones(segs[last], right); }
                else { LinkZones(right, segs[last]); LinkZones(segs[0], left); }
            }
        }

        private void LinkZones(int fromId, int toId)
        {
            if (fromId == toId) return;   // spine corner zone IS the vertical connector zone
            if (!zoneById.TryGetValue(fromId, out var from) || !zoneById.TryGetValue(toId, out var to)) return;
            if (!from.Downstream.Contains(toId)) from.Downstream.Add(toId);
            if (!to.Upstream.Contains(fromId)) to.Upstream.Add(fromId);
        }

        /// @brief Calculates and registers AGV interaction points for machines and belts.
        /// @details Resolves the handshake and approach positions for every physical machine 
        /// and maps them to the nearest reservable traffic zone.
        /// @post TrafficZone.DockPoints dictionaries are populated for relevant zones.
        /// @param rowLaneN, rowLaneS The lane beside the aisle's north edge (hosts the south-face docks of the
        /// machine row above) and beside its south edge (north-face docks of the row below). The same array in
        /// one-way. A dock's approach point lies in the lane beside it, so it is registered on that lane only.
        private void RegisterDockPoints(int[][] rowLaneN, int[][] rowLaneS, int[] topSpine, int[] botSpine,
                                int rows, int cols, int[] leftVert, int[] rightVert)
        {
            // Siding / bypass register the moved belt docks on their sidings instead (BuildIoSidings); bypass keeps
            // the output belt on its corner.
            bool inCorner = layoutManager.ActiveIoDocks == IoDockMethod.Corner;
            bool outCorner = layoutManager.ActiveIoDocks != IoDockMethod.Siding;
            if (inCorner && topSpine.Length > 0 && layoutManager.IncomingBelt != null)
            {
                TrafficZone inZone = zoneById[topSpine[0]];
                Vector3 handshake = layoutManager.IncomingBelt.OutputEndPosition;
                inZone.DockPoints[IncomingBeltId] = new DockPoint { ApproachPosition = handshake - Vector3.forward * 1.5f, HandshakePosition = handshake, FacingDirection = Vector3.forward, IsPickup = true };
            }

            if (outCorner && botSpine.Length > 0 && layoutManager.OutgoingBelt != null)
            {
                TrafficZone outZone = zoneById[botSpine[botSpine.Length - 1]];
                Vector3 handshake = layoutManager.OutgoingBelt.InputEndPosition;
                outZone.DockPoints[OutgoingBeltId] = new DockPoint { ApproachPosition = handshake + Vector3.forward * 1.5f, HandshakePosition = handshake, FacingDirection = -Vector3.forward, IsPickup = false };
            }

            if (layoutManager.ActiveParkingMethod == ParkingMethod.Single)
            {
                BuildSingleParkingZone(botSpine, leftVert);
            }
            else if (layoutManager.ActiveParkingMethod == ParkingMethod.Lane)
            {
                BuildParkingLane(leftVert, rightVert);
            }
            else
            {
                BuildMultipleParkingZones(leftVert, rightVert);
            }


            float standoff = 1.5f;
            for (int i = 0; i < layoutManager.MachineCount; i++)
            {
                int row = i / cols; int col = i % cols;
                Vector3 machinePos = layoutManager.Machines[i].transform.position;

                // Layout A registers docks on both sides wherever a zone exists (its interior machines carry an
                // unused secondary belt pair). Other layouts register a dock only on a side that has a belt, so
                // the hop-distance dispatch heuristic never seeds a dock nobody drives to, and the last row can
                // dock on the bottom spine (layout C).
                LayoutSpec layout = layoutManager.ActiveLayout;
                bool southDock = layout.IsLegacy ? row < rowLaneN.Length : layout.HasBeltOn(row, 'S');
                bool northDock = layout.IsLegacy || layout.HasBeltOn(row, 'N');

                var (inputSide, outputSide) = layout.BeltSides(row);
                int southZone = -1, northZone = -1;

                if (southDock)
                {
                    int zId = row < rowLaneN.Length
                        ? rowLaneN[row][col * 2]                     // dock zones sit at even indices — see BuildRowAisleZones
                        : botSpine[SpineDockIndex(col)];             // last row's south side is the bottom spine

                    Vector3 conveyorEnd = machinePos - Vector3.forward * (layoutManager.MachineDepth / 2f + layoutManager.ConveyorReach);
                    zoneById[zId].DockPoints[i] = new DockPoint { ApproachPosition = conveyorEnd - Vector3.forward * standoff, HandshakePosition = conveyorEnd, FacingDirection = Vector3.forward, IsPickup = layout.IsLegacy ? false : outputSide == 'S' };
                    southZone = zId;
                    if (!machineToZones.ContainsKey(i)) machineToZones[i] = new List<int>();
                    machineToZones[i].Add(zId);
                }

                if (northDock) // north side: the aisle above, or the top spine for row 0
                {
                    int zId = -1;
                    if (row > 0) zId = rowLaneS[row - 1][col * 2];    // dock zones sit at even indices
                    else if (row == 0) zId = topSpine[SpineDockIndex(col)];

                    if (zId != -1)
                    {
                        Vector3 conveyorEnd = machinePos + Vector3.forward * (layoutManager.MachineDepth / 2f + layoutManager.ConveyorReach);
                        zoneById[zId].DockPoints[i] = new DockPoint { ApproachPosition = conveyorEnd + Vector3.forward * standoff, HandshakePosition = conveyorEnd, FacingDirection = -Vector3.forward, IsPickup = layout.IsLegacy ? true : outputSide == 'N' };
                        northZone = zId;
                        if (!machineToZones.ContainsKey(i)) machineToZones[i] = new List<int>();
                        machineToZones[i].Add(zId);
                    }
                }

                // Passthrough: a pickup is only ever served from the output-side dock. Elsewhere every dock
                // serves both roles, so the pickup list is the machine's full dock list (unchanged behaviour).
                if (layout.IsPassthrough)
                {
                    int outZone = outputSide == 'N' ? northZone : southZone;
                    if (outZone >= 0) machinePickupZones[i] = new List<int> { outZone };
                }
                else if (machineToZones.TryGetValue(i, out var allZones))
                    machinePickupZones[i] = allZones;
            }
        }
        /// @brief I/O sidings (IoDockMethod.Siding): per belt, a two-zone one-way siding outside the side wall.
        /// @details Input, west of the left vertical (flows north): leftChain[1] (the zone just south of TopConn)
        ///          -> InSiding_Entry -> InSiding_Dock -> LeftVert_TopConn. Output, east of the right vertical
        ///          (flows south): the zone just north of BotConn -> OutSiding_Entry -> OutSiding_Dock ->
        ///          RightVert_BotConn. Entry sits beside the branch zone, Dock beside the corner; both are Capacity 1.
        ///          A docked AGV holds Dock (and Entry as its held previous zone under holdPrevious), never the
        ///          corner, so through traffic and turns at the corner are not blocked by loading. The siding is a
        ///          forward bypass of one vertical hop, so it adds no new cycle (girth unchanged) and no two-zone
        ///          loop (which a dead-end bay shared by several AGVs would, see LAYOUT_CONFIGURATION_SCOPE.md s19).
        /// @details Bypass (input only): InSiding_Dock does not rejoin at the corner. It continues north into a strip
        ///          beyond the top spine and east above it (InSiding_Exit above the dock, InSiding_Over above
        ///          TopConn, InSiding_N{k} above top-spine zone k) and drops into the top spine at Transit0 (index 2;
        ///          Dock0 hosts row 0's north docks, so merging there would queue behind them). The return strip is
        ///          one lane width north of the spine, so it never shares floor with TopConn.
        private void BuildIoSidings(int[] leftChain, int[] rightChain, int[] topSpine)
        {
            if (leftChain.Length < 2 || rightChain.Length < 2) return;
            bool bypass = layoutManager.ActiveIoDocks == IoDockMethod.Bypass;
            float w = FactoryLayoutManager.SidingWidth;
            const float standoff = 1.5f;   // belt end -> approach point, same as every other dock

            (int entry, int dock) Build(string prefix, bool left, int branchId, int cornerId, FlowDirection flow, bool rejoin = true)
            {
                TrafficZone branch = zoneById[branchId], corner = zoneById[cornerId];
                // Beside the vertical: its centre, out by half the vertical plus half the siding.
                float x = branch.Centre.x + (left ? -1f : 1f) * (layoutManager.VerticalAisleWidth + w) / 2f;
                var e = new TrafficZone { ZoneId = nextZoneId++, Name = $"{prefix}_Entry", AisleType = AisleType.VerticalAisle,
                    Flow = flow, Centre = new Vector3(x, 0.01f, branch.Centre.z), Size = new Vector3(w, 0.1f, branch.Size.z), Capacity = 1 };
                var d = new TrafficZone { ZoneId = nextZoneId++, Name = $"{prefix}_Dock", AisleType = AisleType.VerticalAisle,
                    Flow = flow, Centre = new Vector3(x, 0.01f, corner.Centre.z), Size = new Vector3(w, 0.1f, corner.Size.z), Capacity = 1 };
                RegisterZone(e); RegisterZone(d);
                LinkZones(branchId, e.ZoneId); LinkZones(e.ZoneId, d.ZoneId);
                if (rejoin) LinkZones(d.ZoneId, cornerId);
                return (e.ZoneId, d.ZoneId);
            }

            var (_, inDock) = Build("InSiding", true, leftChain[1], leftChain[0], FlowDirection.North, rejoin: !bypass);
            int outDock = -1;
            if (!bypass)
                (_, outDock) = Build("OutSiding", false, rightChain[rightChain.Length - 2], rightChain[rightChain.Length - 1], FlowDirection.South);
            else
            {
                TrafficZone corner = zoneById[leftChain[0]], dockZ = zoneById[inDock];
                float zN = corner.Centre.z + corner.Size.z / 2f + w / 2f;          // strip centre, one lane north of the spine
                int target = topSpine[Mathf.Min(2, topSpine.Length - 2)];          // Transit0 (Dock0 if a 1-column floor)
                var xs = new List<(string name, float x)> { ("Exit", dockZ.Centre.x), ("Over", corner.Centre.x) };
                for (int k = 1; k < topSpine.Length - 1 && topSpine[k - 1] != target; k++)
                    xs.Add(($"N{k}", zoneById[topSpine[k]].Centre.x));
                int prev = inDock;
                foreach (var (name, x) in xs)
                {
                    var z = new TrafficZone { ZoneId = nextZoneId++, Name = $"InSiding_{name}", AisleType = AisleType.SpineAisle,
                        Flow = FlowDirection.East, Centre = new Vector3(x, 0.01f, zN), Size = new Vector3(w, 0.1f, w), Capacity = 1 };
                    RegisterZone(z); LinkZones(prev, z.ZoneId); prev = z.ZoneId;
                }
                LinkZones(prev, target);
            }

            if (layoutManager.IncomingBelt != null)
            {
                Vector3 h = layoutManager.IncomingBelt.OutputEndPosition;   // west edge of the input siding
                zoneById[inDock].DockPoints[IncomingBeltId] = new DockPoint
                    { ApproachPosition = h + Vector3.right * standoff, HandshakePosition = h, FacingDirection = Vector3.left, IsPickup = true };
            }
            if (outDock >= 0 && layoutManager.OutgoingBelt != null)
            {
                Vector3 h = layoutManager.OutgoingBelt.InputEndPosition;    // east edge of the output siding
                zoneById[outDock].DockPoints[OutgoingBeltId] = new DockPoint
                    { ApproachPosition = h + Vector3.left * standoff, HandshakePosition = h, FacingDirection = Vector3.right, IsPickup = false };
            }
            foreach (int z in new[] { inDock, outDock }.Where(id => id >= 0))
                foreach (var kv in zoneById[z].DockPoints)
                    if (!zoneById[z].Contains(kv.Value.ApproachPosition))
                        SimLogger.Error($"[TrafficZones] I/O siding dock approach lies outside {zoneById[z].Name}.");
        }

        private void BuildSingleParkingZone(int[] botSpine, int[] leftVert)
        {
            if (botSpine.Length == 0) return;

            var parkingZone = new TrafficZone
            {
                ZoneId = nextZoneId++,
                Name = "Parking_Alcove",
                AisleType = AisleType.SpineAisle,
                Flow = FlowDirection.West,
                Centre = layoutManager.AGVParkingPosition,
                Size = new Vector3(layoutManager.FloorSize.x, 0.1f, layoutManager.ParkingAlcoveDepth),
                Capacity = 64
            };
            parkingZone.DockPoints[ParkingAreaId] = new DockPoint
            {
                ApproachPosition = layoutManager.AGVParkingPosition,
                HandshakePosition = layoutManager.AGVParkingPosition,
                FacingDirection = -Vector3.left,
                IsPickup = false
            };
            RegisterZone(parkingZone);
            parkingZoneIds.Add(parkingZone.ZoneId);

            LinkZones(botSpine[0], parkingZone.ZoneId);                 // drive in
            LinkZones(parkingZone.ZoneId, leftVert[leftVert.Length - 1]); // drive out
        }

        /// @brief Builds the parking lane: a one-way run of Capacity 1 zones south of the bottom spine with
        ///        one dedicated Capacity 1 bay per AGV opening onto it (either side).
        /// @details Entry: RightVert_BotConn -> Lane_0 (east end). Exit: last lane zone -> LeftVert_BotConn.
        ///          Bays are leaves (lane <-> bay), so a departing AGV pulls out into the lane and keeps
        ///          going west; nothing in the lane is ever passed. Parking is fully reserved: there is no
        ///          straight-line leg across unreserved floor.
        private void BuildParkingLane(int[] leftVert, int[] rightVert)
        {
            var shape = layoutManager.LaneShape;
            if (shape == null || leftVert.Length == 0 || rightVert.Length == 0) return;

            var laneIds = new int[shape.LaneCentres.Length];
            for (int k = 0; k < laneIds.Length; k++)
            {
                var z = new TrafficZone
                {
                    ZoneId = nextZoneId++,
                    Name = $"Lane_{k}",
                    AisleType = AisleType.SpineAisle,
                    Flow = FlowDirection.West,
                    Centre = shape.LaneCentres[k],
                    Size = new Vector3(shape.Pitch, 0.1f, shape.RowDepth),
                    Capacity = 1,
                    IsParkingLane = true
                };
                RegisterZone(z);
                laneIds[k] = z.ZoneId;
            }
            for (int k = 0; k < laneIds.Length - 1; k++) LinkZones(laneIds[k], laneIds[k + 1]);

            for (int i = 0; i < shape.BayCentres.Length; i++)
            {
                var bay = new TrafficZone
                {
                    ZoneId = nextZoneId++,
                    Name = $"Bay_{i}",
                    AisleType = AisleType.SpineAisle,
                    Flow = FlowDirection.West,
                    Centre = shape.BayCentres[i],
                    Size = new Vector3(shape.Pitch, 0.1f, shape.RowDepth),
                    Capacity = 1,
                    IsParkingLane = true
                };
                RegisterZone(bay);
                int lane = laneIds[shape.BayLaneIndex[i]];
                LinkZones(lane, bay.ZoneId);   // pull in
                LinkZones(bay.ZoneId, lane);   // pull out
            }

            // Exit: straight into LeftVert_BotConn, or through a reserved corridor when the lane runs
            // west of the connector (see FactoryLayoutManager.ComputeLaneShape).
            int prev = laneIds[laneIds.Length - 1];
            for (int j = 0; j < shape.ExitCentres.Length; j++)
            {
                var ez = new TrafficZone
                {
                    ZoneId = nextZoneId++,
                    Name = $"LaneExit_{j}",
                    AisleType = AisleType.SpineAisle,
                    Flow = j == 0 ? FlowDirection.North : FlowDirection.East,
                    Centre = shape.ExitCentres[j],
                    Size = new Vector3(shape.Pitch, 0.1f, shape.RowDepth),
                    Capacity = 1,
                    IsParkingLane = true
                };
                RegisterZone(ez);
                LinkZones(prev, ez.ZoneId);
                prev = ez.ZoneId;
            }

            LinkZones(rightVert[rightVert.Length - 1], laneIds[0]);                    // entry
            LinkZones(prev, leftVert[leftVert.Length - 1]);                            // exit
        }

        private void BuildMultipleParkingZones(int[] leftVert, int[] rightVert)
        {
            foreach (var pa in layoutManager.ParkingAreas)
            {
                if (pa.RowAisleIndex < 0) continue;   // safety: skip a south-style entry

                var pz = new TrafficZone
                {
                    ZoneId = nextZoneId++,
                    Name = $"Parking_Aisle{pa.RowAisleIndex}_{(pa.IsLeftSide ? "L" : "R")}",
                    AisleType = AisleType.SpineAisle,  // (could add a dedicated AisleType.Parking for gizmo colour)
                    Flow = pa.IsLeftSide ? FlowDirection.West : FlowDirection.East,
                    Centre = pa.Position,
                    Size = new Vector3(layoutManager.ParkingAlcoveDepth, 0.1f, layoutManager.RowAisleWidth),
                    Capacity = 8   // tunable; split across alcoves instead of one pool of 64
                };
                pz.DockPoints[ParkingAreaId] = new DockPoint
                {
                    ApproachPosition = pa.Position,
                    HandshakePosition = pa.Position,
                    FacingDirection = pa.IsLeftSide ? Vector3.right : Vector3.left, // face inward
                    IsPickup = false
                };
                RegisterZone(pz);
                parkingZoneIds.Add(pz.ZoneId);

                // Hang the pocket off the exit-side vertical zone aligned with this aisle.
                int vIdx = pa.RowAisleIndex + 1;   // vertical seg 0=TopConn, 1..N=row levels, last=BotConn
                int vZone = pa.IsLeftSide ? leftVert[vIdx] : rightVert[vIdx];
                LinkZones(vZone, pz.ZoneId);   // pull in
                LinkZones(pz.ZoneId, vZone);   // pull back out
            }
        }
        /// @brief Multi-source reverse BFS over one-way flow: the minimum number of zone hops
        ///        FROM every zone TO the nearest of the given target zones.
        /// @param targetZoneIds One or more destination zones (e.g. a machine's dock zones).
        /// @return Map of zoneId → hop count. Zones absent from the map cannot reach any target.
        /// @details Traverses Upstream so a single pass covers the whole fleet instead of one
        ///          forward BFS per AGV. Hop count mirrors GetRoute length minus one.
        public Dictionary<int, int> GetHopDistancesToNearest(IEnumerable<int> targetZoneIds)
        {
            var dist = new Dictionary<int, int>();
            var queue = new Queue<int>();
            bool intoLaneAllowed = false;
            foreach (int t in targetZoneIds) if (ZoneIsParkingLane(t)) intoLaneAllowed = true;

            foreach (int t in targetZoneIds)
            {
                if (!zoneById.ContainsKey(t) || dist.ContainsKey(t)) continue;
                dist[t] = 0;
                queue.Enqueue(t);
            }

            while (queue.Count > 0)
            {
                int cur = queue.Dequeue();
                int d = dist[cur];
                foreach (int pred in zoneById[cur].Upstream)
                {
                    if (dist.ContainsKey(pred)) continue;
                    // Same rule as GetRoute: the lane is not a shortcut for non-parking trips.
                    if (!intoLaneAllowed && zoneById[cur].IsParkingLane && !zoneById[pred].IsParkingLane) continue;
                    dist[pred] = d + 1;
                    queue.Enqueue(pred);
                }
            }
            return dist;
        }

        private bool ZoneIsParkingLane(int zoneId) => zoneById.TryGetValue(zoneId, out var z) && z.IsParkingLane;

        /// @brief Returns the zone hosting a given special dock (e.g. IncomingBeltId), or -1.
        public int GetZoneIdForDock(int dockKey)
        {
            foreach (TrafficZone zone in zones)
                if (zone.DockPoints.ContainsKey(dockKey)) return zone.ZoneId;
            return -1;
        }

        /// @brief Attempts to secure a spot in a zone for an AGV.
        /// @param zoneId The ID of the zone to enter.
        /// @param agvId The ID of the AGV requesting entry.
        /// @return True if capacity is available or AGV is already registered; false if full.
        /// @post If true, the AGV ID is added to the zone's occupant set.
        public bool TryReserve(int zoneId, int agvId)
        {
            if (!zoneById.TryGetValue(zoneId, out TrafficZone zone)) return false;
            if (zone.OccupantAgvIds.Contains(agvId)) return true;   // re-entry, no stat change

            if (zone.IsFull)
            {
                zone.BlockEvents++;    // ← NEW — zone was contended
                return false;
            }

            zone.OccupantAgvIds.Add(agvId);
            zone.TraversalCount++;     // ← NEW — successful entry
            return true;
        }
        /// <summary>
        /// Called by AGVController when it finally acquires a previously-blocked zone.
        /// Accumulates the wait duration on the zone that caused the block.
        /// </summary>
        public void RecordBlockTime(int zoneId, float blockTime)
        {
            if (blockTime <= 0f) return;
            if (zoneById.TryGetValue(zoneId, out TrafficZone zone))
                zone.TotalBlockTime += blockTime;
        }
        /// <summary>
        /// Zeros per-episode congestion counters on all zones.
        /// Called from FactoryOrchestrator.StartEpisode() before each run
        /// so that stats from prior episodes don't bleed through when the
        /// factory is reused (IsFactoryReady = true).
        /// </summary>
        public void ResetEpisodeStats()
        {
            foreach (TrafficZone zone in zones)
            {
                zone.TraversalCount = 0;
                zone.BlockEvents = 0;
                zone.TotalBlockTime = 0f;
            }
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

        /// @brief Zones a pickup from this machine is served from (its output-side dock in passthrough layouts,
        ///        otherwise all of its docks). Used to seed the AGV-to-pickup hop distance.
        public List<int> GetPickupZonesForMachine(int machineId)
        {
            return machinePickupZones.TryGetValue(machineId, out var list) ? list : GetZonesForMachine(machineId);
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
            bool intoLaneAllowed = ZoneIsParkingLane(toZoneId) || ZoneIsParkingLane(fromZoneId);
            var visited = new HashSet<int>(); var parent = new Dictionary<int, int>(); var queue = new Queue<int>();
            queue.Enqueue(fromZoneId); visited.Add(fromZoneId);

            while (queue.Count > 0)
            {
                int current = queue.Dequeue();
                foreach (int next in zoneById[current].Downstream)
                {
                    if (visited.Contains(next)) continue;
                    if (!intoLaneAllowed && zoneById[next].IsParkingLane && !zoneById[current].IsParkingLane) continue;
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