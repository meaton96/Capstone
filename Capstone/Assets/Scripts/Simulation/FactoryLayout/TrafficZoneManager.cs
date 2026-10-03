using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using Assets.Scripts.Simulation.Logging;
using Assets.Scripts.Simulation.Machines;
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
        [NonSerialized] public int BlockEvents;       // AGV waits that began on this zone (one per wait, not per retry)
        [NonSerialized] public float TotalBlockTime;    // cumulative wait time, every wait (incl. stalls, cancels, open at episode end)


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
        /// <summary>Special dock key of a tile's input / output belt. Tile 0 keeps -1 / -2 (IncomingBeltId /
        /// OutgoingBeltId); tile t uses -1-3t / -2-3t, which never collide with ParkingAreaId (-3).</summary>
        public static int IncomingDockKey(int tile) => IncomingBeltId - 3 * tile;
        public static int OutgoingDockKey(int tile) => OutgoingBeltId - 3 * tile;
        /// <summary>Zone id range of each tile's graph: [tile] = (first zone index, zone count).</summary>
        private readonly List<(int start, int count)> tileZoneRanges = new List<(int start, int count)>();
        /// <summary>Each tile's four perimeter corner zones, for the seam bridges of a linked floor.</summary>
        private readonly List<(int leftTop, int leftBot, int rightTop, int rightBot)> tileCorners = new List<(int, int, int, int)>();

        private readonly List<TrafficZone> zones = new List<TrafficZone>();
        private readonly Dictionary<int, TrafficZone> zoneById = new Dictionary<int, TrafficZone>();
        private int nextZoneId;
        private readonly Dictionary<int, List<int>> machineToZones = new Dictionary<int, List<int>>();
        // Zones a pickup from each machine is served from. Same list as machineToZones except in passthrough
        // layouts, where only the output-side dock counts (the input-side dock is for dropoffs).
        private readonly Dictionary<int, List<int>> machinePickupZones = new Dictionary<int, List<int>>();
        // EstimatePathLength cache, keyed (source dock key, target machine); cleared with the graph.
        private readonly Dictionary<(int, int), float> pathLengthCache = new Dictionary<(int, int), float>();

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
        /// Exit code when the zone graph fails CheckZoneGraph (user decision D6, 10-02: every check is fatal).
        public const int InvalidZoneGraphExitCode = 4;

        /// Linked-floor tiles whose parking-lane exit corridor runs into a seam (audit D1); fatal in CheckZoneGraph.
        private int _seamCorridorTiles;

        public void BuildZoneGraph()
        {
            zones.Clear();
            zoneById.Clear();
            machineToZones.Clear();
            machinePickupZones.Clear();
            pathLengthCache.Clear();
            parkingZoneIds.Clear();
            nextZoneId = 0;
            _seamCorridorTiles = 0;

            if (layoutManager == null || layoutManager.LayoutRows == 0)
            {
                SimLogger.Error("[TrafficZones] Layout manager missing or layout not built.");
                return;
            }

            // Tiled floor: one independent graph per tile, built by the same code around each tile's centre
            // (layoutManager.BuildTile), zone names prefixed T{t}_. One tile: built exactly as before, no prefix.
            tileZoneRanges.Clear();
            tileCorners.Clear();
            int tiles = layoutManager.TileCount;
            for (int t = 0; t < tiles; t++)
            {
                layoutManager.BuildTile = t;
                int start = zones.Count;
                BuildTileGraph(t);
                if (tiles > 1)
                    for (int k = start; k < zones.Count; k++) zones[k].Name = $"T{t}_{zones[k].Name}";
                tileZoneRanges.Add((start, zones.Count - start));
            }
            layoutManager.BuildTile = 0;
            if (layoutManager.AgvsPooled) BuildSeamBridges();
            SimLogger.Medium($"[TrafficZones] Built zone graph: {zones.Count} zones" +
                             $"{(layoutManager.ActiveLayout != null && layoutManager.ActiveLayout.Aisles == Types.AisleTopology.TwoWay ? " (two-way row aisles)" : "")}" +
                             $"{(layoutManager.IsPerimeterTwoWay ? " (two-way perimeter)" : "")}" +
                             $"{(tiles > 1 ? $" in {tiles} {(layoutManager.AgvsPooled ? "linked " : "")}tiles" : "")}.");
            CheckZoneGraph();
        }

        /// @brief Builds one tile's zones, links, docks and parking (the whole floor when untiled).
        private void BuildTileGraph(int tile)
        {
            int rows = layoutManager.LayoutRows;
            int cols = layoutManager.LayoutCols;
            if (layoutManager.IsPerimeterTwoWay) { BuildPerimeterTwoWayGraph(tile, rows, cols); return; }
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
            RegisterDockPoints(tile, rowLaneN, rowLaneS, topSpineZones, botSpineZones, rows, cols, leftVertZones, rightVertZones,
                               topSpineZones[0], botSpineZones[botSpineZones.Length - 1]);
            tileCorners.Add((leftVertZones[0], leftVertZones[leftVertZones.Length - 1],
                             rightVertZones[0], rightVertZones[rightVertZones.Length - 1]));
            if (layoutManager.ActiveIoDocks != IoDockMethod.Corner) BuildIoSidings(leftChain, rightChain, topSpineZones);
        }

        /// @brief Two-way perimeter (layouts K-O): one-way row aisles inside two concentric one-way perimeter rings.
        /// @details Every spine and vertical is two stacked lanes. The OUTER lane (away from the machines) is the
        ///          clockwise loop every other layout has (left north, top east, right south, bottom west) and keeps
        ///          the corner zone names (LeftVert_TopConn ...), the belts, the parking lane and the seam bridges. The
        ///          INNER lane (beside the machines, zone names with "In") runs counter-clockwise and carries everything
        ///          that touches the machine area: row aisles start and end on it, and the row-0 / last-row spine docks
        ///          are on it.
        ///
        ///          Lane changes are exit-only, so no short cycle forms (docs/LAYOUT_CONFIGURATION_SCOPE.md section 20):
        ///          - a row aisle starts on the inner lane (outer traffic reaches it through the changes below);
        ///          - at its downstream end a row leaves onto the inner lane, or crosses it onto the outer lane
        ///            (inner Row junction -> outer Row junction). Inner through traffic may take that crossing too;
        ///          - the outer ring feeds the inner ring at two corner crossovers: south-west (the outer left
        ///            lane, heading north, into the inner south-west corner, which heads east) and north-east (outer
        ///            right, heading south, into the inner north-east corner, which heads west); and at a row's
        ///            upstream end only where that cannot close a short cycle (right vertical: above every exit
        ///            change; left: below every one). On the 4 x 4 tile that is aisle 0's east end.
        ///          On the left vertical the inner lane flows south and the outer north, so a short cycle would need an
        ///          inner-to-outer change BELOW an outer-to-inner one; the only outer-to-inner change there is at the
        ///          bottom. The right vertical mirrors this with its crossover at the top. Every remaining cycle runs
        ///          through a row aisle or around a ring, so the girth is not below the one-way floor's.
        ///
        ///          Outer vertical junctions: [corner, spine-row pass, Row0.., spine-row pass, corner]; inner vertical
        ///          junctions: [inner corner, Row0.., inner corner] (the same shape as a one-way vertical). The outer
        ///          spine lanes have a pass zone above / below each inner vertical lane.
        private void BuildPerimeterTwoWayGraph(int tile, int rows, int cols)
        {
            int[][] rowAisles = BuildRowAisleZones(rows, cols);
            Vector3 floorCentre = layoutManager.TileOrigin;
            float halfW = ((cols - 1) * layoutManager.MachineSpacingX) / 2f + layoutManager.MachineDepth / 2f;
            float xc = halfW + layoutManager.VerticalAisleWidth / 2f, vo = layoutManager.VerticalLaneOffset;
            float xIn = xc - vo, xOut = xc + vo;
            float so = layoutManager.SpineLaneOffset, sw = layoutManager.SpineLaneWidth;
            float zInT = layoutManager.GetTopSpineZ() - so, zOutT = layoutManager.GetTopSpineZ() + so;
            float zInB = layoutManager.GetBottomSpineZ() + so, zOutB = layoutManager.GetBottomSpineZ() - so;

            (int[] junctions, int[] chain) Vertical(bool left, bool outer)
            {
                var names = new List<string>(); var zs = new List<float>(); var hs = new List<float>();
                void J(string n, float z, float h) { names.Add(n); zs.Add(z); hs.Add(h); }
                if (outer) { J("TopConn", zOutT, sw); J("TopPass", zInT, sw); }
                else J("TopConn", zInT, sw);
                for (int a = 0; a < rows - 1; a++)
                    J($"Row{a}", layoutManager.GetRowAisleCentre(a).z - floorCentre.z, layoutManager.RowAisleWidth);
                if (outer) { J("BotPass", zInB, sw); J("BotConn", zOutB, sw); }
                else J("BotConn", zInB, sw);
                string side = (left ? "LeftVert" : "RightVert") + (outer ? "" : "In");
                // Outer ring clockwise: left north, right south. Inner ring counter-clockwise: left south, right north.
                FlowDirection flow = left == outer ? FlowDirection.North : FlowDirection.South;
                return BuildVerticalChain(side, (left ? -1f : 1f) * (outer ? xOut : xIn), flow, names.ToArray(), zs.ToArray(), hs.ToArray());
            }
            var (loJ, loChain) = Vertical(true, true);
            var (roJ, roChain) = Vertical(false, true);
            var (liJ, liChain) = Vertical(true, false);
            var (riJ, riChain) = Vertical(false, false);
            int last = loJ.Length - 1;   // outer: [TopConn, TopPass, Row0.., BotPass, BotConn]; inner: [TopConn, Row0.., BotConn]

            int[] topOut = BuildSpineChain("TopSpine", cols, zOutT, FlowDirection.East, loJ[0], roJ[0], xIn);
            int[] botOut = BuildSpineChain("BotSpine", cols, zOutB, FlowDirection.West, loJ[last], roJ[last], xIn);
            int[] topIn = BuildSpineChain("TopSpineIn", cols, zInT, FlowDirection.West, liJ[0], riJ[0], float.NaN);
            int[] botIn = BuildSpineChain("BotSpineIn", cols, zInB, FlowDirection.East, liJ[liJ.Length - 1], riJ[riJ.Length - 1], float.NaN);

            void Forward(int[] c) { for (int k = 0; k < c.Length - 1; k++) LinkZones(c[k], c[k + 1]); }
            void Backward(int[] c) { for (int k = c.Length - 1; k > 0; k--) LinkZones(c[k], c[k - 1]); }

            for (int a = 0; a < rowAisles.Length; a++) WireRowLanes(a, rowAisles, rowAisles, false, liJ, riJ, junctions: false);
            // Outer ring, clockwise.
            Backward(loChain); Forward(topOut); Forward(roChain); Backward(botOut);
            // Inner ring, counter-clockwise.
            Forward(liChain); Forward(botIn); Backward(riChain); Backward(topIn);
            // Rows start and end on the inner lane.
            for (int a = 0; a < rowAisles.Length; a++) WireRowLanes(a, rowAisles, rowAisles, false, liJ, riJ, junctions: true);
            // Exit-only lane changes: at a row's downstream end, inner lane -> outer lane.
            for (int a = 0; a < rowAisles.Length; a++)
            {
                if (layoutManager.GetRowAisleDirection(a).x > 0f) LinkZones(riJ[a + 1], roJ[a + 2]);
                else LinkZones(liJ[a + 1], loJ[a + 2]);
            }
            // Corner crossovers, outer -> inner: south-west (outer left BotPass -> inner BotConn) and
            // north-east (outer right TopPass -> inner TopConn).
            LinkZones(loJ[last - 1], liJ[liJ.Length - 1]);
            LinkZones(roJ[1], riJ[0]);
            // Entry changes, outer -> inner at a row's upstream end, only where no short cycle can form: on the right
            // vertical above its highest exit change, on the left below its lowest one (the corner crossovers are the
            // limiting case). This lets outer-lane traffic enter aisle 0 from the right without lapping the inner ring.
            int aisles = rowAisles.Length, topExitRight = aisles, lowExitLeft = -1;
            for (int a = 0; a < aisles; a++)
            {
                if (layoutManager.GetRowAisleDirection(a).x > 0f) topExitRight = Mathf.Min(topExitRight, a);
                else lowExitLeft = a;
            }
            for (int a = 0; a < aisles; a++)
            {
                bool east = layoutManager.GetRowAisleDirection(a).x > 0f;
                if (!east && a < topExitRight) LinkZones(roJ[a + 2], riJ[a + 1]);
                if (east && a > lowExitLeft) LinkZones(loJ[a + 2], liJ[a + 1]);
            }

            RegisterDockPoints(tile, rowAisles, rowAisles, topIn, botIn, rows, cols, loJ, roJ, loJ[0], roJ[last]);
            tileCorners.Add((loJ[0], loJ[last], roJ[0], roJ[last]));
        }

        /// @brief Linked floor (TilingSpec.AgvsPooled): joins neighbouring tiles' perimeters across each seam.
        /// @details Every tile's loop runs clockwise, so the top spines all flow east and the bottom spines west
        ///          (TILED_LAYOUT_SCOPE.md s2.1). Tile t's north-east corner (RightVert_TopConn) continues east over
        ///          the seam into tile t+1's north-west corner (LeftVert_TopConn), and tile t+1's south-west corner
        ///          continues west into tile t's south-east corner. Each corner becomes a split (turn south or carry
        ///          on east) or a merge; nothing crosses. The floor is then one eastbound top spine, one westbound
        ///          bottom spine and alternating north/south verticals, two per tile. Every new cycle runs through two
        ///          or more tiles, so it is longer than a tile's own loop and the girth does not drop. The seam floor
        ///          between the two corner zones is covered by Capacity 1 bridge zones at least MinVerticalPitch
        ///          wide (none when the seam is narrower, the corners are then linked directly).
        private void BuildSeamBridges()
        {
            for (int t = 0; t + 1 < tileCorners.Count; t++)
            {
                Bridge($"Seam{t}_Top", tileCorners[t].rightTop, tileCorners[t + 1].leftTop, FlowDirection.East);
                Bridge($"Seam{t}_Bot", tileCorners[t + 1].leftBot, tileCorners[t].rightBot, FlowDirection.West);
            }

            void Bridge(string name, int fromId, int toId, FlowDirection flow)
            {
                TrafficZone from = zoneById[fromId], to = zoneById[toId];
                float x0 = from.Centre.x + Mathf.Sign(to.Centre.x - from.Centre.x) * from.Size.x / 2f;   // seam edges
                float x1 = to.Centre.x - Mathf.Sign(to.Centre.x - from.Centre.x) * to.Size.x / 2f;
                float gap = Mathf.Abs(x1 - x0);
                int n = Mathf.FloorToInt(gap / MinVerticalPitch);
                int prev = fromId;
                for (int k = 0; k < n; k++)
                {
                    var z = new TrafficZone
                    {
                        ZoneId = nextZoneId++,
                        Name = $"{name}{k}",
                        AisleType = AisleType.SpineAisle,
                        Flow = flow,
                        Centre = new Vector3(Mathf.Lerp(x0, x1, (k + 0.5f) / n), from.Centre.y, from.Centre.z),
                        Size = new Vector3(gap / n, 0.1f, from.Size.z),
                        Capacity = 1
                    };
                    RegisterZone(z);
                    LinkZones(prev, z.ZoneId);
                    prev = z.ZoneId;
                }
                LinkZones(prev, toId);
            }
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

            int Reach(int from, bool forward)
            {
                var seen = new HashSet<int> { zones[from].ZoneId }; var q = new Queue<int>(seen);
                while (q.Count > 0)
                {
                    var z = zoneById[q.Dequeue()];
                    foreach (int n in forward ? z.Downstream : z.Upstream)
                        if (seen.Add(n)) q.Enqueue(n);
                }
                return seen.Count;
            }
            // Per tile: a tiled floor is deliberately several disconnected graphs (AGVs never leave their tile).
            // A linked floor is one graph, bridges included.
            var ranges = layoutManager.AgvsPooled ? new List<(int start, int count)> { (0, zones.Count) } : tileZoneRanges;
            bool stronglyConnected = true;
            foreach (var (start, count) in ranges)
                if (count > 0 && (Reach(start, true) != count || Reach(start, false) != count)) stronglyConnected = false;

            int misplacedDocks = 0;
            foreach (var z in zones)
                foreach (var kv in z.DockPoints)
                    if (kv.Key != ParkingAreaId && !z.Contains(kv.Value.ApproachPosition))
                    {
                        misplacedDocks++;
                        SimLogger.Error($"[TrafficZones] Dock {(kv.Key >= 0 ? $"for machine {kv.Key}" : $"for belt key {kv.Key}")} approaches outside its zone {z.Name}.");
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

            // (3) Zone centres inside another zone (audit D1). AGVs drive centre to centre, so a centre inside a
            // second zone means two AGVs holding different reservations can stand on the same floor. Edge slivers
            // where aisles meet (every layout, 0.25-0.75 m) are not counted. Known before D1: layout J's vertical
            // gap zones sat inside its row zones (12 per floor, 0 recorded collisions); fixed 10-02 in
            // BuildVerticalChain.
            const float centreEps = 0.05f;
            int centresInside = 0;
            foreach (var za in zones)
            {
                if (za.Name == "Parking_Alcove") continue;
                foreach (var zb in zones)
                {
                    if (ReferenceEquals(za, zb) || zb.Name == "Parking_Alcove") continue;
                    if (Mathf.Abs(za.Centre.x - zb.Centre.x) >= zb.Size.x / 2f - centreEps ||
                        Mathf.Abs(za.Centre.z - zb.Centre.z) >= zb.Size.z / 2f - centreEps) continue;
                    if (centresInside++ < 10)
                        SimLogger.Error($"[TrafficZones] Zone {za.Name}'s centre lies inside zone {zb.Name}.");
                }
            }

            string msg = $"[TrafficZones] Graph check: strongly connected{(ranges.Count > 1 ? " (per tile)" : "")}={stronglyConnected}, misplaced docks={misplacedDocks}, zone centres inside another zone={centresInside}, " +
                         (girth == int.MaxValue ? "no cycle" : $"girth={girth} (no deadlock below {(girth + 1) / 2} AGVs under holdPrevious).");
            msg += $" Seam-corridor tiles={_seamCorridorTiles}.";
            // D6 (user decision 10-02): any failed check stops the player. A floor with an unreachable zone, a
            // dock outside its zone or two zones sharing floor would run, but its results would not mean what
            // the layout claims, so no run is better than a wrong one.
            if (stronglyConnected && misplacedDocks == 0 && centresInside == 0 && _seamCorridorTiles == 0)
            {
                SimLogger.Medium(msg);
                return;
            }
            SimLogger.LogError($"{msg} Zone graph INVALID, stopping the player (exit code {InvalidZoneGraphExitCode}).");
#if UNITY_EDITOR
            UnityEditor.EditorApplication.isPlaying = false;
#else
            Application.Quit(InvalidZoneGraphExitCode);
#endif
            throw new InvalidOperationException("Zone graph failed CheckZoneGraph: " + msg);
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
            => BuildSpineChain(isTop ? "TopSpine" : "BotSpine", cols,
                               isTop ? layoutManager.GetTopSpineZ() : layoutManager.GetBottomSpineZ(),
                               isTop ? FlowDirection.East : FlowDirection.West, cornerLeftZoneId, cornerRightZoneId, float.NaN);

        /// @brief One spine lane: [cornerL, (PassL), Dock0, Transit0, ..., DockN-1, (PassR), cornerR] at local z @p z.
        /// @param passX NaN for none; otherwise the |x| of the inner vertical lane, where a two-way perimeter's outer
        /// spine lane gets a zone of its own (PassL / PassR) between its corner and the dock run. Dock zones then sit
        /// at indices 2 + 2k, so SpineDockIndex applies only to spines without pass zones (the only ones with docks).
        private int[] BuildSpineChain(string side, int cols, float z, FlowDirection flow, int cornerLeftZoneId, int cornerRightZoneId, float passX)
        {
            bool pass = !float.IsNaN(passX);
            int numDockTransit = 2 * cols - 1;
            var result = new List<int> { cornerLeftZoneId };
            Vector3 floorCentre = layoutManager.TileOrigin;
            float segWidth = layoutManager.MachineSpacingX;
            float subWidth = segWidth / 2f;
            float halfTotalWidth = ((cols - 1) * segWidth) / 2f;

            int AddZone(string name, float x, float width)
            {
                var zone = new TrafficZone
                {
                    ZoneId = nextZoneId++,
                    Name = $"{side}_{name}",
                    AisleType = AisleType.SpineAisle,
                    Flow = flow,
                    Centre = new Vector3(floorCentre.x + x, 0.01f, floorCentre.z + z),
                    Size = new Vector3(width, 0.1f, layoutManager.SpineLaneWidth),
                    Capacity = 1
                };
                RegisterZone(zone);
                return zone.ZoneId;
            }

            if (pass) result.Add(AddZone("PassL", -passX, layoutManager.VerticalLaneWidth));
            for (int k = 0; k < numDockTransit; k++)   // 0-based within the dock/transit run
            {
                bool isDock = k % 2 == 0;
                result.Add(AddZone(isDock ? $"Dock{k / 2}" : $"Transit{k / 2}", -halfTotalWidth + k * subWidth, subWidth));
            }
            if (pass) result.Add(AddZone("PassR", passX, layoutManager.VerticalLaneWidth));
            result.Add(cornerRightZoneId);
            return result.ToArray();
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
            float halfMachineAreaW = ((layoutManager.LayoutCols - 1) * layoutManager.MachineSpacingX) / 2f + layoutManager.MachineDepth / 2f;
            float x = isLeft ? -(halfMachineAreaW + layoutManager.VerticalAisleWidth / 2f) : (halfMachineAreaW + layoutManager.VerticalAisleWidth / 2f);
            FlowDirection flow = isLeft ? FlowDirection.North : FlowDirection.South;
            Vector3 floorCentre = layoutManager.TileOrigin;
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
            return BuildVerticalChain(side, x, flow, names, zs, heights);
        }

        /// @brief One vertical lane at local x @p x: a junction zone per (name, z, height), top to bottom, with evenly
        /// spaced gap zones (pitch >= MinVerticalPitch) between consecutive junctions. See BuildVerticalZones.
        private (int[] junctions, int[] chain) BuildVerticalChain(string side, float x, FlowDirection flow, string[] names, float[] zs, float[] heights)
        {
            int numJunctions = names.Length;
            int[] junctions = new int[numJunctions];
            var chain = new List<int>();
            Vector3 floorCentre = layoutManager.TileOrigin;

            // Gap zones per stretch (junction s to s+1), spaced centre to centre.
            int[] segments = new int[numJunctions - 1];
            float[] pitch = new float[numJunctions - 1];
            for (int s = 0; s + 1 < numJunctions; s++)
            {
                float gap = zs[s] - zs[s + 1];
                segments[s] = Mathf.Max(1, Mathf.FloorToInt(gap / MinVerticalPitch));
                pitch[s] = gap / segments[s];
            }

            for (int s = 0; s < numJunctions; s++)
            {
                // A junction taller than the gap pitch (two-way rows, layouts F-J: 6 m) used to contain the centre
                // of the neighbouring gap zone, so GetZoneAtPosition resolved an AGV standing in that gap zone to
                // the junction (registered first). Clip the junction box on such a side to end where the gap zone
                // begins, symmetrically so the centre (where AGVs drive) does not move. Centres, links and
                // registration order are unchanged; layouts whose junctions never contained a gap centre keep
                // their boxes exactly.
                float half = heights[s] / 2f;
                if (s > 0 && segments[s - 1] > 1 && half >= pitch[s - 1])
                    half = Mathf.Min(half, pitch[s - 1] / 2f);
                if (s + 1 < numJunctions && segments[s] > 1 && heights[s] / 2f >= pitch[s])
                    half = Mathf.Min(half, pitch[s] / 2f);
                junctions[s] = AddVerticalZone($"{side}_{names[s]}", x, zs[s], 2f * half, flow, floorCentre);
                chain.Add(junctions[s]);

                if (s == numJunctions - 1) break;
                for (int k = 1; k < segments[s]; k++)
                    chain.Add(AddVerticalZone($"{side}_Gap{s}_{k}", x, zs[s] - k * pitch[s], pitch[s], flow, floorCentre));
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
                Size = new Vector3(layoutManager.VerticalLaneWidth, 0.1f, height),
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
        /// @param inBeltZone, outBeltZone The zones serving the input belt (north-west corner) and output belt (south-east
        /// corner): the spine ends on a one-way perimeter, the outer lane's corners on a two-way one.
        private void RegisterDockPoints(int tile, int[][] rowLaneN, int[][] rowLaneS, int[] topSpine, int[] botSpine,
                                int rows, int cols, int[] leftVert, int[] rightVert, int inBeltZone, int outBeltZone)
        {
            ConveyorBelt inBelt = layoutManager.IncomingBeltOf(tile), outBelt = layoutManager.OutgoingBeltOf(tile);
            // Siding / bypass register the moved belt docks on their sidings instead (BuildIoSidings); bypass keeps
            // the output belt on its corner.
            bool inCorner = layoutManager.ActiveIoDocks == IoDockMethod.Corner;
            bool outCorner = layoutManager.ActiveIoDocks != IoDockMethod.Siding;
            if (inCorner && topSpine.Length > 0 && inBelt != null)
            {
                TrafficZone inZone = zoneById[inBeltZone];
                Vector3 handshake = inBelt.OutputEndPosition;
                inZone.DockPoints[IncomingDockKey(tile)] = new DockPoint { ApproachPosition = handshake - Vector3.forward * 1.5f, HandshakePosition = handshake, FacingDirection = Vector3.forward, IsPickup = true };
            }

            if (outCorner && botSpine.Length > 0 && outBelt != null)
            {
                TrafficZone outZone = zoneById[outBeltZone];
                Vector3 handshake = outBelt.InputEndPosition;
                outZone.DockPoints[OutgoingDockKey(tile)] = new DockPoint { ApproachPosition = handshake + Vector3.forward * 1.5f, HandshakePosition = handshake, FacingDirection = -Vector3.forward, IsPickup = false };
            }

            if (layoutManager.ActiveParkingMethod == ParkingMethod.Single)
            {
                BuildSingleParkingZone(botSpine, leftVert);
            }
            else if (layoutManager.ActiveParkingMethod == ParkingMethod.Lane)
            {
                var laneShape = layoutManager.LaneShapeOf(tile);
                // A fleet too big for the lane band routes its exit west of the left connector (LaneShapeOf). On a
                // linked floor, west of tile t>0's connector is the seam, where the bottom bridge zones sit: the
                // two sets of zones overlap, so two AGVs can hold the same floor under different reservations
                // (audit D1; latent, >= ~17 AGVs per tile, no reported run came close). CheckZoneGraph counts it.
                if (layoutManager.AgvsPooled && tile > 0 && laneShape.ExitCentres.Length > 0)
                {
                    _seamCorridorTiles++;
                    SimLogger.Error($"[TrafficZones] Tile {tile}: the parking-lane exit corridor runs into the seam " +
                                    $"of a linked floor (fleet too large for the lane band). Zones will overlap the " +
                                    $"seam bridges; use fewer AGVs per tile or an unlinked floor.");
                }
                BuildParkingLane(laneShape, leftVert, rightVert);
            }
            else
            {
                BuildMultipleParkingZones(leftVert, rightVert);
            }


            float standoff = 1.5f;
            int perTile = layoutManager.MachinesPerTile, first = tile * perTile;
            for (int i = first; i < first + perTile; i++)
            {
                int row = (i - first) / cols; int col = (i - first) % cols;
                Vector3 machinePos = layoutManager.Machines[i].transform.position;

                // Layout A registers a dock on each side that has a belt: interior machines (4-belt prefab) on
                // both sides, row 0 (2-belt, turned 180 deg) on the south only, the last row on the north only.
                // Before 10-02 (audit D2) row 0 also got a phantom top-spine dock with no belt. Other layouts
                // register a dock only on a side that has a belt, so the hop-distance dispatch heuristic never
                // seeds a dock nobody drives to, and the last row can dock on the bottom spine (layout C).
                LayoutSpec layout = layoutManager.ActiveLayout;
                bool southDock = layout.IsLegacy ? row < rowLaneN.Length : layout.HasBeltOn(row, 'S');
                bool northDock = layout.IsLegacy ? row > 0 : layout.HasBeltOn(row, 'N');

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
        private void BuildParkingLane(FactoryLayoutManager.ParkingLaneShape shape, int[] leftVert, int[] rightVert)
        {
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

            if (zone.IsFull) return false;

            zone.OccupantAgvIds.Add(agvId);
            zone.TraversalCount++;     // ← NEW — successful entry
            return true;
        }
        /// <summary>
        /// Called by AGVController once when a wait for this zone begins (not on each retry).
        /// </summary>
        public void RecordBlockStart(int zoneId)
        {
            if (zoneById.TryGetValue(zoneId, out TrafficZone zone))
                zone.BlockEvents++;
        }

        /// <summary>
        /// Called by AGVController whenever a wait for this zone ends, however it ends
        /// (acquired, stall recovery, redispatch, cancel) or at episode end. a previously-blocked zone.
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

        /// @brief Zone-graph path length (sum of zone-centre hops) from where a job is picked up to where it
        ///        would be dropped at a machine: the shortest over the source's pickup docks and the target's
        ///        dropoff docks. Ignores congestion. Cached per (source dock key, target) until the graph is rebuilt.
        /// @param fromMachineId The job's current machine, or -1 for the incoming belt of @p tile.
        /// @param tile          The job's home tile (used only when @p fromMachineId is -1).
        /// @param toMachineId   The candidate target machine.
        /// @return Path length in world units; float.MaxValue when no route exists.
        public float EstimatePathLength(int fromMachineId, int tile, int toMachineId)
        {
            int srcKey = fromMachineId >= 0 ? fromMachineId : IncomingDockKey(tile);
            var key = (srcKey, toMachineId);
            if (pathLengthCache.TryGetValue(key, out float cached)) return cached;

            List<int> srcZones;
            if (fromMachineId >= 0) srcZones = GetPickupZonesForMachine(fromMachineId);
            else
            {
                int z = GetZoneIdForDock(srcKey);
                srcZones = z >= 0 ? new List<int> { z } : new List<int>();
            }

            // Passthrough layouts drop off only at the input-side dock (the one that is not the pickup dock).
            List<int> allDst = GetZonesForMachine(toMachineId);
            List<int> pickupDst = GetPickupZonesForMachine(toMachineId);
            var dstZones = allDst.Where(z => !(pickupDst != allDst && pickupDst.Contains(z))).ToList();
            if (dstZones.Count == 0) dstZones = allDst;

            float best = float.MaxValue;
            foreach (int s in srcZones)
                foreach (int d in dstZones)
                {
                    List<int> route = GetRoute(s, d);
                    if (route.Count == 0) continue;
                    float len = 0f;
                    for (int k = 1; k < route.Count; k++)
                        len += Vector3.Distance(zoneById[route[k - 1]].Centre, zoneById[route[k]].Centre);
                    if (len < best) best = len;
                }

            pathLengthCache[key] = best;
            return best;
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