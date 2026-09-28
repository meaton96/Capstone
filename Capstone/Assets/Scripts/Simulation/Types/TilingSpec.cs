using System;
using System.Collections.Generic;
using System.Linq;
using Newtonsoft.Json.Linq;
using Assets.Scripts.Simulation.Machines;

namespace Assets.Scripts.Simulation.Types
{
    /// <summary>Which tile an arriving job is released into (TilingSpec.Release).</summary>
    public enum ReleaseRule { RoundRobin, LeastWip }

    /// <summary>
    /// Tiled floor (docs/features/TILED_LAYOUT_SCOPE.md): the floor is <see cref="Tiles"/> copies of one layout,
    /// side by side west to east, each with its own machines, belts, parking lane, zone graph and AGVs. Phase 1
    /// (jobScope "tile", agvAssignment "tile"): a job's operations stay in its home tile and an AGV serves one tile
    /// only. Linked floor (agvAssignment "pooled", scope section 10): the tiles' spines are joined across the
    /// seams into one zone graph and any AGV serves any tile; with jobScope "open" a job may also use any machine
    /// on the floor, so the floor is one coupled problem. Open jobs need pooled AGVs (tile-bound AGVs would need
    /// phase 2's transfer belts). Tiles = 1 is today's floor. Immutable, shared by reference across per-seed
    /// config clones (like LayoutSpec).
    /// </summary>
    public sealed class TilingSpec
    {
        /// <summary>Space between adjacent tiles' outer walls: room for phase 2's transfer belts.</summary>
        public const float DefaultSeamGap = 3f;

        public static readonly TilingSpec Single = new TilingSpec(1, 0, ReleaseRule.RoundRobin, DefaultSeamGap, false, false);

        private static readonly string[] JsonKeys = { "tiles", "machinesPerTile", "jobScope", "agvAssignment", "releaseRule", "seamGap" };

        public int Tiles { get; }
        /// <summary>Machines per tile; 0 = machine count / Tiles.</summary>
        public int MachinesPerTile { get; }
        public ReleaseRule Release { get; }
        public float SeamGap { get; }

        /// <summary>True for jobScope "open": a job's eligible machines are not confined to its home tile.</summary>
        public bool JobsOpen { get; }
        /// <summary>True for agvAssignment "pooled": spines linked across seams, any AGV serves any tile.</summary>
        public bool AgvsPooled { get; }

        /// <summary>"tile" (phase 1) or "open" (jobs use any machine; needs pooled AGVs).</summary>
        public string JobScope => JobsOpen ? "open" : "tile";
        /// <summary>"tile" (phase 1) or "pooled" (linked spines, shared fleet).</summary>
        public string AgvAssignment => AgvsPooled ? "pooled" : "tile";
        public bool IsTiled => Tiles > 1;
        public string ReleaseString => ReleaseToString(Release);

        private TilingSpec(int tiles, int machinesPerTile, ReleaseRule release, float seamGap, bool jobsOpen, bool agvsPooled)
        {
            if (tiles < 1) throw new ArgumentException($"tiling.tiles must be >= 1 (got {tiles}).");
            if (machinesPerTile < 0) throw new ArgumentException($"tiling.machinesPerTile must be >= 0 (got {machinesPerTile}).");
            if (seamGap < 0f) throw new ArgumentException($"tiling.seamGap must be >= 0 (got {seamGap}).");
            if (jobsOpen && !agvsPooled)
                throw new NotSupportedException("tiling.jobScope \"open\" needs agvAssignment \"pooled\": with tile-bound AGVs a " +
                                                "cross-tile job needs transfer belts (phase 2), which are not built.");
            Tiles = tiles; MachinesPerTile = machinesPerTile; Release = release; SeamGap = seamGap;
            JobsOpen = jobsOpen; AgvsPooled = agvsPooled;
        }

        public TilingSpec WithTiles(int tiles) => new TilingSpec(tiles, tiles == Tiles ? MachinesPerTile : 0, Release, SeamGap, JobsOpen, AgvsPooled);
        public TilingSpec WithRelease(ReleaseRule release) => new TilingSpec(Tiles, MachinesPerTile, release, SeamGap, JobsOpen, AgvsPooled);
        /// <summary>Overrides job scope and/or AGV assignment (null = keep); "open" alone implies "pooled".</summary>
        public TilingSpec WithScope(string jobScope, string agvAssignment)
        {
            bool open = jobScope == null ? JobsOpen : ParseJobScope(jobScope);
            bool pooled = agvAssignment == null ? (AgvsPooled || (open && jobScope != null)) : ParseAgvAssignment(agvAssignment);
            return new TilingSpec(Tiles, MachinesPerTile, Release, SeamGap, open, pooled);
        }

        /// <summary>Machines in each tile for a floor of <paramref name="machineCount"/> machines.</summary>
        public int MachinesPerTileFor(int machineCount) => MachinesPerTile > 0 ? MachinesPerTile : machineCount / Tiles;

        /// <summary>Parses the optional scenario / config "tiling" object; null or absent = one tile.</summary>
        public static TilingSpec FromJson(JToken token)
        {
            if (token == null || token.Type == JTokenType.Null) return Single;
            if (!(token is JObject obj))
                throw new ArgumentException("\"tiling\" must be an object, e.g. { \"tiles\": 7 }.");
            foreach (var prop in obj.Properties())
                if (!JsonKeys.Contains(prop.Name))
                    throw new ArgumentException($"Unknown key \"{prop.Name}\" in \"tiling\". Valid keys: {string.Join(", ", JsonKeys)}.");

            return new TilingSpec(
                obj["tiles"]?.Value<int>() ?? 1,
                obj["machinesPerTile"]?.Value<int>() ?? 0,
                ParseRelease(obj["releaseRule"]?.Value<string>()),
                obj["seamGap"]?.Value<float>() ?? DefaultSeamGap,
                ParseJobScope(obj["jobScope"]?.Value<string>()),
                ParseAgvAssignment(obj["agvAssignment"]?.Value<string>()));
        }

        /// <summary>"tile" (default) or "open", case-insensitive: true for open.</summary>
        public static bool ParseJobScope(string value)
        {
            string v = (value ?? "tile").Trim().ToLowerInvariant();
            if (v == "tile") return false;
            if (v == "open") return true;
            throw new ArgumentException($"Invalid tiling.jobScope '{value}'. Valid values: tile, open.");
        }

        /// <summary>"tile" (default) or "pooled", case-insensitive: true for pooled.</summary>
        public static bool ParseAgvAssignment(string value)
        {
            string v = (value ?? "tile").Trim().ToLowerInvariant();
            if (v == "tile") return false;
            if (v == "pooled") return true;
            throw new ArgumentException($"Invalid tiling.agvAssignment '{value}'. Valid values: tile, pooled.");
        }

        /// <summary>"roundRobin" (default) or "leastWip", case-insensitive; throws on anything else.</summary>
        public static ReleaseRule ParseRelease(string value)
        {
            string v = (value ?? "roundRobin").Trim().ToLowerInvariant();
            if (v == "roundrobin") return ReleaseRule.RoundRobin;
            if (v == "leastwip") return ReleaseRule.LeastWip;
            throw new ArgumentException($"Invalid tiling.releaseRule '{value}'. Valid values: roundRobin, leastWip.");
        }

        public static string ReleaseToString(ReleaseRule r) => r == ReleaseRule.LeastWip ? "leastWip" : "roundRobin";

        /// <summary>
        /// Checks a tiled floor can be built from this config: every tile gets the same machines (each type's
        /// count divides evenly) and the same fleet, and only the options tiling supports are used. Called by
        /// FactoryLayoutManager.BuildFloor after CLI overrides are applied. No-op for one tile.
        /// </summary>
        public void Validate(MachineType[] machineTypes, int agvCount, string parkingMethod, string ioDocks)
        {
            if (!IsTiled) return;
            int machineCount = machineTypes.Length;
            int perTile = MachinesPerTileFor(machineCount);
            if (perTile < 1 || perTile * Tiles != machineCount)
                throw new ArgumentException($"tiling: {machineCount} machines cannot be split into {Tiles} tiles" +
                                            (MachinesPerTile > 0 ? $" of {MachinesPerTile}." : " evenly."));
            foreach (var g in machineTypes.GroupBy(t => t))
                if (g.Count() % Tiles != 0)
                    throw new ArgumentException($"tiling: {g.Count()} {g.Key} machines do not divide evenly over {Tiles} tiles.");
            if (agvCount % Tiles != 0 || agvCount < Tiles)
                throw new ArgumentException($"tiling: the AGV count ({agvCount}) must be a positive multiple of the tile count ({Tiles}).");
            if (!string.Equals(parkingMethod, "lane", StringComparison.OrdinalIgnoreCase))
                throw new NotSupportedException($"tiling needs parkingMethod \"lane\" (got \"{parkingMethod}\").");
            if (!string.Equals(ioDocks ?? "corner", "corner", StringComparison.OrdinalIgnoreCase))
                throw new NotSupportedException($"tiling needs ioDocks \"corner\" (got \"{ioDocks}\").");
        }
    }
}
