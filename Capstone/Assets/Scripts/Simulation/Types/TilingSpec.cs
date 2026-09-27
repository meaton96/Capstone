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
    /// side by side west to east, each with its own machines, belts, parking lane, zone graph and AGVs. Phase 1:
    /// a job's operations stay in its home tile and an AGV serves one tile only. Tiles = 1 is today's floor.
    /// Immutable, shared by reference across per-seed config clones (like LayoutSpec).
    /// </summary>
    public sealed class TilingSpec
    {
        /// <summary>Space between adjacent tiles' outer walls: room for phase 2's transfer belts.</summary>
        public const float DefaultSeamGap = 3f;

        public static readonly TilingSpec Single = new TilingSpec(1, 0, ReleaseRule.RoundRobin, DefaultSeamGap);

        private static readonly string[] JsonKeys = { "tiles", "machinesPerTile", "jobScope", "agvAssignment", "releaseRule", "seamGap" };

        public int Tiles { get; }
        /// <summary>Machines per tile; 0 = machine count / Tiles.</summary>
        public int MachinesPerTile { get; }
        public ReleaseRule Release { get; }
        public float SeamGap { get; }

        /// <summary>Only "tile" is built (phase 1); "open" is phase 2 (cross-tile jobs via transfer belts).</summary>
        public string JobScope => "tile";
        /// <summary>Only "tile" is built (phase 1); "pooled" is phase 3 (linked spines).</summary>
        public string AgvAssignment => "tile";
        public bool IsTiled => Tiles > 1;
        public string ReleaseString => ReleaseToString(Release);

        private TilingSpec(int tiles, int machinesPerTile, ReleaseRule release, float seamGap)
        {
            if (tiles < 1) throw new ArgumentException($"tiling.tiles must be >= 1 (got {tiles}).");
            if (machinesPerTile < 0) throw new ArgumentException($"tiling.machinesPerTile must be >= 0 (got {machinesPerTile}).");
            if (seamGap < 0f) throw new ArgumentException($"tiling.seamGap must be >= 0 (got {seamGap}).");
            Tiles = tiles; MachinesPerTile = machinesPerTile; Release = release; SeamGap = seamGap;
        }

        public TilingSpec WithTiles(int tiles) => new TilingSpec(tiles, tiles == Tiles ? MachinesPerTile : 0, Release, SeamGap);
        public TilingSpec WithRelease(ReleaseRule release) => new TilingSpec(Tiles, MachinesPerTile, release, SeamGap);

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

            string scope = obj["jobScope"]?.Value<string>() ?? "tile";
            if (scope != "tile")
                throw new NotSupportedException($"tiling.jobScope \"{scope}\" is not built yet (phase 2); only \"tile\" is supported.");
            string agvs = obj["agvAssignment"]?.Value<string>() ?? "tile";
            if (agvs != "tile")
                throw new NotSupportedException($"tiling.agvAssignment \"{agvs}\" is not built yet (phase 3); only \"tile\" is supported.");

            return new TilingSpec(
                obj["tiles"]?.Value<int>() ?? 1,
                obj["machinesPerTile"]?.Value<int>() ?? 0,
                ParseRelease(obj["releaseRule"]?.Value<string>()),
                obj["seamGap"]?.Value<float>() ?? DefaultSeamGap);
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
        /// count divides evenly) and the same fleet, and only the options phase 1 supports are used. Called by
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
