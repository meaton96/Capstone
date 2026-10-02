using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using Newtonsoft.Json.Linq;
using Assets.Scripts.Simulation.Machines;

namespace Assets.Scripts.Simulation.Types
{
    /// <summary>
    /// Which tile an arriving job is released into (TilingSpec.Release). Weighted sends jobs to tiles in proportion
    /// to TilingSpec.ReleaseWeights, deterministically (smooth weighted round-robin), so a floor can be loaded
    /// unevenly on purpose.
    /// </summary>
    public enum ReleaseRule { RoundRobin, LeastWip, Weighted }

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

        public static readonly TilingSpec Single = new TilingSpec(1, 0, ReleaseRule.RoundRobin, DefaultSeamGap, false, false, null);

        private static readonly string[] JsonKeys =
            { "tiles", "machinesPerTile", "jobScope", "agvAssignment", "releaseRule", "releaseWeights", "seamGap" };

        public int Tiles { get; }
        /// <summary>Machines per tile; 0 = machine count / Tiles.</summary>
        public int MachinesPerTile { get; }
        public ReleaseRule Release { get; }
        /// <summary>One weight per tile for ReleaseRule.Weighted (null otherwise): tile t receives weight[t] / sum of
        ///          the arriving jobs.</summary>
        public IReadOnlyList<float> ReleaseWeights { get; }
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
        /// <summary>The weights as "3;2;1" (results.csv), "" when the release rule is not weighted.</summary>
        public string ReleaseWeightsString => ReleaseWeights == null ? ""
            : string.Join(";", ReleaseWeights.Select(w => w.ToString("R", CultureInfo.InvariantCulture)));

        private TilingSpec(int tiles, int machinesPerTile, ReleaseRule release, float seamGap, bool jobsOpen, bool agvsPooled,
                           IReadOnlyList<float> releaseWeights)
        {
            if (tiles < 1) throw new ArgumentException($"tiling.tiles must be >= 1 (got {tiles}).");
            if (machinesPerTile < 0) throw new ArgumentException($"tiling.machinesPerTile must be >= 0 (got {machinesPerTile}).");
            if (seamGap < 0f) throw new ArgumentException($"tiling.seamGap must be >= 0 (got {seamGap}).");
            if (jobsOpen && !agvsPooled)
                throw new NotSupportedException("tiling.jobScope \"open\" needs agvAssignment \"pooled\": with tile-bound AGVs a " +
                                                "cross-tile job needs transfer belts (phase 2), which are not built.");
            if (release == ReleaseRule.Weighted)
            {
                if (releaseWeights == null || releaseWeights.Count != tiles)
                    throw new ArgumentException($"tiling.releaseRule \"weighted\" needs releaseWeights with one weight per tile " +
                                                $"({tiles}), got {releaseWeights?.Count ?? 0}.");
                if (releaseWeights.Any(w => !(w > 0f) || float.IsInfinity(w)))
                    throw new ArgumentException($"tiling.releaseWeights must all be > 0 and finite (got {string.Join(", ", releaseWeights)}).");
            }
            else if (releaseWeights != null)
                throw new ArgumentException("tiling.releaseWeights needs releaseRule \"weighted\".");
            Tiles = tiles; MachinesPerTile = machinesPerTile; Release = release; SeamGap = seamGap;
            JobsOpen = jobsOpen; AgvsPooled = agvsPooled;
            ReleaseWeights = releaseWeights?.ToArray();
        }

        public TilingSpec WithTiles(int tiles) => new TilingSpec(tiles, tiles == Tiles ? MachinesPerTile : 0, Release, SeamGap, JobsOpen, AgvsPooled, ReleaseWeights);
        /// <summary>Overrides the release rule; @p weights is required for Weighted unless this spec already has weights.</summary>
        public TilingSpec WithRelease(ReleaseRule release, IReadOnlyList<float> weights = null)
            => new TilingSpec(Tiles, MachinesPerTile, release, SeamGap, JobsOpen, AgvsPooled,
                              release == ReleaseRule.Weighted ? weights ?? ReleaseWeights : null);
        /// <summary>Overrides job scope and/or AGV assignment (null = keep); "open" alone implies "pooled".</summary>
        public TilingSpec WithScope(string jobScope, string agvAssignment)
        {
            bool open = jobScope == null ? JobsOpen : ParseJobScope(jobScope);
            bool pooled = agvAssignment == null ? (AgvsPooled || (open && jobScope != null)) : ParseAgvAssignment(agvAssignment);
            return new TilingSpec(Tiles, MachinesPerTile, Release, SeamGap, open, pooled, ReleaseWeights);
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

            float[] weights = null;
            JToken w = obj["releaseWeights"];
            if (w != null && w.Type != JTokenType.Null)
            {
                if (!(w is JArray arr))
                    throw new ArgumentException("tiling.releaseWeights must be an array of numbers, one per tile.");
                weights = arr.Select(t => t.Value<float>()).ToArray();
            }

            return new TilingSpec(
                obj["tiles"]?.Value<int>() ?? 1,
                obj["machinesPerTile"]?.Value<int>() ?? 0,
                ParseRelease(obj["releaseRule"]?.Value<string>()),
                obj["seamGap"]?.Value<float>() ?? DefaultSeamGap,
                ParseJobScope(obj["jobScope"]?.Value<string>()),
                ParseAgvAssignment(obj["agvAssignment"]?.Value<string>()),
                weights);
        }

        /// <summary>Comma-separated release weights ("3,2,1,1"), for the -releaseweights CLI flag.</summary>
        public static float[] ParseWeights(string csv)
        {
            string[] parts = (csv ?? "").Split(new[] { ',', ';' }, StringSplitOptions.RemoveEmptyEntries);
            var weights = new float[parts.Length];
            for (int i = 0; i < parts.Length; i++)
                if (!float.TryParse(parts[i].Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out weights[i]))
                    throw new ArgumentException($"Invalid release weight '{parts[i]}' in '{csv}': expected numbers > 0.");
            return weights;
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

        /// <summary>"roundRobin" (default), "leastWip" or "weighted", case-insensitive; throws on anything else.</summary>
        public static ReleaseRule ParseRelease(string value)
        {
            string v = (value ?? "roundRobin").Trim().ToLowerInvariant();
            if (v == "roundrobin") return ReleaseRule.RoundRobin;
            if (v == "leastwip") return ReleaseRule.LeastWip;
            if (v == "weighted") return ReleaseRule.Weighted;
            throw new ArgumentException($"Invalid tiling.releaseRule '{value}'. Valid values: roundRobin, leastWip, weighted.");
        }

        public static string ReleaseToString(ReleaseRule r)
            => r == ReleaseRule.LeastWip ? "leastWip" : r == ReleaseRule.Weighted ? "weighted" : "roundRobin";

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
