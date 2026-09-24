using System;
using System.Collections.Generic;
using System.Linq;
using Newtonsoft.Json.Linq;

namespace Assets.Scripts.Simulation.Types
{
    /// <summary>Aisle topology axis of a layout. One-way (A-E) and two-way row aisles (F-J) are both built.</summary>
    public enum AisleTopology { OneWay, TwoWay }

    /// <summary>
    /// One of the named factory layouts A-J (docs/LAYOUT_CONFIGURATION_SCOPE.md). Immutable, so configs and their
    /// per-seed clones share instances. All layouts use the same machine grid and machine positions; they differ in
    /// which side of a machine its belts are on and (F-J) in the aisles.
    /// </summary>
    /// <remarks>
    /// <code>
    /// A  current layout: row 0 belts south, other rows north (interior machines are 4-belt, secondary pair unused)
    /// B  one-way aisles, every machine's belts north, 2-belt machines only
    /// C  as B, every machine's belts south
    /// D  passthrough, one-way aisles: input north, output south (the default layout — see Default)
    /// E  as D with input south, output north
    /// F-J  A-E's belt patterns (2-belt machines, F included) with two-way row aisles: each row aisle is two
    ///      stacked one-way lanes (north lane west, south lane east); the perimeter (spines and verticals)
    ///      stays one-way. A dock is served only from the lane beside it. See TrafficZoneManager.BuildZoneGraph.
    /// </code>
    /// JSON (optional; absent = <see cref="Default"/> (D), so a config only reproduces the pre-2026-09-21 floor
    /// (A) if it says so explicitly): <c>"layout": "B"</c> or <c>"layout": { "preset": "B" }</c>. "legacy" is
    /// accepted as an alias for A. A layout that is named here but whose machinery is not built yet is rejected
    /// at load by <see cref="EnsureBuildable"/>, never silently run as the default.
    /// </remarks>
    public sealed class LayoutSpec
    {
        /// <summary>Layout ids in canonical spelling.</summary>
        public static readonly string[] PresetNames = { "A", "B", "C", "D", "E", "F", "G", "H", "I", "J" };

        /// <summary>Layouts whose machinery exists. Widen as each one is built.</summary>
        public static readonly string[] BuiltIds = { "A", "B", "C", "D", "E", "F", "G", "H", "I", "J" };

        /// <summary>Keys allowed inside a "layout" block. Anything else is an error (a typo must not run the default layout).</summary>
        public static readonly string[] JsonKeys = { "preset" };

        /// <summary>Layout A: the pre-layout-config floor, kept selectable by name ("A" or "legacy") to reproduce old results.</summary>
        public static readonly LayoutSpec Legacy = new LayoutSpec("A");

        /// <summary>
        /// The layout a config gets when it omits "layout" entirely: D (passthrough, one-way), chosen 2026-09-22
        /// after the full A-E failures-on comparison (ties the most gridlock-robust layout at a 1:1 AGV:machine
        /// ratio, fastest flow time at moderate fleet sizes — see docs/LAYOUT_CONFIGURATION_SCOPE.md section 16).
        /// Old results (all layout A) need "layout": "A" (or -layout A) to reproduce.
        /// </summary>
        public static readonly LayoutSpec Default = new LayoutSpec("D");

        /// <summary>Canonical id used in logs and results ("A" .. "J").</summary>
        public string Id { get; }

        public AisleTopology Aisles => Id[0] >= 'F' ? AisleTopology.TwoWay : AisleTopology.OneWay;

        /// <summary>True for layout A (the pre-layout-config floor, including its 4-belt interior machines).</summary>
        public bool IsLegacy => Id == "A";

        private LayoutSpec(string id) { Id = id; }

        // ── Grid ──

        /// <summary>
        /// Machine grid for a machine count: the same formula FactoryLayoutManager.BuildFloor uses, so anything that
        /// needs the row count can compute it at load time.
        /// </summary>
        public static (int cols, int rows) GridFor(int machineCount)
        {
            if (machineCount < 1) throw new ArgumentException($"Cannot lay out {machineCount} machines.");
            int cols = (int)Math.Ceiling(Math.Sqrt(machineCount));
            int rows = (int)Math.Ceiling(machineCount / (double)cols);
            return (cols, rows);
        }

        // ── Resolution ──

        /// <summary>Resolves a layout name (case-insensitive; "legacy" = A) and checks it is buildable. CLI override path.</summary>
        public static LayoutSpec FromPreset(string preset)
        {
            LayoutSpec spec = Parse(preset);
            spec.EnsureBuildable();
            return spec;
        }

        /// <summary>
        /// Parses an optional JSON "layout" value (a string, an object with "preset", or null/absent for
        /// <see cref="Default"/>) and checks it is buildable. Shared by ConfigLoader and ScenarioLoader so both
        /// regimes validate identically.
        /// </summary>
        public static LayoutSpec FromJson(JToken token)
        {
            if (token == null || token.Type == JTokenType.Null) return Default;

            if (token.Type == JTokenType.String) return FromPreset((string)token);

            if (!(token is JObject obj))
                throw new ArgumentException("\"layout\" must be a layout name (\"B\") or an object ({ \"preset\": \"B\" }).");

            foreach (var prop in obj.Properties())
                if (!JsonKeys.Contains(prop.Name))
                    throw new ArgumentException(
                        $"Unknown key \"{prop.Name}\" in \"layout\". Valid keys: {string.Join(", ", JsonKeys)}.");

            JToken p = obj["preset"];
            if (p == null || p.Type == JTokenType.Null) return Default;
            if (p.Type != JTokenType.String) throw new ArgumentException("\"preset\" must be a string.");
            return FromPreset((string)p);
        }

        /// <summary>Parses a layout name without the buildable check.</summary>
        public static LayoutSpec Parse(string preset)
        {
            string name = (preset ?? "").Trim();
            if (name.Equals("legacy", StringComparison.OrdinalIgnoreCase)) name = "A";
            string canonical = PresetNames.FirstOrDefault(p => string.Equals(p, name, StringComparison.OrdinalIgnoreCase));
            if (canonical == null)
                throw new ArgumentException(
                    $"Unknown layout '{preset}'. Valid layouts: {string.Join(", ", PresetNames)} (\"legacy\" = A).");
            return canonical == "A" ? Legacy : new LayoutSpec(canonical);
        }

        /// <summary>Rejects layouts whose machinery is not built yet (see <see cref="BuiltIds"/>).</summary>
        public void EnsureBuildable()
        {
            if (BuiltIds.Contains(Id)) return;
            string what = Aisles == AisleTopology.TwoWay ? "two-way aisles are not built yet"
                                                          : "this belt layout is not built yet";
            throw new NotSupportedException(
                $"Layout '{Id}' cannot run: {what}. Built layouts: {string.Join(", ", BuiltIds)}. " +
                "See docs/LAYOUT_CONFIGURATION_SCOPE.md.");
        }

        // ── Belt sides ──

        /// <summary>Belt side ('N' or 'S') of the input and output belts of a machine in the given grid row.</summary>
        public (char input, char output) BeltSides(int row)
        {
            // Two-way variants (F-J) reuse the belt pattern of A-E; only the aisles differ.
            char basis = Aisles == AisleTopology.TwoWay ? (char)(Id[0] - 5) : Id[0];
            switch (basis)
            {
                case 'A': return row == 0 ? ('S', 'S') : ('N', 'N');
                case 'B': return ('N', 'N');
                case 'C': return ('S', 'S');
                case 'D': return ('N', 'S');
                case 'E': return ('S', 'N');
                default: throw new InvalidOperationException($"No belt pattern for layout '{Id}'.");
            }
        }

        /// <summary>True for passthrough layouts (D, E, and their two-way variants): input and output belts on opposite sides.</summary>
        public bool IsPassthrough { get { var (i, o) = BeltSides(0); return i != o; } }

        /// <summary>True if the machine in this row has a belt on the north (or south) side.</summary>
        public bool HasBeltOn(int row, char side)
        {
            var (i, o) = BeltSides(row);
            return i == side || o == side;
        }

        /// <summary>Belt sides as "SS/NN/NN/..." (input,output per row) for results.</summary>
        public string DescribeBelts(int machineRows)
            => string.Join("/", Enumerable.Range(0, machineRows).Select(r => { var (i, o) = BeltSides(r); return $"{i}{o}"; }));

        public string AislesString => Aisles == AisleTopology.TwoWay ? "twoway" : "oneway";

        public override string ToString() => Id;
    }

    /// <summary>
    /// Typo guard for JSON configs. JsonUtility (ConfigLoader) silently ignores unknown keys, so a misspelt
    /// "layuot" would run layout A and poison a layout comparison; this reports them instead.
    /// </summary>
    public static class ConfigKeyCheck
    {
        /// <summary>
        /// Logs a warning for every top-level key not in <paramref name="known"/>. Keys starting with "_" are
        /// comments (batch files carry "_comment", "_phases") and are exempt.
        /// </summary>
        public static void WarnUnknownTopLevel(JObject obj, IEnumerable<string> known, string where)
        {
            var knownSet = new HashSet<string>(known);
            foreach (var prop in obj.Properties())
            {
                if (prop.Name.StartsWith("_") || knownSet.Contains(prop.Name)) continue;
                Logging.SimLogger.LogWarning(
                    $"[Config] {where}: unknown key \"{prop.Name}\" ignored (typo? it will NOT take effect).");
            }
        }
    }
}
