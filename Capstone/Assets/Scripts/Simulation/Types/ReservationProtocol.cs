using System;

namespace Assets.Scripts.Simulation.Types
{
    /// <summary>
    /// How an AGV holds traffic-zone reservations as it moves (AGVController.OnEnteredZone).
    /// Independent of the floor layout: layout decides which zones exist, this decides how many an
    /// AGV keeps reserved behind it. Kept a separate setting so their effects stay separable.
    /// </summary>
    public enum ReservationProtocol
    {
        /// <summary>Hold the current AND previous zone (released one zone later). Two slots per AGV. The default
        /// until 2026-09-26; deadlocks at 12+ AGVs on most layouts (docs/experiments/E1_E2_findings.md section 3).</summary>
        HoldPrevious,
        /// <summary>
        /// Default since 2026-09-26. Release the zone just left as soon as the next zone's centre is reached (or once
        /// the AGV is clear of it). One slot per AGV. Its body-overlap bug was fixed 2026-09-24; the full E1 grid
        /// (864 runs) then had 0 collisions and 0 deadlocks from 3 to 15 AGVs on every layout
        /// (docs/experiments/E1_E2_findings.md section 6b).
        /// </summary>
        ReleasePrevious,
    }

    public static class ReservationProtocolParser
    {
        public const string Default = "releasePrevious";

        /// <summary>Parses a config/CLI string. Throws on unknown values so a typo cannot silently run the default.</summary>
        public static ReservationProtocol Parse(string value)
        {
            switch ((value ?? Default).Trim().ToLowerInvariant())
            {
                case "holdprevious": return ReservationProtocol.HoldPrevious;
                case "releaseprevious": return ReservationProtocol.ReleasePrevious;
                default:
                    throw new ArgumentException(
                        $"Invalid reservationProtocol '{value}'. Valid values: holdPrevious, releasePrevious.");
            }
        }

        /// <summary>Returns the value (or the default when null) after checking it parses; throws otherwise.</summary>
        public static string Validated(string value)
        {
            if (string.IsNullOrWhiteSpace(value)) return Default;   // absent (JsonUtility may give "")
            Parse(value);
            return value;
        }

        public static string ToConfigString(ReservationProtocol p) =>
            p == ReservationProtocol.ReleasePrevious ? "releasePrevious" : "holdPrevious";
    }

    /// <summary>
    /// Command-line overrides applied to every config as it is loaded, from a single point
    /// (FactoryOrchestrator.ApplyConfigOverrides), so an override cannot be missed on one of the
    /// several batch/scenario/Python entry points. Null = no override. Set once by HeadlessBatchRunner.
    /// </summary>
    public static class ConfigOverrides
    {
        public static string ReservationProtocol;
        /// <summary>Routing trigger ("onTransport" | "onReady"); overrides every config's routingTrigger.</summary>
        public static string RoutingTrigger;
        /// <summary>Parking method ("single" | "multiple" | "lane"); overrides every config's parkingMethod.</summary>
        public static string ParkingMethod;
        /// <summary>Layout name A-J (see LayoutSpec); overrides every config's "layout" block. CLI over JSON over legacy.</summary>
        public static string Layout;
        /// <summary>Input/output belt docks ("corner" | "siding" | "bypass"); overrides every config's ioDocks.</summary>
        public static string IoDocks;
        /// <summary>Tile count (TilingSpec.Tiles); overrides every config's tiling.tiles. Null = no override.</summary>
        public static int? Tiles;
        /// <summary>Tiled floors: which tile an arriving job enters ("roundRobin" | "leastWip" | "weighted").</summary>
        public static string ReleaseRule;
        /// <summary>tiling.releaseWeights for every config (one per tile; needs release rule "weighted"). Null = keep.</summary>
        public static float[] ReleaseWeights;
        /// <summary>TECT travel price λ (FJSSPConfig.TravelPrice, >= 0); overrides every config's travelPrice.</summary>
        public static float? TravelPrice;
        /// <summary>tiling.jobScope ("tile" | "open") for every config. Null = no override.</summary>
        public static string JobScope;
        /// <summary>tiling.agvAssignment ("tile" | "pooled") for every config. Null = no override.</summary>
        public static string AgvAssignment;
        /// <summary>Machine flexibility probability [0, 1]; overrides every config's machineFlexibilityProbability.</summary>
        public static float? MachineFlexibility;
        /// <summary>Secondary-capability processing-time factor (> 0); overrides every config's secondaryTimeMultiplier.</summary>
        public static float? SecondaryTimeMultiplier;

        /// <summary>Throws on an unknown parking method so a typo cannot silently run the default.</summary>
        public static string ValidatedParkingMethod(string value)
        {
            string v = (value ?? "").Trim().ToLowerInvariant();
            if (v != "single" && v != "multiple" && v != "lane")
                throw new ArgumentException($"Invalid parkingMethod '{value}'. Valid values: single, multiple, lane.");
            return v;
        }

        /// <summary>Throws on an unknown ioDocks value so a typo cannot silently run the default.</summary>
        public static string ValidatedIoDocks(string value)
        {
            string v = (value ?? "").Trim().ToLowerInvariant();
            if (v != "corner" && v != "siding" && v != "bypass")
                throw new ArgumentException($"Invalid ioDocks '{value}'. Valid values: corner, siding, bypass.");
            return v;
        }
    }
}
