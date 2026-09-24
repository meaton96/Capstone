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
        /// <summary>Default. Hold the current AND previous zone (released one zone later). Two slots per AGV.</summary>
        HoldPrevious,
        /// <summary>
        /// Release the zone just left as soon as the next zone's centre is reached. One slot per AGV.
        /// Raises the gridlock threshold but produces AGV-AGV body overlaps (see
        /// docs/GRIDLOCK_INVESTIGATION_2026-09-19.md, section 8c), so treat results as non-physical
        /// until the overlap is designed out.
        /// </summary>
        ReleasePrevious,
    }

    public static class ReservationProtocolParser
    {
        public const string Default = "holdPrevious";

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
            if (value == null) return Default;
            Parse(value);
            return value;
        }

        public static string ToConfigString(ReservationProtocol p) =>
            p == ReservationProtocol.ReleasePrevious ? "releasePrevious" : Default;
    }

    /// <summary>
    /// Command-line overrides applied to every config as it is loaded, from a single point
    /// (FactoryOrchestrator.ApplyConfigOverrides), so an override cannot be missed on one of the
    /// several batch/scenario/Python entry points. Null = no override. Set once by HeadlessBatchRunner.
    /// </summary>
    public static class ConfigOverrides
    {
        public static string ReservationProtocol;
        /// <summary>Parking method ("single" | "multiple" | "lane"); overrides every config's parkingMethod.</summary>
        public static string ParkingMethod;
        /// <summary>Layout name A-J (see LayoutSpec); overrides every config's "layout" block. CLI over JSON over legacy.</summary>
        public static string Layout;
        /// <summary>Input/output belt docks ("corner" | "siding" | "bypass"); overrides every config's ioDocks.</summary>
        public static string IoDocks;

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
