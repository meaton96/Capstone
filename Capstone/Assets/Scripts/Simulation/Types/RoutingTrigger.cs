using System;

namespace Assets.Scripts.Simulation.Types
{
    /// <summary>
    /// When a job that has finished an operation gets its routing decision (which job moves next, and to which
    /// machine). Part of scenario identity: never compare runs with different values.
    /// </summary>
    public enum RoutingTrigger
    {
        /// <summary>
        /// Default. Route only when transport exists: an AGV is idle or returning to parking (or one is already
        /// pre-dispatched to the job). Jobs that finish while every AGV is busy wait as NeedsRouting, and when an
        /// AGV frees up the rule ranks the whole waiting pool, so a later job can go first. The chosen job gets
        /// the nearest free AGV in the same step, and its machine is chosen against the queues of that moment.
        /// </summary>
        OnTransport,
        /// <summary>
        /// Legacy (every run before 2026-09-25). Route the moment a job finishes an operation; the job then waits
        /// for pickup, and free AGVs go to waiting jobs in the order they entered the system, whatever the rule.
        /// The routing pool is almost always a single job, so the job-priority half of a rule rarely applies.
        /// </summary>
        OnReady,
    }

    public static class RoutingTriggerParser
    {
        public const string Default = "onTransport";

        /// <summary>Parses a config/CLI string. Throws on unknown values so a typo cannot silently run the default.</summary>
        public static RoutingTrigger Parse(string value)
        {
            switch ((value ?? Default).Trim().ToLowerInvariant())
            {
                case "ontransport": return RoutingTrigger.OnTransport;
                case "onready": return RoutingTrigger.OnReady;
                default:
                    throw new ArgumentException(
                        $"Invalid routingTrigger '{value}'. Valid values: onTransport, onReady.");
            }
        }

        /// <summary>Returns the value (or the default when null) after checking it parses; throws otherwise.</summary>
        public static string Validated(string value)
        {
            if (value == null) return Default;
            Parse(value);
            return value;
        }
    }
}
