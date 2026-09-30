using System;
using System.Collections.Generic;

namespace Assets.Scripts.Simulation.Types
{
    /// <summary>
    /// Physical-bounds check for configs received from Python over EpisodeConfigChannel (thesis section 7.1).
    /// A config that fails is rejected and the player stops (FactoryOrchestrator.RejectPythonConfig); nothing
    /// falls back to a default or to the previous config, so a bad scenario shows up as a crash instead of a
    /// quietly shifted result. Mirrors env/channels/config_schema.py, which checks the same bounds before sending.
    /// CLI batch runs do not go through this: the fleet-size study deliberately runs past the gridlock threshold.
    /// </summary>
    public static class ConfigValidator
    {
        /// <summary>Floor size the fleet thresholds below were measured on (layout D, 15 machines).</summary>
        public const int ReferenceMachines = 15;

        /// <summary>holdPrevious: 0 gridlocks to 8 AGVs, 79/80 runs gridlocked at 9 (sweep 2026-09-21,
        /// docs/GRIDLOCK_INVESTIGATION_2026-09-19.md section 8e).</summary>
        public const int HoldPreviousMaxAgvsPerReference = 8;

        /// <summary>releasePrevious: 0 collisions and 0 deadlocks from 3 to 15 AGVs on every layout (E1_rel3, 864 runs,
        /// docs/experiments/E1_E2_findings.md section 6b). The largest fleet measured deadlock-free on all layouts.</summary>
        public const int ReleasePreviousMaxAgvsPerReference = 15;

        /// <summary>Set by the -allowunsafefleet CLI flag: lifts the AGV upper bound (every other bound still applies).
        /// A process argument, not a config field, so it cannot arrive over the config channel.</summary>
        public static bool AllowUnsafeFleet =
            Array.Exists(Environment.GetCommandLineArgs(), a => string.Equals(a, "-allowunsafefleet", StringComparison.OrdinalIgnoreCase));

        /// <summary>
        /// Largest AGV fleet accepted for this config: the measured threshold for its reservation protocol, scaled
        /// linearly by machine count from the 15-machine reference floor (tiled floors are copies of that floor, so
        /// the per-tile threshold carries over; larger monolithic floors are an extrapolation).
        /// </summary>
        public static int MaxAgvCount(FJSSPConfig cfg)
        {
            int perReference = ReservationProtocolParser.Parse(cfg.reservationProtocol) == ReservationProtocol.HoldPrevious
                ? HoldPreviousMaxAgvsPerReference
                : ReleasePreviousMaxAgvsPerReference;
            return Math.Max(1, cfg.TotalMachines * perReference / ReferenceMachines);
        }

        /// <summary>Every bound the config breaks; empty when it is valid.</summary>
        public static List<string> Validate(FJSSPConfig cfg)
        {
            var errors = new List<string>();
            if (cfg == null)
            {
                errors.Add("config is null");
                return errors;
            }

            void Check(bool ok, string message) { if (!ok) errors.Add(message); }
            bool Finite(double v) => !double.IsNaN(v) && !double.IsInfinity(v);

            Check(cfg.Seed >= 0, $"seed must be >= 0 (got {cfg.Seed})");
            Check(cfg.JobCount >= 1, $"jobCount must be >= 1 (got {cfg.JobCount})");
            Check(cfg.TotalMachines >= 1, "the floor has no machines (machineTypes / machineTypeLayout empty)");
            Check(Finite(cfg.MinProcTime) && cfg.MinProcTime > 0f, $"minProcTime must be > 0 (got {cfg.MinProcTime})");
            Check(Finite(cfg.MaxProcTime) && cfg.MaxProcTime >= cfg.MinProcTime,
                  $"maxProcTime must be >= minProcTime (got {cfg.MaxProcTime} < {cfg.MinProcTime})");
            Check(cfg.MinOpsPerJob >= 1, $"minOpsPerJob must be >= 1 (got {cfg.MinOpsPerJob})");
            Check(cfg.MaxOpsPerJob >= cfg.MinOpsPerJob,
                  $"maxOpsPerJob must be >= minOpsPerJob (got {cfg.MaxOpsPerJob} < {cfg.MinOpsPerJob})");

            try
            {
                int maxAgvs = MaxAgvCount(cfg);
                Check(cfg.AGVCount >= 1, $"agvCount must be >= 1 (got {cfg.AGVCount})");
                Check(AllowUnsafeFleet || cfg.AGVCount <= maxAgvs,
                      $"agvCount {cfg.AGVCount} is above the gridlock-safe maximum {maxAgvs} for {cfg.TotalMachines} " +
                      $"machines under {cfg.reservationProtocol} (launch the player with -allowunsafefleet to run it anyway)");
            }
            catch (ArgumentException ex) { errors.Add(ex.Message); }

            Check(cfg.AGVMoveSpeed == null || (Finite(cfg.AGVMoveSpeed.Value) && cfg.AGVMoveSpeed.Value > 0f),
                  $"agvMoveSpeed must be > 0 (got {cfg.AGVMoveSpeed})");
            Check(cfg.AGVHandshakeDuration == null || (Finite(cfg.AGVHandshakeDuration.Value) && cfg.AGVHandshakeDuration.Value >= 0f),
                  $"agvHandshakeDuration must be >= 0 (got {cfg.AGVHandshakeDuration})");

            foreach (var kvp in cfg.ProcTimeParams)
            {
                Check(Finite(kvp.Value.mu) && kvp.Value.mu > 0f, $"procTimeParams.{kvp.Key}.mu must be > 0 (got {kvp.Value.mu})");
                Check(Finite(kvp.Value.sigma) && kvp.Value.sigma >= 0f, $"procTimeParams.{kvp.Key}.sigma must be >= 0 (got {kvp.Value.sigma})");
            }

            ValidateStochastic(cfg.Stochastic, errors);

            // Checks that already exist elsewhere and throw; run them here so they reject instead of crashing mid-build.
            TryCheck(cfg.ValidateFlexibility, errors);
            TryCheck(() => ConfigOverrides.ValidatedParkingMethod(cfg.parkingMethod), errors);
            TryCheck(() => ConfigOverrides.ValidatedIoDocks(cfg.ioDocks), errors);
            TryCheck(() => RoutingTriggerParser.Parse(cfg.routingTrigger), errors);
            TryCheck(() => (cfg.Layout ?? LayoutSpec.Default).EnsureBuildable(), errors);
            if (cfg.MachineTypeLayout != null)
                TryCheck(() => (cfg.Tiling ?? TilingSpec.Single).Validate(cfg.MachineTypeLayout, cfg.AGVCount, cfg.parkingMethod, cfg.ioDocks), errors);

            return errors;
        }

        private static void ValidateStochastic(StochasticConfig s, List<string> errors)
        {
            if (s == null) return;
            void Check(bool ok, string message) { if (!ok) errors.Add("stochastic." + message); }
            bool Positive(double v) => !double.IsNaN(v) && !double.IsInfinity(v) && v > 0.0;
            bool NonNegative(double v) => !double.IsNaN(v) && !double.IsInfinity(v) && v >= 0.0;

            Check(NonNegative(s.InitialArrivalSpread), $"initialArrivalSpread must be >= 0 (got {s.InitialArrivalSpread})");
            if (s.MachineFailuresEnabled)
            {
                Check(Positive(s.WeibullK), $"weibullK must be > 0 (got {s.WeibullK})");
                Check(Positive(s.WeibullLambda), $"weibullLambda must be > 0 (got {s.WeibullLambda})");
                Check(NonNegative(s.RepairLogSigma), $"repairLogSigma must be >= 0 (got {s.RepairLogSigma})");
            }
            if (s.AGVFailuresEnabled)
            {
                Check(Positive(s.AGVWeibullLambda), $"agvWeibullLambda must be > 0 (got {s.AGVWeibullLambda})");
                Check(NonNegative(s.AGVRepairLogSigma), $"agvRepairLogSigma must be >= 0 (got {s.AGVRepairLogSigma})");
            }
            if (s.DynamicArrivalsEnabled)
                Check(Positive(s.ArrivalLambda), $"arrivalLambda must be > 0 (got {s.ArrivalLambda})");
            Check(s.DynamicJobCap >= 0, $"dynamicJobCap must be >= 0 (got {s.DynamicJobCap})");
            if (s.BurstArrivalsEnabled)
                Check(!float.IsNaN(s.BurstSizeMean) && s.BurstSizeMean >= 1f, $"burstSizeMean must be >= 1 (got {s.BurstSizeMean})");
            Check(NonNegative(s.EpisodeDurationSeconds), $"episodeDurationSeconds must be >= 0 (got {s.EpisodeDurationSeconds})");
            Check(NonNegative(s.WarmupSeconds), $"warmupSeconds must be >= 0 (got {s.WarmupSeconds})");
        }

        private static void TryCheck(Action check, List<string> errors)
        {
            try { check(); }
            catch (Exception ex) when (ex is ArgumentException || ex is NotSupportedException) { errors.Add(ex.Message); }
        }

        private static void TryCheck<T>(Func<T> check, List<string> errors) => TryCheck(() => { check(); }, errors);
    }
}
