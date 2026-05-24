using System;
using UnityEngine;
using Assets.Scripts.Simulation.Types;
using Assets.Scripts.Simulation.Logging;

namespace Assets.Scripts.Simulation.Stochastic
{
    /// <summary>
    /// Singleton MonoBehaviour that owns the seeded RNG for all stochastic
    /// disruption events. All failure and arrival systems draw from this
    /// manager — never from UnityEngine.Random or a local System.Random —
    /// so that the entire stochastic stream is deterministic given a seed.
    ///
    /// Lifecycle:
    ///   1. SimulationBridge.LoadConfig() calls StochasticEventManager.Instance.Initialize(config).
    ///   2. Machine/AGV controllers call SampleMachineTTF() / SampleAGVTTF() on episode start
    ///      and after each repair to schedule their next failure.
    ///   3. PoissonClock calls SampleInterArrivalTime() each time it needs the next gap.
    ///   4. All repair durations are obtained via SampleMachineRepair() / SampleAGVRepair().
    ///
    /// When StochasticConfig is null or AnyEnabled == false, IsActive returns false
    /// and all Sample* methods return float.MaxValue (= effectively never fail).
    /// This lets callers skip null-checks — the deterministic path is just a no-op TTF.
    /// </summary>
    public class StochasticEventManager : MonoBehaviour
    {
        // ── Singleton ────────────────────────────────────────────────────────

        public static StochasticEventManager Instance { get; private set; }

        private void Awake()
        {
            if (Instance != null && Instance != this)
            {
                Destroy(gameObject);
                return;
            }
            Instance = this;
        }

        // ── State ────────────────────────────────────────────────────────────

        private System.Random _rng;
        private StochasticConfig _cfg;

        /// @brief True when a non-null StochasticConfig with AnyEnabled=true is loaded.
        public bool IsActive => _cfg != null && _cfg.AnyEnabled;

        /// @brief Convenience passthrough — safe to call even when not active.
        public bool MachineFailuresEnabled => IsActive && _cfg.MachineFailuresEnabled;
        public bool AGVFailuresEnabled => IsActive && _cfg.AGVFailuresEnabled;
        public bool DynamicArrivalsEnabled => IsActive && _cfg.DynamicArrivalsEnabled;

        // ── Initialisation ───────────────────────────────────────────────────

        /// <summary>
        /// Called by SimulationBridge.LoadConfig(). Re-seeds the RNG and caches
        /// the stochastic parameters. Safe to call with a null stochastic field
        /// (marks manager as inactive for that episode).
        /// </summary>
        public void Initialize(FJSSPConfig config)
        {
            _cfg = config?.Stochastic;
            _rng = new System.Random(config?.Seed ?? 0);

            if (IsActive)
                SimLogger.Low($"[StochasticMgr] Initialized — seed={config.Seed} " +
                              $"mode=[{_cfg.Tag}] " +
                              $"WeibullK={_cfg.WeibullK} λ_machine={_cfg.WeibullLambda} " +
                              $"λ_agv={_cfg.AGVWeibullLambda} " +
                              $"repairMu={_cfg.RepairLogMu} repairSigma={_cfg.RepairLogSigma} " +
                              $"arrivalLambda={_cfg.ArrivalLambda}");
            else
                SimLogger.Low("[StochasticMgr] Deterministic mode (no stochastic config).");
        }

        // ── Public sampling API ──────────────────────────────────────────────

        /// <summary>
        /// Sample time-to-failure for a machine from Weibull(k, λ_machine).
        /// Returns float.MaxValue when machine failures are disabled.
        /// </summary>
        public float SampleMachineTTF()
        {
            if (!MachineFailuresEnabled) return float.MaxValue;
            return SampleWeibull(_cfg.WeibullK, _cfg.WeibullLambda);
        }

        /// <summary>
        /// Sample repair duration for a machine from LogNormal(μ, σ).
        /// Returns 0 when machine failures are disabled (should not be called).
        /// </summary>
        public float SampleMachineRepair()
        {
            if (!MachineFailuresEnabled) return 0f;
            return SampleLogNormal(_cfg.RepairLogMu, _cfg.RepairLogSigma);
        }

        /// <summary>
        /// Sample time-to-failure for an AGV from Weibull(k, λ_agv).
        /// Returns float.MaxValue when AGV failures are disabled.
        /// </summary>
        public float SampleAGVTTF()
        {
            if (!AGVFailuresEnabled) return float.MaxValue;
            return SampleWeibull(_cfg.WeibullK, _cfg.AGVWeibullLambda);
        }

        /// <summary>
        /// Sample repair duration for an AGV from LogNormal(μ_agv, σ_agv).
        /// </summary>
        public float SampleAGVRepair()
        {
            if (!AGVFailuresEnabled) return 0f;
            return SampleLogNormal(_cfg.AGVRepairLogMu, _cfg.AGVRepairLogSigma);
        }

        /// <summary>
        /// Sample the time until the next job arrival from Exponential(λ_arrival).
        /// Inter-arrival time = -ln(U) / λ, giving a homogeneous Poisson process.
        /// Returns float.MaxValue when dynamic arrivals are disabled.
        /// </summary>
        public float SampleInterArrivalTime()
        {
            if (!DynamicArrivalsEnabled) return float.MaxValue;
            return SampleExponential(_cfg.ArrivalLambda);
        }

        // ── Distribution implementations ─────────────────────────────────────

        /// <summary>
        /// Weibull inverse-CDF: X = λ × (−ln(1−U))^(1/k), U ~ Uniform(0,1).
        /// </summary>
        private float SampleWeibull(float k, float lambda)
        {
            // Guard against degenerate params
            if (k <= 0f || lambda <= 0f)
            {
                SimLogger.LogWarning("[StochasticMgr] SampleWeibull: degenerate params, returning MaxValue.");
                return float.MaxValue;
            }

            double u = NextNonZeroUniform();
            double x = lambda * Math.Pow(-Math.Log(1.0 - u), 1.0 / k);
            return (float)x;
        }

        /// <summary>
        /// Log-normal sample: X = exp(μ + σ × Z), Z ~ N(0,1) via Box-Muller.
        /// </summary>
        private float SampleLogNormal(float mu, float sigma)
        {
            if (sigma < 0f)
            {
                SimLogger.LogWarning("[StochasticMgr] SampleLogNormal: negative sigma, clamping to 0.");
                sigma = 0f;
            }

            double z = SampleStandardNormal();
            double x = Math.Exp(mu + sigma * z);
            return (float)x;
        }

        /// <summary>
        /// Exponential sample: X = −ln(U) / λ. Used for Poisson inter-arrival times.
        /// </summary>
        private float SampleExponential(float lambda)
        {
            if (lambda <= 0f)
            {
                SimLogger.LogWarning("[StochasticMgr] SampleExponential: λ <= 0, returning MaxValue.");
                return float.MaxValue;
            }

            double u = NextNonZeroUniform();
            return (float)(-Math.Log(u) / lambda);
        }

        /// <summary>
        /// Box-Muller transform: produces one N(0,1) sample.
        /// Uses two independent Uniform(0,1) draws from the seeded RNG.
        /// </summary>
        private double SampleStandardNormal()
        {
            double u1 = NextNonZeroUniform();
            double u2 = NextNonZeroUniform();
            return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
        }

        /// <summary>
        /// Returns a Uniform(0,1) draw from the seeded RNG, guaranteed > 0
        /// to prevent log(0) in inverse-CDF and Box-Muller transforms.
        /// </summary>
        private double NextNonZeroUniform()
        {
            double u;
            do { u = _rng.NextDouble(); } while (u <= 0.0);
            return u;
        }
    }
}
