using System;
using UnityEngine;
using Assets.Scripts.Simulation.Types;
using Assets.Scripts.Simulation.Logging;

namespace Assets.Scripts.Simulation.Stochastic
{
    /// <summary>
    /// Singleton MonoBehaviour that owns the seeded random number generators (RNGs) for all
    /// stochastic disruption events in the simulation. All failure and arrival systems draw
    /// exclusively from this manager, which keeps failure/repair sampling and arrival/burst
    /// sampling on two INDEPENDENT streams (see <see cref="_rng"/> / <see cref="_arrivalRng"/>)
    /// so the dynamic-arrival sequence for a given seed is identical regardless of which
    /// dispatching rule is running, even when machine failures are enabled.
    ///
    /// <para>Responsibilities:</para>
    ///   <list type="bullet">
    ///     <item><description>Provide seeded RNG access — never UnityEngine.Random or System.Random directly.</description></item>
    ///     <item><description>Sample time-to-failure (TTF) from Weibull distributions for machines and AGVs.</description></item>
    ///     <item><description>Sample repair durations from LogNormal distributions.</description></item>
    ///     <item><description>Sample inter-arrival times from Exponential distribution for dynamic job arrivals.</description></item>
    ///   </list>
    ///
    /// <para>Lifecycle:</para>
    ///   <list type="ordered">
    ///     <item>SimulationBridge.LoadConfig() calls StochasticEventManager.Instance.Initialize(config).</item>
    ///     <item>Machine/AGV controllers call SampleMachineTTF() / SampleAGVTTF() on episode start
    ///           and after each repair to schedule their next failure.</item>
    ///     <item>PoissonClock calls SampleInterArrivalTime() each time it needs the next job arrival gap.</item>
    ///     <item>All repair durations are obtained via SampleMachineRepair() / SampleAGVRepair().</item>
    ///   </list>
    ///
    /// <para>Graceful degradation:</para>
    /// When StochasticConfig is null or AnyEnabled == false, IsActive returns false and all
    /// Sample* methods return float.MaxValue (meaning "never fail"). This lets callers skip
    /// null-checks — the deterministic path is simply a no-op with infinite time-to-failure.
    /// </summary>
    public class StochasticEventManager : MonoBehaviour
    {
        // ── Singleton ────────────────────────────────────────────────────────

        /// <summary>
        /// Global singleton instance. Null until first Awake() call creates it.
        /// </summary>
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

        /// <summary>
        /// Seeded pseudo-random number generator for machine/AGV failure and repair sampling.
        /// Deliberately kept SEPARATE from <see cref="_arrivalRng"/> — see that field's remarks.
        /// </summary>
        private System.Random _rng;

        /// <summary>
        /// Seeded pseudo-random number generator for the dynamic-arrival Poisson clock
        /// (inter-arrival times and burst sizes) — independent of <see cref="_rng"/>.
        /// </summary>
        /// <remarks>
        /// Failure/repair sampling is triggered by simulation events (a machine going idle
        /// or finishing repair) whose timing depends on the dispatching rule in effect. If
        /// arrivals drew from the SAME stream, the number of failure-sampling draws consumed
        /// before each arrival draw would vary by rule, desyncing the arrival sequence
        /// per-rule even under an identical seed — silently breaking "same seed → same
        /// environment" comparisons across rules whenever machine failures are enabled.
        /// Splitting the streams keeps the arrival process reproducible and rule-independent.
        /// </remarks>
        private System.Random _arrivalRng;

        /// <summary>
        /// Fixed offset XORed into the episode seed to derive <see cref="_arrivalRng"/>'s seed.
        /// Any distinct constant works — this just needs to reliably decorrelate the arrival
        /// stream's seed from the failure stream's seed (which uses the raw episode seed).
        /// </summary>
        private const int ArrivalStreamSeedOffset = unchecked((int)0x9E3779B9); // golden-ratio constant, arbitrary but fixed

        /// <summary>
        /// Cached stochastic configuration. Null when no config has been loaded or when the
        /// current episode has stochastic features disabled.
        /// </summary>
        private StochasticConfig _cfg;

        /// <summary>
        /// True when a non-null StochasticConfig with AnyEnabled=true is currently loaded.
        /// Use this to quickly check whether any stochastic behavior is active.
        /// </summary>
        public bool IsActive => _cfg != null && _cfg.AnyEnabled;

        /// <summary>Convenience passthrough — safe to call even when the manager is inactive.</summary>
        public bool MachineFailuresEnabled => IsActive && _cfg.MachineFailuresEnabled;

        /// <summary>Convenience passthrough — safe to call even when the manager is inactive.</summary>
        public bool AGVFailuresEnabled => IsActive && _cfg.AGVFailuresEnabled;

        /// <summary>Convenience passthrough — safe to call even when the manager is inactive.</summary>
        public bool DynamicArrivalsEnabled => IsActive && _cfg.DynamicArrivalsEnabled;

        // ── Initialisation ───────────────────────────────────────────────────

        /// <summary>
        /// Initializes or re-initializes the stochastic event manager with a new simulation config.
        /// This method re-seeds the RNG and caches the stochastic parameters from the provided config.
        /// It is safe to call with a null Stochastic field, which marks the manager as inactive.
        /// </summary>
        /// <param name="config">The full FJSSP config containing the Stochastic sub-config and seed.
        ///                       If null, the manager becomes inactive (deterministic mode).</param>
        public void Initialize(FJSSPConfig config)
        {
            _cfg = config?.Stochastic;
            int seed = config?.Seed ?? 0;
            _rng = new System.Random(seed);
            _arrivalRng = new System.Random(seed ^ ArrivalStreamSeedOffset);

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
        /// Samples the time-to-failure (TTF) for a machine from a Weibull distribution
        /// with parameters k (shape) and λ (scale) from the current config.
        /// </summary>
        /// <returns>A sample in simulation time units, or float.MaxValue if machine failures are disabled.</returns>
        public float SampleMachineTTF()
        {
            if (!MachineFailuresEnabled) return float.MaxValue;
            return SampleWeibull(_cfg.WeibullK, _cfg.WeibullLambda);
        }

        /// <summary>
        /// Samples the repair duration for a machine from a LogNormal distribution
        /// with parameters μ and σ from the current config.
        /// </summary>
        /// <returns>A sample in simulation time units, or 0f if machine failures are disabled.</returns>
        public float SampleMachineRepair()
        {
            if (!MachineFailuresEnabled) return 0f;
            return SampleLogNormal(_cfg.RepairLogMu, _cfg.RepairLogSigma);
        }

        /// <summary>
        /// Samples the time-to-failure (TTF) for an AGV from a Weibull distribution
        /// with shape parameter k and AGV-specific scale parameter λ_agv.
        /// </summary>
        /// <returns>A sample in simulation time units, or float.MaxValue if AGV failures are disabled.</returns>
        public float SampleAGVTTF()
        {
            if (!AGVFailuresEnabled) return float.MaxValue;
            return SampleWeibull(_cfg.WeibullK, _cfg.AGVWeibullLambda);
        }

        /// <summary>
        /// Samples the repair duration for an AGV from a LogNormal distribution
        /// with AGV-specific parameters μ_agv and σ_agv.
        /// </summary>
        /// <returns>A sample in simulation time units, or 0f if AGV failures are disabled.</returns>
        public float SampleAGVRepair()
        {
            if (!AGVFailuresEnabled) return 0f;
            return SampleLogNormal(_cfg.AGVRepairLogMu, _cfg.AGVRepairLogSigma);
        }

        /// <summary>
        /// Samples the time until the next job arrival from an Exponential distribution
        /// with rate λ_arrival. This implements a homogeneous Poisson process for
        /// dynamic job arrivals.
        /// </summary>
        /// <returns>
        /// The inter-arrival time in simulation time units, computed as -ln(U) / λ where U ~ Uniform(0,1).
        /// Returns float.MaxValue if dynamic arrivals are disabled.
        /// </returns>
        public float SampleInterArrivalTime()
        {
            if (!DynamicArrivalsEnabled) return float.MaxValue;
            return SampleExponential(_cfg.ArrivalLambda);
        }

        /// <summary>
        /// Samples the number of jobs to inject for a single arrival event. When
        /// BurstArrivalsEnabled is false (the default), always returns 1 — every arrival
        /// event injects exactly one job, matching the original single-job Poisson process.
        /// When enabled, returns 1 + Poisson(BurstSizeMean - 1), so at least one job always
        /// arrives and any additional jobs on top are Poisson-distributed with mean
        /// (BurstSizeMean - 1).
        /// </summary>
        /// <returns>The number of jobs to spawn for this arrival event (always >= 1).</returns>
        public int SampleBurstSize()
        {
            if (!DynamicArrivalsEnabled || !_cfg.BurstArrivalsEnabled) return 1;
            double poissonMean = Math.Max(0.0, _cfg.BurstSizeMean - 1.0);
            return 1 + SamplePoisson(poissonMean);
        }

        // ── Distribution implementations ─────────────────────────────────────

        /// <summary>
        /// Generates a sample from a Weibull distribution using the inverse-CDF (quantile) method.
        /// Formula: X = λ × (−ln(1−U))^(1/k), where U ~ Uniform(0,1).
        /// </summary>
        /// <param name="k">Shape parameter (k > 0). Controls the failure rate trend.</param>
        /// <param name="lambda">Scale parameter (λ > 0). Characteristic life parameter.</param>
        /// <returns>A Weibull-distributed random variate, or float.MaxValue if parameters are invalid.</returns>
        /// <remarks>
        /// When k = 1, this reduces to an Exponential distribution with rate 1/λ.
        /// When k > 1, the failure rate increases over time (wear-out failures).
        /// When k < 1, the failure rate decreases over time (infant mortality).
        /// </remarks>
        private float SampleWeibull(float k, float lambda)
        {
            if (k <= 0f || lambda <= 0f)
            {
                SimLogger.LogWarning("[StochasticMgr] SampleWeibull: degenerate params, returning MaxValue.");
                return float.MaxValue;
            }

            double u = NextNonZeroUniform(_rng);
            double x = lambda * Math.Pow(-Math.Log(1.0 - u), 1.0 / k);
            return (float)x;
        }

        /// <summary>
        /// Generates a sample from a LogNormal distribution. A variable X is LogNormal-distributed
        /// if ln(X) follows a Normal(μ, σ) distribution.
        /// </summary>
        /// <param name="mu">Mean of the underlying normal distribution (μ).</param>
        /// <param name="sigma">Standard deviation of the underlying normal distribution (σ ≥ 0).</param>
        /// <returns>A LogNormal-distributed random variate.</returns>
        /// <remarks>
        /// Uses the Box-Muller transform to generate the underlying normal sample.
        /// Sigma is clamped to 0 if negative to prevent NaN outputs.
        /// </remarks>
        private float SampleLogNormal(float mu, float sigma)
        {
            if (sigma < 0f)
            {
                SimLogger.LogWarning("[StochasticMgr] SampleLogNormal: negative sigma, clamping to 0.");
                sigma = 0f;
            }

            double z = SampleStandardNormal(_rng);
            double x = Math.Exp(mu + sigma * z);
            return (float)x;
        }

        /// <summary>
        /// Generates a sample from an Exponential distribution using the inverse-CDF method.
        /// Formula: X = −ln(U) / λ, where U ~ Uniform(0,1).
        /// </summary>
        /// <param name="lambda">Rate parameter (λ > 0). For inter-arrival times, this is the arrival rate.</param>
        /// <returns>
        /// An Exponential-distributed random variate, or float.MaxValue if λ is non-positive.
        /// </returns>
        /// <remarks>
        /// This is used to generate inter-arrival times in a homogeneous Poisson process.
        /// The expected inter-arrival time is 1/λ.
        /// </remarks>
        private float SampleExponential(float lambda)
        {
            if (lambda <= 0f)
            {
                SimLogger.LogWarning("[StochasticMgr] SampleExponential: λ <= 0, returning MaxValue.");
                return float.MaxValue;
            }

            double u = NextNonZeroUniform(_arrivalRng);
            return (float)(-Math.Log(u) / lambda);
        }

        /// <summary>
        /// Generates a sample from a Poisson distribution using Knuth's algorithm:
        /// multiply successive Uniform(0,1) draws until the running product drops below
        /// e^(-mean), counting the number of draws needed.
        /// </summary>
        /// <param name="mean">Mean of the distribution (mean >= 0). 0 always returns 0.</param>
        /// <returns>A non-negative integer sample from Poisson(mean).</returns>
        /// <remarks>
        /// Adequate for the small means (a handful of extra jobs per burst) this simulation
        /// uses it for; not intended for large-mean, high-throughput sampling.
        /// </remarks>
        private int SamplePoisson(double mean)
        {
            if (mean <= 0.0) return 0;

            double l = Math.Exp(-mean);
            int k = 0;
            double p = 1.0;
            do
            {
                k++;
                p *= NextNonZeroUniform(_arrivalRng);
            } while (p > l);
            return k - 1;
        }

        /// <summary>
        /// Generates a standard normal random variate N(0,1) using the Box-Muller transform.
        /// This method consumes two independent Uniform(0,1) draws and produces one
        /// standard normal sample via the transformation:
        ///   Z = sqrt(−2 × ln(U1)) × cos(2π × U2)
        /// </summary>
        /// <returns>A sample from the standard normal distribution.</returns>
        /// <remarks>
        /// The Box-Muller transform produces two independent normal samples from two uniforms.
        /// This implementation returns only the first (cosine) sample; the sine sample is discarded.
        /// For production use with high throughput, a method that caches and returns both
        /// samples would be more efficient.
        /// </remarks>
        private double SampleStandardNormal(System.Random rng)
        {
            double u1 = NextNonZeroUniform(rng);
            double u2 = NextNonZeroUniform(rng);
            return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
        }

        /// <summary>
        /// Returns a Uniform(0,1) draw from the given RNG, guaranteed to be strictly greater than 0.
        /// If the RNG returns exactly 0, it is discarded and a new sample is drawn.
        /// </summary>
        /// <param name="rng">Which stream to draw from — <see cref="_rng"/> for failure/repair
        /// sampling, <see cref="_arrivalRng"/> for arrival/burst sampling. Callers must pass the
        /// stream matching their sample's purpose; mixing them reintroduces the cross-rule
        /// arrival desync this split exists to prevent.</param>
        /// <returns>A double in the open interval (0, 1).</returns>
        /// <remarks>
        /// This guard prevents log(0) in inverse-CDF methods (Weibull, Exponential) and
        /// sqrt(log(0)) in the Box-Muller transform, which would produce NaN or Infinity.
        /// The probability of NextDouble() returning exactly 0 is negligible in practice,
        /// but the guard is retained for robustness.
        /// </remarks>
        private double NextNonZeroUniform(System.Random rng)
        {
            double u;
            do { u = rng.NextDouble(); } while (u <= 0.0);
            return u;
        }
    }
}