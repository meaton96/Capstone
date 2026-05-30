namespace Assets.Scripts.Simulation.Types
{
    /// @brief Optional stochastic disruption parameters attached to FJSSPConfig.
    ///
    /// @details When FJSSPConfig.Stochastic is null, the episode is fully deterministic
    ///          (no random failures or dynamic arrivals). When non-null, activates the
    ///          @c StochasticEventManager for the subset of disruption types flagged
    ///          via the enabled booleans.
    ///
    /// Attach to a config via JSON:
    /// @code
    ///   "stochastic": {
    ///     "machineFailuresEnabled": true,
    ///     "weibullK": 1.5,
    ///     "weibullLambda": 900.0,
    ///     "repairLogMu": 4.0,
    ///     "repairLogSigma": 0.5,
    ///     "agvFailuresEnabled": false,
    ///     "agvWeibullLambda": 700.0,
    ///     "dynamicArrivalsEnabled": false,
    ///     "arrivalLambda": 0.005
    ///   }
    /// @endcode
    ///
    /// @see FJSSPConfig.Stochastic
    /// @see StochasticEventManager
    public class StochasticConfig
    {
        // ── Machine failures ──

        /// @brief Enable Weibull-distributed machine time-to-failure sampling.
        public bool MachineFailuresEnabled = false;

        /// @brief Weibull shape parameter k.
        /// @details k=1.5 → wear-out regime (increasing failure rate).
        ///          k=1.0 = exponential (memoryless).
        ///          k>1 = wear-out, k<1 = infant mortality.
        public float WeibullK = 1.5f;

        /// @brief Weibull scale parameter λ (characteristic life) in simulation-seconds.
        ///
        /// @details Mean TTF ≈ λ × Γ(1 + 1/k). At k=1.5, mean ≈ 0.903 × λ.
        ///          Tune against typical episode length. A value of ~3× mean episode
        ///          length gives roughly 1 failure per 3 episodes on average per machine.
        public float WeibullLambda = 900.0f;

        // ── Repair times ──

        /// @brief Log-normal μ for repair duration (ln-space mean).
        ///
        /// @details Repair duration X = exp(RepairLogMu + RepairLogSigma * Z) where Z~N(0,1).
        ///          Real-space mean = exp(μ + σ²/2). At μ=4.0, σ=0.5: mean ≈ 60 sim-seconds.
        public float RepairLogMu = 4.0f;

        /// @brief Log-normal σ for repair duration (ln-space standard deviation).
        ///
        /// @details Higher σ produces heavier right tail (occasional very long repairs).
        public float RepairLogSigma = 0.5f;

        // ── AGV failures ──

        /// @brief Enable Weibull-distributed AGV time-to-failure.
        ///
        /// @details AGVs share @c WeibullK with machines but have their own scale parameter.
        ///          AGVs fail more frequently due to higher mechanical stress from continuous movement.
        public bool AGVFailuresEnabled = false;

        /// @brief Weibull scale for AGV failures in simulation-seconds.
        ///
        /// @details Typically lower than @c WeibullLambda since AGVs operate continuously.
        ///          Default is ~78% of machine λ.
        public float AGVWeibullLambda = 700.0f;

        /// @brief AGV repair log-normal μ.
        ///
        /// @details AGV repairs are typically faster than machine repairs.
        ///          Default real-space mean ≈ 30 sim-seconds.
        public float AGVRepairLogMu = 3.4f;

        /// @brief AGV repair log-normal σ.
        public float AGVRepairLogSigma = 0.4f;

        // ── Dynamic job arrivals ──

        /// @brief Enable homogeneous Poisson job arrival process mid-episode.
        ///
        /// @details When enabled, new jobs arrive according to a Poisson process after the
        ///          initial batch is released. Inter-arrival times are exponentially distributed.
        public bool DynamicArrivalsEnabled = false;

        /// @brief Arrival rate λ in jobs per simulation-second.
        ///
        /// @details Inter-arrival times are drawn from Exponential(λ) = -ln(U) / λ.
        ///          A value of 0.005 gives ~1 arrival per 200 sim-seconds on average.
        public float ArrivalLambda = 0.005f;

        // ── Convenience ──

        /// @brief True if any disruption source is active.
        public bool AnyEnabled =>
            MachineFailuresEnabled || AGVFailuresEnabled || DynamicArrivalsEnabled;

        /// @brief Descriptive tag for log output and CSV labelling.
        ///
        /// @details Composed from active disruption types: "mf" (machine failures),
        ///          "agv", "arr" (dynamic arrivals). Multiple types joined with "+".
        ///          Returns "none" when no disruptions are active.
        public string Tag
        {
            get
            {
                if (!AnyEnabled) return "none";
                var parts = new System.Collections.Generic.List<string>();
                if (MachineFailuresEnabled) parts.Add("mf");
                if (AGVFailuresEnabled) parts.Add("agv");
                if (DynamicArrivalsEnabled) parts.Add("arr");
                return string.Join("+", parts);
            }
        }
    }
}
