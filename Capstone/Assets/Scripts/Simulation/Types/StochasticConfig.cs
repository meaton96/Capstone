namespace Assets.Scripts.Simulation.Types
{
    /// <summary>
    /// Optional stochastic disruption parameters attached to FJSSPConfig.
    /// Null reference on FJSSPConfig.Stochastic = fully deterministic episode.
    ///
    /// Attach to a config via JSON:
    /// <code>
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
    /// </code>
    /// </summary>
    public class StochasticConfig
    {
        // ── Machine failures ─────────────────────────────────────────────────
        
        /// @brief Enable Weibull-distributed machine time-to-failure sampling.
        public bool MachineFailuresEnabled = false;

        /// @brief Weibull shape parameter k. k=1.5 → wear-out regime (increasing failure rate).
        /// k=1.0 = exponential (memoryless), k>1 = wear-out, k<1 = infant mortality.
        public float WeibullK = 1.5f;

        /// @brief Weibull scale parameter λ (characteristic life) in simulation-seconds.
        /// Mean TTF ≈ λ × Γ(1 + 1/k). At k=1.5, mean ≈ 0.903 × λ.
        /// Tune this against your typical episode length. A value of ~3× mean episode
        /// length gives roughly 1 failure per 3 episodes on average per machine.
        public float WeibullLambda = 900.0f;

        // ── Repair times ─────────────────────────────────────────────────────

        /// @brief Log-normal μ for repair duration (ln-space mean).
        /// Repair duration X = exp(RepairLogMu + RepairLogSigma * Z) where Z~N(0,1).
        /// Real-space mean = exp(μ + σ²/2). At μ=4.0, σ=0.5: mean ≈ 60 sim-seconds.
        public float RepairLogMu = 4.0f;

        /// @brief Log-normal σ for repair duration (ln-space std dev).
        /// Higher σ produces heavier right tail (occasional very long repairs).
        public float RepairLogSigma = 0.5f;

        // ── AGV failures ─────────────────────────────────────────────────────

        /// @brief Enable Weibull-distributed AGV time-to-failure. Uses same k as machines.
        /// AGVs share WeibullK but have their own scale (they fail more frequently
        /// due to higher mechanical stress from continuous movement).
        public bool AGVFailuresEnabled = false;

        /// @brief Weibull scale for AGV failures. Typically lower than WeibullLambda
        /// since AGVs operate continuously. Defaults to ~78% of machine λ.
        public float AGVWeibullLambda = 700.0f;

        /// @brief AGV repair log-normal μ. AGV repairs are typically faster than machines.
        /// Default real-space mean ≈ 30 sim-seconds.
        public float AGVRepairLogMu = 3.4f;

        /// @brief AGV repair log-normal σ.
        public float AGVRepairLogSigma = 0.4f;

        // ── Dynamic job arrivals ─────────────────────────────────────────────

        /// @brief Enable homogeneous Poisson job arrival process mid-episode.
        public bool DynamicArrivalsEnabled = false;

        /// @brief Arrival rate λ in jobs per simulation-second.
        /// Inter-arrival times are drawn from Exponential(λ) = -ln(U) / λ.
        /// A value of 0.005 gives ~1 arrival per 200 sim-seconds on average.
        public float ArrivalLambda = 0.005f;

        // ── Convenience ──────────────────────────────────────────────────────

        /// @brief True if any disruption source is active.
        public bool AnyEnabled =>
            MachineFailuresEnabled || AGVFailuresEnabled || DynamicArrivalsEnabled;

        /// @brief Descriptive tag for log output and CSV labelling.
        /// E.g. "mf+agv", "arrivals", "none", "all"
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
