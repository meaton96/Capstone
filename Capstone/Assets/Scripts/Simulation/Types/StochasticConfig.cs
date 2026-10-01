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
    ///     "agvWeibullK": 1.5,
    ///     "agvWeibullLambda": 8400.0,
    ///     "dynamicArrivalsEnabled": false,
    ///     "arrivalLambda": 0.005
    ///   }
    /// @endcode
    ///
    /// @see FJSSPConfig.Stochastic
    /// @see StochasticEventManager
    public class StochasticConfig
    {
        /// @brief Uniform spread window for the initial job batch (sim-seconds).
        /// @details Jobs in the initial batch are released uniformly in [0, InitialArrivalSpread].
        /// 0 = all jobs released at t=0 (default). Distinct from DynamicArrivalsEnabled,
        /// which controls mid-episode Poisson arrivals after the initial batch.
        public float InitialArrivalSpread = 0f;
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

        /// @brief Enable Weibull-distributed AGV breakdowns (see AGVController, "Breakdowns").
        ///
        /// @details A failed AGV stops where it is and keeps its zone reservations for the whole
        ///          repair, so on one-way aisles every AGV queued behind it waits too. A job it was
        ///          only on its way to collect is handed back for another AGV; a job it is carrying
        ///          stays on board and is delivered after the repair.
        ///
        ///          Time to failure counts OPERATING time only (travelling, loading, unloading), not
        ///          calendar time: a parked AGV, or one waiting for a zone, does not age. Machines,
        ///          by contrast, age on calendar time (PhysicalMachine.TickTTF).
        public bool AGVFailuresEnabled = false;

        /// @brief Weibull shape for AGV time to failure (operating seconds). Separate from the
        ///        machines' @c WeibullK so the two failure models can be set independently.
        public float AGVWeibullK = 1.5f;

        /// @brief Weibull scale for AGV time to failure, in operating seconds.
        ///
        /// @details Mean TTF = λ Γ(1 + 1/k), ≈ 0.903 λ at k = 1.5: the default 8400 gives ~7,600
        ///          operating seconds between failures, i.e. about 3 failures per 5,400 s window for
        ///          7 AGVs at ~60% busy. Placeholder pending calibration against measured AGV
        ///          operating time (docs/THESIS_GAP_PLAN_2026-09-30.md, D2).
        public float AGVWeibullLambda = 8400.0f;

        /// @brief AGV repair log-normal μ.
        ///
        /// @details A broken AGV blocks its aisle for the whole repair, so repairs are short
        ///          (a fault reset, not a machine overhaul). Default μ = 4.6, σ = 0.5 gives a mean of
        ///          exp(μ + σ²/2) ≈ 113 sim-seconds.
        public float AGVRepairLogMu = 4.6f;

        /// @brief AGV repair log-normal σ.
        public float AGVRepairLogSigma = 0.5f;

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

        /// @brief Maximum number of jobs the Poisson clock may inject per episode.
        ///
        /// @details 0 = unlimited — the clock fires indefinitely and the episode is bounded
        ///          only by the orchestrator's MAX_EPISODE_SIM_SECONDS timeout. Use 0 during
        ///          DRL training where episode length is controlled externally.
        ///
        ///          Positive values disarm the clock once that many dynamic jobs have arrived,
        ///          after which the episode ends naturally when all jobs exit. Recommended for
        ///          headless benchmark sweeps: set to a small multiple of the initial job count
        ///          (e.g. 2× gives a 3× total workload per episode).
        public int DynamicJobCap = 0;

        /// @brief Enable burst (compound-Poisson) arrivals: each arrival event injects a
        ///        cluster of jobs simultaneously instead of exactly one.
        ///
        /// @details Models e.g. a truck dropping off a multi-item order at the same
        ///          sim-time. Independent of the event RATE (still governed by
        ///          ArrivalLambda) — this only controls how many jobs land per event.
        ///          When false, every arrival event injects exactly 1 job (unchanged
        ///          behavior).
        public bool BurstArrivalsEnabled = false;

        /// @brief Mean number of jobs injected per arrival event when BurstArrivalsEnabled.
        ///
        /// @details Must be >= 1.0. Burst size B = 1 + Poisson(BurstSizeMean - 1), so at
        ///          least one job always arrives and any additional jobs on top are
        ///          Poisson-distributed. BurstSizeMean=1.0 degenerates to exactly 1 job
        ///          per event, identical to BurstArrivalsEnabled=false.
        public float BurstSizeMean = 1.0f;

        // ── Fixed-duration (steady-state) episodes ──

        /// @brief Sim-seconds after which the episode ends regardless of job completion, 0 = disabled.
        ///
        /// @details For steady-state measurement: pair with DynamicJobCap=0 (unlimited arrivals) so
        ///          the Poisson process runs for the full duration instead of the default
        ///          "cap-then-drain" shape (arrivals stop, WIP drains to zero, episode ends).
        ///          Jobs still in-flight when the duration elapses are recorded as censored
        ///          (JobCompletionRecord.Completed = false), identical to the existing
        ///          MAX_EPISODE_SIM_SECONDS timeout path. Analysis should exclude jobs arriving
        ///          near t=0 (warm-up) AND near the episode end (right-censored, incomplete data)
        ///          when computing steady-state statistics.
        public double EpisodeDurationSeconds = 0.0;

        // ── Warm-start (mid-cycle) episodes ──

        /// @brief Sim-seconds run under the config's dispatchingRule before the RL agent takes
        ///        over, 0 = disabled (agent controls from t=0, the default).
        ///
        /// @details Lets an episode "start" mid-scenario with realistic WIP already built up
        ///          (machines busy, AGVs in transit, queues populated) instead of an empty
        ///          floor, without any extra Python round-trips during the warm-up window —
        ///          decisions in [0, WarmupSeconds) are resolved the same way BaselineDrainMode
        ///          resolves them (see FactoryOrchestrator.InWarmup). The agent's first real
        ///          decision, first observation, and first reward baseline all originate at
        ///          SimTime=WarmupSeconds, so no reward-accounting changes are needed on the
        ///          Python side — env/env_wrappers/unity_env.py seeds its reward baseline from
        ///          whatever the first decision observation actually is.
        public double WarmupSeconds = 0.0;

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
                if (DynamicArrivalsEnabled && BurstArrivalsEnabled) parts.Add("burst");
                return string.Join("+", parts);
            }
        }
    }
}