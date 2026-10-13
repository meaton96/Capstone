using System.Collections.Generic;
using Assets.Scripts.Simulation.Machines;
using Assets.Scripts.Simulation.Stochastic;

namespace Assets.Scripts.Simulation.Types
{
    /// @brief Configuration parameters for a Flexible Job Shop Scheduling Problem (FJSSP) instance.
    ///
    /// @details Defines all static parameters governing job shop generation: machine topology,
    ///          processing time ranges, operation counts, AGV fleet size, and stochastic
    ///          disruption settings. Used by @c ConfigLoader to parse from JSON and by
    ///          @c FJSSPJobGenerator to create problem instances.
    ///
    /// @see ConfigLoader
    /// @see FJSSPJobGenerator
    /// @see StochasticConfig
    public class FJSSPConfig
    {
        // ── Identity ──

        /// @brief Human-readable label used in logs and CSV output (e.g. "20j_15m_baseline").
        public string Name = "unnamed";

        // ── Random seed ──

        /// @brief Seed for deterministic job shop generation. Default: 42.
        public int Seed = 42;

        // ── Problem scale ──

        /// @brief Number of jobs to generate per episode.
        public int JobCount;

        /// @brief Number of machines of each type in the layout.
        public int MachinesPerType;

        /// @brief Ordered array of machine types defining the floor layout.
        /// @details Length equals @c TotalMachines. Each entry specifies the type of the
        ///          corresponding machine on the floor.
        public MachineType[] MachineTypeLayout;

        // ── Processing time parameters ──

        /// @brief Minimum processing time (uniform fallback).
        /// @details Used only when a machine type has no entry in @c ProcTimeParams.
        public float MinProcTime;

        /// @brief Maximum processing time (uniform fallback).
        /// @details Used only when a machine type has no entry in @c ProcTimeParams.
        public float MaxProcTime;

        // ── Operation count range ──

        /// @brief Minimum number of operations per job.
        public int MinOpsPerJob;

        /// @brief Maximum number of operations per job.
        public int MaxOpsPerJob;

        /// @brief Maximum arrival time for initial job batch (time window).
        /// @details Controls how spread out initial job releases are.
       // public float MaxArrivalTime;

        // ── AGV fleet ──

        /// @brief Number of AGVs in the fleet for material transport.
        public int AGVCount;

        /// @brief Fleet schedule (2026-10-04, scenario "agvSchedule"): from sim time AgvScheduleStarts[i] only the first
        ///        AgvScheduleCounts[i] AGVs take new tasks (AGVPool.IsOffDuty); an AGV going off duty finishes its task
        ///        and returns to its bay. Null = the whole fleet throughout (left out of the config hash then).
        [FingerprintOmitIfNull] public float[] AgvScheduleStarts;
        [FingerprintOmitIfNull] public int[] AgvScheduleCounts;

        /// @brief AGV travel speed (units/sim-second), overriding the AGV prefab's own
        ///        serialized default (3.5) when set. Null = use the prefab's value.
        /// @details Lets a config raise/lower physical transit time relative to job processing
        ///          time — e.g. faster AGVs shrink inter-arrival gaps at a machine without
        ///          changing the scripted job-generation rate, so decisions are less often
        ///          degenerate (0-1 real candidates) purely because AGV transit spaced jobs out.
        public float? AGVMoveSpeed = null;

        /// @brief AGV pickup/dropoff handshake duration (sim-seconds), overriding the AGV
        ///        prefab's own serialized default (1.5) when set. Null = use the prefab's value.
        public float? AGVHandshakeDuration = null;

        // ── Flexibility ──

        /// @brief Probability [0,1] that a machine gains each non-primary type as a
        /// secondary capability during floor construction.
        /// @details 0 = fully typed (default, backward-compatible).
        ///          1 = fully flexible (every machine processes every operation type).
        ///          Sampled once per machine (per tile position on a tiled floor, so tiles stay identical).
        public float MachineFlexibilityProbability = 0f;

        /// @brief Processing-time factor for an operation run on a machine whose PRIMARY type differs from the
        ///        operation's type (a secondary capability): duration = base duration x this.
        /// @details 1 = a secondary capability is as fast as a dedicated machine (flexibility only adds
        ///          capacity). Above 1 gives machine-dependent processing times, the standard FJSP form, where
        ///          sending an op to an idle generalist trades speed for waiting. Only matters when
        ///          MachineFlexibilityProbability &gt; 0.
        public float SecondaryTimeMultiplier = 1f;

        /// @brief Throws on an out-of-range flexibility setting (called by FactoryLayoutManager.BuildFloor).
        public void ValidateFlexibility()
        {
            if (!(MachineFlexibilityProbability >= 0f && MachineFlexibilityProbability <= 1f))
                throw new System.ArgumentException($"machineFlexibilityProbability must be in [0, 1] (got {MachineFlexibilityProbability}).");
            if (!(SecondaryTimeMultiplier > 0f))
                throw new System.ArgumentException($"secondaryTimeMultiplier must be > 0 (got {SecondaryTimeMultiplier}).");
        }

        // ── Throughput timing ──

        /// @brief Timing window (in simulation time units) used for throughput calculations.
        /// @details Controls the observation window over which throughput is measured.
        public float ThroughputTimingWindow;

        // ── Scheduling policy ──

        /// @brief Default dispatching rule applied when no agent policy is active.
        public DispatchingRule dispatchingRule = DispatchingRule.SRT_SRWT;

        /// @brief Price per second of loaded AGV travel in the TECT machine rule (λ):
        ///        score = max(travel, queued work) + processing time + λ x travel.
        /// @details 0 = plain TECT (default, unchanged). TECT treats a trip as free whenever the machine's queue is
        ///          longer than the trip, but the AGV is busy for the whole trip; λ &gt; 0 charges for that, so a
        ///          far machine has to save more queueing to be picked (large λ ≈ stay local). Applies to every
        ///          TECT decision of the episode, warm-up included; other machine rules ignore it.
        public float TravelPrice = 0f;

        // ── Machine buffers ──

        /// @brief Input buffer size per machine: the most jobs that can wait at a machine (queued there or already
        ///        routed to it and on their way), not counting the one it is processing. 0 = unbounded (default).
        /// @details A slot is claimed when a job is routed, so an AGV never arrives at a full buffer: a machine
        ///          whose buffer is full is left out of the routing candidates, and a job with no machine that
        ///          has room waits in the routing pool. See DecisionCoordinator.HasInputRoom.
        public int InputBufferCapacity = 0;

        /// @brief Output buffer size per machine: finished jobs waiting at a machine for pickup. 0 = unbounded.
        /// @details Blocking after service: when the buffer is full the next finished job stays on the machine
        ///          and the machine cannot start another operation until an AGV collects one.
        ///          See DecisionCoordinator.IsOutputBlocked.
        public int OutputBufferCapacity = 0;

        // ── Processing-time-weighted machine rules (PECT / PTECT, 2026-10-10) ──

        /// @brief Parameters of the PECT / PTECT machine rules (alpha, shape, tau, re-route cost, availability).
        /// @details null = the defaults (PtimeRuleParams.Default: quad, alpha 3), the values the twin
        ///          (env/des_twin TwinConfig.ptime_*) and the RL action heads use. Set only when a scenario or config
        ///          message gives a ptime* key, and left out of the config fingerprint while null, so configs that do
        ///          not use it keep their config_hash.
        [FingerprintOmitIfNull]
        public PtimeRuleParams PtimeRule = null;

        /// @brief The PECT / PTECT parameters in effect: PtimeRule, or the defaults. A method, not a property:
        ///        ConfigFingerprint hashes every public property, and this one must not change config hashes.
        public PtimeRuleParams GetEffectivePtimeRule() => PtimeRule ?? PtimeRuleParams.Default;

        // ── Parking method ──

        /// @brief Parking layout: "lane" (default; reserved lane, one dedicated bay per AGV),
        ///        "single" (one abstract alcove) or "multiple" (per-aisle alcoves).
        public string parkingMethod = "lane";

        /// @brief Input/output belt docks: "corner" (default; the belt docks on the corner zone of the loop, so a
        ///        loading AGV blocks the corner) or "siding" (a two-zone one-way siding outside each side wall, so
        ///        the corner stays free for through traffic) or "bypass" (input siding that wraps around the corner and
        ///        merges into the top spine past it; output stays on its corner). See FactoryLayoutManager.IoDockMethod.
        public string ioDocks = "corner";

        /// @brief Factory layout (belt sides per row x aisle topology). Default legacy = today's floor, so a
        ///        config that omits it is unchanged. Immutable, shared by reference across per-seed clones.
        public LayoutSpec Layout = LayoutSpec.Default;

        /// @brief AGV zone-reservation protocol: "releasePrevious" (default since 2026-09-26) or "holdPrevious".
        ///        See ReservationProtocol. Validated at load by ReservationProtocolParser.
        public string reservationProtocol = ReservationProtocolParser.Default;

        /// @brief When routing decisions are made: "onTransport" (default) or "onReady" (legacy). See RoutingTrigger.
        public string routingTrigger = RoutingTriggerParser.Default;

        /// @brief Tiled floor: copies of the layout side by side, each with its own machines, belts, lane and AGVs
        ///        (docs/features/TILED_LAYOUT_SCOPE.md). Default one tile = today's floor. Immutable, shared by reference.
        public TilingSpec Tiling = TilingSpec.Single;

        // ── Pre-dispatching method ──

        /// @brief Strategy used for pre-dispatching decisions.
        /// @details Controls how jobs are pre-dispatched before execution.
        ///          Defaults to "fixed" (fixed dispatching schedule).
        public string preDispatchingMethod = "fixed";

        // ── Processing time distributions ──

        /// @brief Per-machine-type normal distribution parameters (mu, sigma) for processing time sampling.
        ///
        /// @details When populated, @c FJSSPJobGenerator will sample processing times from
        /// N(mu, sigma) for each type. Types not present in this dictionary fall back to
        /// a uniform sample within [MinProcTime, MaxProcTime].
        public Dictionary<MachineType, (float mu, float sigma)> ProcTimeParams
            = new Dictionary<MachineType, (float mu, float sigma)>();

        // ── Stochastic disruptions ──

        /// @brief Optional stochastic disruption parameters.
        /// @details Null = fully deterministic episode. Non-null activates @c StochasticEventManager
        ///          for the subset of disruption types flagged in the config. Deserialised from
        ///          the optional "stochastic" block in batch JSON configs.
        public StochasticConfig Stochastic = null;

        // ── Computed properties ──

        /// @brief Total number of machines in this configuration.
        /// @details Equals the length of @c MachineTypeLayout, or 0 if layout is null.
        public int TotalMachines => MachineTypeLayout?.Length ?? 0;

        // ── Methods ──

        /// @brief Returns a deep clone with a new seed.
        ///
        /// @param newSeed  Random seed for the cloned configuration.
        /// @returns         Deep clone with identical parameters except for the updated seed.
        ///
        /// @details Used by @c HeadlessBatchRunner to generate varied instances across
        ///          repeated runs while keeping all other parameters identical.
        ///          Note: @c Stochastic is shared by reference (not deep-cloned).
        public FJSSPConfig CloneWithSeed(int newSeed)
        {
            return new FJSSPConfig
            {
                Name = Name,
                Seed = newSeed,
                JobCount = JobCount,
                MachinesPerType = MachinesPerType,
                MachineTypeLayout = (MachineType[])MachineTypeLayout.Clone(),
                MinProcTime = MinProcTime,
                MaxProcTime = MaxProcTime,
                MinOpsPerJob = MinOpsPerJob,
                MaxOpsPerJob = MaxOpsPerJob,
                AGVCount = AGVCount,
                AgvScheduleStarts = (float[])AgvScheduleStarts?.Clone(),
                AgvScheduleCounts = (int[])AgvScheduleCounts?.Clone(),
                AGVMoveSpeed = AGVMoveSpeed,
                AGVHandshakeDuration = AGVHandshakeDuration,
                dispatchingRule = dispatchingRule,
                TravelPrice = TravelPrice,
                InputBufferCapacity = InputBufferCapacity,
                OutputBufferCapacity = OutputBufferCapacity,
                PtimeRule = PtimeRule,
                ProcTimeParams = new Dictionary<MachineType, (float mu, float sigma)>(ProcTimeParams),
                Stochastic = Stochastic,
                MachineFlexibilityProbability = MachineFlexibilityProbability,
                SecondaryTimeMultiplier = SecondaryTimeMultiplier,
                parkingMethod = parkingMethod,
                ioDocks = ioDocks,
                Layout = Layout,
                reservationProtocol = reservationProtocol,
                routingTrigger = routingTrigger,
                Tiling = Tiling,
                preDispatchingMethod = preDispatchingMethod,
                ThroughputTimingWindow = ThroughputTimingWindow,
            };
        }
    }

    /// @brief Parameters of the processing-time-weighted machine rules PECT and PTECT (2026-10-10, ported from
    ///        env/des_twin/rules.py ptime_scores; results/dev-pect-rule chose quad, alpha 3).
    /// @details Scores, lower first, over the routed job's candidate machines (p = processing time there, q = queued
    ///          work, tr = TECT's travel estimate, p_min / p_mean = the fastest / mean p over the candidates):
    ///            PECT  = q / A + p + alpha g(p)
    ///            PTECT = max(tr, q / A) + p + alpha g(p)
    ///          g by Shape: "lin" p; "quad" p^2 / p_mean; "ratio" p (p / p_min - 1); "exp" (1 - e^(-p / tau)) (p / 2 + R).
    ///          Doubles, as the twin's Python floats, so both sides compute the same float32 scores.
    public class PtimeRuleParams
    {
        public static readonly string[] Shapes = { "lin", "quad", "ratio", "exp" };

        /// @brief Scenario / config-message keys (camelCase JSON).
        public static readonly string[] JsonKeys = { "ptimeAlpha", "ptimeShape", "ptimeTau", "ptimeReroute", "ptimeAvail" };

        /// @brief The defaults: twin TwinConfig.ptime_* and env/config.py PTIME_RULE_PARAMS.
        public static readonly PtimeRuleParams Default = new PtimeRuleParams();

        public double Alpha = 3.0;
        public string Shape = "quad";
        public double Tau = 5544.0;
        public double Reroute = 0.0;
        public double Avail = 1.0;

        /// @brief The params a JSON object sets: null when it has none of JsonKeys (the defaults apply), otherwise
        ///        the defaults with the given keys replaced.
        public static PtimeRuleParams FromJson(Newtonsoft.Json.Linq.JObject root)
        {
            bool any = false;
            foreach (string k in JsonKeys) any |= root[k] != null;
            if (!any) return null;
            PtimeRuleParams d = Default;
            return new PtimeRuleParams
            {
                Alpha = root["ptimeAlpha"] != null ? (double)root["ptimeAlpha"] : d.Alpha,
                Shape = root["ptimeShape"] != null ? (string)root["ptimeShape"] : d.Shape,
                Tau = root["ptimeTau"] != null ? (double)root["ptimeTau"] : d.Tau,
                Reroute = root["ptimeReroute"] != null ? (double)root["ptimeReroute"] : d.Reroute,
                Avail = root["ptimeAvail"] != null ? (double)root["ptimeAvail"] : d.Avail,
            };
        }

        /// @brief Problems with these values (empty when valid); ConfigValidator reports them.
        public System.Collections.Generic.List<string> Problems()
        {
            var errors = new System.Collections.Generic.List<string>();
            if (System.Array.IndexOf(Shapes, Shape) < 0)
                errors.Add($"ptimeShape must be one of {string.Join(", ", Shapes)} (got '{Shape}')");
            if (double.IsNaN(Alpha) || double.IsInfinity(Alpha) || Alpha < 0) errors.Add($"ptimeAlpha must be >= 0 (got {Alpha})");
            if (double.IsNaN(Tau) || double.IsInfinity(Tau) || Tau <= 0) errors.Add($"ptimeTau must be > 0 (got {Tau})");
            if (double.IsNaN(Reroute) || double.IsInfinity(Reroute) || Reroute < 0) errors.Add($"ptimeReroute must be >= 0 (got {Reroute})");
            if (double.IsNaN(Avail) || double.IsInfinity(Avail) || Avail <= 0) errors.Add($"ptimeAvail must be > 0 (got {Avail})");
            return errors;
        }

        public override string ToString() =>
            $"{Shape} alpha={Alpha.ToString("R", System.Globalization.CultureInfo.InvariantCulture)}";
    }
}
