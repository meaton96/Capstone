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
                AGVMoveSpeed = AGVMoveSpeed,
                AGVHandshakeDuration = AGVHandshakeDuration,
                dispatchingRule = dispatchingRule,
                TravelPrice = TravelPrice,
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
}
