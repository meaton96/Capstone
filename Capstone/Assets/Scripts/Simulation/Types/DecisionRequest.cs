using Assets.Scripts.Simulation.Machines;
namespace Assets.Scripts.Simulation.Types
{
    /// @brief Enumerates the two categories of scheduling decisions the agent can make.
    public enum DecisionType { Dispatch, Routing }

    /// @brief Snapshot of the simulation state presented to the agent when a scheduling decision is required.
    ///
    /// @details Carries both shared state (time, job counts) and decision-specific fields.
    ///          Only the fields relevant to @c Type are populated; unused arrays are null.
    ///          - Dispatch: @c MachineId identifies the idle machine, @c QueuedJobIds/@c QueuedDurations
    ///            list candidate jobs waiting in that machine's queue.
    ///          - Routing: @c JobId identifies the completed operation, @c RequiredType specifies the
    ///            machine type needed next, and @c CandidateMachineIds lists eligible destinations.
    public class DecisionRequest
    {
        /// @brief Type of decision: Dispatch (assign job to idle machine) or Routing (route finished job to next machine).
        public DecisionType Type;

        // ── Shared fields (populated for both decision types) ──

        /// @brief Current simulation time when this decision was generated.
        public double SimTime;

        /// @brief Sequential index of this decision point within the episode.
        public int DecisionIndex;

        /// @brief Total number of jobs in the episode.
        public int TotalJobs;

        /// @brief Number of jobs fully completed at the time of this decision.
        public int CompletedJobs;

        /// @brief ID of the machine that triggered this decision (idle machine for Dispatch,
        ///        finished machine for Routing).
        public int SourceMachineId;

        // ── Dispatch decision fields (populated when Type == Dispatch) ──

        /// @brief ID of the idle machine awaiting job assignment.
        public int MachineId;

        /// @brief IDs of jobs currently queued for this machine.
        public int[] QueuedJobIds;

        /// @brief Processing durations for each queued job at this machine (parallel to @c QueuedJobIds).
        public double[] QueuedDurations;

        // ── Routing decision fields (populated when Type == Routing) ──

        /// @brief ID of the job that just finished processing and needs routing.
        public int JobId;

        /// @brief Machine type required for the next operation of @c JobId.
        public MachineType RequiredType;

        /// @brief IDs of candidate machines of the required type that are available for routing.
        public int[] CandidateMachineIds;

        /// @brief IDs of all jobs that were simultaneously ready for a routing decision (had
        ///        >=1 available eligible machine) when this one was selected via job-priority
        ///        scoring (DispatchingEngine.SelectRoutingJob). Length 1 = JobId was the only
        ///        ready job (degenerate case, no rule ran). Diagnostic-only, not consumed by
        ///        SelectMachine.
        public int[] JobCandidateIds;

        /// @brief Current queue lengths at each candidate machine (parallel to @c CandidateMachineIds).
        public float[] CandidateQueueLengths;

        /// @brief Processing time of @c JobId at each candidate machine (parallel to @c CandidateMachineIds).
        public float[] CandidateJobTimes;

        /// @brief Live cumulative utilization ratio (busy time / operational time so far, in
        ///        [0, 1]) at each candidate machine (parallel to @c CandidateMachineIds). Used by
        ///        the MMUR (Minimum Machine Utilization) routing rule.
        public float[] CandidateUtilization;
    }
}
