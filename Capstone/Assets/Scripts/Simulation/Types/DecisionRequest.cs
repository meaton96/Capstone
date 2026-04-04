namespace Assets.Scripts.Simulation.Types
{
    /// @brief Snapshot of the state presented to the agent when a scheduling decision is needed.
    public struct DecisionRequest
    {
        /// @brief ID of the machine waiting for a job to be dispatched.
        public int MachineId;

        /// @brief Current simulation time in seconds when the decision was raised.
        public double SimTime;

        /// @brief Job IDs currently queued at the machine.
        public int[] QueuedJobIds;

        /// @brief Processing durations (in sim-seconds) corresponding to each queued job.
        public double[] QueuedDurations;

        /// @brief Sequential index of this decision point across the episode.
        public int DecisionIndex;

        /// @brief Total number of jobs in the current instance.
        public int TotalJobs;

        /// @brief Number of jobs that have completed all operations so far.
        public int CompletedJobs;
    }
}