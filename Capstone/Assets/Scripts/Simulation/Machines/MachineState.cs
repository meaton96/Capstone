namespace Assets.Scripts.Simulation.Machines
{
    /// @brief Represents the operational state of a machine within the simulation.
    ///
    /// @details This enum is used by @c MachineVisual to determine indicator colors and
    /// status text, and by @c PhysicalMachine to track processing lifecycle. Each state
    /// maps to a corresponding material in the @c indicatorMaterials array on @c MachineVisual.
    public enum MachineState
    {
        /// @brief Machine is idle and available to accept new jobs.
        Idle,

        /// @brief Machine is actively processing a job.
        Busy,

        /// @brief Machine has finished processing but cannot release the job (outgoing belt/full).
        /// The job is held on the machine until space becomes available.
        Blocked,

        /// @brief Machine has experienced a failure and is non-operational.
        /// Requires repair before it can resume processing.
        Failed,

        /// @brief Machine is undergoing repair and is unavailable for work.
        /// Progress is tracked via the associated progress bar.
        Repair,
    }
}
