namespace Assets.Scripts.Simulation.Machines
{
    /// @brief Represents the health lifecycle of a physical machine.
    ///
    /// @details Used by @c PhysicalMachine to gate processing, by @c SimulationBridge
    /// to filter routing candidates and dispatch decisions, and encoded as a 4th channel
    /// in the 64×64 spatial occupancy tensor for the RL observation.
    ///
    /// State transitions (driven by @c SimulationBridge, not self-initiated):
    ///   Operational → Failed        (TTF countdown expires in PhysicalMachine.Update)
    ///   Failed       → Repairing    (SimulationBridge calls AcknowledgeFailure after job return)
    ///   Repairing    → Operational  (SimulationBridge calls AcknowledgeRepairComplete)
    public enum MachineHealthState
    {
        /// @brief Normal processing state. TTF countdown is running.
        Operational,

        /// @brief TTF has expired. FailedFlag is set; SimulationBridge is handling job return.
        /// Repair duration is already sampled and available via SampledRepairDuration.
        Failed,

        /// @brief Repair is in progress. RemainingRepairTime is counting down.
        /// Machine is idle but unavailable for new work.
        Repairing,
    }
}