namespace Assets.Scripts.Simulation.Machines
{
    /// @brief Defines the functional type of a machine in the factory simulation.
    ///
    /// @details Each machine type corresponds to a specific manufacturing operation
    /// and is associated with a unique color in @c MachineVisual for visual identification.
    /// Machine types are used for job routing, capability matching, and scheduling decisions.
    public enum MachineType
    {
        /// @brief Milling machine — removes material from a workpiece using rotating cutters.
        Mill = 0,

        /// @brief Lathe — rotates the workpiece to cut with a stationary tool.
        Lathe = 1,

        /// @brief Welding station — joins workpieces through welding processes.
        Weld = 2,

        /// @brief Inspection station — performs quality checks on workpieces.
        Inspect = 3,

        /// @brief Assembly station — assembles multiple components into a final product.
        Assemble = 4
    }
}

