using System.Collections.Generic;
using Assets.Scripts.Simulation.Machines;

namespace Assets.Scripts.Simulation.Jobs
{
    /// <summary>
    /// Defines a static job specification for the Flexible Job Shop Scheduling Problem (FJSSP).
    /// </summary>
    /// <remarks>
    /// This class is a blueprint used during job initialization. It is converted into
    /// live <see cref="JobData"/> instances at runtime by <see cref="JobStore"/>.
    /// </remarks>
    public class FJSSPJobDefinition
    {
        /// <summary>
        /// Unique identifier for this job within the simulation.
        /// </summary>
        public int JobId;

        /// <summary>
        /// Time at which the job becomes available in the system.
        /// A value of 0 indicates the job is available at episode start.
        /// </summary>
        public float ArrivalTime;

        /// <summary>
        /// Ordered array of <see cref="MachineType"/> values representing the required
        /// sequence of operations for this job.
        /// </summary>
        public MachineType[] OperationSequence;

        /// <summary>
        /// Per-operation mapping of eligible machine IDs to their processing times.
        /// Index corresponds to <see cref="OperationSequence"/> position.
        /// </summary>
        /// <remarks>
        /// Each dictionary maps a runtime machine ID to the processing time (in
        /// simulation seconds) that machine would take for the corresponding operation.
        /// </remarks>
        public Dictionary<int, float>[] EligibleMachinesPerOp;
    }
}
