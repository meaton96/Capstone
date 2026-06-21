namespace Assets.Scripts.Scheduling.Core
{
    /// @brief Represents a single operation within a job: process on a given machine for a fixed duration.
    ///
    /// @details Operations are immutable after construction except for @ref StartTime and @ref EndTime,
    /// which are set to @c -1 at creation and populated by @ref DESSimulator during the simulation run.
    /// Each operation belongs to exactly one job and targets exactly one machine.
    public class Operation
    {
        /// @brief ID of the job this operation belongs to.
        public int JobId { get; }

        /// @brief Zero-based index of this operation within its parent job's operation sequence.
        public int OperationIndex { get; }

        /// @brief ID of the machine on which this operation must be processed.
        public int MachineId { get; }

        /// @brief Processing time required to complete this operation, in simulation time units.
        public int Duration { get; }

        /// @brief Simulation time at which this operation began processing.
        /// @details Initialised to @c -1 and set by @ref Machine.StartProcessing during simulation.
        public double StartTime { get; set; } = -1;

        /// @brief Simulation time at which this operation finished processing.
        /// @details Initialised to @c -1 and set by @ref Machine.StartProcessing during simulation.
        /// Equal to @ref StartTime + @ref Duration once the operation has been scheduled.
        public double EndTime { get; set; } = -1;

        /// @brief Constructs a fully specified, immutable operation.
        ///
        /// @param jobId ID of the owning job.
        /// @param opIndex Zero-based position of this operation in the job's sequence.
        /// @param machineId ID of the machine that must process this operation.
        /// @param duration Processing time in simulation time units.
        public Operation(int jobId, int opIndex, int machineId, int duration)
        {
            JobId = jobId;
            OperationIndex = opIndex;
            MachineId = machineId;
            Duration = duration;
        }
    }

    /// @brief An ordered sequence of @ref Operation objects representing a single job.
    ///
    /// @details Tracks execution progress via @ref NextOperationIndex, which advances each
    /// time an operation completes inside @ref DESSimulator.HandleOperationComplete.
    /// @ref CurrentOperation and @ref IsComplete are derived directly from this index and
    /// require no additional state.
    public class Job
    {
        /// @brief Unique identifier for this job, matching its index in @ref DESSimulator.Jobs.
        public int Id { get; }

        /// @brief Ordered array of operations that must be executed sequentially.
        public Operation[] Operations { get; }

        /// @brief Index of the next operation to be dispatched or currently in progress.
        /// @details Starts at @c 0 and is incremented by @ref DESSimulator after each
        /// @ref EventType.OperationComplete event. When equal to @c Operations.Length
        /// the job is considered complete.
        public int NextOperationIndex { get; set; } = 0;

        /// @brief Simulation time at which this job entered the system.
        /// @details Currently fixed at @c 0 for all jobs in Taillard benchmark instances.
        /// Reserved for future dynamic arrival scheduling.
        public double ArrivalTime { get; }

        /// @brief Simulation time at which the job's final operation completed.
        /// @details Initialised to @c -1 and set by @ref DESSimulator.HandleOperationComplete
        /// when @ref IsComplete first becomes @c true. Used to compute the makespan.
        public double CompletionTime { get; set; } = -1;

        /// @brief @c true when all operations have been processed; @c false otherwise.
        public bool IsComplete => NextOperationIndex >= Operations.Length;

        /// @brief The next operation awaiting dispatch, or @c null if the job is complete.
        public Operation CurrentOperation => IsComplete ? null : Operations[NextOperationIndex];

        /// @brief Constructs a job with a fixed operation sequence and an optional arrival time.
        ///
        /// @param id Unique job identifier.
        /// @param operations Ordered array of operations to execute. Must not be null or empty.
        /// @param arrivalTime Simulation time at which the job enters the system. Defaults to @c 0.
        public Job(int id, Operation[] operations, double arrivalTime = 0)
        {
            Id = id;
            Operations = operations;
            ArrivalTime = arrivalTime;
        }
    }
}