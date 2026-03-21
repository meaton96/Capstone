namespace Scheduling.Core
{
    /// <summary>
    /// Represents a single operation within a job: 
    /// "process on machine X for Y time units."
    /// </summary>
    public class Operation
    {
        public int JobId { get; }
        public int OperationIndex { get; }
        public int MachineId { get; }
        public int Duration { get; }

        // Filled in during simulation
        public double StartTime { get; set; } = -1;
        public double EndTime { get; set; } = -1;

        public Operation(int jobId, int opIndex, int machineId, int duration)
        {
            JobId = jobId;
            OperationIndex = opIndex;
            MachineId = machineId;
            Duration = duration;
        }
    }

    /// <summary>
    /// A job is an ordered sequence of operations.
    /// Tracks which operation is next.
    /// </summary>
    public class Job
    {
        public int Id { get; }
        public Operation[] Operations { get; }
        public int NextOperationIndex { get; set; } = 0;

        public double ArrivalTime { get; }
        public double CompletionTime { get; set; } = -1;

        public bool IsComplete => NextOperationIndex >= Operations.Length;
        public Operation CurrentOperation => IsComplete ? null : Operations[NextOperationIndex];

        public Job(int id, Operation[] operations, double arrivalTime = 0)
        {
            Id = id;
            Operations = operations;
            ArrivalTime = arrivalTime;
        }
    }
}
