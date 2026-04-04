using Assets.Scripts.Simulation.Machines;
namespace Assets.Scripts.Simulation.Types
{
    public enum DecisionType { Dispatch, Routing }
    /// @brief Snapshot of the state presented to the agent when a scheduling decision is needed.
    public struct DecisionRequest
    {
        public DecisionType Type;

        // --- Shared ---
        public double SimTime;
        public int DecisionIndex;
        public int TotalJobs;
        public int CompletedJobs;
        public int SourceMachineId;

        // --- Dispatch decision (idle machine, pick a job) ---
        public int MachineId;
        public int[] QueuedJobIds;
        public double[] QueuedDurations;

        // --- Routing decision (finished job, pick a machine) ---
        public int JobId;
        public MachineType RequiredType;
        public int[] CandidateMachineIds;   // eligible machines of required type
        public float[] CandidateQueueLengths;
        public float[] CandidateJobTimes;   // this job's proc time at each candidate
    }
}