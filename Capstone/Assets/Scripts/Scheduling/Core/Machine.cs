using System.Collections.Generic;

namespace Scheduling.Core
{
    public enum MachineState { Idle, Busy, Failed, Repair }

    /// <summary>
    /// Represents a single machine on the shop floor.
    /// Maintains a waiting queue of operations and tracks state.
    /// </summary>
    public class Machine
    {
        public int Id { get; }
        public MachineState State { get; set; } = MachineState.Idle;
        public Operation CurrentOperation { get; set; }
        public double BusyUntil { get; set; } = 0;

        // Jobs waiting to use this machine
        public List<Operation> WaitingQueue { get; } = new List<Operation>();

        public Machine(int id)
        {
            Id = id;
        }

        public void StartProcessing(Operation op, double currentTime)
        {
            State = MachineState.Busy;
            CurrentOperation = op;
            op.StartTime = currentTime;
            op.EndTime = currentTime + op.Duration;
            BusyUntil = op.EndTime;
        }

        public void FinishProcessing()
        {
            State = MachineState.Idle;
            CurrentOperation = null;
        }
    }
}
