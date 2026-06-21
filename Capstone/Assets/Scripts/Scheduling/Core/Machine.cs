using System.Collections.Generic;
namespace Assets.Scripts.Scheduling.Core
{
    /// @brief Operational states a machine can occupy during simulation.
    ///
    /// @details @ref MachineState.Failed and @ref MachineState.Repair are reserved for future
    /// fault-injection extensions and are not currently handled by @ref DESSimulator.
    public enum MachineState
    {
        Idle,   ///< Machine is free and ready to accept a new operation.
        Busy,   ///< Machine is currently processing an operation.
        Blocked,
        Failed, ///< Machine has experienced a fault and cannot accept work. Reserved for future use.
        Repair, ///< Machine is undergoing repair after a fault. Reserved for future use.
    }

    /// @brief Represents a single machine on the shop floor.
    ///
    /// @details Maintains the machine's current @ref MachineState, the @ref Operation it is
    /// actively processing, and a @ref WaitingQueue of operations that have arrived but cannot
    /// yet start. State transitions are driven exclusively by @ref DESSimulator via
    /// @ref StartProcessing and @ref FinishProcessing.
    public class Machine
    {
        /// @brief Unique identifier for this machine, matching its index in @ref DESSimulator.Machines.
        public int Id { get; }

        /// @brief Current operational state of the machine.
        /// @details Initialised to @ref MachineState.Idle. Transitions to @ref MachineState.Busy
        /// on @ref StartProcessing and back to @ref MachineState.Idle on @ref FinishProcessing.
        public MachineState State { get; set; } = MachineState.Idle;

        /// @brief The operation currently being processed, or @c null when the machine is idle.
        /// @details Set by @ref StartProcessing and cleared to @c null by @ref FinishProcessing.
        public Operation CurrentOperation { get; set; }

        /// @brief Simulation time at which the current operation will finish.
        /// @details Mirrors @ref Operation.EndTime of @ref CurrentOperation and is used by
        /// @ref DESSimulator to schedule the corresponding @ref EventType.OperationComplete event.
        /// Reset to @c 0 by @ref DESSimulator.Reset between runs.
        public double BusyUntil { get; set; } = 0;

        /// @brief Ordered list of operations waiting to be processed on this machine.
        /// @details Operations are appended by @ref DESSimulator.TryDispatchJob when the machine
        /// is busy, and removed by @ref DESSimulator.TryStartNextOnMachine when the machine
        /// becomes free. Selection order is determined by @ref DispatchingRules.SelectNext.
        public List<Operation> WaitingQueue { get; } = new List<Operation>();

        /// @brief Constructs an idle machine with an empty waiting queue.
        ///
        /// @param id Unique machine identifier matching its index in @ref DESSimulator.Machines.
        public Machine(int id)
        {
            Id = id;
        }

        /// @brief Transitions the machine to @ref MachineState.Busy and begins processing an operation.
        ///
        /// @details Sets @ref State to @ref MachineState.Busy, assigns @p op to
        /// @ref CurrentOperation, stamps @ref Operation.StartTime with @p currentTime,
        /// and computes @ref Operation.EndTime as @p currentTime + @ref Operation.Duration.
        /// @ref BusyUntil is updated to match @ref Operation.EndTime so the simulator can
        /// schedule the @ref EventType.OperationComplete event.
        ///
        /// @param op The operation to begin processing. Must not be @c null.
        /// @param currentTime The current simulation time at which processing starts.
        public void StartProcessing(Operation op, double currentTime)
        {
            State = MachineState.Busy;
            CurrentOperation = op;
            op.StartTime = currentTime;
            op.EndTime = currentTime + op.Duration;
            BusyUntil = op.EndTime;
        }

        /// @brief Transitions the machine to @ref MachineState.Idle after an operation completes.
        ///
        /// @details Clears @ref CurrentOperation and sets @ref State to @ref MachineState.Idle.
        /// Does not select the next operation from @ref WaitingQueue — that responsibility
        /// belongs to @ref DESSimulator.TryStartNextOnMachine, which is called immediately
        /// after this method in @ref DESSimulator.HandleOperationComplete.
        public void FinishProcessing()
        {
            State = MachineState.Idle;
            CurrentOperation = null;
        }
    }
}