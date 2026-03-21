using System;
using System.Collections.Generic;
namespace Assets.Scripts.Scheduling.Core
{
    /// @brief Discriminated union of simulation event types processed by @ref DESSimulator.
    public enum EventType
    {
        JobArrived,         ///< A job has arrived at its next required machine and is ready to be queued or started.
        OperationComplete,  ///< An operation that was running on a machine has finished processing.
    }

    /// @brief Immutable record representing a single scheduled event in the simulation timeline.
    ///
    /// @details Implements @c IComparable<SimEvent> so that instances can be stored directly
    /// in a @c SortedSet inside @ref EventQueue. Ordering is by @ref Time ascending, with
    /// @ref SequenceNumber as a FIFO tiebreaker for events scheduled at the same instant.
    public class SimEvent : IComparable<SimEvent>
    {
        /// @brief Simulation time at which this event is scheduled to occur.
        public double Time { get; }

        /// @brief The category of this event, determining how @ref DESSimulator dispatches it.
        public EventType Type { get; }

        /// @brief ID of the job associated with this event. Set to @c -1 if not applicable.
        public int JobId { get; }

        /// @brief ID of the machine associated with this event. Set to @c -1 if not applicable.
        public int MachineId { get; }

        /// @brief Monotonically increasing counter assigned at enqueue time.
        ///
        /// @details Used as a FIFO tiebreaker so that events with identical @ref Time values
        /// are processed in the order they were enqueued. A lower sequence number indicates
        /// higher priority.
        public long SequenceNumber { get; }

        /// @brief Constructs a fully specified, immutable simulation event.
        ///
        /// @param time Simulation time at which the event fires.
        /// @param type The @ref EventType that classifies this event.
        /// @param jobId ID of the relevant job, or @c -1 if not applicable.
        /// @param machineId ID of the relevant machine, or @c -1 if not applicable.
        /// @param seq Sequence number assigned by @ref EventQueue to enforce FIFO ordering.
        public SimEvent(double time, EventType type, int jobId, int machineId, long seq)
        {
            Time = time;
            Type = type;
            JobId = jobId;
            MachineId = machineId;
            SequenceNumber = seq;
        }

        /// @brief Compares this event to another for priority queue ordering.
        ///
        /// @details Primary sort is by @ref Time ascending. When two events share the same
        /// time, @ref SequenceNumber is used as a FIFO tiebreaker, ensuring earlier-enqueued
        /// events are processed first.
        ///
        /// @param other The event to compare against.
        /// @returns A negative value if this event should be processed before @p other,
        /// zero if they are equal, or a positive value if @p other has higher priority.
        public int CompareTo(SimEvent other)
        {
            int cmp = Time.CompareTo(other.Time);
            if (cmp != 0) return cmp;
            return SequenceNumber.CompareTo(other.SequenceNumber);
        }
    }

    /// @brief Min-heap priority queue for simulation events, ordered by time with FIFO tiebreaking.
    ///
    /// @details Backed by a @c SortedSet<SimEvent> which exploits @ref SimEvent's
    /// @c IComparable implementation to maintain heap order without a separate comparator.
    /// A monotonically increasing @ref _sequence counter is stamped onto each event at
    /// enqueue time to guarantee stable FIFO ordering among events with identical timestamps.
    public class EventQueue
    {
        private readonly SortedSet<SimEvent> _events = new SortedSet<SimEvent>();
        private long _sequence = 0;

        /// @brief Number of events currently in the queue.
        public int Count => _events.Count;

        /// @brief @c true when the queue contains at least one event; @c false when empty.
        public bool HasEvents => _events.Count > 0;

        /// @brief Creates and enqueues a new @ref SimEvent at the specified simulation time.
        ///
        /// @details Atomically stamps the event with the next available sequence number
        /// before insertion, ensuring FIFO ordering is preserved for same-time events
        /// regardless of call order.
        ///
        /// @param time Simulation time at which the event should fire.
        /// @param type The @ref EventType classifying this event.
        /// @param jobId ID of the associated job. Defaults to @c -1 if not applicable.
        /// @param machineId ID of the associated machine. Defaults to @c -1 if not applicable.
        ///
        /// @returns The newly created and inserted @ref SimEvent.
        public SimEvent Enqueue(double time, EventType type, int jobId = -1, int machineId = -1)
        {
            var evt = new SimEvent(time, type, jobId, machineId, _sequence++);
            _events.Add(evt);
            return evt;
        }

        /// @brief Removes and returns the earliest-scheduled event in the queue.
        ///
        /// @details Retrieves the minimum element from the underlying @c SortedSet,
        /// which corresponds to the event with the lowest @ref SimEvent.Time and,
        /// among ties, the lowest @ref SimEvent.SequenceNumber.
        ///
        /// @returns The @ref SimEvent with the highest dispatch priority.
        public SimEvent Dequeue()
        {
            var first = _events.Min;
            _events.Remove(first);
            return first;
        }

        /// @brief Returns the earliest-scheduled event without removing it from the queue.
        ///
        /// @returns The @ref SimEvent with the highest dispatch priority, or @c null if the queue is empty.
        public SimEvent Peek() => _events.Min;

        /// @brief Removes all events from the queue and resets the sequence counter to zero.
        ///
        /// @details Should be called during @ref DESSimulator.Reset() to ensure sequence
        /// numbers restart cleanly between simulation runs, preventing counter overflow
        /// across long multi-run benchmark sessions.
        public void Clear()
        {
            _events.Clear();
            _sequence = 0;
        }
    }
}