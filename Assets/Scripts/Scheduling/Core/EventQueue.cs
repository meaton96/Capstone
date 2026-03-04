using System;
using System.Collections.Generic;

namespace Scheduling.Core
{
    public enum EventType
    {
        JobArrived,
        OperationComplete,
    }

    public class SimEvent : IComparable<SimEvent>
    {
        public double Time { get; }
        public EventType Type { get; }
        public int JobId { get; }
        public int MachineId { get; }

        // Tiebreaker: lower sequence number = higher priority (FIFO for same time)
        public long SequenceNumber { get; }

        public SimEvent(double time, EventType type, int jobId, int machineId, long seq)
        {
            Time = time;
            Type = type;
            JobId = jobId;
            MachineId = machineId;
            SequenceNumber = seq;
        }

        public int CompareTo(SimEvent other)
        {
            int cmp = Time.CompareTo(other.Time);
            if (cmp != 0) return cmp;
            return SequenceNumber.CompareTo(other.SequenceNumber);
        }
    }

    /// <summary>
    /// Min-heap priority queue for simulation events.
    /// Events are ordered by time, with FIFO tiebreaking via sequence numbers.
    /// </summary>
    public class EventQueue
    {
        private readonly SortedSet<SimEvent> _events = new SortedSet<SimEvent>();
        private long _sequence = 0;

        public int Count => _events.Count;
        public bool HasEvents => _events.Count > 0;

        public SimEvent Enqueue(double time, EventType type, int jobId = -1, int machineId = -1)
        {
            var evt = new SimEvent(time, type, jobId, machineId, _sequence++);
            _events.Add(evt);
            return evt;
        }

        public SimEvent Dequeue()
        {
            var first = _events.Min;
            _events.Remove(first);
            return first;
        }

        public SimEvent Peek() => _events.Min;

        public void Clear()
        {
            _events.Clear();
            _sequence = 0;
        }
    }
}
