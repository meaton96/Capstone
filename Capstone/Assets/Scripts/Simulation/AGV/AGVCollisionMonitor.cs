using System;
using System.Collections.Generic;
using UnityEngine;
using Assets.Scripts.Simulation.Types;

namespace Assets.Scripts.Simulation.AGV
{
    /// <summary>
    /// Detects AGV-AGV physical overlaps. The simulation has NO inter-AGV collision avoidance
    /// (AGVController drives straight at a zone centre; the Rigidbody is kinematic, so bodies pass
    /// through each other silently), and traffic-zone reservations are the only thing keeping AGVs
    /// apart. This monitor is purely diagnostic: it checks every AGV pair each fixed tick and never
    /// changes behaviour.
    ///
    /// Two thresholds are tracked, because they answer different questions:
    ///   * BODY OVERLAP   - the two oriented rectangles (BoxCollider size x lossyScale, top-down)
    ///                      intersect. This is an actual physical collision.
    ///   * CLEARANCE      - centre distance is below the sum of the two NavMeshAgent radii
    ///                      (2 x 1.18 = 2.36 units). The zone sizes were designed around this figure,
    ///                      so a violation means the zone design's own safety margin is broken even
    ///                      if the bodies do not touch.
    ///
    /// Any pair where either AGV is parking-related (Idle, or in/just out of the parking alcove;
    /// see AGVController.IsParkingRelated) is reported separately as a parking overlap and excluded
    /// from the headline numbers. Parking is an abstraction with no queueing model, so overlaps
    /// there are expected and out of scope; the headline counts are floor (aisle/spine) collisions.
    /// </summary>
    public class AGVCollisionMonitor
    {
        private const int MaxLoggedEvents = 2000;

        private struct OpenEvent
        {
            public int Id;
            public double Start;
            public float MinDistance;
            public string ZoneA, ZoneB, StateA, StateB;
            public Vector2 PosA, PosB;
        }

        private readonly Dictionary<long, OpenEvent> _open = new Dictionary<long, OpenEvent>();
        private readonly HashSet<long> _clearanceOpen = new HashSet<long>();
        private readonly HashSet<long> _parkingOpen = new HashSet<long>();
        private int _nextEventId;

        public readonly List<CollisionRecord> Events = new List<CollisionRecord>();

        /// <summary>Distinct contiguous body-overlap episodes between a pair, not both Idle.</summary>
        public int OverlapEvents { get; private set; }
        /// <summary>Pair-seconds of body overlap (two AGVs overlapping for 1 s = 1 pair-second).</summary>
        public double OverlapPairSeconds { get; private set; }
        /// <summary>Distinct contiguous clearance violations between a pair, not both Idle.</summary>
        public int ClearanceEvents { get; private set; }
        public double ClearancePairSeconds { get; private set; }
        /// <summary>Body-overlap episodes involving a parking-related AGV (excluded from the floor counts).</summary>
        public int ParkingOverlapEvents { get; private set; }
        /// <summary>Smallest centre-to-centre distance seen between any non-static pair; -1 if none.</summary>
        public float MinCentreDistance { get; private set; } = float.MaxValue;
        /// <summary>Events dropped from the CSV list because MaxLoggedEvents was reached (counters stay exact).</summary>
        public int DroppedEvents { get; private set; }

        public void Tick(IReadOnlyList<AGVController> fleet, double simTime, float dt,
                         Func<Vector3, string> zoneNameAt)
        {
            int n = fleet.Count;
            for (int i = 0; i < n; i++)
            {
                AGVController a = fleet[i];
                a.GetFootprint(out Vector2 ca, out Vector2 ax_a, out Vector2 az_a, out Vector2 ha);
                for (int j = i + 1; j < n; j++)
                {
                    AGVController b = fleet[j];
                    b.GetFootprint(out Vector2 cb, out Vector2 ax_b, out Vector2 az_b, out Vector2 hb);

                    long key = (long)a.AgvId * 100000L + b.AgvId;
                    bool parkingRelated = a.IsParkingRelated || b.IsParkingRelated;
                    float dist = Vector2.Distance(ca, cb);

                    bool overlap = dist < (ha.magnitude + hb.magnitude) &&
                                   OrientedBoxesOverlap(ca, ax_a, az_a, ha, cb, ax_b, az_b, hb);

                    if (parkingRelated)
                    {
                        // A floor overlap that was open when an AGV entered the alcove ends here.
                        if (_open.TryGetValue(key, out OpenEvent leaving)) Close(key, leaving, simTime);
                        _clearanceOpen.Remove(key);
                        if (overlap) { if (_parkingOpen.Add(key)) ParkingOverlapEvents++; }
                        else _parkingOpen.Remove(key);
                        continue;
                    }
                    _parkingOpen.Remove(key);

                    if (dist < MinCentreDistance) MinCentreDistance = dist;

                    bool clearance = dist < a.ClearanceRadius + b.ClearanceRadius;
                    if (clearance)
                    {
                        ClearancePairSeconds += dt;
                        if (_clearanceOpen.Add(key)) ClearanceEvents++;
                    }
                    else _clearanceOpen.Remove(key);

                    if (overlap)
                    {
                        OverlapPairSeconds += dt;
                        if (_open.TryGetValue(key, out OpenEvent ev))
                        {
                            if (dist < ev.MinDistance) { ev.MinDistance = dist; _open[key] = ev; }
                        }
                        else
                        {
                            OverlapEvents++;
                            _open[key] = new OpenEvent
                            {
                                Id = _nextEventId++, Start = simTime, MinDistance = dist,
                                ZoneA = zoneNameAt(a.transform.position) ?? "-",
                                ZoneB = zoneNameAt(b.transform.position) ?? "-",
                                StateA = a.State.ToString(), StateB = b.State.ToString(),
                                PosA = ca, PosB = cb,
                            };
                        }
                    }
                    else if (_open.TryGetValue(key, out OpenEvent done))
                    {
                        Close(key, done, simTime);
                    }
                }
            }
        }

        /// <summary>Closes any still-open overlaps at episode end so they are recorded.</summary>
        public void Finish(double simTime)
        {
            foreach (var kv in new List<KeyValuePair<long, OpenEvent>>(_open))
                Close(kv.Key, kv.Value, simTime);
        }

        private void Close(long key, OpenEvent ev, double endTime)
        {
            _open.Remove(key);
            if (Events.Count >= MaxLoggedEvents) { DroppedEvents++; return; }
            Events.Add(new CollisionRecord
            {
                EventId = ev.Id,
                StartTime = ev.Start,
                Duration = endTime - ev.Start,
                AgvA = (int)(key / 100000L),
                AgvB = (int)(key % 100000L),
                MinCentreDistance = ev.MinDistance,
                ZoneA = ev.ZoneA, ZoneB = ev.ZoneB,
                StateA = ev.StateA, StateB = ev.StateB,
                PosAx = ev.PosA.x, PosAz = ev.PosA.y, PosBx = ev.PosB.x, PosBz = ev.PosB.y,
            });
        }

        /// <summary>2D separating-axis test for two oriented rectangles (centre, unit axes, half extents).</summary>
        private static bool OrientedBoxesOverlap(Vector2 ca, Vector2 ax1, Vector2 az1, Vector2 ha,
                                                 Vector2 cb, Vector2 ax2, Vector2 az2, Vector2 hb)
        {
            Vector2 d = cb - ca;
            return !SeparatedOn(ax1, d, ax1, az1, ha, ax2, az2, hb)
                && !SeparatedOn(az1, d, ax1, az1, ha, ax2, az2, hb)
                && !SeparatedOn(ax2, d, ax1, az1, ha, ax2, az2, hb)
                && !SeparatedOn(az2, d, ax1, az1, ha, ax2, az2, hb);
        }

        private static bool SeparatedOn(Vector2 axis, Vector2 d,
                                        Vector2 ax1, Vector2 az1, Vector2 ha,
                                        Vector2 ax2, Vector2 az2, Vector2 hb)
        {
            float ra = ha.x * Mathf.Abs(Vector2.Dot(ax1, axis)) + ha.y * Mathf.Abs(Vector2.Dot(az1, axis));
            float rb = hb.x * Mathf.Abs(Vector2.Dot(ax2, axis)) + hb.y * Mathf.Abs(Vector2.Dot(az2, axis));
            return Mathf.Abs(Vector2.Dot(d, axis)) > ra + rb;
        }
    }
}
