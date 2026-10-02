using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using Newtonsoft.Json;
using UnityEngine;
using Assets.Scripts.Simulation.AGV;
using Assets.Scripts.Simulation.FactoryLayout;
using Assets.Scripts.Simulation.Jobs;
using Assets.Scripts.Simulation.Types;

namespace Assets.Scripts.Simulation.Logging
{
    /// <summary>
    /// Exports what the event-based twin (env/des_twin) needs to rebuild a run without the physical floor
    /// (thesis: event-based vs physically simulated training environments). Enabled by "-destrace".
    ///
    ///   des_floor.json  zone graph (centres, boxes, one-way links, docks), machine docks and belts, parking
    ///                   bays, AGV start poses and motion parameters, written at episode start.
    ///   des_jobs.json   the resolved job list (arrival, per-op eligible machine -> duration) exactly as the
    ///                   episode loaded it, so the twin never re-implements ScenarioLoader's machine mapping.
    ///   agv_events.csv  transport milestones (AGVController.TraceEvent, written by ResultsLogger).
    ///
    /// The twin then runs the same jobs and dispatching logic with travel taken from the graph and the AGV
    /// kinematics but no zone reservations, so its gap to this run is the effect of AGVs sharing the floor.
    /// </summary>
    public static class DesTwinExport
    {
        /// <summary>Set by HeadlessBatchRunner from "-destrace". Off by default: agv_events.csv is ~1 MB per run.</summary>
        public static bool Enabled;

        public const string FloorFile = "des_floor.json";
        public const string JobsFile = "des_jobs.json";

        private static float[] XZ(Vector3 v) => new[] { R(v.x), R(v.z) };
        private static float R(float f) => (float)System.Math.Round(f, 4);
        // Observation frame: unrounded, so the twin's grid cells and distances match ObservationBuilder's.
        private static float[] XYZ(Vector3 v) => new[] { v.x, v.y, v.z };
        private static int[] Cell(FactoryLayoutManager layout, Vector3 v)
        {
            Vector2Int c = ObservationBuilder.GridCellOf(layout, v);
            return new[] { c.x, c.y };
        }

        /// <summary>Writes des_floor.json and des_jobs.json for the episode that is starting.</summary>
        public static void Write(FJSSPConfig config, FactoryLayoutManager layout, TrafficZoneManager traffic,
                                 AGVPool pool, IEnumerable<FJSSPJobDefinition> jobs, int preDispatchLeadTime,
                                 int episodeSeed)
        {
            if (!Enabled || layout == null || traffic == null || pool == null) return;
            try
            {
                File.WriteAllText(ResultsLogger.PathFor(FloorFile),
                    JsonConvert.SerializeObject(BuildFloor(config, layout, traffic, pool, preDispatchLeadTime), Formatting.Indented));
                File.WriteAllText(ResultsLogger.PathFor(JobsFile),
                    JsonConvert.SerializeObject(BuildJobs(config, jobs, episodeSeed)));
            }
            catch (System.Exception ex)
            {
                SimLogger.Error($"[DesTwinExport] Export failed: {ex.Message}");
            }
        }

        private static object BuildFloor(FJSSPConfig config, FactoryLayoutManager layout, TrafficZoneManager traffic,
                                         AGVPool pool, int preDispatchLeadTime)
        {
            AGVKinematics k = pool.AllAGVs.Count > 0 ? pool.AllAGVs[0].Kinematics : default;

            var zones = traffic.Zones.Select(z => new
            {
                id = z.ZoneId,
                name = z.Name,
                centre = XZ(z.Centre),
                size = XZ(z.Size),
                capacity = z.Capacity,
                parking_lane = z.IsParkingLane,
                down = z.Downstream.ToArray(),
                up = z.Upstream.ToArray(),
                docks = z.DockPoints.Select(kv => new
                {
                    key = kv.Key,
                    approach = XZ(kv.Value.ApproachPosition),
                    handshake = XZ(kv.Value.HandshakePosition),
                    facing = XZ(kv.Value.FacingDirection),
                    pickup = kv.Value.IsPickup,
                }).ToArray(),
            }).ToArray();

            var machines = layout.Machines.Select(m => new
            {
                id = m.MachineId,
                type = m.PrimaryType.ToString(),
                tile = layout.TileOfMachine(m.MachineId),
                zones = traffic.GetZonesForMachine(m.MachineId).ToArray(),
                pickup_zones = traffic.GetPickupZonesForMachine(m.MachineId).ToArray(),
                pickup_pos = XZ(m.GetPickupPosition()),
                // GetDropoffPosition is the input end of the first non-full incoming belt (primary when both are
                // full), so the twin tracks belt fill; dropoff_pos is the answer with every belt empty.
                dropoff_pos = XZ(m.GetDropoffPosition()),
                in_belts = m.IncomingBeltsInOrder().Select(b => new
                {
                    input_pos = XZ(b.InputEndPosition),
                    capacity = b.Capacity,
                }).ToArray(),
                // Observation (des_twin.observation): machine position, its grid cell, every capability.
                pos3 = XYZ(m.transform.position),
                cell = Cell(layout, m.transform.position),
                capabilities = m.Capabilities.OrderBy(c => c == m.PrimaryType ? 0 : 1).ThenBy(c => (int)c)
                                .Select(c => c.ToString()).ToArray(),
            }).ToArray();

            var tiles = Enumerable.Range(0, layout.TileCount).Select(t => new
            {
                tile = t,
                incoming_key = TrafficZoneManager.IncomingDockKey(t),
                outgoing_key = TrafficZoneManager.OutgoingDockKey(t),
                incoming_zone = traffic.GetZoneIdForDock(TrafficZoneManager.IncomingDockKey(t)),
                outgoing_zone = traffic.GetZoneIdForDock(TrafficZoneManager.OutgoingDockKey(t)),
                incoming_pos = XZ(layout.IncomingBeltPositionOf(t)),
                outgoing_pos = XZ(layout.OutgoingBeltPositionOf(t)),
                incoming_pos3 = XYZ(layout.IncomingBeltPositionOf(t)),
                incoming_cell = Cell(layout, layout.IncomingBeltPositionOf(t)),
            }).ToArray();

            ObservationBuilder.TryGridFrame(layout, out float gcx, out float gcz, out float ghw, out float ghd);
            var obs = new
            {
                grid_centre = new[] { gcx, gcz },
                grid_half = new[] { ghw, ghd },
                floor_size = new[] { layout.FloorSize.x, layout.FloorSize.y },
            };

            var agvs = pool.AllAGVs.Select(a =>
            {
                Vector3 park = pool.GetParkingPosition(a.AgvId);
                return new
                {
                    id = a.AgvId,
                    tile = a.TileId,
                    park_pos = XZ(park),
                    park_zone = traffic.GetZoneAtPosition(park)?.ZoneId ?? -1,
                    pos = XZ(a.transform.position),
                    yaw = R(a.transform.eulerAngles.y),
                    zone = a.CurrentZoneId,
                };
            }).ToArray();

            return new
            {
                schema = "des_floor/1",
                instance = config.Name,
                layout = layout.ActiveLayout?.Id,
                agv_count = pool.AllAGVs.Count,
                tile_count = layout.TileCount,
                agvs_pooled = layout.AgvsPooled,
                parking_method = config.parkingMethod,
                routing_trigger = config.routingTrigger,
                reservation_protocol = config.reservationProtocol,
                fixed_dt = Time.fixedDeltaTime,
                pre_dispatch_lead = preDispatchLeadTime,
                obs,
                parking_bayed = traffic.ParkingIsBayed,
                agv = new
                {
                    speed = k.MoveSpeed,
                    turn_speed = k.TurnSpeed,
                    path_turn_threshold = k.PathTurnThreshold,
                    alignment_threshold = k.AlignmentThreshold,
                    waypoint_arrival_dist = k.WaypointArrivalDist,
                    dock_arrival_dist = k.DockArrivalDist,
                    handshake = k.HandshakeDuration,
                },
                zones,
                machines,
                tiles,
                agvs,
            };
        }

        private static object BuildJobs(FJSSPConfig config, IEnumerable<FJSSPJobDefinition> jobs, int episodeSeed)
        {
            return new
            {
                schema = "des_jobs/1",
                instance = config.Name,
                seed = episodeSeed >= 0 ? episodeSeed : config.Seed,
                jobs = jobs.OrderBy(j => j.ArrivalTime).ThenBy(j => j.JobId).Select(j => new
                {
                    id = j.JobId,
                    arrival = j.ArrivalTime,
                    ops = Enumerable.Range(0, j.EligibleMachinesPerOp.Length).Select(o => new
                    {
                        type = j.OperationSequence[o].ToString(),
                        // Insertion order is kept: ties in the machine rules resolve by candidate order.
                        eligible = j.EligibleMachinesPerOp[o].Select(kv => new object[] { kv.Key, kv.Value }).ToArray(),
                    }).ToArray(),
                }).ToArray(),
            };
        }
    }
}
