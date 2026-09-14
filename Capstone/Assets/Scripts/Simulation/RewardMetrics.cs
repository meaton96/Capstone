using System;
using System.Collections.Generic;
using Assets.Scripts.Simulation.AGV;
using Assets.Scripts.Simulation.FactoryLayout;
using Assets.Scripts.Simulation.Jobs;
using Assets.Scripts.Simulation.Logging;
using Assets.Scripts.Simulation.Machines;
using Assets.Scripts.Simulation.Types;

namespace Assets.Scripts.Simulation
{
    /// <summary>
    /// Reward-agnostic snapshot of simulation state, sent to Python on every agent step (each
    /// decision and the terminal step) through <see cref="RewardMetricsSensor"/>. The scalar reward
    /// is computed entirely in Python (env/rewards) from differences between consecutive snapshots,
    /// so reward functions can be changed or swapped without rebuilding the player.
    ///
    /// The layout is a contract with env/rewards/metrics.py (METRIC_NAMES, checked by
    /// env/tests/test_rewards.py): only ever append new metrics at the end and bump
    /// <see cref="SchemaVersion"/>; never reorder or remove. "_sum" metrics are running totals since
    /// episode start, so a per-step term is simply curr - prev. Values travel as float32 — fine for
    /// per-step deltas at current episode scales, but very long episodes with hundreds of jobs will
    /// lose sub-second resolution on the largest sums.
    /// </summary>
    public static class RewardMetrics
    {
        public const int SchemaVersion = 2;

        public static readonly string[] Names =
        {
            "schema_version",
            "sim_time",
            "episode_active",
            "decision_count",

            "jobs_total",
            "jobs_exited",
            "wip",
            "ops_total",
            "ops_completed",

            "flow_time_exited_sum",       // sum of (exit - arrival) over exited jobs
            "time_in_system_sum",         // sum of time in system over all jobs, live (integral of WIP)

            "jobs_needs_routing",
            "jobs_waiting_pickup",
            "jobs_in_transit",
            "jobs_queued",
            "jobs_processing",

            "time_needs_routing_sum",
            "time_waiting_pickup_sum",
            "time_in_transit_sum",
            "time_queued_sum",
            "time_processing_sum",

            "machines_total",
            "machines_busy",
            "machines_down",
            "machine_downtime_sum",

            "agvs_total",
            "agvs_idle",
            "agv_time_traveling_sum",
            "agv_time_waiting_route_sum",
            "agv_time_idle_sum",
            "agv_trips_total",

            "zone_block_events_sum",

            "deadlock",
            "timed_out",
            "all_jobs_exited",

            // v2
            "episode_seed",               // instance seed from EpisodeSeedChannel, -1 if none was queued
            "episode_seed_index",         // that seed's position in the queue since the last clear, -1 if none
        };

        public static int Count => Names.Length;

        /// <summary>
        /// Writes one snapshot into <paramref name="buffer"/> in <see cref="Names"/> order. Null
        /// collections (before the factory has spawned) are treated as empty.
        /// </summary>
        public static void Fill(float[] buffer, double simTime, bool episodeActive, int decisionCount,
                                JobStore jobs, IReadOnlyList<PhysicalMachine> machines,
                                IReadOnlyList<AGVController> agvs, IReadOnlyList<TrafficZone> zones,
                                EpisodeTracker tracker, bool deadlock, bool timedOut,
                                int episodeSeed, int episodeSeedIndex)
        {
            int jobsTotal = 0, exited = 0, opsTotal = 0, opsDone = 0;
            int nRouting = 0, nWaiting = 0, nTransit = 0, nQueued = 0, nProcessing = 0;
            double flowExited = 0, timeInSystem = 0;
            double tRouting = 0, tWaiting = 0, tTransit = 0, tQueued = 0, tProcessing = 0;

            if (jobs != null)
            {
                foreach (JobData job in jobs.AllJobs)
                {
                    jobsTotal++;
                    opsTotal += job.TotalOperations;
                    opsDone += job.CompletedOps;
                    tRouting += job.TimeNeedsRouting;
                    tWaiting += job.TimeWaitingPickup;
                    tTransit += job.TimeInTransit;
                    tQueued += job.TimeQueued;
                    tProcessing += job.TimeProcessing;

                    // Per-state times only accumulate on TransitionTo, so add the still-open
                    // interval of the current state to make the sums live.
                    double open = Math.Max(0.0, simTime - job.StateEntryTime);
                    switch (job.State)
                    {
                        case JobState.NeedsRouting: nRouting++; tRouting += open; break;
                        case JobState.WaitingForPickup: nWaiting++; tWaiting += open; break;
                        case JobState.InTransit: nTransit++; tTransit += open; break;
                        case JobState.Queued: nQueued++; tQueued += open; break;
                        case JobState.Processing: nProcessing++; tProcessing += open; break;
                    }

                    if (job.State == JobState.Exited)
                    {
                        exited++;
                        double flow = job.ExitTime - job.ArrivalTime;
                        flowExited += flow;
                        timeInSystem += flow;
                    }
                    else
                    {
                        timeInSystem += Math.Max(0.0, simTime - job.ArrivalTime);
                    }
                }
            }

            int machinesBusy = 0, machinesDown = 0;
            double downtime = 0;
            if (machines != null)
            {
                foreach (PhysicalMachine machine in machines)
                {
                    if (!machine.IsIdle) machinesBusy++;
                    if (machine.HealthState != MachineHealthState.Operational) machinesDown++;
                    if (tracker != null) downtime += tracker.DowntimeSoFar(machine.MachineId, simTime);
                }
            }

            int agvsIdle = 0, trips = 0;
            double travel = 0, waitRoute = 0, idle = 0;
            if (agvs != null)
            {
                foreach (AGVController agv in agvs)
                {
                    if (agv.IsIdle) agvsIdle++;
                    AGVRecord record = agv.GetRecord(simTime);
                    travel += record.TimeTraveling;
                    waitRoute += record.TimeWaitingRoute;
                    idle += record.TimeIdle;
                    trips += record.TotalTrips;
                }
            }

            int blockEvents = 0;
            if (zones != null)
            {
                foreach (TrafficZone zone in zones)
                    blockEvents += zone.BlockEvents;
            }

            int i = 0;
            buffer[i++] = SchemaVersion;
            buffer[i++] = (float)simTime;
            buffer[i++] = episodeActive ? 1f : 0f;
            buffer[i++] = decisionCount;

            buffer[i++] = jobsTotal;
            buffer[i++] = exited;
            buffer[i++] = jobsTotal - exited;
            buffer[i++] = opsTotal;
            buffer[i++] = opsDone;

            buffer[i++] = (float)flowExited;
            buffer[i++] = (float)timeInSystem;

            buffer[i++] = nRouting;
            buffer[i++] = nWaiting;
            buffer[i++] = nTransit;
            buffer[i++] = nQueued;
            buffer[i++] = nProcessing;

            buffer[i++] = (float)tRouting;
            buffer[i++] = (float)tWaiting;
            buffer[i++] = (float)tTransit;
            buffer[i++] = (float)tQueued;
            buffer[i++] = (float)tProcessing;

            buffer[i++] = machines?.Count ?? 0;
            buffer[i++] = machinesBusy;
            buffer[i++] = machinesDown;
            buffer[i++] = (float)downtime;

            buffer[i++] = agvs?.Count ?? 0;
            buffer[i++] = agvsIdle;
            buffer[i++] = (float)travel;
            buffer[i++] = (float)waitRoute;
            buffer[i++] = (float)idle;
            buffer[i++] = trips;

            buffer[i++] = blockEvents;

            buffer[i++] = deadlock ? 1f : 0f;
            buffer[i++] = timedOut ? 1f : 0f;
            buffer[i++] = jobsTotal > 0 && exited == jobsTotal ? 1f : 0f;

            buffer[i++] = episodeSeed;
            buffer[i++] = episodeSeedIndex;

            if (i != Count)
                SimLogger.Error($"[RewardMetrics] Fill wrote {i} values but Names has {Count} — they are out of sync.");
        }
    }
}
