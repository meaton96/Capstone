using System.Collections.Generic;
using UnityEngine;
using Assets.Scripts.Simulation.Machines;
using Assets.Scripts.Simulation.Types;
using Assets.Scripts.Simulation.Logging;

namespace Assets.Scripts.Simulation
{
    /// <summary>
    /// Accumulates per-episode and per-machine statistics during a simulation run.
    /// 
    /// This class is owned as a field by SimulationBridge and manages all statistical
    /// tracking including processing times, failure counts, repair durations, and
    /// machine availability metrics.
    /// 
    /// Lifecycle:
    ///   - StartEpisode: Reset() clears all accumulated state
    ///   - Update: RecordX() methods accumulate statistics each frame
    ///   - FinaliseEpisode: Build() constructs the final EpisodeRecord and fires OnEpisodeFinished
    /// </summary>
    public class EpisodeTracker
    {
        // ── Episode-level accumulators ────────────────────────────────────────

        /// <summary>Total number of machine failures during the current episode.</summary>
        private int _machineFailureCount;

        /// <summary>Total machine repair time (in simulation seconds) during the current episode.</summary>
        private float _machineRepairTime;

        // ── Per-machine accumulators ──────────────────────────────────────────

        /// <summary>Cumulative processing time per machine ID.</summary>
        private readonly Dictionary<int, double> _processingTime = new();

        /// <summary>Cumulative downtime (repair time) per machine ID.</summary>
        private readonly Dictionary<int, double> _totalDowntime = new();

        /// <summary>Simulation time when a machine entered a failed state (keyed by machine ID).</summary>
        private readonly Dictionary<int, double> _downtimeStart = new();

        /// <summary>Number of failures per machine ID.</summary>
        private readonly Dictionary<int, int> _failureCount = new();

        /// <summary>Total repair time per machine ID.</summary>
        private readonly Dictionary<int, float> _repairTime = new();

        /// <summary>Number of operations completed per machine ID.</summary>
        private readonly Dictionary<int, int> _opsCompleted = new();

        /// <summary>Simulation time of the last completed operation per machine ID (used for TTF validation).</summary>
        private readonly Dictionary<int, double> _lastOpTime = new();

        /// <summary>Job exits within the currently-open throughput window. Reset on each window close.</summary>
        private int _windowExits;

        /// <summary>Cumulative job exits this episode.</summary>
        private int _totalExits;

        /// <summary>One record per closed throughput window. Attached to the EpisodeRecord in Build().</summary>
        private readonly List<ThroughputWindowRecord> _throughputWindows = new();

        // ── Reset ─────────────────────────────────────────────────────────────

        /// <summary>
        /// Clears all accumulated statistics. Call at the start of every episode.
        /// </summary>
        public void Reset()
        {
            _machineFailureCount = 0;
            _machineRepairTime = 0f;

            _processingTime.Clear();
            _totalDowntime.Clear();
            _downtimeStart.Clear();
            _failureCount.Clear();
            _repairTime.Clear();
            _opsCompleted.Clear();
            _lastOpTime.Clear();
            _windowExits = 0;
            _totalExits = 0;
            _throughputWindows.Clear();
        }

        // ── Recording API — called from SimBridge Update/handlers ─────────────

        /// <summary>
        /// Records that a job has reached JobState.Exited. Called from FlagHarvester.HarvestAGVFlags
        /// when a delivery completes with machineId &lt; 0.
        /// </summary>
        public void RecordJobExit()
        {
            _windowExits++;
            _totalExits++;
        }

        /// <summary>
        /// Snapshots throughput for the window [windowStart, windowEnd) and resets the per-window
        /// exit counter. Driven by FactoryOrchestrator's window clock.
        /// </summary>
        /// <param name="windowStart">Sim-time at the start of the window.</param>
        /// <param name="windowEnd">Sim-time at window close (current SimTime for the trailing partial window).</param>
        /// <param name="workInProgress">Jobs in the system (not Exited) at close time.</param>
        public void CloseThroughputWindow(double windowStart, double windowEnd, int workInProgress)
        {
            _throughputWindows.Add(new ThroughputWindowRecord
            {
                WindowStartTime = windowStart,
                WindowEndTime = windowEnd,
                WindowSeconds = (float)(windowEnd - windowStart),
                JobsCompleted = _windowExits,
                CumulativeCompleted = _totalExits,
                WorkInProgress = workInProgress,
            });
            _windowExits = 0;
        }

        /// <summary>
        /// Records that a machine has started an operation. Called when StartJob fires.
        /// </summary>
        /// <param name="machineId">The ID of the machine starting the operation.</param>
        public void RecordOperationStart(int machineId)
        {
            // Processing time is measured in TickProcessing — no action needed here.
        }

        /// <summary>
        /// Accumulates processing time for a machine during a frame where it is active.
        /// Called from HarvestMachineFlags or a dedicated update pass.
        /// </summary>
        /// <param name="machineId">The ID of the machine.</param>
        /// <param name="dt">Delta time in simulation seconds (Time.deltaTime at current timescale).</param>
        public void AddProcessingTime(int machineId, double dt)
        {
            _processingTime.TryAdd(machineId, 0.0);
            _processingTime[machineId] += dt;
        }

        /// <summary>
        /// Records that an operation has completed on a machine.
        /// Called from HarvestMachineFlags when FinishedFlag is detected.
        /// </summary>
        /// <param name="machineId">The ID of the machine that completed the operation.</param>
        public void RecordOperationComplete(int machineId)
        {
            _opsCompleted.TryAdd(machineId, 0);
            _opsCompleted[machineId]++;
        }

        /// <summary>
        /// Records a machine failure event. Called immediately after the failure flag is detected.
        /// </summary>
        /// <param name="machineId">The ID of the failed machine.</param>
        /// <param name="repairDuration">Estimated repair duration in simulation seconds.</param>
        /// <param name="simTime">Current simulation time at the moment of failure.</param>
        public void RecordMachineFailure(int machineId, float repairDuration, double simTime)
        {
            // Episode totals
            _machineFailureCount++;
            _machineRepairTime += repairDuration;

            // Per-machine totals
            _failureCount.TryAdd(machineId, 0);
            _failureCount[machineId]++;

            _repairTime.TryAdd(machineId, 0f);
            _repairTime[machineId] += repairDuration;

            // Open a downtime interval — closed in RecordRepairComplete
            _downtimeStart[machineId] = simTime;

            SimLogger.Low($"[EpisodeTracker] Machine {machineId} failed. " +
                          $"Episode total failures={_machineFailureCount}");
        }

        /// <summary>
        /// Records that a machine has returned to operational state after repair.
        /// Called from HandleMachineRepairComplete.
        /// </summary>
        /// <param name="machineId">The ID of the repaired machine.</param>
        /// <param name="simTime">Current simulation time when the repair completed.</param>
        public void RecordRepairComplete(int machineId, double simTime)
        {
            if (_downtimeStart.TryGetValue(machineId, out double start))
            {
                _totalDowntime.TryAdd(machineId, 0.0);
                _totalDowntime[machineId] += simTime - start;
                _downtimeStart.Remove(machineId);
            }

            _lastOpTime[machineId] = simTime;
        }

        // ── Build ─────────────────────────────────────────────────────────────

        /// <summary>
        /// Constructs the final EpisodeRecord from all accumulated statistics.
        /// Call once at FinaliseEpisode, after closing any open downtime intervals.
        /// </summary>
        /// <param name="config">Simulation configuration containing instance parameters.</param>
        /// <param name="simTime">Total simulation time (makespan) for the episode.</param>
        /// <param name="ruleName">Name of the dispatching rule used.</param>
        /// <param name="completedJobs">Number of jobs fully completed during the episode.</param>
        /// <param name="totalOps">Total number of operations scheduled in the episode.</param>
        /// <param name="decisionPoints">Number of dispatching decisions made.</param>
        /// <param name="totalReward">Cumulative reward accumulated during the episode.</param>
        /// <param name="agvCount">Number of AGVs in the simulation.</param>
        /// <param name="machines">Collection of all physical machines for per-machine record generation.</param>
        /// <param name="averageTimeScale">Average simulation timescale used during the episode.</param>
        /// <returns>Constructed EpisodeRecord containing all accumulated statistics.</returns>
        public EpisodeRecord Build(
            FJSSPConfig config,
            double simTime,
            string ruleName,
            int completedJobs,
            int totalOps,
            int decisionPoints,
            double totalReward,
            int agvCount,
            IEnumerable<PhysicalMachine> machines,
            float averageTimeScale = 100f)
        {
            // Close any downtime intervals still open (episode ended before repair completed).
            foreach (var machine in machines)
            {
                int mid = machine.MachineId;
                if (_downtimeStart.TryGetValue(mid, out double start))
                {
                    _totalDowntime.TryAdd(mid, 0.0);
                    _totalDowntime[mid] += simTime - start;
                    _downtimeStart.Remove(mid);
                }
            }

            var record = new EpisodeRecord
            {
                InstanceName = config?.Name ?? string.Empty,
                RuleName = ruleName,
                Seed = config?.Seed ?? 0,
                Makespan = simTime,
                TotalReward = totalReward,
                JobCount = config?.JobCount ?? 0,
                MachineCount = config?.TotalMachines ?? 0,
                AGVCount = agvCount,
                TotalOperations = totalOps,
                CompletedJobs = completedJobs,
                DecisionPoints = decisionPoints,
                AverageTimeScale = averageTimeScale,
                Stochastic = config?.Stochastic,
                MachineFailureCount = _machineFailureCount,
                MachineRepairTime = _machineRepairTime,
            };

            // Build per-machine records
            foreach (var machine in machines)
            {
                int mid = machine.MachineId;
                double timeProc = _processingTime.TryGetValue(mid, out double tp) ? tp : 0.0;
                double downtime = _totalDowntime.TryGetValue(mid, out double td) ? td : 0.0;
                double timeOp = simTime - downtime;
                double availability = simTime > 0 ? timeOp / simTime : 1.0;

                record.MachineRecords.Add(new MachineRecord
                {
                    MachineId = mid,
                    MachineType = machine.PrimaryType.ToString(),
                    OpsCompleted = _opsCompleted.TryGetValue(mid, out int ops) ? ops : 0,
                    TimeProcessing = timeProc,
                    TimeOperational = timeOp,
                    AvailabilityRate = availability,
                    FailureCount = _failureCount.TryGetValue(mid, out int fc) ? fc : 0,
                    RepairTime = _repairTime.TryGetValue(mid, out float rt) ? rt : 0f,
                });
            }

            record.ThroughputRecords.AddRange(_throughputWindows);

            return record;
        }

        // ── Observation helpers (used by ObservationBuilder) ──────────────────

        /// <summary>
        /// Gets the total machine failure count for the current episode.
        /// Used by Global Scalars observation.
        /// </summary>
        public int EpisodeMachineFailures => _machineFailureCount;

        /// <summary>
        /// Gets the total machine repair time for the current episode.
        /// Used by Global Scalars observation.
        /// </summary>
        public float EpisodeMachineRepairTime => _machineRepairTime;

        /// <summary>
        /// Computes the theoretical mean time-to-failure (TTF) for a Weibull distribution
        /// with shape parameter k=1.5 and scale parameter lambda.
        /// </summary>
        /// <param name="lambda">Scale parameter (lambda) of the Weibull distribution.</param>
        /// <returns>Theoretical mean TTF: lambda × Γ(1 + 1/k) ≈ lambda × 0.9027 for k=1.5.</returns>
        public static float TheoreticalMeanTTF(float lambda) => lambda * 0.9027f;
    }
}