using System.Collections.Generic;
using UnityEngine;
using Assets.Scripts.Simulation.Machines;
using Assets.Scripts.Simulation.Types;
using Assets.Scripts.Simulation.Logging;

namespace Assets.Scripts.Simulation
{
    /// <summary>
    /// Accumulates all per-episode and per-machine statistics during a simulation run.
    /// Owned by SimulationBridge as a field; completely replaces the scattered
    /// _episode* and _machine* tracking dictionaries that previously lived in the bridge.
    ///
    /// Lifecycle:
    ///   bridge.StartEpisode()  → tracker.Reset(config)
    ///   bridge.Update()        → tracker.RecordX() calls
    ///   bridge.FinaliseEpisode() → record = tracker.Build(layoutManager, ...)
    ///                           → OnEpisodeFinished.Invoke(record)
    ///
    /// Adding a new stochastic event (e.g. AGV failures):
    ///   1. Add per-episode and per-machine fields to this class
    ///   2. Add a RecordAGVFailure() method
    ///   3. Add AGV fields to EpisodeRecord / MachineRecord
    ///   4. Populate them in Build()
    ///   5. Add columns in ResultsLogger
    ///   Nothing else changes.
    /// </summary>
    public class EpisodeTracker
    {
        // ── Episode-level accumulators ────────────────────────────────────────

        private int _machineFailureCount;
        private float _machineRepairTime;

        // Phase 3 — uncomment when AGV failures are implemented:
        // private int   _agvFailureCount;
        // private float _agvRepairTime;

        // Phase 4 — uncomment when dynamic arrivals are implemented:
        // private int   _dynamicArrivals;

        // ── Per-machine accumulators ──────────────────────────────────────────

        // Processing time: how long each machine was actively processing a job
        private readonly Dictionary<int, double> _processingTime = new();

        // Downtime: cumulative repair time (only > 0 in stochastic runs)
        private readonly Dictionary<int, double> _totalDowntime = new();

        // Open downtime interval start (machine is currently repairing)
        private readonly Dictionary<int, double> _downtimeStart = new();

        // Per-machine failure counts and repair time totals
        private readonly Dictionary<int, int> _failureCount = new();
        private readonly Dictionary<int, float> _repairTime = new();

        // Ops completed per machine
        private readonly Dictionary<int, int> _opsCompleted = new();

        // TTF observations — for stochastic validation logging
        private readonly Dictionary<int, double> _lastOpTime = new();

        // ── Reset ─────────────────────────────────────────────────────────────

        /// <summary>
        /// Call at the start of every episode to wipe accumulated state.
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
        }

        // ── Recording API — called from SimBridge Update/handlers ─────────────

        /// <summary>
        /// Record a machine starting an operation. Call when StartJob fires.
        /// </summary>
        public void RecordOperationStart(int machineId)
        {
            // Processing time is measured in TickProcessing — no action needed here.
            // Provided as a hook if you want to measure scheduling latency.
        }

        /// <summary>
        /// Accumulate processing time each frame a machine is active.
        /// Call from HarvestMachineFlags or a dedicated update pass.
        /// dt = Time.deltaTime (already in sim-seconds at current timescale).
        /// </summary>
        public void AddProcessingTime(int machineId, double dt)
        {
            _processingTime.TryAdd(machineId, 0.0);
            _processingTime[machineId] += dt;
        }

        /// <summary>
        /// Record a completed operation on a machine.
        /// Call from HarvestMachineFlags when FinishedFlag is detected.
        /// </summary>
        public void RecordOperationComplete(int machineId)
        {
            _opsCompleted.TryAdd(machineId, 0);
            _opsCompleted[machineId]++;
        }

        /// <summary>
        /// Record a machine failure.
        /// Call from HandleMachineFailure immediately after the flag is detected.
        /// simTime = SimulationBridge.SimTime at the moment of failure.
        /// </summary>
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
        /// Record a machine returning to operational after repair.
        /// Call from HandleMachineRepairComplete.
        /// </summary>
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

        // Phase 3 — AGV failure recording (add when implementing Phase 3):
        // public void RecordAGVFailure(int agvId, float repairDuration, double simTime) { ... }
        // public void RecordAGVRepairComplete(int agvId, double simTime) { ... }

        // Phase 4 — dynamic arrival recording:
        // public void RecordJobArrival(double simTime) { _dynamicArrivals++; }

        // ── Build ─────────────────────────────────────────────────────────────

        /// <summary>
        /// Constructs the final EpisodeRecord from all accumulated data.
        /// Call once at FinaliseEpisode, after closing any open downtime intervals.
        /// </summary>
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
            // Close any downtime intervals still open (machine failed, episode ended
            // before repair completed — valid in high-disruption stochastic runs).
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

            return record;
        }

        // ── Observation helpers (used by ObservationBuilder) ──────────────────

        /// <summary>
        /// Episode-level failure count — exposed for Global Scalars observation.
        /// </summary>
        public int EpisodeMachineFailures => _machineFailureCount;

        /// <summary>
        /// Episode-level repair time — exposed for Global Scalars observation.
        /// </summary>
        public float EpisodeMachineRepairTime => _machineRepairTime;

        /// <summary>
        /// Theoretical mean TTF for validation logging.
        /// k=1.5 Weibull: mean = lambda × Γ(1+1/k) ≈ lambda × 0.9027
        /// </summary>
        public static float TheoreticalMeanTTF(float lambda) => lambda * 0.9027f;
    }
}