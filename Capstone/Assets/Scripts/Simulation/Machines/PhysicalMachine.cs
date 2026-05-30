using UnityEngine;
using Assets.Scripts.Simulation.Jobs;
using Assets.Scripts.Simulation.Stochastic;
using Assets.Scripts.Simulation.Logging;
using Assets.Scripts.Simulation.Channels;
using System.Collections.Generic;

namespace Assets.Scripts.Simulation.Machines
{
    /// @brief Represents a physical processing unit within the factory simulation.
    ///
    /// @details Manages processing timers, state flags, and (when stochastic mode is
    /// active) a Weibull time-to-failure countdown and a log-normal repair timer.
    ///
    /// This component is strictly passive: it never initiates state transitions or
    /// communicates directly with the orchestrator. @c SimulationBridge polls
    /// @c FinishedFlag, @c FailedFlag, and @c RepairCompleteFlag each frame to drive
    /// the factory lifecycle. All mutation of @c JobData remains in SimulationBridge.
    ///
    /// Failure lifecycle (stochastic mode only):
    ///   1. @c _ttfCountdown reaches zero in @c Update.
    ///   2. Repair duration is sampled immediately; @c FailedFlag is raised.
    ///   3. @c SimulationBridge detects @c FailedFlag, handles job return and AGV
    ///      re-routing, then calls @c AcknowledgeFailure() to move to Repairing.
    ///   4. @c _repairCountdown counts down in @c Update; @c RepairCompleteFlag is raised.
    ///   5. @c SimulationBridge calls @c AcknowledgeRepairComplete(); machine samples a
    ///      fresh TTF (age reset to zero post-repair) and returns to Operational.
    [RequireComponent(typeof(Collider))]
    [RequireComponent(typeof(Rigidbody))]
    public class PhysicalMachine : MonoBehaviour
    {
        // ── Identity ─────────────────────────────────────────────────────────

        /// @brief The unique identifier for this machine instance.
        public int MachineId { get; private set; }

        /// @brief The primary functional type of this machine.
        public MachineType PrimaryType { get; private set; }

        /// @brief The set of machine types this machine is capable of processing.
        public HashSet<MachineType> Capabilities { get; private set; }

        /// @brief Checks whether this machine can process the specified operation type.
        ///
        /// @param opType The @c MachineType to check.
        /// @return True if the machine's @c Capabilities set contains @p opType.
        public bool CanProcess(MachineType opType) => Capabilities.Contains(opType);

        // ── Normal processing state ──────────────────────────────────────────

        /// @brief True when the machine is idle and available to accept new jobs.
        public bool IsIdle { get; private set; } = true;

        /// @brief Set to true when the active job is complete. Polled by @c SimulationBridge.
        public bool FinishedFlag { get; private set; }

        /// @brief The ID of the job currently being processed, or -1 if idle.
        public int ActiveJobId { get; private set; } = -1;

        /// @brief Set to true when the active job is nearing completion (within @c PreDispatchLeadTime).
        public bool AlmostDoneFlag { get; private set; }

        /// @brief The ID of the job that is nearing completion.
        public int AlmostDoneJobId { get; private set; } = -1;

        /// @brief Remaining time to complete the current job.
        private float remainingTime;

        /// @brief Total time required for the current job.
        private float totalDuration;

        /// @brief True if the @c AlmostDoneFlag has already been fired for the current job.
        private bool almostDoneFired;

        // ── Health state machine ──────────────────────────────────────────────

        /// @brief Current health state of the machine.
        ///
        /// @details Encoded as a 4th channel in the spatial occupancy tensor:
        ///   Operational = 0.0,  Repairing = 0.5,  Failed = 1.0
        public MachineHealthState HealthState { get; private set; } = MachineHealthState.Operational;

        /// @brief Set to true when the time-to-failure (TTF) countdown expires.
        ///
        /// @details Polled by @c SimulationBridge, which handles job return and AGV
        /// re-routing before calling @c AcknowledgeFailure() to transition to Repairing.
        public bool FailedFlag { get; private set; }

        /// @brief Set to true when the repair countdown reaches zero.
        ///
        /// @details Polled by @c SimulationBridge, which calls @c AcknowledgeRepairComplete()
        /// to transition back to Operational.
        public bool RepairCompleteFlag { get; private set; }

        /// @brief Repair duration sampled at the moment of failure using a log-normal distribution.
        ///
        /// @details Available immediately once @c FailedFlag is raised, so the observation
        /// builder can read it before @c AcknowledgeFailure() is called.
        public float SampledRepairDuration { get; private set; }

        /// @brief Remaining repair time in seconds. Counts down in @c Update while @c HealthState is Repairing.
        ///
        /// @details Normalize against @c SampledRepairDuration for the Global Scalars observation channel.
        public float RemainingRepairTime { get; private set; }

        /// @brief True when this machine can accept new work.
        ///
        /// @details Use this property to filter routing candidates and dispatch decisions
        /// in @c SimulationBridge. Returns false when the machine is Failed or Repairing.
        public bool IsAvailableForWork => HealthState == MachineHealthState.Operational;

        /// @brief Returns the health state encoded as a float for the spatial occupancy tensor.
        ///
        /// @details Operational → 0.0f, Repairing → 0.5f, Failed → 1.0f.
        public float HealthStateEncoded => HealthState switch
        {
            MachineHealthState.Failed => 1.0f,
            MachineHealthState.Repairing => 0.5f,
            _ => 0.0f,
        };

        /// @brief Time-to-failure (TTF) countdown in seconds. When at @c float.MaxValue,
        /// the machine is effectively failure-free.
        private float _ttfCountdown = float.MaxValue;

        /// @brief Accumulated operational time since the last repair completion.
        private float _ageSinceLastRepair = 0f;

        // ── Visual & conveyor references ──────────────────────────────────────

        /// @brief Serialized references to connected conveyor belts (used for job visual management).
        [Header("Conveyor Belts (visual only)")]
        [SerializeField] private ConveyorBelt incomingConveyor;
        [SerializeField] private ConveyorBelt outgoingConveyor;
        [SerializeField] private ConveyorBelt secondaryIncomingConveyor;
        [SerializeField] private ConveyorBelt secondaryOutgoingConveyor;

        /// @brief Reference to the paired visual layer component.
        private MachineVisual visualLayer;

        // ── Initialisation ────────────────────────────────────────────────────

        /// @brief Initializes the machine with its identity, capabilities, and visual layer.
        ///
        /// @details Resets all processing and health state flags to their initial values.
        /// The capabilities set defaults to a single-element set containing @p primary
        /// if no explicit capabilities are provided.
        ///
        /// @param id Unique identifier for the machine instance.
        /// @param primary The functional @c MachineType (e.g., Mill, Lathe).
        /// @param capabilities Optional set of additional machine types this machine can process.
        public void Initialize(int id, MachineType primary,
                        IEnumerable<MachineType> capabilities = null)
        {
            MachineId = id;
            PrimaryType = primary;
            Capabilities = capabilities != null
                ? new HashSet<MachineType>(capabilities)
                : new HashSet<MachineType> { primary };
            IsIdle = true;
            FinishedFlag = false;
            ActiveJobId = -1;
            HealthState = MachineHealthState.Operational;
            FailedFlag = false;
            RepairCompleteFlag = false;
            _ttfCountdown = float.MaxValue;
            RemainingRepairTime = 0f;
            SampledRepairDuration = 0f;

            visualLayer = GetComponent<MachineVisual>();
            visualLayer?.Initialise(id, PrimaryType);
            ClearConveyors();
        }

#if UNITY_EDITOR
        /// @brief Forces an immediate machine failure for editor testing.
        ///
        /// @details Sets the TTF countdown to zero so that @c TickTTF fires on the
        /// next Update. Only operates when the machine is currently Operational.
        public void DEBUG_ForceFailure()
        {
            if (HealthState != MachineHealthState.Operational) return;
            _ttfCountdown = 0f; // TickTTF will fire on the next Update
        }
#endif

        /// @brief Seeds this machine's time-to-failure (TTF) countdown for the current episode.
        ///
        /// @details Called by @c SimulationBridge.StartEpisode() after the job store
        /// is initialized. Applies initial age randomization: each machine starts at a
        /// random point in its first wear-out cycle rather than all failing simultaneously
        /// after one full TTF.
        ///
        /// When @c StochasticEventManager.Instance.MachineFailuresEnabled is false, the
        /// countdown is set to @c float.MaxValue, making the machine effectively immortal
        /// for that episode.
        public void InitializeStochastic()
        {
            // Reset health to operational in case this is a new episode after a previous
            // stochastic one that ended mid-repair.
            HealthState = MachineHealthState.Operational;
            FailedFlag = false;
            RepairCompleteFlag = false;
            RemainingRepairTime = 0f;
            SampledRepairDuration = 0f;

            if (StochasticEventManager.Instance == null ||
                !StochasticEventManager.Instance.MachineFailuresEnabled)
            {
                _ttfCountdown = float.MaxValue;
                return;
            }

            // Sample a full Weibull TTF, then draw the starting countdown from
            // Uniform(0, fullTtf) so machines are not all fresh at episode start.
            // This prevents the policy from exploiting a failure-free warmup period.
            float fullTtf = StochasticEventManager.Instance.SampleMachineTTF();
            _ttfCountdown = Random.Range(0f, fullTtf);
            _ageSinceLastRepair = fullTtf - _ttfCountdown;
        }

        // ── Processing control ────────────────────────────────────────────────

        /// @brief Begins processing a specific job for a defined duration.
        ///
        /// @details Removes the job from any incoming conveyor, snaps the visual to
        /// the machine position, and notifies the visual layer. The processing timer
        /// begins counting down in @c TickProcessing.
        ///
        /// @param jobId The identifier of the job being processed.
        /// @param duration The time in simulation seconds to complete the operation.
        /// @param visual The 3D representation of the job to be snapped to the machine.
        public void StartJob(int jobId, float duration, JobVisual visual = null)
        {
            ActiveJobId = jobId;
            remainingTime = duration;
            totalDuration = duration;
            IsIdle = false;
            FinishedFlag = false;
            AlmostDoneFlag = false;
            AlmostDoneJobId = -1;
            almostDoneFired = false;

            RemoveFromAnyIncoming(jobId);

            if (visual != null)
            {
                visual.SetOnConveyor(false);
                visual.SnapToPosition(transform.position);
            }
            visualLayer.BeginOperation(jobId, Time.time, duration);
        }

        /// @brief Clears the finished flag and resets processing state after the orchestrator acknowledges completion.
        ///
        /// @details Called by @c SimulationBridge after a job is removed from the machine.
        /// Resets @c IsIdle, @c ActiveJobId, and notifies the visual layer.
        public void ClearFinished()
        {
            FinishedFlag = false;
            IsIdle = true;
            ActiveJobId = -1;
            visualLayer?.CompleteOperation(-1);
        }

        /// @brief Clears the pre-dispatch signaling flags after the near-complete job has been handled.
        ///
        /// @details Called by @c SimulationBridge after processing the @c AlmostDoneFlag.
        public void ClearAlmostDone()
        {
            AlmostDoneFlag = false;
            AlmostDoneJobId = -1;
        }

        // ── Failure acknowledgement (called by SimulationBridge) ─────────────

        /// @brief Transitions the machine from Failed to Repairing state.
        ///
        /// @details Called by @c SimulationBridge after it has handled job return
        /// and AGV re-routing. The repair countdown (already sampled into @c SampledRepairDuration)
        /// begins here. The machine is left idle but unavailable for new work until repair completes.
        ///
        /// @post @c FailedFlag is cleared, @c HealthState is set to Repairing,
        /// and @c visualLayer.BeginRepair() is called with the sampled duration.
        public void AcknowledgeFailure()
        {
            FailedFlag = false;
            HealthState = MachineHealthState.Repairing;
            SimLogger.Medium($"[PhysicalMachine] Machine [{MachineId}] failure");
            // Clear any processing state — the active job has been returned by SimulationBridge.
            IsIdle = true;
            ActiveJobId = -1;
            FinishedFlag = false;
            AlmostDoneFlag = false;
            AlmostDoneJobId = -1;
            almostDoneFired = false;
            remainingTime = 0f;



            // exposes a repair animation (e.g. a maintenance icon / colour tint).
            visualLayer?.CompleteOperation(-1); // at minimum, stop the processing animation
            visualLayer?.BeginRepair(SampledRepairDuration);
            SimLogger.Medium($"[PhysicalMachine] Machine [{MachineId}] begin repair");
        }

        /// @brief Transitions the machine from Repairing to Operational and arms the next TTF countdown.
        ///
        /// @details Called by @c SimulationBridge after detecting @c RepairCompleteFlag.
        /// The machine's age is considered zero post-repair, so a fresh Weibull TTF
        /// is sampled rather than resuming accumulated lifetime.
        ///
        /// @post @c RepairCompleteFlag is cleared, @c HealthState is set to Operational,
        /// @c _ageSinceLastRepair is reset to zero, and a new TTF is sampled.
        public void AcknowledgeRepairComplete()
        {
            RepairCompleteFlag = false;
            RemainingRepairTime = 0f;
            HealthState = MachineHealthState.Operational;

            // Fresh TTF from repaired state — age counter resets to zero.
            _ttfCountdown = StochasticEventManager.Instance != null
                ? StochasticEventManager.Instance.SampleMachineTTF()
                : float.MaxValue;
            float actualRepair = SampledRepairDuration - RemainingRepairTime; // already 0 here, so use SampledRepairDuration
            EpisodeTelemetryChannel.Instance?.RecordMachineRepairComplete(MachineId, SampledRepairDuration);
            _ageSinceLastRepair = 0f;   // age resets — machine is "new" post-repair

            SimLogger.Medium($"Machine [{MachineId}] repair complete");
            visualLayer?.EndRepair();
        }

        // ── Conveyor helpers ──────────────────────────────────────────────────

        /// @brief Places a job visual onto the most appropriate available incoming conveyor belt.
        ///
        /// @details Prefers @ref incomingConveyor, falls back to @ref secondaryIncomingConveyor.
        /// The job visual is flagged as being on a conveyor.
        ///
        /// @param jobId The ID of the job to place.
        /// @param visual The visual component to enqueue.
        public void PlaceOnIncoming(int jobId, JobVisual visual)
        {
            ConveyorBelt belt = PickIncomingBelt();
            if (belt != null && visual != null)
            {
                belt.TryEnqueue(jobId, visual);
                visual.SetOnConveyor(true);
            }
        }

        /// @brief Removes a job from the outgoing conveyor belt systems.
        ///
        /// @details Checks both @ref outgoingConveyor and @ref secondaryOutgoingConveyor.
        ///
        /// @param jobId The ID of the job to remove.
        public void RemoveFromOutgoing(int jobId)
        {
            if (outgoingConveyor != null && outgoingConveyor.Contains(jobId))
                outgoingConveyor.RemoveJob(jobId);
            else if (secondaryOutgoingConveyor != null && secondaryOutgoingConveyor.Contains(jobId))
                secondaryOutgoingConveyor.RemoveJob(jobId);
        }

        /// @brief Transfers a finished job visual from the machine center to an outgoing conveyor belt.
        ///
        /// @details Prefers @ref outgoingConveyor, falls back to @ref secondaryOutgoingConveyor.
        /// The job visual is flagged as being on a conveyor.
        ///
        /// @param jobId The ID of the job to place.
        /// @param visual The visual component to enqueue.
        public void PlaceOnOutgoing(int jobId, JobVisual visual)
        {
            ConveyorBelt belt = PickOutgoingBelt();
            if (belt != null && visual != null)
            {
                belt.TryEnqueue(jobId, visual);
                visual.SetOnConveyor(true);
            }
        }

        /// @brief Returns the world position where AGVs should drop off jobs for this machine.
        ///
        /// @details Uses the input conveyor's @c InputEndPosition if available,
        /// otherwise returns a default position behind the machine.
        ///
        /// @return The world-space drop-off position for AGVs.
        public Vector3 GetDropoffPosition()
        {
            ConveyorBelt belt = PickIncomingBelt();
            if (belt != null) return belt.InputEndPosition;
            return transform.position + transform.TransformDirection(new Vector3(-2.5f, 0.5f, 0f));
        }

        /// @brief Returns the world position where AGVs should pick up completed jobs from this machine.
        ///
        /// @details Uses the outgoing conveyor's @c OutputEndPosition if available,
        /// otherwise returns a default position in front of the machine.
        ///
        /// @return The world-space pickup position for AGVs.
        public Vector3 GetPickupPosition()
        {
            if (outgoingConveyor != null) return outgoingConveyor.OutputEndPosition;
            if (secondaryOutgoingConveyor != null) return secondaryOutgoingConveyor.OutputEndPosition;
            return transform.position + transform.TransformDirection(new Vector3(2.5f, 0.5f, 0f));
        }

        /// @brief Updates the numerical UI labels for the machine's current queue state.
        ///
        /// @param incomingCount The number of jobs waiting to enter this machine.
        /// @param outgoingCount The number of jobs waiting at the machine's output.
        public void RefreshQueueLabels(int incomingCount, int outgoingCount)
        {
            visualLayer.UpdateIncomingQueueLabel(incomingCount);
            visualLayer.UpdateOutgoingQueueLabel(outgoingCount);
        }

        // ── Reset ─────────────────────────────────────────────────────────────

        /// @brief Performs a full reset of the machine for episode teardown.
        ///
        /// @details Resets all processing state, health state, and conveyor belts.
        /// The machine returns to Operational/Idle with an infinite TTF countdown.
        public void FullReset()
        {
            IsIdle = true;
            FinishedFlag = false;
            ActiveJobId = -1;
            remainingTime = 0f;
            HealthState = MachineHealthState.Operational;
            FailedFlag = false;
            RepairCompleteFlag = false;
            _ttfCountdown = float.MaxValue;
            RemainingRepairTime = 0f;
            SampledRepairDuration = 0f;
            ClearConveyors();
        }

        // ── Update ────────────────────────────────────────────────────────────

        /// @brief Unity Update loop. Ticks all time-based countdowns each frame.
        private void Update()
        {
            TickTTF();
            TickProcessing();
            TickRepair();
        }

        /// @brief Decrements the time-to-failure (TTF) countdown while the machine is Operational.
        ///
        /// @details Runs regardless of whether the machine is idle or processing —
        /// a machine can fail mid-job or while idle. When the countdown expires:
        ///   - Repair duration is sampled immediately so it is available to the
        ///     observation builder before @c SimulationBridge processes the flag.
        ///   - @c FailedFlag is set; processing halts until @c AcknowledgeFailure().
        private void TickTTF()
        {
            if (HealthState != MachineHealthState.Operational) return;

            _ttfCountdown -= Time.deltaTime;
            if (_ttfCountdown > 0f) return;

            // Clamp to prevent multiple triggers if SimulationBridge is slow to poll.
            _ttfCountdown = float.MaxValue;
            _ageSinceLastRepair += Time.deltaTime;

            // Sample repair duration now so the observation can read it in the same frame.
            SampledRepairDuration = StochasticEventManager.Instance != null
                ? StochasticEventManager.Instance.SampleMachineRepair()
                : 0f;
            RemainingRepairTime = SampledRepairDuration;

            HealthState = MachineHealthState.Failed;
            FailedFlag = true;
            visualLayer?.BeginFailure();
            EpisodeTelemetryChannel.Instance?.RecordMachineFailure(
                MachineId,
                observedTtf: _ageSinceLastRepair,
                repairDuration: SampledRepairDuration
            );


        }

        /// @brief Advances the active job's processing timer when Operational and busy.
        ///
        /// @details Decrements @c remainingTime, updates the visual progress bar, fires
        /// the @c AlmostDoneFlag when within @c PreDispatchLeadTime, and sets @c FinishedFlag
        /// when processing completes. Guarded by health state so processing halts
        /// the frame a failure occurs.
        private void TickProcessing()
        {
            if (HealthState != MachineHealthState.Operational) return;
            if (IsIdle || FinishedFlag) return;

            remainingTime -= Time.deltaTime;

            if (visualLayer != null && remainingTime > 0f)
                visualLayer.UpdateProgress(1f - (remainingTime / Mathf.Max(totalDuration, 0.001f)));

            if (!almostDoneFired && remainingTime <= FactoryOrchestrator.Instance.PreDispatchLeadTime)
            {
                almostDoneFired = true;
                AlmostDoneFlag = true;
                AlmostDoneJobId = ActiveJobId;
            }

            if (remainingTime <= 0f)
                FinishedFlag = true;
        }

        /// @brief Decrements the repair countdown while the machine is in Repairing state.
        ///
        /// @details Updates the visual progress bar to show repair progress and sets
        /// @c RepairCompleteFlag when the countdown reaches zero.
        private void TickRepair()
        {
            if (HealthState != MachineHealthState.Repairing) return;

            RemainingRepairTime -= Time.deltaTime;

            if (visualLayer != null && SampledRepairDuration > 0f)
            {
                visualLayer.UpdateProgress(1f - (RemainingRepairTime / SampledRepairDuration));
            }

            if (RemainingRepairTime <= 0f)
            {
                RemainingRepairTime = 0f;
                RepairCompleteFlag = true;
            }
        }

        // ── Private helpers ───────────────────────────────────────────────────

        /// @brief Removes a job from any incoming conveyor belt.
        ///
        /// @param jobId The ID of the job to remove.
        private void RemoveFromAnyIncoming(int jobId)
        {
            if (incomingConveyor != null && incomingConveyor.Contains(jobId))
                incomingConveyor.RemoveJob(jobId);
            else if (secondaryIncomingConveyor != null && secondaryIncomingConveyor.Contains(jobId))
                secondaryIncomingConveyor.RemoveJob(jobId);
        }

        /// @brief Picks the most appropriate incoming conveyor belt.
        ///
        /// @details Prefers @ref incomingConveyor, falls back to @ref secondaryIncomingConveyor.
        /// Returns null if neither is assigned.
        ///
        /// @return The selected conveyor belt, or null if none available.
        private ConveyorBelt PickIncomingBelt()
        {
            if (incomingConveyor != null && !incomingConveyor.IsFull) return incomingConveyor;
            if (secondaryIncomingConveyor != null && !secondaryIncomingConveyor.IsFull) return secondaryIncomingConveyor;
            return incomingConveyor ?? secondaryIncomingConveyor;
        }

        /// @brief Picks the most appropriate outgoing conveyor belt.
        ///
        /// @details Prefers @ref outgoingConveyor, falls back to @ref secondaryOutgoingConveyor.
        /// Returns null if neither is assigned.
        ///
        /// @return The selected conveyor belt, or null if none available.
        private ConveyorBelt PickOutgoingBelt()
        {
            if (outgoingConveyor != null && !outgoingConveyor.IsFull) return outgoingConveyor;
            if (secondaryOutgoingConveyor != null && !secondaryOutgoingConveyor.IsFull) return secondaryOutgoingConveyor;
            return outgoingConveyor ?? secondaryOutgoingConveyor;
        }

        /// @brief Clears all connected conveyor belts.
        private void ClearConveyors()
        {
            incomingConveyor?.Clear();
            outgoingConveyor?.Clear();
            secondaryIncomingConveyor?.Clear();
            secondaryOutgoingConveyor?.Clear();
        }
    }
}