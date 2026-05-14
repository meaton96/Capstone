using UnityEngine;
using Assets.Scripts.Simulation.Jobs;
using Assets.Scripts.Simulation.Stochastic;
using Assets.Scripts.Logging;

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

        public int MachineId { get; private set; }
        public MachineType MachineType { get; private set; }

        // ── Normal processing state (unchanged) ──────────────────────────────

        public bool IsIdle { get; private set; } = true;
        public bool FinishedFlag { get; private set; }
        public int ActiveJobId { get; private set; } = -1;
        public bool AlmostDoneFlag { get; private set; }
        public int AlmostDoneJobId { get; private set; } = -1;

        private float remainingTime;
        private float totalDuration;
        private bool almostDoneFired;

        // ── Health state machine ──────────────────────────────────────────────

        /// @brief Current health state of the machine.
        /// Encoded as a 4th channel in the spatial occupancy tensor:
        ///   Operational = 0.0,  Repairing = 0.5,  Failed = 1.0
        public MachineHealthState HealthState { get; private set; } = MachineHealthState.Operational;

        /// @brief Set when the TTF countdown expires. Polled by @c SimulationBridge,
        /// which handles job return before calling @c AcknowledgeFailure().
        public bool FailedFlag { get; private set; }

        /// @brief Set when the repair countdown reaches zero. Polled by @c SimulationBridge,
        /// which calls @c AcknowledgeRepairComplete() to return to Operational.
        public bool RepairCompleteFlag { get; private set; }

        /// @brief Repair duration sampled at the moment of failure (log-normal).
        /// Available immediately once @c FailedFlag is raised, so the observation
        /// builder can read it before @c AcknowledgeFailure() is called.
        public float SampledRepairDuration { get; private set; }

        /// @brief Remaining repair time. Counts down in @c Update while Repairing.
        /// Normalise against @c SampledRepairDuration for the Global Scalars observation.
        public float RemainingRepairTime { get; private set; }

        /// @brief True when this machine can accept new work.
        /// Use this to filter routing candidates and dispatch decisions in SimulationBridge.
        public bool IsAvailableForWork => HealthState == MachineHealthState.Operational;

        /// @brief Returns the health state encoded as a float for the spatial occupancy tensor.
        public float HealthStateEncoded => HealthState switch
        {
            MachineHealthState.Failed => 1.0f,
            MachineHealthState.Repairing => 0.5f,
            _ => 0.0f,
        };



        private float _ttfCountdown = float.MaxValue;

        // ── Visual & conveyor references (unchanged) ──────────────────────────

        [Header("Conveyor Belts (visual only)")]
        [SerializeField] private ConveyorBelt incomingConveyor;
        [SerializeField] private ConveyorBelt outgoingConveyor;
        [SerializeField] private ConveyorBelt secondaryIncomingConveyor;
        [SerializeField] private ConveyorBelt secondaryOutgoingConveyor;

        private MachineVisual visualLayer;

        // ── Initialisation ────────────────────────────────────────────────────

        /// @brief Sets the machine identity and resets the visual layer.
        ///
        /// @param id   Unique identifier for the machine instance.
        /// @param type The functional @c MachineType (e.g., Mill, Lathe).
        public void Initialize(int id, MachineType type)
        {
            MachineId = id;
            MachineType = type;
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
            visualLayer?.Initialise(id, type);
            ClearConveyors();
        }

        // Add inside PhysicalMachine.cs, inside #if UNITY_EDITOR guard
#if UNITY_EDITOR
        /// @brief Forces an immediate failure. Editor/testing use only.
        public void DEBUG_ForceFailure()
        {
            if (HealthState != MachineHealthState.Operational) return;
            _ttfCountdown = 0f; // TickTTF will fire on the next Update
        }
#endif

        /// @brief Seeds this machine's TTF countdown for the current episode.
        ///
        /// @details Called by @c SimulationBridge.StartEpisode() after the job store
        /// is initialised. Applies initial age randomisation per the roadmap: each
        /// machine starts at a random point in its first wear-out cycle rather than
        /// all failing simultaneously after one full TTF.
        ///
        /// When @c MachineFailuresEnabled is false the countdown is set to
        /// @c float.MaxValue, making the machine effectively immortal for that episode.
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
        }

        // ── Processing control (unchanged public API) ─────────────────────────

        /// @brief Commences the processing of a specific job for a defined duration.
        ///
        /// @param jobId    The identifier of the job being processed.
        /// @param duration The time in simulation seconds to complete the operation.
        /// @param visual   The 3D representation of the job to be snapped to the machine.
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

        /// @brief Resets the finished status after the orchestrator acknowledges completion.
        public void ClearFinished()
        {
            FinishedFlag = false;
            IsIdle = true;
            ActiveJobId = -1;
            visualLayer?.CompleteOperation(-1);
        }

        /// @brief Resets the pre-dispatch signaling flags.
        public void ClearAlmostDone()
        {
            AlmostDoneFlag = false;
            AlmostDoneJobId = -1;
        }

        // ── Failure acknowledgement (called by SimulationBridge) ─────────────

        /// @brief Transitions from Failed → Repairing.
        ///
        /// @details Called by @c SimulationBridge after it has handled job return
        /// and AGV re-routing. The repair countdown (already sampled) begins here.
        /// The machine is left idle but unavailable for new work until repair completes.
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

        /// @brief Transitions from Repairing → Operational and arms the next TTF countdown.
        ///
        /// @details Called by @c SimulationBridge after detecting @c RepairCompleteFlag.
        /// The machine's age is considered zero post-repair, so a fresh Weibull TTF
        /// is sampled rather than resuming accumulated lifetime.
        public void AcknowledgeRepairComplete()
        {
            RepairCompleteFlag = false;
            RemainingRepairTime = 0f;
            HealthState = MachineHealthState.Operational;

            // Fresh TTF from repaired state — age counter resets to zero.
            _ttfCountdown = StochasticEventManager.Instance != null
                ? StochasticEventManager.Instance.SampleMachineTTF()
                : float.MaxValue;
            SimLogger.Medium($"Machine [{MachineId}] repair complete");
            visualLayer?.EndRepair();
        }

        // ── Conveyor helpers (unchanged) ──────────────────────────────────────

        /// @brief Places a job visual onto the most appropriate incoming conveyor belt.
        public void PlaceOnIncoming(int jobId, JobVisual visual)
        {
            ConveyorBelt belt = PickIncomingBelt();
            if (belt != null && visual != null)
            {
                belt.TryEnqueue(jobId, visual);
                visual.SetOnConveyor(true);
            }
        }

        /// @brief Removes a job from the outgoing belt systems.
        public void RemoveFromOutgoing(int jobId)
        {
            if (outgoingConveyor != null && outgoingConveyor.Contains(jobId))
                outgoingConveyor.RemoveJob(jobId);
            else if (secondaryOutgoingConveyor != null && secondaryOutgoingConveyor.Contains(jobId))
                secondaryOutgoingConveyor.RemoveJob(jobId);
        }

        /// @brief Transfers a finished job visual from the machine center to an outgoing belt.
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
        public Vector3 GetDropoffPosition()
        {
            ConveyorBelt belt = PickIncomingBelt();
            if (belt != null) return belt.InputEndPosition;
            return transform.position + transform.TransformDirection(new Vector3(-2.5f, 0.5f, 0f));
        }

        /// @brief Returns the world position where AGVs should pick up jobs from this machine.
        public Vector3 GetPickupPosition()
        {
            if (outgoingConveyor != null) return outgoingConveyor.OutputEndPosition;
            if (secondaryOutgoingConveyor != null) return secondaryOutgoingConveyor.OutputEndPosition;
            return transform.position + transform.TransformDirection(new Vector3(2.5f, 0.5f, 0f));
        }

        /// @brief Updates the numerical UI labels for the machine's current queue state.
        public void RefreshQueueLabels(int incomingCount, int outgoingCount)
        {
            visualLayer.UpdateIncomingQueueLabel(incomingCount);
            visualLayer.UpdateOutgoingQueueLabel(outgoingCount);
        }

        // ── Forces the machine into an idle state and clears all belt visuals ──

        /// @brief Full reset for episode teardown. Resets health state to Operational.
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

        private void Update()
        {
            TickTTF();
            TickProcessing();
            TickRepair();
        }

        /// @brief Decrements the TTF countdown while the machine is Operational.
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

            // Sample repair duration now so the observation can read it in the same frame.
            SampledRepairDuration = StochasticEventManager.Instance != null
                ? StochasticEventManager.Instance.SampleMachineRepair()
                : 0f;
            RemainingRepairTime = SampledRepairDuration;

            HealthState = MachineHealthState.Failed;
            FailedFlag = true;
            visualLayer?.BeginFailure();
        }

        /// @brief Advances the active job's processing timer when Operational and busy.
        ///
        /// @details Unchanged from the original implementation; guarded by health state
        /// so processing halts the frame a failure occurs.
        private void TickProcessing()
        {
            if (HealthState != MachineHealthState.Operational) return;
            if (IsIdle || FinishedFlag) return;

            remainingTime -= Time.deltaTime;

            if (visualLayer != null && remainingTime > 0f)
                visualLayer.UpdateProgress(1f - (remainingTime / Mathf.Max(totalDuration, 0.001f)));

            if (!almostDoneFired && remainingTime <= SimulationBridge.Instance.PreDispatchLeadTime)
            {
                almostDoneFired = true;
                AlmostDoneFlag = true;
                AlmostDoneJobId = ActiveJobId;
            }

            if (remainingTime <= 0f)
                FinishedFlag = true;
        }

        /// @brief Decrements the repair countdown while the machine is Repairing.
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

        // ── Private helpers (unchanged) ───────────────────────────────────────

        private void RemoveFromAnyIncoming(int jobId)
        {
            if (incomingConveyor != null && incomingConveyor.Contains(jobId))
                incomingConveyor.RemoveJob(jobId);
            else if (secondaryIncomingConveyor != null && secondaryIncomingConveyor.Contains(jobId))
                secondaryIncomingConveyor.RemoveJob(jobId);
        }

        private ConveyorBelt PickIncomingBelt()
        {
            if (incomingConveyor != null && !incomingConveyor.IsFull) return incomingConveyor;
            if (secondaryIncomingConveyor != null && !secondaryIncomingConveyor.IsFull) return secondaryIncomingConveyor;
            return incomingConveyor ?? secondaryIncomingConveyor;
        }

        private ConveyorBelt PickOutgoingBelt()
        {
            if (outgoingConveyor != null && !outgoingConveyor.IsFull) return outgoingConveyor;
            if (secondaryOutgoingConveyor != null && !secondaryOutgoingConveyor.IsFull) return secondaryOutgoingConveyor;
            return outgoingConveyor ?? secondaryOutgoingConveyor;
        }

        private void ClearConveyors()
        {
            incomingConveyor?.Clear();
            outgoingConveyor?.Clear();
            secondaryIncomingConveyor?.Clear();
            secondaryOutgoingConveyor?.Clear();
        }
    }
}