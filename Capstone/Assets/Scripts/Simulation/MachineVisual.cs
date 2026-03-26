using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using Assets.Scripts.Scheduling.Core;
using TMPro;




namespace Assets.Scripts.Simulation
{
    /**
     * @file MachineVisual.cs
     * @brief Unity MonoBehaviour that sits on each machine cube prefab.
     *
     * Owns all visual state: color, overhead UI, progress bar,
     * queue slots, and decision-point flash.
     *
     * @par Architecture references
     * - Phase1 Steps 1-2 (prefab + state machine)
     * - Phase1 Step 6    (queue visualisation)
     * - Phase1 Step 8    (decision-point marking)
     */

    /// <summary>
    /// Machine operating states — mirrors the core DES Machine states.
    /// </summary>
    public enum MachineState
    {
        Idle,       ///< Machine is idle and waiting for work.
        Busy,       ///< Machine is actively processing a job.
        Failed,     ///< Machine has encountered a failure.
        Repairing   ///< Machine is undergoing repair.
    }

    /// <summary>
    /// Visual representation of a single machine on the factory floor.
    /// </summary>
    /// <remarks>
    /// Attach this to the MachinePrefab cube alongside a <see cref="MeshRenderer"/>
    /// and a child world-space Canvas for the overhead UI.
    /// </remarks>
    public class MachineVisual : MonoBehaviour
    {
        /// @name Inspector References — Identity
        /// @{

        [Header("Identity")]
        [Tooltip("Set at runtime by FactoryLayoutManager.")]
        [SerializeField] private int machineId;

        /// @}

        /// @name Inspector References — Rendering
        /// @{

        [Header("Rendering")]
        [SerializeField] private MeshRenderer meshRenderer;

        /// @}

        /// @name Inspector References — Overhead UI (world-space canvas children)
        /// @{

        [Header("Overhead UI (world-space canvas children)")]
        [SerializeField] private TextMeshProUGUI labelText;      ///< Label displaying machine ID, e.g. "M3".
        [SerializeField] private TextMeshProUGUI statusText;      ///< Status text displaying current state, e.g. "IDLE".
        [SerializeField] private Slider progressBar;   ///< Slider showing operation completion percentage.
        [SerializeField] private TextMeshProUGUI queueLengthText; ///< Text showing queue depth, e.g. "Q: 4".

        /// @}

        /// @name Inspector References — State Colours
        /// @{

        [Header("State Colours")]
        [SerializeField] private Color idleColour = new Color(0.30f, 0.85f, 0.40f); ///< Green.
        [SerializeField] private Color busyColour = new Color(0.95f, 0.80f, 0.20f); ///< Yellow.
        [SerializeField] private Color failedColour = new Color(0.90f, 0.25f, 0.25f); ///< Red.
        [SerializeField] private Color repairingColour = new Color(0.30f, 0.55f, 0.95f); ///< Blue.

        /// @}

        /// @name Inspector References — Decision-Point Flash (Step 8)
        /// @{

        [Header("Decision-Point Flash (Step 8)")]
        [SerializeField] private Color flashColour = Color.white;  ///< Colour used during the decision-point flash.
        [SerializeField] private float flashDuration = 0.25f;      ///< Duration of the flash in seconds.

        /// @}

        /// @name Inspector References — Queue Layout (Step 6)
        /// @{

        [Header("Queue Layout (Step 6)")]
        [Tooltip("Local-space offset direction for queued job slots.")]
        [SerializeField] private Vector3 queueDirection = Vector3.back;
        [Tooltip("Spacing between queued job visuals.")]
        [SerializeField] private float queueSpacing = 1.2f;
        [Tooltip("Offset from machine centre to first queue slot.")]
        [SerializeField] private float queueStartOffset = 1.5f;

        /// @}

        /// @name Runtime State
        /// @{

        private MachineState currentState = MachineState.Idle;
        private Material instanceMaterial; ///< Per-instance material so colour changes don't leak.

        /// @}

        /// @name Progress Tracking
        /// @{

        private float operationStartTime;
        private float operationDuration;
        private bool isTrackingProgress;

        /// @}

        /// @name Queue State
        /// @{

        private readonly List<GameObject> queuedJobs = new List<GameObject>();

        /// @}

        /// @name Decision-Point Stats
        /// @{

        private int decisionPointCount;

        /// @}

        /// <summary>
        /// Handle to the active flash coroutine, allowing cancellation of overlapping flashes.
        /// </summary>
        private Coroutine activeFlash;

        /// <summary>
        /// Reference to the core DES Machine object, wired by SimulationBridge.
        /// </summary>
        /// <remarks>
        /// Stored as <see cref="System.Object"/> so this file compiles without a hard
        /// dependency on the Core assembly. Cast to your
        /// <c>Scheduling.Core.Machine</c> when you wire it up.
        /// </remarks>
        private Machine coreMachine;

        /// <summary>
        /// Gets the core DES Machine reference.
        /// </summary>
        public Machine CoreMachine => coreMachine;

        /// <summary>
        /// History log entries for the machine inspector (Step 9).
        /// </summary>
        private readonly List<string> historyLog = new List<string>();

        /// @name Public Properties
        /// @{

        /// <summary>Machine index inside the Taillard instance.</summary>
        public int MachineId => machineId;

        /// <summary>Current visual state.</summary>
        public MachineState CurrentState => currentState;

        /// <summary>Number of jobs visually queued at this machine.</summary>
        public int QueueLength => queuedJobs.Count;

        /// <summary>Total dispatching decisions made at this machine.</summary>
        public int DecisionPointCount => decisionPointCount;

        /// <summary>Read-only view of the history log.</summary>
        public IReadOnlyList<string> HistoryLog => historyLog;

        /// @}

        /// @name Lifecycle
        /// @{

        /// <summary>
        /// Creates a per-instance material clone so <see cref="SetState"/> colour
        /// changes are isolated to this machine, and hides the progress bar.
        /// </summary>
        private void Awake()
        {
            if (meshRenderer != null)
            {
                instanceMaterial = new Material(meshRenderer.sharedMaterial);
                meshRenderer.material = instanceMaterial;
            }

            SetProgressBarVisible(false);
        }

        /// <summary>
        /// Drives the fallback real-time progress bar each frame when active.
        /// </summary>
        private void Update()
        {
            if (isTrackingProgress)
            {
                UpdateProgressBar();
            }
        }

        /// <summary>
        /// Cleans up the cloned material to avoid memory leaks.
        /// </summary>
        private void OnDestroy()
        {
            if (instanceMaterial != null)
            {
                Destroy(instanceMaterial);
            }
        }

        /// @}

        /// @name Initialisation
        /// @{

        /// <summary>
        /// One-time setup after instantiation. Called by FactoryLayoutManager.
        /// </summary>
        /// <param name="id">Machine index from the TaillardInstance.</param>
        /// <param name="coreMachineRef">
        /// Reference to the core DES Machine object (<c>Scheduling.Core.Machine</c>).
        /// Stored as <see cref="System.Object"/> to avoid a compile-time dependency.
        /// </param>
        public void Initialise(int id, Machine coreMachineRef = null)
        {
            machineId = id;
            coreMachine = coreMachineRef;

            if (labelText != null)
                labelText.text = $"M{id}";

            SetState(MachineState.Idle);
            Log($"Initialised at {transform.position}");
        }

        /// @}

        /// @name State Management (Step 2)
        /// @{

        /// <summary>
        /// Transitions to a new visual state. Updates colour,
        /// status label, and progress bar visibility.
        /// </summary>
        /// <param name="newState">The target <see cref="MachineState"/>.</param>
        public void SetState(MachineState newState)
        {
            MachineState previous = currentState;
            currentState = newState;

            if (instanceMaterial != null)
            {
                instanceMaterial.color = GetColourForState(newState);
            }

            if (statusText != null)
            {
                statusText.text = newState.ToString().ToUpper();
            }

            if (newState != MachineState.Busy)
            {
                StopProgress();
            }

            Log($"State: {previous} → {newState}");
        }

        /// <summary>
        /// Begins an operation — sets state to <see cref="MachineState.Busy"/> and starts
        /// the progress bar. Called by SimulationBridge on OperationStarted events.
        /// </summary>
        /// <param name="jobId">Job being processed.</param>
        /// <param name="simStartTime">DES simulation start time of the operation.</param>
        /// <param name="duration">Processing time in sim-time units.</param>
        public void BeginOperation(int jobId, float simStartTime, float duration)
        {
            SetState(MachineState.Busy);
            StartProgress(simStartTime, duration);
            Log($"Op started: Job {jobId}, dur={duration:F1}");
        }

        /// <summary>
        /// Completes the current operation — sets state to <see cref="MachineState.Idle"/>.
        /// Called by SimulationBridge on OperationCompleted events.
        /// </summary>
        /// <param name="jobId">Job that finished.</param>
        public void CompleteOperation(int jobId)
        {
            StopProgress();
            SetState(MachineState.Idle);
            Log($"Op completed: Job {jobId}");
        }

        /// @}

        /// @name Progress Bar
        /// @{

        /// <summary>
        /// Initialises and shows the progress bar for a new operation.
        /// </summary>
        /// <param name="startTime">Simulation time when the operation begins.</param>
        /// <param name="duration">Total duration of the operation in sim-time units.</param>
        private void StartProgress(float startTime, float duration)
        {
            operationStartTime = startTime;
            operationDuration = Mathf.Max(duration, 0.001f);
            isTrackingProgress = true;
            SetProgressBarVisible(true);

            if (progressBar != null)
                progressBar.value = 0f;
        }

        /// <summary>
        /// Stops tracking progress and hides the progress bar.
        /// </summary>
        private void StopProgress()
        {
            isTrackingProgress = false;
            SetProgressBarVisible(false);
        }

        /// <summary>
        /// Drives the progress bar from an external simulation time.
        /// The SimulationBridge should call this each frame with the
        /// current simulation clock value.
        /// </summary>
        /// <param name="currentSimTime">Current simulation clock time.</param>
        public void UpdateProgress(float currentSimTime)
        {
            if (!isTrackingProgress || progressBar == null) return;

            float elapsed = currentSimTime - operationStartTime;
            progressBar.value = Mathf.Clamp01(elapsed / operationDuration);
        }

        /// <summary>
        /// Fallback progress update using Unity real-time.
        /// </summary>
        /// <remarks>
        /// This path is only hit when <see cref="UpdateProgress"/> is NOT
        /// being called by the bridge. Override with sim time for accurate display.
        /// </remarks>
        private void UpdateProgressBar()
        {
            if (progressBar != null)
            {
                float t = progressBar.value + Time.deltaTime / operationDuration;
                progressBar.value = Mathf.Clamp01(t);
            }
        }

        /// <summary>
        /// Shows or hides the progress bar GameObject.
        /// </summary>
        /// <param name="visible"><c>true</c> to show; <c>false</c> to hide.</param>
        private void SetProgressBarVisible(bool visible)
        {
            if (progressBar != null)
                progressBar.gameObject.SetActive(visible);
        }

        /// @}

        /// @name Queue Visualisation (Step 6)
        /// @{

        /// <summary>
        /// Adds a job visual to this machine's queue.
        /// </summary>
        /// <param name="jobVisual">The GameObject representing the queued job.</param>
        /// <returns>The world-space position the job should move to.</returns>
        public Vector3 EnqueueJob(GameObject jobVisual)
        {
            queuedJobs.Add(jobVisual);
            UpdateQueueLabel();
            return GetQueueSlotPosition(queuedJobs.Count - 1);
        }

        /// <summary>
        /// Removes a specific job from the queue (it is being processed or rerouted).
        /// Remaining jobs shift forward to fill the gap.
        /// </summary>
        /// <param name="jobVisual">The GameObject to remove.</param>
        public void DequeueJob(GameObject jobVisual)
        {
            queuedJobs.Remove(jobVisual);
            RebuildQueuePositions();
            UpdateQueueLabel();
        }

        /// <summary>
        /// Gets the world-space position of the "processing" slot where
        /// the active job sits while the machine is busy.
        /// </summary>
        /// <returns>World-space position above the machine.</returns>
        public Vector3 GetProcessingPosition()
        {
            return transform.position + Vector3.up * 0.8f;
        }

        /// <summary>
        /// Gets the world-space position for the Nth queued job slot.
        /// </summary>
        /// <param name="slotIndex">Zero-based index in the queue.</param>
        /// <returns>World-space position for the slot.</returns>
        public Vector3 GetQueueSlotPosition(int slotIndex)
        {
            Vector3 dir = transform.TransformDirection(queueDirection.normalized);
            return transform.position
                   + dir * (queueStartOffset + slotIndex * queueSpacing);
        }

        /// <summary>
        /// Snaps all queued job visuals to their correct slot positions.
        /// </summary>
        /// <remarks>
        /// Currently snaps instantly. Replace with a tween or coroutine for polish.
        /// </remarks>
        private void RebuildQueuePositions()
        {
            for (int i = 0; i < queuedJobs.Count; i++)
            {
                if (queuedJobs[i] != null)
                {
                    queuedJobs[i].transform.position = GetQueueSlotPosition(i);
                }
            }
        }

        /// <summary>
        /// Updates the queue-length UI label to reflect the current count.
        /// </summary>
        private void UpdateQueueLabel()
        {
            if (queueLengthText != null)
                queueLengthText.text = $"Q: {queuedJobs.Count}";
        }

        /// @}

        /// @name Decision-Point Flash (Step 8)
        /// @{

        /// <summary>
        /// Records a dispatching decision at this machine and optionally
        /// flashes the visual. Called when the DES makes a dispatching decision
        /// (i.e., queue is non-empty when the machine becomes free).
        /// </summary>
        /// <param name="simTime">Current simulation time.</param>
        /// <param name="queuedJobIds">IDs of jobs in the queue at decision time.</param>
        /// <param name="chosenJobId">The job the dispatching rule selected.</param>
        /// <param name="ruleName">Name of the dispatching rule.</param>
        /// <param name="flash">If <c>true</c>, briefly flashes the machine white.</param>
        public void RecordDecisionPoint(
            float simTime,
            int[] queuedJobIds,
            int chosenJobId,
            string ruleName,
            bool flash = true)
        {
            decisionPointCount++;
            string queueStr = string.Join(", ", Array.ConvertAll(queuedJobIds, id => $"Job {id}"));
            string entry = $"[t={simTime:F1}] M{machineId} free, queue=[{queueStr}], " +
                           $"rule chose Job {chosenJobId} ({ruleName})";
            Log(entry);
            Debug.Log($"[DecisionPoint] {entry}");

            if (flash)
            {
                Flash();
            }
        }

        /// <summary>
        /// Triggers a brief white flash on the machine to visually mark
        /// a dispatching decision.
        /// </summary>
        public void Flash()
        {
            if (instanceMaterial == null) return;

            if (activeFlash != null)
                StopCoroutine(activeFlash);

            activeFlash = StartCoroutine(FlashRoutine());
        }

        /// <summary>
        /// Coroutine that sets the material to the flash colour, waits,
        /// then restores the state-based colour.
        /// </summary>
        /// <returns>An <see cref="IEnumerator"/> for the coroutine.</returns>
        private IEnumerator FlashRoutine()
        {
            Color baseColour = GetColourForState(currentState);
            instanceMaterial.color = flashColour;
            yield return new WaitForSeconds(flashDuration);
            instanceMaterial.color = baseColour;
            activeFlash = null;
        }

        /// @}

        /// @name Reset
        /// @{

        /// <summary>
        /// Resets all runtime state for a new episode. Called between episodes.
        /// </summary>
        public void ResetMachine()
        {
            SetState(MachineState.Idle);
            StopProgress();
            queuedJobs.Clear();
            UpdateQueueLabel();
            decisionPointCount = 0;
            historyLog.Clear();
            Log("Reset for new episode");
        }

        /// @}

        /// @name Helpers
        /// @{

        /// <summary>
        /// Maps a <see cref="MachineState"/> to its corresponding display colour.
        /// </summary>
        /// <param name="state">The state to look up.</param>
        /// <returns>The <see cref="Color"/> assigned to the given state.</returns>
        private Color GetColourForState(MachineState state)
        {
            return state switch
            {
                MachineState.Idle => idleColour,
                MachineState.Busy => busyColour,
                MachineState.Failed => failedColour,
                MachineState.Repairing => repairingColour,
                _ => Color.magenta
            };
        }

        /// <summary>
        /// Appends a timestamped message to the history log.
        /// </summary>
        /// <param name="message">The message to record.</param>
        private void Log(string message)
        {
            historyLog.Add($"[{Time.time:F2}] {message}");
        }

        /// @}
    }
}