using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using Assets.Scripts.Scheduling.Core;
using TMPro;

namespace Assets.Scripts.Simulation
{
    /// @brief Handles all visual feedback for a single machine on the factory floor.
    ///
    /// @details Manages the machine's mesh colour, overhead UI labels, progress bar,
    /// queue-length display, and decision-point flash effect. All state changes are
    /// driven externally by @c PhysicalMachine; this class contains no scheduling logic.
    public class MachineVisual : MonoBehaviour
    {
        [Header("Identity")]
        [SerializeField] private int machineId;

        [Header("Rendering")]
        [SerializeField] private MeshRenderer meshRenderer;

        [Header("Overhead UI")]
        [SerializeField] private TextMeshProUGUI labelText;
        [SerializeField] private TextMeshProUGUI statusText;
        [SerializeField] private Slider progressBar;
        [SerializeField] private TextMeshProUGUI queueLengthText;

        [Header("State Colours")]
        [SerializeField] private Color idleColour = new Color(0.30f, 0.85f, 0.40f);
        [SerializeField] private Color busyColour = new Color(0.95f, 0.80f, 0.20f);
        [SerializeField] private Color failedColour = new Color(0.90f, 0.25f, 0.25f);
        [SerializeField] private Color repairingColour = new Color(0.30f, 0.55f, 0.95f);

        [Header("Decision-Point Flash")]
        [SerializeField] private Color flashColour = Color.white;
        [SerializeField] private float flashDuration = 0.25f;

        private MachineState currentState = MachineState.Idle;
        private Material instanceMaterial;
        private Coroutine activeFlash;
        private Machine coreMachine;
        private int decisionPointCount;
        private readonly List<string> historyLog = new List<string>();

        /// @brief The logical @c Machine data this visual is bound to.
        public Machine CoreMachine => coreMachine;

        /// @brief Unique machine index matching the simulation's machine array.
        public int MachineId => machineId;

        /// @brief The machine's current @c MachineState.
        public MachineState CurrentState => currentState;

        /// @brief Total number of decision points recorded for this machine this episode.
        public int DecisionPointCount => decisionPointCount;

        /// @brief Timestamped log of state transitions and decision events.
        public IReadOnlyList<string> HistoryLog => historyLog;

        // ─────────────────────────────────────────────────────────
        //  Unity Lifecycle
        // ─────────────────────────────────────────────────────────

        /// @brief Creates a per-instance material and hides the progress bar on startup.
        private void Awake()
        {
            if (meshRenderer != null)
            {
                instanceMaterial = new Material(meshRenderer.sharedMaterial);
                meshRenderer.material = instanceMaterial;
            }
            SetProgressBarVisible(false);
            UpdateQueueLabel(0);
        }

        /// @brief Destroys the per-instance material to prevent memory leaks.
        private void OnDestroy()
        {
            if (instanceMaterial != null) Destroy(instanceMaterial);
        }

        // ─────────────────────────────────────────────────────────
        //  Initialization
        // ─────────────────────────────────────────────────────────

        /// @brief Binds this visual to a machine ID and optional logical machine reference.
        /// @param id              Zero-based machine index.
        /// @param coreMachineRef  Optional logical @c Machine carrying scheduling metadata.
        public void Initialise(int id, Machine coreMachineRef = null)
        {
            machineId = id;
            coreMachine = coreMachineRef;

            if (labelText != null) labelText.text = $"M{id}";

            SetState(MachineState.Idle);
            Log($"Initialised at {transform.position}");
        }

        // ─────────────────────────────────────────────────────────
        //  State Management
        // ─────────────────────────────────────────────────────────

        /// @brief Transitions the machine to a new state, updating colour and status text.
        /// @details Hides the progress bar for any state other than @c Busy.
        /// @param newState  The @c MachineState to transition to.
        public void SetState(MachineState newState)
        {
            MachineState previous = currentState;
            currentState = newState;

            if (instanceMaterial != null)
                instanceMaterial.color = GetColourForState(newState);

            if (statusText != null)
                statusText.text = newState.ToString().ToUpper();

            if (newState != MachineState.Busy)
                SetProgressBarVisible(false);

            Log($"State: {previous} → {newState}");
        }

        /// @brief Transitions to @c Busy, shows the progress bar, and logs the operation start.
        /// @param jobId        The job whose operation is beginning.
        /// @param simStartTime Simulation time at which the operation started.
        /// @param duration     Expected duration of the operation in simulation seconds.
        public void BeginOperation(int jobId, float simStartTime, float duration)
        {
            SetState(MachineState.Busy);
            SetProgressBarVisible(true);
            if (progressBar != null) progressBar.value = 0f;
            Log($"Op started: Job {jobId}, dur={duration:F1}");
        }

        /// @brief Hides the progress bar, transitions to @c Idle, and logs the completion.
        /// @param jobId  The job whose operation just finished.
        public void CompleteOperation(int jobId)
        {
            SetProgressBarVisible(false);
            SetState(MachineState.Idle);
            Log($"Op completed: Job {jobId}");
        }

        // ─────────────────────────────────────────────────────────
        //  Progress Bar
        // ─────────────────────────────────────────────────────────

        /// @brief Sets the progress bar fill to a normalised value.
        /// @details Called each frame by @c PhysicalMachine during job processing.
        /// @param normalizedProgress  Clamped fill amount in [0, 1].
        public void UpdateProgress(float normalizedProgress)
        {
            if (progressBar != null)
                progressBar.value = Mathf.Clamp01(normalizedProgress);
        }

        /// @brief Shows or hides the progress bar GameObject.
        /// @param visible  True to show, false to hide.
        private void SetProgressBarVisible(bool visible)
        {
            if (progressBar != null) progressBar.gameObject.SetActive(visible);
        }

        // ─────────────────────────────────────────────────────────
        //  Queue UI
        // ─────────────────────────────────────────────────────────

        /// @brief Updates the overhead queue-length label.
        /// @details Called by @c PhysicalMachine whenever its physical queue changes.
        /// @param queueCount  Current number of jobs waiting in the machine's trigger zone.
        public void UpdateQueueLabel(int queueCount)
        {
            if (queueLengthText != null)
                queueLengthText.text = $"Q: {queueCount}";
        }

        // ─────────────────────────────────────────────────────────
        //  Decision-Point Flash
        // ─────────────────────────────────────────────────────────

        /// @brief Logs a scheduling decision and optionally triggers a flash effect.
        /// @param simTime       Simulation time at which the decision was made.
        /// @param queuedJobIds  Jobs that were present in the queue at decision time.
        /// @param chosenJobId   The job selected by the dispatching rule.
        /// @param ruleName      Human-readable name of the rule that was applied.
        /// @param flash         When true, triggers a brief white flash on the mesh.
        public void RecordDecisionPoint(float simTime, int[] queuedJobIds, int chosenJobId, string ruleName, bool flash = true)
        {
            decisionPointCount++;
            string queueStr = string.Join(", ", Array.ConvertAll(queuedJobIds, id => $"Job {id}"));
            string entry = $"[t={simTime:F1}] M{machineId} free, queue=[{queueStr}], rule chose Job {chosenJobId} ({ruleName})";
            Log(entry);

            if (flash) Flash();
        }

        /// @brief Immediately triggers the white flash coroutine, interrupting any active flash.
        public void Flash()
        {
            if (instanceMaterial == null) return;
            if (activeFlash != null) StopCoroutine(activeFlash);
            activeFlash = StartCoroutine(FlashRoutine());
        }

        /// @brief Coroutine that sets the mesh to @c flashColour for @c flashDuration seconds,
        ///        then restores the colour for the current state.
        private IEnumerator FlashRoutine()
        {
            Color baseColour = GetColourForState(currentState);
            instanceMaterial.color = flashColour;
            yield return new WaitForSeconds(flashDuration);
            instanceMaterial.color = baseColour;
            activeFlash = null;
        }

        // ─────────────────────────────────────────────────────────
        //  Helpers
        // ─────────────────────────────────────────────────────────

        /// @brief Returns the configured colour associated with a given @c MachineState.
        /// @param state  The state to look up.
        /// @return       The corresponding @c Color, or magenta for unhandled states.
        private Color GetColourForState(MachineState state)
        {
            return state switch
            {
                MachineState.Idle => idleColour,
                MachineState.Busy => busyColour,
                MachineState.Failed => failedColour,
                MachineState.Repair => repairingColour,
                _ => Color.magenta
            };
        }

        /// @brief Appends a timestamped message to the history log.
        /// @param message  The message to record.
        private void Log(string message)
        {
            historyLog.Add($"[{Time.time:F2}] {message}");
        }
    }
}