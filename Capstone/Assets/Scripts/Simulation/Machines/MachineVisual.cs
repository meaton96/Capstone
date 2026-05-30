using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using TMPro;
using Assets.Scripts.Simulation.Types;
using Assets.Scripts.Simulation.Logging;

namespace Assets.Scripts.Simulation.Machines
{
    /// @brief Visual representation of a machine in the factory simulation.
    ///
    /// @details Manages the visual appearance of a machine including body color (by type),
    /// indicator light (by state), overhead UI labels, progress bar, and decision-point
    /// flash effects. Each machine type receives a unique identity color, and each
    /// operational state maps to a distinct indicator material.
    ///
    /// @remarks This component is paired with @c PhysicalMachine, which handles the
    /// logical state machine. All state changes flow through the public callback methods
    /// (e.g., @c SetState, @c BeginOperation, @c BeginFailure).
    public class MachineVisual : MonoBehaviour
    {
        // ── Serialized Fields ───────────────────────────────────────────────

        /// @brief Unique identifier for this machine instance.
        [SerializeField] private int machineId;

        /// @brief Renderer for the machine body mesh.
        [SerializeField] private MeshRenderer machineBodyRenderer;

        /// @brief Renderer for the state indicator light.
        [SerializeField] private MeshRenderer indicatorRenderer;

        /// @brief World offset position for incoming job visuals.
        [SerializeField] private Vector3 incomingOffset = new Vector3(-2.5f, -.5f, 0f);

        /// @brief World offset position for outgoing job visuals.
        [SerializeField] private Vector3 outgoingOffset = new Vector3(2.5f, -.5f, 0f);

        /// @brief Array of indicator materials, one per @c MachineState.
        ///
        /// @details Assigned in the Inspector. The index into this array must match
        /// the @c MachineState enum value (Idle=0, Busy=1, Blocked=2, Failed=3, Repair=4).
        [SerializeField] private Material[] indicatorMaterials;

        /// @brief Label displaying the machine ID and type above the machine.
        [SerializeField] private TextMeshProUGUI labelText;

        /// @brief Label displaying the current operational state.
        [SerializeField] private TextMeshProUGUI statusText;

        /// @brief Progress bar showing operation or repair progress.
        [SerializeField] private Slider progressBar;

        /// @brief Label showing the count of jobs waiting to enter.
        [SerializeField] private TextMeshProUGUI incomingQueueLabel;

        /// @brief Label showing the count of jobs waiting to exit.
        [SerializeField] private TextMeshProUGUI outgoingQueueLabel;

        /// @brief Color used for the decision-point flash effect.
        [SerializeField] private Color flashColour = Color.white;

        /// @brief Duration of the decision-point flash in seconds.
        [SerializeField] private float flashDuration = 0.25f;

        // ── Type identity colours ───────────────────────────────────────────

        /// @brief Mapping of machine types to their unique identity colors.
        ///
        /// @details Used in @c Initialise to set the machine body color.
        private static readonly Dictionary<MachineType, Color> TypeColors = new()
        {
            { MachineType.Mill,     new Color(0.20f, 0.40f, 0.80f) },   // steel blue
            { MachineType.Lathe,    new Color(0.80f, 0.50f, 0.10f) },   // amber
            { MachineType.Weld,     new Color(0.70f, 0.20f, 0.20f) },   // deep red
            { MachineType.Inspect,  new Color(0.20f, 0.70f, 0.40f) },   // green
            { MachineType.Assemble, new Color(0.55f, 0.25f, 0.70f) },   // purple
        };

        // ── Private Fields ──────────────────────────────────────────────────

        private MachineState currentState = MachineState.Idle;
        private Material bodyInstanceMaterial;
        private Material indicatorInstanceMaterial;
        private Coroutine activeFlash;
        private int decisionPointCount;
        private readonly List<string> historyLog = new List<string>();

        // ── Public Properties ───────────────────────────────────────────────

        /// @brief The unique identifier for this machine.
        public int MachineId => machineId;

        /// @brief The current operational state of this machine.
        public MachineState CurrentState => currentState;

        /// @brief The total number of scheduling decisions recorded at this machine.
        public int DecisionPointCount => decisionPointCount;

        /// @brief Read-only log of visual state changes and events.
        public IReadOnlyList<string> HistoryLog => historyLog;

        // ── Unity Lifecycle ─────────────────────────────────────────────────

        /// @brief Called on initialization. Creates material instances and sets initial state.
        ///
        /// @details Creates an independent material instance for the body renderer so that
        /// the type color does not affect other machines sharing the same prefab material.
        /// The indicator is initialized to the Idle state material.
        private void Awake()
        {
            if (machineBodyRenderer != null)
            {
                bodyInstanceMaterial = new Material(machineBodyRenderer.sharedMaterial);
                machineBodyRenderer.material = bodyInstanceMaterial;
            }

            if (indicatorRenderer != null)
                ApplyIndicatorMaterial(MachineState.Idle);

            SetProgressBarVisible(false);
            UpdateIncomingQueueLabel(0);
            UpdateOutgoingQueueLabel(0);
        }

        /// @brief Called when the component is destroyed. Cleans up instance materials.
        private void OnDestroy()
        {
            if (bodyInstanceMaterial != null) Destroy(bodyInstanceMaterial);
            if (indicatorInstanceMaterial != null) Destroy(indicatorInstanceMaterial);
        }

        // ── Initialisation ──────────────────────────────────────────────────

        /// @brief Initializes the machine with its identity and type color.
        ///
        /// @details Sets the machine label, applies the type-specific body color (darkened
        /// to 60% so the indicator light stands out), and resets to the Idle state.
        ///
        /// @param id The unique machine identifier.
        /// @param type The functional machine type (e.g., Mill, Lathe).
        public void Initialise(int id, MachineType type)
        {
            machineId = id;
            SetLabel($"M{id}\n{type}");

            if (bodyInstanceMaterial != null && TypeColors.TryGetValue(type, out Color bodyColor))
                bodyInstanceMaterial.color = bodyColor * 0.6f;  // darken so indicator pops

            SetState(MachineState.Idle);
            Log($"Initialised as {type} at {transform.position}");
        }

        // ── State ───────────────────────────────────────────────────────────

        /// @brief Transitions the machine to a new operational state.
        ///
        /// @details Updates the indicator light material via @c ApplyIndicatorMaterial,
        /// updates the status text label, and hides the progress bar unless the state
        /// is Busy. The machine body color is never modified after @c Initialise.
        ///
        /// @param newState The target operational state.
        public void SetState(MachineState newState)
        {
            MachineState previous = currentState;
            currentState = newState;

            ApplyIndicatorMaterial(newState);

            if (statusText != null)
                statusText.text = newState.ToString().ToUpper();

            if (newState != MachineState.Busy)
                SetProgressBarVisible(false);

            Log($"State: {previous} → {newState}");
        }

        // ── Operation Callbacks ─────────────────────────────────────────────

        /// @brief Notified that this machine has begun processing a job.
        ///
        /// @details Transitions to Busy state, shows the progress bar, and resets it to zero.
        ///
        /// @param jobId The ID of the job being processed.
        /// @param simStartTime The simulation time when the operation started.
        /// @param duration The total time required to complete the operation.
        public void BeginOperation(int jobId, float simStartTime, float duration)
        {
            SetState(MachineState.Busy);
            SetProgressBarVisible(true);
            if (progressBar != null) progressBar.value = 0f;
            Log($"Op started: Job {jobId}, dur={duration:F1}", true);
        }

        /// @brief Notified that this machine has completed processing a job.
        ///
        /// @details Hides the progress bar and transitions back to Idle state.
        ///
        /// @param jobId The ID of the completed job.
        public void CompleteOperation(int jobId)
        {
            SetProgressBarVisible(false);
            SetState(MachineState.Idle);
            Log($"Op completed: Job {jobId}", true);
        }

        /// @brief Notified that this machine is blocked after processing.
        ///
        /// @details The machine has finished a job but cannot release it because the
        /// outgoing conveyor is full. Transitions to Blocked state.
        ///
        /// @param jobId The ID of the job being held.
        public void SetBlockedAfterProcessing(int jobId)
        {
            SetProgressBarVisible(false);
            SetState(MachineState.Blocked);
            Log($"Blocked: outgoing full, holding Job {jobId}");
        }

        /// @brief Updates the progress bar to the specified normalized value.
        ///
        /// @param normalizedProgress The progress value clamped between 0.0 (not started)
        ///   and 1.0 (complete).
        public void UpdateProgress(float normalizedProgress)
        {
            if (progressBar != null)
                progressBar.value = Mathf.Clamp01(normalizedProgress);
        }

        // ── Queue Labels ────────────────────────────────────────────────────

        /// @brief Updates the incoming queue count label.
        ///
        /// @param count The number of jobs waiting to enter this machine.
        public void UpdateIncomingQueueLabel(int count)
        {
            if (incomingQueueLabel != null) incomingQueueLabel.text = $"IN: {count}";
        }

        /// @brief Updates the outgoing queue count label.
        ///
        /// @param count The number of jobs waiting at the machine's output.
        public void UpdateOutgoingQueueLabel(int count)
        {
            if (outgoingQueueLabel != null) outgoingQueueLabel.text = $"OUT: {count}";
        }

        // ── Decision Flash ──────────────────────────────────────────────────

        /// @brief Records a scheduling decision point and optionally flashes the indicator.
        ///
        /// @details Increments the decision counter, logs the queued jobs and chosen job
        /// with the rule name, and triggers a brief white flash of the indicator light.
        ///
        /// @param simTime The current simulation time.
        /// @param queuedJobIds Array of job IDs currently queued for this machine.
        /// @param chosenJobId The ID of the job selected by the scheduling rule.
        /// @param ruleName The name of the scheduling rule that was applied.
        /// @param flash If true, triggers a visual flash effect on the indicator.
        public void RecordDecisionPoint(float simTime, int[] queuedJobIds, int chosenJobId, string ruleName, bool flash = true)
        {
            decisionPointCount++;
            string queueStr = string.Join(", ", Array.ConvertAll(queuedJobIds, id => $"Job {id}"));
            Log($"[t={simTime:F1}] queue=[{queueStr}], chose Job {chosenJobId} ({ruleName})");
            if (flash) Flash();
        }

        /// @brief Triggers a brief white flash of the indicator light.
        ///
        /// @details Stops any existing flash coroutine and starts a new flash effect
        /// that lasts for @ref flashDuration seconds before restoring the current state material.
        public void Flash()
        {
            if (indicatorRenderer == null) return;
            if (activeFlash != null) StopCoroutine(activeFlash);
            activeFlash = StartCoroutine(FlashRoutine());
        }

        // ── Health/Failure Callbacks ────────────────────────────────────────

        /// @brief Notified that this machine has experienced a failure.
        ///
        /// @details Hides the progress bar and transitions to Failed state.
        public void BeginFailure()
        {
            SetProgressBarVisible(false);
            SetState(MachineState.Failed);
            Log("Machine failed!", true);
        }

        /// @brief Notified that repair has begun on this machine.
        ///
        /// @details Transitions to Repair state and shows the progress bar
        /// to track repair completion.
        ///
        /// @param duration The total time required to complete the repair.
        public void BeginRepair(float duration)
        {
            SetState(MachineState.Repair);
            SetProgressBarVisible(true);
            if (progressBar != null) progressBar.value = 0f;
            Log($"Repair started, dur={duration:F1}", true);
        }

        /// @brief Notified that repair has been completed.
        ///
        /// @details Hides the progress bar and transitions back to Idle state.
        public void EndRepair()
        {
            SetProgressBarVisible(false);
            SetState(MachineState.Idle);
            Log("Repair completed", true);
        }

        // ── Private Helpers ─────────────────────────────────────────────────

        /// @brief Swaps the indicator renderer to the material matching the specified state.
        ///
        /// @details Creates an instance copy of the material so that emission tweaks
        /// do not affect the shared asset referenced in the inspector. The emission
        /// is driven from the material's base color at 1.8x intensity.
        ///
        /// @param state The machine state whose corresponding material should be applied.
        private void ApplyIndicatorMaterial(MachineState state)
        {
            if (indicatorRenderer == null) return;

            int index = (int)state;
            if (indicatorMaterials == null || index >= indicatorMaterials.Length || indicatorMaterials[index] == null)
            {
                SimLogger.LogWarning($"[MachineVisual] No indicator material for state {state} on M{machineId}");
                return;
            }

            // Re-use the instance if we already created one for this state,
            // otherwise make a fresh owned copy so we can safely tweak emission
            if (indicatorInstanceMaterial == null ||
                indicatorInstanceMaterial.name != indicatorMaterials[index].name + " (Instance)")
            {
                if (indicatorInstanceMaterial != null) Destroy(indicatorInstanceMaterial);
                indicatorInstanceMaterial = new Material(indicatorMaterials[index]);
            }

            // Drive emission from the material's base color so the inspector
            // controls both tint and glow in one place
            Color baseColor = indicatorInstanceMaterial.color;
            indicatorInstanceMaterial.SetColor("_EmissionColor", baseColor * 1.8f);
            indicatorInstanceMaterial.EnableKeyword("_EMISSION");

            indicatorRenderer.material = indicatorInstanceMaterial;
        }

        /// @brief Toggles the progress bar's visibility.
        ///
        /// @param visible True to show the progress bar, false to hide it.
        private void SetProgressBarVisible(bool visible)
        {
            if (progressBar != null) progressBar.gameObject.SetActive(visible);
        }

        /// @brief Coroutine that performs the white flash effect, then restores the state material.
        ///
        /// @details Sets the indicator material color to @ref flashColour, waits for
        /// @ref flashDuration seconds, then re-applies the correct state material.
        ///
        /// @yields A single frame wait for the flash duration.
        private IEnumerator FlashRoutine()
        {
            if (indicatorInstanceMaterial != null)
                indicatorInstanceMaterial.color = flashColour;

            yield return new WaitForSeconds(flashDuration);

            ApplyIndicatorMaterial(currentState);
            activeFlash = null;
        }

        /// @brief Updates the machine label text.
        ///
        /// @param label The text to display, typically in format "M{id}\n{type}".
        private void SetLabel(string label)
        {
            if (labelText != null) labelText.text = label;
        }

        /// @brief Logs a message to the internal history and optionally to the simulation logger.
        ///
        /// @param message The log message text.
        /// @param toLog If true, also writes to @c SimLogger.High for simulation-wide logging.
        private void Log(string message, bool toLog = false)
        {
            if (toLog)
                SimLogger.High($"[{Time.time:F2}] {message}");
            historyLog.Add($"[{Time.time:F2}] {message}");
        }
    }
}
