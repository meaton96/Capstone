using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using Assets.Scripts.Scheduling.Core;
using TMPro;

namespace Assets.Scripts.Simulation.Machines
{
    /// @brief Handles all visual feedback for a single machine on the factory floor.
    /// @details Manages mesh colour, overhead UI, progress bars, and decision flashes. 
    /// State changes are driven externally by PhysicalMachine.
    public class MachineVisual : MonoBehaviour
    {
        [Header("Identity")]
        [SerializeField] private int machineId;

        [Header("Rendering")]
        [SerializeField] private MeshRenderer meshRenderer;
        [SerializeField] private Vector3 incomingOffset = new Vector3(-2.5f, -.5f, 0f);
        [SerializeField] private Vector3 outgoingOffset = new Vector3(2.5f, -.5f, 0f);

        [Header("Overhead UI")]
        [SerializeField] private TextMeshProUGUI labelText;
        [SerializeField] private TextMeshProUGUI statusText;
        [SerializeField] private Slider progressBar;
        [SerializeField] private TextMeshProUGUI incomingQueueLabel;
        [SerializeField] private TextMeshProUGUI outgoingQueueLabel;

        [Header("State Colours")]
        [SerializeField] private Color idleColour = new Color(0.30f, 0.85f, 0.40f);
        [SerializeField] private Color busyColour = new Color(0.95f, 0.80f, 0.20f);
        [SerializeField] private Color blockedColour = new Color(1.00f, 0.55f, 0.10f);
        [SerializeField] private Color failedColour = new Color(0.90f, 0.25f, 0.25f);
        [SerializeField] private Color repairingColour = new Color(0.30f, 0.55f, 0.95f);

        [Header("Decision-Point Flash")]
        [SerializeField] private Color flashColour = Color.white;
        [SerializeField] private float flashDuration = 0.25f;

        private MachineState currentState = MachineState.Idle;
        private Material instanceMaterial;
        private Coroutine activeFlash;
        // private Machine coreMachine;
        private int decisionPointCount;
        private readonly List<string> historyLog = new List<string>();

        // public Machine CoreMachine => coreMachine;
        public int MachineId => machineId;
        public MachineState CurrentState => currentState;
        public int DecisionPointCount => decisionPointCount;
        public IReadOnlyList<string> HistoryLog => historyLog;

        private void Awake()
        {
            if (meshRenderer != null)
            {
                instanceMaterial = new Material(meshRenderer.sharedMaterial);
                meshRenderer.material = instanceMaterial;
            }
            SetProgressBarVisible(false);
            UpdateIncomingQueueLabel(0);
            UpdateOutgoingQueueLabel(0);
        }

        private void OnDestroy()
        {
            if (instanceMaterial != null) Destroy(instanceMaterial);
        }

        /// @brief Initializes the machine visual identity and binds it to the simulation core.
        /// @param id The unique machine index.
        /// @param coreMachineRef Reference to the logical machine data.
        /// @post State is set to Idle and UI labels are updated.
        public void Initialise(int id)
        {
            machineId = id;
            // coreMachine = coreMachineRef;

            if (labelText != null) labelText.text = $"M{id}";

            SetState(MachineState.Idle);
            Log($"Initialised at {transform.position}");
        }

        /// @brief Transitions the machine to a new operational state and updates visuals.
        /// @details Updates the mesh color, status text label, and disables progress bars if no longer busy.
        /// @param newState The target @ref MachineState to apply.
        /// @post currentState is updated and the history log is appended.
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

        /// @brief Initiates the visual processing phase for a job.
        /// @param jobId ID of the job being processed.
        /// @param simStartTime The simulation time at the start of the operation.
        /// @param duration The calculated processing time.
        /// @post state becomes Busy and the overhead progress bar is enabled.
        public void BeginOperation(int jobId, float simStartTime, float duration)
        {
            SetState(MachineState.Busy);
            SetProgressBarVisible(true);
            if (progressBar != null) progressBar.value = 0f;
            Log($"Op started: Job {jobId}, dur={duration:F1}");
        }

        /// @brief Resets the machine visual state after a job is successfully released.
        /// @param jobId ID of the job that finished.
        /// @post Progress bar is hidden and state returns to Idle.
        public void CompleteOperation(int jobId)
        {
            SetProgressBarVisible(false);
            SetState(MachineState.Idle);
            Log($"Op completed: Job {jobId}");
        }

        /// @brief Sets the visual state to Blocked when the output end is obstructed.
        /// @details Triggered when processing is 100% complete but the outgoing conveyor is at capacity.
        /// @param jobId The job currently being held inside the machine.
        /// @post State becomes Blocked (orange) and progress bar is hidden.
        public void SetBlockedAfterProcessing(int jobId)
        {
            SetProgressBarVisible(false);
            SetState(MachineState.Blocked);
            Log($"Blocked: outgoing conveyor full, holding Job {jobId}");
        }

        /// @brief Updates the overhead slider value.
        /// @param normalizedProgress Value between 0.0 and 1.0.
        public void UpdateProgress(float normalizedProgress)
        {
            if (progressBar != null)
                progressBar.value = Mathf.Clamp01(normalizedProgress);
        }

        private void SetProgressBarVisible(bool visible)
        {
            if (progressBar != null) progressBar.gameObject.SetActive(visible);
        }

        /// @brief Updates the text label for the incoming machine buffer.
        /// @param count Number of jobs waiting to be processed.
        public void UpdateIncomingQueueLabel(int count)
        {
            if (incomingQueueLabel != null) incomingQueueLabel.text = $"IN: {count}";
        }

        /// @brief Updates the text label for the outgoing machine buffer.
        /// @param count Number of jobs waiting for AGV pickup.
        public void UpdateOutgoingQueueLabel(int count)
        {
            if (outgoingQueueLabel != null) outgoingQueueLabel.text = $"OUT: {count}";
        }

        /// @brief Logs a scheduling decision point and triggers visual feedback.
        /// @details Records state of the queue and the rule choice into the @ref historyLog.
        /// @param simTime Current simulation clock.
        /// @param queuedJobIds Array of all jobs currently in the buffer.
        /// @param chosenJobId The job selected for processing.
        /// @param ruleName Name of the dispatching rule applied.
        /// @param flash If true, triggers the mesh color flash effect.
        /// @post decisionPointCount is incremented.
        public void RecordDecisionPoint(float simTime, int[] queuedJobIds, int chosenJobId, string ruleName, bool flash = true)
        {
            decisionPointCount++;
            string queueStr = string.Join(", ", Array.ConvertAll(queuedJobIds, id => $"Job {id}"));
            string entry = $"[t={simTime:F1}] M{machineId} free, queue=[{queueStr}], rule chose Job {chosenJobId} ({ruleName})";
            Log(entry);

            if (flash) Flash();
        }

        /// @brief Initiates a temporary color flash on the machine mesh.
        /// @post Starts the FlashRoutine coroutine, stopping any existing flash.
        public void Flash()
        {
            if (instanceMaterial == null) return;
            if (activeFlash != null) StopCoroutine(activeFlash);
            activeFlash = StartCoroutine(FlashRoutine());
        }

        private IEnumerator FlashRoutine()
        {
            Color baseColour = GetColourForState(currentState);
            instanceMaterial.color = flashColour;
            yield return new WaitForSeconds(flashDuration);
            instanceMaterial.color = baseColour;
            activeFlash = null;
        }

        private Color GetColourForState(MachineState state)
        {
            return state switch
            {
                MachineState.Idle => idleColour,
                MachineState.Busy => busyColour,
                MachineState.Blocked => blockedColour,
                MachineState.Failed => failedColour,
                MachineState.Repair => repairingColour,
                _ => Color.magenta
            };
        }

        private void Log(string message)
        {
            historyLog.Add($"[{Time.time:F2}] {message}");
        }
    }
}