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
    ///
    /// @details Manages the machine's mesh colour, overhead UI labels, progress bar,
    /// queue-length display, and decision-point flash effect. All state changes are
    /// driven externally by @c PhysicalMachine; this class contains no scheduling logic.
    ///
    /// <para><b>Blocked state:</b> When the outgoing conveyor is full after processing,
    /// <see cref="SetBlockedAfterProcessing"/> hides the progress bar and sets the
    /// machine to an orange <c>MachineState.Blocked</c> colour so operators can
    /// immediately see output bottlenecks on the floor.</para>
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

        // ─────────────────────────────────────────────────────────
        //  Initialization
        // ─────────────────────────────────────────────────────────

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

        public void BeginOperation(int jobId, float simStartTime, float duration)
        {
            SetState(MachineState.Busy);
            SetProgressBarVisible(true);
            if (progressBar != null) progressBar.value = 0f;
            Log($"Op started: Job {jobId}, dur={duration:F1}");
        }

        public void CompleteOperation(int jobId)
        {
            SetProgressBarVisible(false);
            SetState(MachineState.Idle);
            Log($"Op completed: Job {jobId}");
        }

        /// <summary>
        /// Called by <see cref="PhysicalMachine"/> when processing finishes but
        /// the outgoing conveyor is full. Hides the progress bar and transitions
        /// to <c>MachineState.Blocked</c> (orange) so the bottleneck is visible.
        /// The machine will stay in this state until an AGV frees a slot on the
        /// outgoing belt, at which point <see cref="CompleteOperation"/> is called.
        /// </summary>
        /// <param name="jobId">The job being held inside the machine.</param>
        public void SetBlockedAfterProcessing(int jobId)
        {
            SetProgressBarVisible(false);
            SetState(MachineState.Blocked);
            Log($"Blocked: outgoing conveyor full, holding Job {jobId}");
        }

        // ─────────────────────────────────────────────────────────
        //  Progress Bar
        // ─────────────────────────────────────────────────────────

        public void UpdateProgress(float normalizedProgress)
        {
            if (progressBar != null)
                progressBar.value = Mathf.Clamp01(normalizedProgress);
        }

        private void SetProgressBarVisible(bool visible)
        {
            if (progressBar != null) progressBar.gameObject.SetActive(visible);
        }

        // ─────────────────────────────────────────────────────────
        //  Queue UI
        // ─────────────────────────────────────────────────────────

        public void UpdateIncomingQueueLabel(int count)
        {
            if (incomingQueueLabel != null) incomingQueueLabel.text = $"IN: {count}";
        }

        public void UpdateOutgoingQueueLabel(int count)
        {
            if (outgoingQueueLabel != null) outgoingQueueLabel.text = $"OUT: {count}";
        }

        // private void OnDrawGizmosSelected()
        // {
        //     Gizmos.color = Color.cyan;
        //     for (int i = 0; i < 4; i++)
        //     {
        //         Vector3 inSlot = transform.position + transform.TransformDirection(incomingOffset)
        //                        + transform.right * (i * 0.75f);
        //         Gizmos.DrawWireCube(inSlot, Vector3.one * 0.3f);
        //     }
        //     Gizmos.color = Color.yellow;
        //     for (int i = 0; i < 4; i++)
        //     {
        //         Vector3 outSlot = transform.position + transform.TransformDirection(outgoingOffset)
        //                         + transform.right * (i * 0.75f);
        //         Gizmos.DrawWireCube(outSlot, Vector3.one * 0.3f);
        //     }
        // }

        // ─────────────────────────────────────────────────────────
        //  Decision-Point Flash
        // ─────────────────────────────────────────────────────────

        public void RecordDecisionPoint(float simTime, int[] queuedJobIds, int chosenJobId, string ruleName, bool flash = true)
        {
            decisionPointCount++;
            string queueStr = string.Join(", ", Array.ConvertAll(queuedJobIds, id => $"Job {id}"));
            string entry = $"[t={simTime:F1}] M{machineId} free, queue=[{queueStr}], rule chose Job {chosenJobId} ({ruleName})";
            Log(entry);

            if (flash) Flash();
        }

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

        // ─────────────────────────────────────────────────────────
        //  Helpers
        // ─────────────────────────────────────────────────────────

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