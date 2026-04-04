using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using TMPro;
using Assets.Scripts.Simulation.Types;

namespace Assets.Scripts.Simulation.Machines
{
    public class MachineVisual : MonoBehaviour
    {
        [Header("Identity")]
        [SerializeField] private int machineId;

        [Header("Rendering")]
        [SerializeField] private MeshRenderer machineBodyRenderer;
        [SerializeField] private MeshRenderer indicatorRenderer;
        [SerializeField] private Vector3 incomingOffset = new Vector3(-2.5f, -.5f, 0f);
        [SerializeField] private Vector3 outgoingOffset = new Vector3(2.5f, -.5f, 0f);

        /// @brief One material per MachineState, assigned in Inspector.
        /// Order must match MachineState enum: Idle, Busy, Blocked, Failed, Repair
        [SerializeField] private Material[] indicatorMaterials;

        [Header("Overhead UI")]
        [SerializeField] private TextMeshProUGUI labelText;
        [SerializeField] private TextMeshProUGUI statusText;
        [SerializeField] private Slider progressBar;
        [SerializeField] private TextMeshProUGUI incomingQueueLabel;
        [SerializeField] private TextMeshProUGUI outgoingQueueLabel;

        [Header("Decision-Point Flash")]
        [SerializeField] private Color flashColour = Color.white;
        [SerializeField] private float flashDuration = 0.25f;

        // ── Type identity colours ───────────────────────────────────────────
        private static readonly Dictionary<MachineType, Color> TypeColors = new()
        {
            { MachineType.Mill,     new Color(0.20f, 0.40f, 0.80f) },   // steel blue
            { MachineType.Lathe,    new Color(0.80f, 0.50f, 0.10f) },   // amber
            { MachineType.Weld,     new Color(0.70f, 0.20f, 0.20f) },   // deep red
            { MachineType.Inspect,  new Color(0.20f, 0.70f, 0.40f) },   // green
            { MachineType.Assemble, new Color(0.55f, 0.25f, 0.70f) },   // purple
        };

        private MachineState currentState = MachineState.Idle;
        private Material bodyInstanceMaterial;
        private Material indicatorInstanceMaterial;
        private Coroutine activeFlash;
        private int decisionPointCount;
        private readonly List<string> historyLog = new List<string>();

        public int MachineId => machineId;
        public MachineState CurrentState => currentState;
        public int DecisionPointCount => decisionPointCount;
        public IReadOnlyList<string> HistoryLog => historyLog;

        // ── Unity Lifecycle ─────────────────────────────────────────────────

        private void Awake()
        {
            // Body gets its own material instance so type color doesn't bleed
            // across all machines sharing the same prefab material
            if (machineBodyRenderer != null)
            {
                bodyInstanceMaterial = new Material(machineBodyRenderer.sharedMaterial);
                machineBodyRenderer.material = bodyInstanceMaterial;
            }

            // Indicator starts on the Idle material from the inspector list
            if (indicatorRenderer != null)
                ApplyIndicatorMaterial(MachineState.Idle);

            SetProgressBarVisible(false);
            UpdateIncomingQueueLabel(0);
            UpdateOutgoingQueueLabel(0);
        }

        private void OnDestroy()
        {
            if (bodyInstanceMaterial != null) Destroy(bodyInstanceMaterial);
            if (indicatorInstanceMaterial != null) Destroy(indicatorInstanceMaterial);
        }

        // ── Initialisation ──────────────────────────────────────────────────

        /// @brief Sets permanent type identity color on the machine body and resets state.
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

        /// @brief Transitions to a new operational state.
        /// @details Updates the indicator light material and status text.
        ///          The machine body color never changes after Initialise.
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

        public void SetBlockedAfterProcessing(int jobId)
        {
            SetProgressBarVisible(false);
            SetState(MachineState.Blocked);
            Log($"Blocked: outgoing full, holding Job {jobId}");
        }

        public void UpdateProgress(float normalizedProgress)
        {
            if (progressBar != null)
                progressBar.value = Mathf.Clamp01(normalizedProgress);
        }

        // ── Queue Labels ────────────────────────────────────────────────────

        public void UpdateIncomingQueueLabel(int count)
        {
            if (incomingQueueLabel != null) incomingQueueLabel.text = $"IN: {count}";
        }

        public void UpdateOutgoingQueueLabel(int count)
        {
            if (outgoingQueueLabel != null) outgoingQueueLabel.text = $"OUT: {count}";
        }

        // ── Decision Flash ──────────────────────────────────────────────────

        /// @brief Records a scheduling decision and flashes the indicator briefly white.
        public void RecordDecisionPoint(float simTime, int[] queuedJobIds, int chosenJobId, string ruleName, bool flash = true)
        {
            decisionPointCount++;
            string queueStr = string.Join(", ", Array.ConvertAll(queuedJobIds, id => $"Job {id}"));
            Log($"[t={simTime:F1}] queue=[{queueStr}], chose Job {chosenJobId} ({ruleName})");
            if (flash) Flash();
        }

        public void Flash()
        {
            if (indicatorRenderer == null) return;
            if (activeFlash != null) StopCoroutine(activeFlash);
            activeFlash = StartCoroutine(FlashRoutine());
        }

        // ── Private Helpers ─────────────────────────────────────────────────

        /// @brief Swaps the indicator renderer to the material matching <paramref name="state"/>.
        /// @details Creates an instance copy so emission tweaks don't affect the shared asset.
        private void ApplyIndicatorMaterial(MachineState state)
        {
            if (indicatorRenderer == null) return;

            int index = (int)state;
            if (indicatorMaterials == null || index >= indicatorMaterials.Length || indicatorMaterials[index] == null)
            {
                Debug.LogWarning($"[MachineVisual] No indicator material for state {state} on M{machineId}");
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
        private void SetProgressBarVisible(bool visible)
        {
            if (progressBar != null) progressBar.gameObject.SetActive(visible);
        }

        private IEnumerator FlashRoutine()
        {
            // Flash white on the indicator, then restore the current state material
            if (indicatorInstanceMaterial != null)
                indicatorInstanceMaterial.color = flashColour;

            yield return new WaitForSeconds(flashDuration);

            // Re-apply the correct state material cleanly
            ApplyIndicatorMaterial(currentState);
            activeFlash = null;
        }

        private void SetLabel(string label)
        {
            if (labelText != null) labelText.text = label;
        }

        private void Log(string message)
        {
            historyLog.Add($"[{Time.time:F2}] {message}");
        }
    }
}