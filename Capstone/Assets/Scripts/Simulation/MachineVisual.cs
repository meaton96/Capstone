using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using Assets.Scripts.Scheduling.Core;
using TMPro;

namespace Assets.Scripts.Simulation
{
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

        public Machine CoreMachine => coreMachine;
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
            UpdateQueueLabel(0); // Initialize UI with 0
        }

        private void OnDestroy()
        {
            if (instanceMaterial != null) Destroy(instanceMaterial);
        }

        public void Initialise(int id, Machine coreMachineRef = null)
        {
            machineId = id;
            coreMachine = coreMachineRef;

            if (labelText != null) labelText.text = $"M{id}";

            SetState(MachineState.Idle);
            Log($"Initialised at {transform.position}");
        }

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

        // ─── PROGRESS BAR (Driven strictly by PhysicalMachine) ───

        public void UpdateProgress(float normalizedProgress)
        {
            if (progressBar != null)
            {
                progressBar.value = Mathf.Clamp01(normalizedProgress);
            }
        }

        private void SetProgressBarVisible(bool visible)
        {
            if (progressBar != null) progressBar.gameObject.SetActive(visible);
        }

        // ─── QUEUE UI (Driven strictly by PhysicalMachine) ───

        public void UpdateQueueLabel(int queueCount)
        {
            if (queueLengthText != null)
            {
                queueLengthText.text = $"Q: {queueCount}";
            }
        }

        // ─── DECISION FLASH ───

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

        private void Log(string message)
        {
            historyLog.Add($"[{Time.time:F2}] {message}");
        }
    }
}