using UnityEngine;

namespace Assets.Scripts.Simulation
{
    /// @file JobVisual.cs
    /// @brief Visual token representing a single job on the factory floor.
    ///
    /// @details Attach this to a small prefab (e.g., a colored cube or sphere).
    /// The JobManager sets its position and state each sync cycle. The token
    /// smoothly interpolates to its target position so it doesn't teleport
    /// when the DES jumps forward in time.
    ///
    /// @par Prefab Setup
    /// Create a small GameObject (0.3-0.5 scale) with a MeshRenderer.
    /// Add this component. The JobManager will call Initialize() at episode start.
    /// Colors are set automatically based on lifecycle state.

    public class JobVisual : MonoBehaviour
    {
        // ─────────────────────────────────────────────────────────
        //  Inspector
        // ─────────────────────────────────────────────────────────

        [Header("Movement")]
        [Tooltip("How quickly the token moves toward its target position.")]
        [SerializeField] private float moveSpeed = 2f;
        [Tooltip("How high the job hops when moving between machines.")]
        [SerializeField] private float hopHeight = 3.0f;

        private Vector3 startPosition;
        private float travelProgress = 1f;

        [Header("State Colors")]
        [SerializeField] private Color notStartedColor = new Color(0.5f, 0.5f, 0.5f, 0.6f);
        [SerializeField] private Color queuedColor = new Color(1.0f, 0.85f, 0.2f, 1f);
        [SerializeField] private Color processingColor = new Color(0.2f, 0.8f, 0.2f, 1f);
        [SerializeField] private Color waitingColor = new Color(1.0f, 0.5f, 0.1f, 1f);
        [SerializeField] private Color inTransitColor = new Color(0.3f, 0.6f, 1.0f, 1f);
        [SerializeField] private Color completeColor = new Color(0.3f, 0.3f, 0.3f, 0.4f);



        // ─────────────────────────────────────────────────────────
        //  Runtime state
        // ─────────────────────────────────────────────────────────

        private int jobId;
        private int totalOperations;
        private Vector3 targetPosition;
        private Renderer meshRenderer;
        private MaterialPropertyBlock propBlock;
        private JobLifecycleState currentState;

        // ─────────────────────────────────────────────────────────
        //  Public accessors
        // ─────────────────────────────────────────────────────────

        public int JobId => jobId;
        public JobLifecycleState CurrentState => currentState;

        // ─────────────────────────────────────────────────────────
        //  Lifecycle
        // ─────────────────────────────────────────────────────────

        /// @brief Called by JobManager when the tracker is created.
        public void Initialize(int id, int opCount)
        {
            jobId = id;
            totalOperations = opCount;
            targetPosition = transform.position;

            meshRenderer = GetComponentInChildren<Renderer>();
            propBlock = new MaterialPropertyBlock();

            SetState(JobLifecycleState.NotStarted);
        }

        private void Update()
        {
            // If we haven't reached the destination yet
            if (travelProgress < 1f)
            {
                travelProgress += Time.deltaTime * moveSpeed;

                // Clamp it so we don't overshoot
                if (travelProgress > 1f) travelProgress = 1f;

                // 1. Move linearly along the X/Z floor
                Vector3 currentPos = Vector3.Lerp(startPosition, targetPosition, travelProgress);

                // 2. Add the vertical hop using a Sine wave
                // Mathf.PI ensures the wave goes from 0 (start) up to 1 (middle) and back to 0 (end)
                float currentHeight = Mathf.Sin(travelProgress * Mathf.PI) * hopHeight;
                currentPos.y += currentHeight;

                transform.position = currentPos;
            }
        }

        // ─────────────────────────────────────────────────────────
        //  State management
        // ─────────────────────────────────────────────────────────

        /// @brief Updates the visual appearance based on lifecycle state.
        public void SetState(JobLifecycleState state)
        {
            currentState = state;

            if (meshRenderer == null) return;

            Color c;
            switch (state)
            {
                case JobLifecycleState.NotStarted: c = notStartedColor; break;
                case JobLifecycleState.Queued: c = queuedColor; break;
                case JobLifecycleState.Processing: c = processingColor; break;
                case JobLifecycleState.WaitingForTransport: c = waitingColor; break;
                case JobLifecycleState.InTransit: c = inTransitColor; break;
                case JobLifecycleState.Complete: c = completeColor; break;
                default: c = notStartedColor; break;
            }

            meshRenderer.GetPropertyBlock(propBlock);
            propBlock.SetColor("_Color", c);
            meshRenderer.SetPropertyBlock(propBlock);
        }

        /// @brief Sets the position the token should move toward.
        public void SetTargetPosition(Vector3 pos)
        {
            startPosition = transform.position;
            targetPosition = pos;
            travelProgress = 0f; // Reset progress to trigger the animation

        }

        /// @brief Updates the token scale to reflect operation progress (0-1).
        /// A subtle vertical stretch so the user can see which jobs are
        /// further along in their current operation.
        public void SetProgress(float progress)
        {
            // Gentle pulse: scale Y from 1.0 to 1.3 as progress advances
            float scaleY = 1f + progress * 0.3f;
            transform.localScale = new Vector3(1f, scaleY, 1f) * 0.4f;
        }
    }
}