using UnityEngine;

namespace Assets.Scripts.Simulation
{
    /// @brief Visual token representing a single job on the factory floor.
    ///
    /// @details Attach this to a small prefab (e.g., a colored cube or sphere).
    /// The @c JobManager sets its target position and state each time a physics
    /// event fires. The token smoothly interpolates toward its destination using
    /// a parabolic arc so it visually hops between machines.
    ///
    /// @par Prefab Setup
    /// Create a small GameObject (0.3–0.5 scale) with a @c MeshRenderer.
    /// Add this component. @c JobManager.Initialize() will call @c Initialize()
    /// at episode start. Colors are set automatically from lifecycle state.
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

        [Header("State Colors")]
        [SerializeField] private Color notStartedColor = new Color(0.5f, 0.5f, 0.5f, 0.6f);
        [SerializeField] private Color queuedColor = new Color(1.0f, 0.85f, 0.2f, 1f);
        [SerializeField] private Color processingColor = new Color(0.2f, 0.8f, 0.2f, 1f);
        [SerializeField] private Color waitingColor = new Color(1.0f, 0.5f, 0.1f, 1f);
        [SerializeField] private Color inTransitColor = new Color(0.3f, 0.6f, 1.0f, 1f);
        [SerializeField] private Color completeColor = new Color(0.3f, 0.3f, 0.3f, 0.4f);

        // ─────────────────────────────────────────────────────────
        //  Runtime State
        // ─────────────────────────────────────────────────────────

        private int jobId;
        private int totalOperations;
        private Vector3 targetPosition;
        private Vector3 startPosition;
        private float travelProgress = 1f;
        private Renderer meshRenderer;
        private MaterialPropertyBlock propBlock;
        private JobLifecycleState currentState;

        // ─────────────────────────────────────────────────────────
        //  Public Accessors
        // ─────────────────────────────────────────────────────────

        /// @brief The job ID this token represents.
        public int JobId => jobId;

        /// @brief The token's current lifecycle state.
        public JobLifecycleState CurrentState => currentState;

        // ─────────────────────────────────────────────────────────
        //  Lifecycle
        // ─────────────────────────────────────────────────────────

        /// @brief Called by @c JobManager when the tracker is created.
        /// @param id       Zero-based job index.
        /// @param opCount  Total number of operations this job must complete.
        public void Initialize(int id, int opCount)
        {
            jobId = id;
            totalOperations = opCount;
            targetPosition = transform.position;

            meshRenderer = GetComponentInChildren<Renderer>();
            propBlock = new MaterialPropertyBlock();

            SetState(JobLifecycleState.NotStarted);
        }

        /// @brief Advances the token along the arc toward its target position each frame.
        private void Update()
        {
            if (travelProgress < 1f)
            {
                travelProgress += Time.deltaTime * moveSpeed;
                if (travelProgress > 1f) travelProgress = 1f;

                Vector3 currentPos = Vector3.Lerp(startPosition, targetPosition, travelProgress);
                float currentHeight = Mathf.Sin(travelProgress * Mathf.PI) * hopHeight;
                currentPos.y += currentHeight;

                transform.position = currentPos;
            }
        }

        // ─────────────────────────────────────────────────────────
        //  Public API
        // ─────────────────────────────────────────────────────────

        /// @brief Updates the token's tint to reflect its current lifecycle state.
        /// @param state  The new @c JobLifecycleState to display.
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

        /// @brief Begins animating the token toward a new world position.
        /// @param pos  The destination position in world space.
        public void SetTargetPosition(Vector3 pos)
        {
            startPosition = transform.position;
            targetPosition = pos;
            travelProgress = 0f;
        }

        /// @brief Scales the token vertically to reflect progress through the current operation.
        /// @details Y scale grows from 1.0 to 1.3 as @p progress advances from 0 to 1,
        ///          providing a subtle visual cue of how far along an operation is.
        /// @param progress  Normalised operation progress in [0, 1].
        public void SetProgress(float progress)
        {
            float scaleY = 1f + progress * 0.3f;
            transform.localScale = new Vector3(1f, scaleY, 1f) * 0.4f;
        }
    }
}