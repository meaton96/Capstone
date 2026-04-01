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



        private bool isCarried = false;

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
        public void SetTargetPosition(Vector3 worldPos)
        {
            if (isCarried) return;  // ignore position commands while on an AGV
            startPosition = transform.position;
            targetPosition = worldPos;
            travelProgress = 0f;
        }
        public void SnapToPosition(Vector3 worldPos)
        {
            transform.position = worldPos;
            startPosition = worldPos;
            targetPosition = worldPos;
            travelProgress = 1f;
        }

        // Replace AttachToCarrier to use local position correctly:
        public void AttachToCarrier(Transform carrier)
        {
            isCarried = true;
            travelProgress = 1f;
            transform.SetParent(carrier);
            transform.localPosition = new Vector3(0f, 0.5f, 0f); // local, not world
            transform.localRotation = Quaternion.identity;
        }

        // Replace DetachFromCarrier to snap cleanly:
        public void DetachFromCarrier(Vector3 worldSnapPos)
        {
            transform.SetParent(null);          // unparent first
            isCarried = false;
            SnapToPosition(worldSnapPos);       // then snap — order matters
        }

        // Guard Update against carrying state and redundant lerps:
        private void Update()
        {
            if (isCarried) return;
            if (travelProgress >= 1f) return;

            travelProgress += Time.deltaTime * moveSpeed;
            if (travelProgress > 1f) travelProgress = 1f;

            transform.position = Vector3.Lerp(startPosition, targetPosition, travelProgress);
        }

        /// @brief Scales the token vertically to reflect progress through the current operation.
        /// @details Y scale grows from 1.0 to 1.3 as @p progress advances from 0 to 1,
        ///          providing a subtle visual cue of how far along an operation is.
        /// @param progress  Normalised operation progress in [0, 1].
        public void SetProgress(float progress)
        {
            // float scaleY = 1f + progress * 0.3f;
            // transform.localScale = new Vector3(1f, scaleY, 1f) * 0.4f;
        }
    }
}