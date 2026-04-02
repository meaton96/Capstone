using UnityEngine;

namespace Assets.Scripts.Simulation
{
    /// <summary>
    /// Visual token representing a single job on the factory floor.
    ///
    /// Attach this to a small prefab (e.g., a colored cube or sphere).
    /// The <see cref="JobManager"/> sets its target position and state each
    /// time a physics event fires. The token smoothly interpolates toward its
    /// destination so it visually glides between machines.
    ///
    /// <para><b>Movement ownership:</b></para>
    /// <list type="bullet">
    ///   <item><b>Self-driven:</b> default — lerps toward <c>targetPosition</c> in Update.</item>
    ///   <item><b>Carried:</b> parented to an AGV — Update is skipped.</item>
    ///   <item><b>Conveyor:</b> a <see cref="ConveyorBelt"/> directly sets
    ///     <c>transform.position</c> each frame — Update is skipped.</item>
    /// </list>
    /// </summary>
    public class JobVisual : MonoBehaviour
    {
        // ─────────────────────────────────────────────────────────
        //  Inspector
        // ─────────────────────────────────────────────────────────

        [Header("Movement")]
        [Tooltip("How quickly the token self-drives toward its target position.")]
        [SerializeField] private float moveSpeed = 2f;

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

        /// <summary>True when parented to an AGV.</summary>
        private bool isCarried = false;

        /// <summary>
        /// True when a <see cref="ConveyorBelt"/> is driving this token's position.
        /// While on a conveyor the token's own Update is skipped — the belt
        /// moves it directly via <c>transform.position</c>.
        /// </summary>
        private bool isOnConveyor = false;

        // ─────────────────────────────────────────────────────────
        //  Public Accessors
        // ─────────────────────────────────────────────────────────

        /// <summary>The job ID this token represents.</summary>
        public int JobId => jobId;

        /// <summary>The token's current lifecycle state.</summary>
        public JobLifecycleState CurrentState => currentState;

        // ─────────────────────────────────────────────────────────
        //  Lifecycle
        // ─────────────────────────────────────────────────────────

        /// <summary>
        /// Called by <see cref="JobManager"/> when the tracker is created.
        /// </summary>
        /// <param name="id">Zero-based job index.</param>
        /// <param name="opCount">Total number of operations this job must complete.</param>
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
        //  State / Color
        // ─────────────────────────────────────────────────────────

        /// <summary>Updates the token's tint to reflect its current lifecycle state.</summary>
        public void SetState(JobLifecycleState state)
        {
            currentState = state;
            if (meshRenderer == null) return;

            Color c = state switch
            {
                JobLifecycleState.NotStarted => notStartedColor,
                JobLifecycleState.Queued => queuedColor,
                JobLifecycleState.Processing => processingColor,
                JobLifecycleState.WaitingForTransport => waitingColor,
                JobLifecycleState.InTransit => inTransitColor,
                JobLifecycleState.Complete => completeColor,
                _ => notStartedColor,
            };

            meshRenderer.GetPropertyBlock(propBlock);
            propBlock.SetColor("_Color", c);
            meshRenderer.SetPropertyBlock(propBlock);
        }

        // ─────────────────────────────────────────────────────────
        //  Position Control
        // ─────────────────────────────────────────────────────────

        /// <summary>
        /// Begins a smooth self-driven lerp toward a new world position.
        /// Ignored while carried by an AGV or managed by a conveyor.
        /// </summary>
        public void SetTargetPosition(Vector3 worldPos)
        {
            if (isCarried || isOnConveyor) return;
            startPosition = transform.position;
            targetPosition = worldPos;
            travelProgress = 0f;
        }

        /// <summary>Teleports the token to a position with no interpolation.</summary>
        public void SnapToPosition(Vector3 worldPos)
        {
            transform.position = worldPos;
            startPosition = worldPos;
            targetPosition = worldPos;
            travelProgress = 1f;
        }

        // ─────────────────────────────────────────────────────────
        //  Conveyor Ownership
        // ─────────────────────────────────────────────────────────

        /// <summary>
        /// Called by <see cref="ConveyorBelt"/> when a job enters or leaves a belt.
        /// While <paramref name="on"/> is true, the conveyor drives
        /// <c>transform.position</c> directly and this token's Update is skipped.
        /// </summary>
        public void SetOnConveyor(bool on)
        {
            isOnConveyor = on;
        }

        // ─────────────────────────────────────────────────────────
        //  AGV Carrier Parenting
        // ─────────────────────────────────────────────────────────

        /// <summary>Parents the token to an AGV carrier transform.</summary>
        public void AttachToCarrier(Transform carrier)
        {
            isCarried = true;
            isOnConveyor = false; // AGV takes priority
            travelProgress = 1f;
            transform.SetParent(carrier);
            transform.localPosition = new Vector3(0f, 0.5f, 0f);
            transform.localRotation = Quaternion.identity;
        }

        /// <summary>Un-parents the token and snaps it to a world position.</summary>
        public void DetachFromCarrier(Vector3 worldSnapPos)
        {
            transform.SetParent(null);
            isCarried = false;
            SnapToPosition(worldSnapPos);
        }

        // ─────────────────────────────────────────────────────────
        //  Update — Self-Driven Lerp
        // ─────────────────────────────────────────────────────────

        private void Update()
        {
            // Skip when another system owns our position.
            if (isCarried || isOnConveyor) return;
            if (travelProgress >= 1f) return;

            travelProgress += Time.deltaTime * moveSpeed;
            if (travelProgress > 1f) travelProgress = 1f;

            transform.position = Vector3.Lerp(startPosition, targetPosition, travelProgress);
        }

        // ─────────────────────────────────────────────────────────
        //  Progress (cosmetic)
        // ─────────────────────────────────────────────────────────

        /// <summary>
        /// Scales the token vertically to reflect operation progress.
        /// Currently disabled (no-op). Un-comment the body to enable.
        /// </summary>
        public void SetProgress(float progress)
        {
            // float scaleY = 1f + progress * 0.3f;
            // transform.localScale = new Vector3(1f, scaleY, 1f) * 0.4f;
        }
    }
}