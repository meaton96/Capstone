using UnityEngine;

namespace Assets.Scripts.Simulation.Jobs
{
    /// <summary>
    /// Visual token representing a single job on the factory floor.
    /// </summary>
    /// <remarks>
    /// Smoothly interpolates toward targets set by the JobManager. Movement ownership
    /// switches between self-driven (lerp), carried (parented to AGV), or conveyor-driven.
    /// </remarks>
    public class JobVisual : MonoBehaviour
    {
        [SerializeField] private float moveSpeed = 2f;

        [Header("State Colors")]
        [SerializeField] private Color notStartedColor = new Color(0.5f, 0.5f, 0.5f, 0.6f);
        [SerializeField] private Color queuedColor = new Color(1.0f, 0.85f, 0.2f, 1f);
        [SerializeField] private Color processingColor = new Color(0.2f, 0.8f, 0.2f, 1f);
        [SerializeField] private Color waitingColor = new Color(1.0f, 0.5f, 0.1f, 1f);
        [SerializeField] private Color inTransitColor = new Color(0.3f, 0.6f, 1.0f, 1f);
        [SerializeField] private Color completeColor = new Color(0.3f, 0.3f, 0.3f, 0.4f);

        private int jobId;
        private int totalOperations;
        private Vector3 targetPosition;
        private Vector3 startPosition;
        private float travelProgress = 1f;
        private Renderer meshRenderer;
        private MaterialPropertyBlock propBlock;
        [SerializeField] private JobState currentState;

        private bool isCarried = false;
        private bool isOnConveyor = false;

        public int JobId => jobId;
        public JobState CurrentState => currentState;

        //public JobLocation jobLocation;

        /// <summary>
        /// Initializes the visual token and caches rendering components.
        /// </summary>
        /// <param name="id">Zero-based job index.</param>
        /// <param name="opCount">Total number of operations this job must complete.</param>
        /// <remarks>
        /// After initialization, the state is set to <see cref="JobState.NeedsRouting"/>
        /// and the target position matches the current transform position.
        /// </remarks>
        public void Initialize(int id, int opCount)
        {
            jobId = id;
            totalOperations = opCount;
            targetPosition = transform.position;

            meshRenderer = GetComponentInChildren<Renderer>();
            propBlock = new MaterialPropertyBlock();

            SetState(JobState.NeedsRouting);
        }

        /// <summary>
        /// Updates the token's tint to reflect its current lifecycle state.
        /// </summary>
        /// <param name="state">The target <see cref="JobState"/>.</param>
        /// <remarks>
        /// Uses a <see cref="MaterialPropertyBlock"/> to update the "_Color" property
        /// without creating material instances, preserving GPU instancing.
        /// </remarks>
        public void SetState(JobState state)
        {
            currentState = state;
            if (meshRenderer == null) return;

            Color c = state switch
            {
                JobState.NeedsRouting => notStartedColor,
                JobState.WaitingForPickup => waitingColor,
                JobState.InTransit => inTransitColor,
                JobState.Queued => queuedColor,
                JobState.Processing => processingColor,
                JobState.Exited => completeColor,
                _ => notStartedColor,
            };

            meshRenderer.GetPropertyBlock(propBlock);
            propBlock.SetColor("_Color", c);
            meshRenderer.SetPropertyBlock(propBlock);
        }

        /// <summary>
        /// Defines a destination for the smooth interpolation logic.
        /// </summary>
        /// <param name="worldPos">The target global coordinates.</param>
        /// <remarks>
        /// If the job is carried or on a conveyor, this request is ignored.
        /// </remarks>
        public void SetTargetPosition(Vector3 worldPos)
        {
            if (isCarried || isOnConveyor) return;
            startPosition = transform.position;
            targetPosition = worldPos;
            travelProgress = 0f;
        }

        /// <summary>
        /// Instantly teleports the token to a position with no interpolation.
        /// </summary>
        /// <param name="worldPos">The destination global coordinates.</param>
        /// <remarks>
        /// Resets <see cref="travelProgress"/> to 1 and finalizes interpolation.
        /// </remarks>
        public void SnapToPosition(Vector3 worldPos)
        {
            transform.position = worldPos;
            startPosition = worldPos;
            targetPosition = worldPos;
            travelProgress = 1f;
        }

        /// <summary>
        /// Toggles external movement control by a conveyor belt.
        /// </summary>
        /// <param name="on">True if a conveyor belt is now driving the transform.</param>
        /// <remarks>
        /// While enabled, the token's internal <c>Update</c> loop will not process movement.
        /// </remarks>
        public void SetOnConveyor(bool on)
        {
            isOnConveyor = on;
        }

        /// <summary>
        /// Parents the token to an AGV carrier for physical transport.
        /// </summary>
        /// <param name="carrier">The <see cref="Transform"/> component of the AGV.</param>
        /// <remarks>
        /// Disables conveyor control and resets interpolation progress. The visual is
        /// parented with a fixed local offset of (0, 0.5, 0).
        /// </remarks>
        public void AttachToCarrier(Transform carrier)
        {
            isCarried = true;
            isOnConveyor = false;
            travelProgress = 1f;
            transform.SetParent(carrier);
            transform.localPosition = new Vector3(0f, 0.5f, 0f);
            transform.localRotation = Quaternion.identity;
        }

        /// <summary>
        /// Detaches the token from an AGV and returns it to world space.
        /// </summary>
        /// <param name="worldSnapPos">The world position to snap to upon release.</param>
        /// <remarks>
        /// Sets <see cref="isCarried"/> to false and removes the parent transform.
        /// </remarks>
        public void DetachFromCarrier(Vector3 worldSnapPos)
        {
            transform.SetParent(null);
            isCarried = false;
            SnapToPosition(worldSnapPos);
        }

        /// <summary>
        /// Internal Unity update loop driving self-driven position interpolation.
        /// </summary>
        /// <remarks>
        /// Interpolates position between <c>startPosition</c> and <c>targetPosition</c>
        /// using <see cref="moveSpeed"/>. Skips processing if carried or on a conveyor.
        /// </remarks>
        private void Update()
        {
            if (isCarried || isOnConveyor) return;
            if (travelProgress >= 1f) return;

            travelProgress += Time.deltaTime * moveSpeed;
            if (travelProgress > 1f) travelProgress = 1f;

            transform.position = Vector3.Lerp(startPosition, targetPosition, travelProgress);
        }
    }
}