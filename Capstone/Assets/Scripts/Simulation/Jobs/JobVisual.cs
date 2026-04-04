using UnityEngine;

namespace Assets.Scripts.Simulation.Jobs
{
    /// @brief Visual token representing a single job on the factory floor.
    /// @details Smoothly interpolates toward targets set by the JobManager. Movement ownership
    /// switches between self-driven (lerp), carried (parented to AGV), or conveyor-driven.
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
        private JobLifecycleState currentState;

        private bool isCarried = false;
        private bool isOnConveyor = false;

        public int JobId => jobId;
        public JobLifecycleState CurrentState => currentState;

        /// @brief Initializes the visual token and caches rendering components.
        /// @param id Zero-based job index.
        /// @param opCount Total number of operations this job must complete.
        /// @post State is set to NotStarted and targetPosition matches current transform.
        public void Initialize(int id, int opCount)
        {
            jobId = id;
            totalOperations = opCount;
            targetPosition = transform.position;

            meshRenderer = GetComponentInChildren<Renderer>();
            propBlock = new MaterialPropertyBlock();

            SetState(JobLifecycleState.NotStarted);
        }

        /// @brief Updates the token's tint to reflect its current lifecycle state.
        /// @details Uses a MaterialPropertyBlock to update the "_Color" property without creating material instances.
        /// @param state The target @ref JobLifecycleState.
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

        /// @brief Defines a destination for the smooth interpolation logic.
        /// @details If the job is carried or on a conveyor, this request is ignored.
        /// @param worldPos The target global coordinates.
        /// @pre @ref isCarried and @ref isOnConveyor must be false.
        /// @post @ref travelProgress is reset to 0 to initiate movement.
        public void SetTargetPosition(Vector3 worldPos)
        {
            if (isCarried || isOnConveyor) return;
            startPosition = transform.position;
            targetPosition = worldPos;
            travelProgress = 0f;
        }

        /// @brief Instantly teleports the token to a position with no interpolation.
        /// @param worldPos The destination global coordinates.
        /// @post @ref travelProgress is set to 1 and interpolation is finalized.
        public void SnapToPosition(Vector3 worldPos)
        {
            transform.position = worldPos;
            startPosition = worldPos;
            targetPosition = worldPos;
            travelProgress = 1f;
        }

        /// @brief Toggles external movement control by a conveyor belt.
        /// @details While enabled, the token's internal Update loop will not process movement.
        /// @param on True if a ConveyorBelt is now driving the transform.
        public void SetOnConveyor(bool on)
        {
            isOnConveyor = on;
        }

        /// @brief Parents the token to an AGV carrier for physical transport.
        /// @details Disables conveyor control and resets interpolation progress.
        /// @param carrier The transform component of the AGV.
        /// @post Visual is parented with a fixed local offset (0, 0.5, 0).
        public void AttachToCarrier(Transform carrier)
        {
            isCarried = true;
            isOnConveyor = false;
            travelProgress = 1f;
            transform.SetParent(carrier);
            transform.localPosition = new Vector3(0f, 0.5f, 0f);
            transform.localRotation = Quaternion.identity;
        }

        /// @brief Detaches the token from an AGV and returns it to world space.
        /// @param worldSnapPos The world position to snap to upon release.
        /// @post @ref isCarried is false and the visual has no parent transform.
        public void DetachFromCarrier(Vector3 worldSnapPos)
        {
            transform.SetParent(null);
            isCarried = false;
            SnapToPosition(worldSnapPos);
        }

        /// @brief Internal Unity update loop driving self-driven lerps.
        /// @details Interpolates position between startPosition and targetPosition using moveSpeed.
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