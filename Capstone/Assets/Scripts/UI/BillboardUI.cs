using UnityEngine;

/// @brief Rotates this GameObject each frame to face the main camera.
///
/// @details Attach to any world-space UI element (e.g., a floating nameplate or
/// overhead label) that should always be readable from the player's viewpoint.
/// Uses @c LateUpdate so the orientation is applied after all camera movement
/// has settled for the frame.
public class BillboardUI : MonoBehaviour
{
    [Header("Settings")]
    [Tooltip("If true, the UI will only rotate on the Y-axis (useful for floating nameplates).")]
    public bool lockVertical = false;

    private Transform _mainCameraTransform;

    /// @brief Caches the main camera's transform to avoid repeated @c Camera.main lookups.
    private void Start()
    {
        if (Camera.main != null)
        {
            _mainCameraTransform = Camera.main.transform;
        }
    }

    /// @brief Orients the UI toward the camera after all movement updates have run.
    /// @details When @c lockVertical is true the direction vector is flattened on Y
    ///          so the element only yaws, preventing tilting on sloped terrain.
    ///          The direction is negated before passing to @c Quaternion.LookRotation
    ///          because the UI's forward axis points away from the camera.
    private void LateUpdate()
    {
        if (_mainCameraTransform == null) return;

        Vector3 targetDirection = transform.position - _mainCameraTransform.position;

        if (lockVertical)
        {
            targetDirection.y = 0;
        }

        if (targetDirection != Vector3.zero)
        {
            transform.rotation = Quaternion.LookRotation(targetDirection);
        }
    }
}