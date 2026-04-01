using UnityEngine;

/// @brief Continuously orbits the camera around a target Transform.
///
/// @details Each @c LateUpdate the camera advances its yaw angle by @c orbitSpeed
/// degrees per real-world second (unaffected by @c Time.timeScale), then positions
/// itself behind and above the target before looking at it. Assign a @c target in
/// the Inspector; the camera will log a warning and skip movement if none is set.
public class OrbitCamera : MonoBehaviour
{
    [Header("Target Settings")]
    [Tooltip("The object the camera will orbit around.")]
    public Transform target;

    [Header("Orbit Settings")]
    [Tooltip("Degrees per second the camera rotates around the target.")]
    public float orbitSpeed = 20f;

    [Tooltip("Distance from the target along the camera's local backward axis.")]
    public float zoomDistance = 10f;

    [Tooltip("World-space height offset applied to both camera position and look-at point.")]
    public float heightOffset = 2f;

    private float _currentAngle;

    /// @brief Advances the orbit angle and repositions the camera after all other updates.
    /// @details The angle increment is divided by @c Time.timeScale so the orbit
    ///          speed remains constant regardless of simulation speed changes.
    ///          The look-at target is lifted by half @c heightOffset so the subject
    ///          sits comfortably in the centre of the frame.
    private void LateUpdate()
    {
        if (target == null)
        {
            Debug.LogWarning("OrbitCamera: No target assigned!");
            return;
        }

        _currentAngle += orbitSpeed * Time.deltaTime / Time.timeScale;
        Quaternion rotation = Quaternion.Euler(0, _currentAngle, 0);

        Vector3 offset = new Vector3(0, heightOffset, -zoomDistance);
        Vector3 finalPosition = target.position + (rotation * offset);

        transform.position = finalPosition;
        transform.LookAt(target.position + Vector3.up * (heightOffset * 0.5f));
    }
}