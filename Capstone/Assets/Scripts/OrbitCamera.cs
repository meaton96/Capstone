using UnityEngine;

public class OrbitCamera : MonoBehaviour
{
    [Header("Target Settings")]
    [Tooltip("The object the camera will orbit around.")]
    public Transform target;

    [Header("Orbit Settings")]
    [Tooltip("How fast the camera rotates around the object.")]
    public float orbitSpeed = 20f;

    [Tooltip("The distance from the target. Moves camera along its local forward/backward axis.")]
    public float zoomDistance = 10f;

    [Tooltip("Optional: Height offset if you want the camera to look from slightly above/below.")]
    public float heightOffset = 2f;

    private float _currentAngle;

    void LateUpdate()
    {
        if (target == null)
        {
            Debug.LogWarning("OrbitCamera: No target assigned!");
            return;
        }

        // 1. Calculate the rotation based on time and speed
        _currentAngle += orbitSpeed * Time.deltaTime;
        Quaternion rotation = Quaternion.Euler(0, _currentAngle, 0);

        // 2. Calculate the new position
        // We start at the target, move 'back' by the zoom distance, and up by the offset
        Vector3 offset = new Vector3(0, heightOffset, -zoomDistance);

        // Rotate the offset so it orbits around the center
        Vector3 finalPosition = target.position + (rotation * offset);

        // 3. Apply the position and look at the target
        transform.position = finalPosition;
        transform.LookAt(target.position + Vector3.up * (heightOffset * 0.5f));
    }
}