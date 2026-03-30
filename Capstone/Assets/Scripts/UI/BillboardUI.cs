using UnityEngine;

public class BillboardUI : MonoBehaviour
{
    [Header("Settings")]
    [Tooltip("If true, the UI will only rotate on the Y-axis (useful for floating nameplates).")]
    public bool lockVertical = false;

    private Transform _mainCameraTransform;

    void Start()
    {
        // Cache the camera transform for better performance
        if (Camera.main != null)
        {
            _mainCameraTransform = Camera.main.transform;
        }
    }

    // LateUpdate is used so the UI follows the camera AFTER the camera has moved
    void LateUpdate()
    {
        if (_mainCameraTransform == null) return;

        // Calculate the direction from the UI to the camera
        Vector3 targetDirection = transform.position - _mainCameraTransform.position;

        if (lockVertical)
        {
            targetDirection.y = 0; // Flatten the vector so it only rotates horizontally
        }

        // Apply the rotation
        // We use -targetDirection because 'forward' for UI is technically facing away from the camera
        if (targetDirection != Vector3.zero)
        {
            transform.rotation = Quaternion.LookRotation(targetDirection);
        }
    }
}