using UnityEngine;
using UnityEngine.InputSystem;

[RequireComponent(typeof(OrbitCamera))]
public class CameraController : MonoBehaviour
{
    private enum CameraState { Orbit, FreeLook }
    private CameraState currentState = CameraState.Orbit;

    private OrbitCamera orbitCamera;

    [Header("Free Look Settings")]
    [Tooltip("Movement speed in Free Look mode.")]
    public float moveSpeed = 15f;

    [Tooltip("Mouse look sensitivity in Free Look mode.")]
    public float lookSensitivity = 0.2f;

    private float pitch;
    private float yaw;

    private void Start()
    {
        // Grab the existing OrbitCamera script on this GameObject
        orbitCamera = GetComponent<OrbitCamera>();

        // Ensure we start in the correct state
        orbitCamera.enabled = (currentState == CameraState.Orbit);
    }

    private void Update()
    {
        // Safety check to ensure the New Input System is active
        if (Keyboard.current == null || Mouse.current == null) return;

        // Toggle camera mode on 'F' key press
        if (Keyboard.current.fKey.wasPressedThisFrame)
        {
            ToggleCameraMode();
        }

        // Handle Free Look behaviour if active
        if (currentState == CameraState.FreeLook)
        {
            HandleFreeLook();
            HandleMovement();
        }
    }

    private void ToggleCameraMode()
    {
        // Switch states
        currentState = currentState == CameraState.Orbit ? CameraState.FreeLook : CameraState.Orbit;

        // Enable or disable the orbit script based on the new state
        orbitCamera.enabled = (currentState == CameraState.Orbit);

        // When switching to Free Look, sync the internal pitch/yaw to where the orbit camera left off
        // This prevents the camera from snapping back to a previous rotation
        if (currentState == CameraState.FreeLook)
        {
            Vector3 euler = transform.eulerAngles;
            pitch = euler.x;
            yaw = euler.y;
        }
    }

    private void HandleFreeLook()
    {
        // Only rotate if the Right Mouse Button is held down
        if (Mouse.current.rightButton.isPressed)
        {
            Vector2 mouseDelta = Mouse.current.delta.ReadValue();

            yaw += mouseDelta.x * lookSensitivity;
            pitch -= mouseDelta.y * lookSensitivity;

            // Clamp the pitch so the camera doesn't flip upside down
            pitch = Mathf.Clamp(pitch, -89f, 89f);

            transform.rotation = Quaternion.Euler(pitch, yaw, 0f);
        }
    }

    private void HandleMovement()
    {
        Vector3 moveDirection = Vector3.zero;

        // Forward / Backward
        if (Keyboard.current.wKey.isPressed) moveDirection += transform.forward;
        if (Keyboard.current.sKey.isPressed) moveDirection -= transform.forward;

        // Left / Right
        if (Keyboard.current.aKey.isPressed) moveDirection -= transform.right;
        if (Keyboard.current.dKey.isPressed) moveDirection += transform.right;

        // Up / Down (Optional standard freecam controls using E and Q)
        if (Keyboard.current.eKey.isPressed) moveDirection += transform.up;
        if (Keyboard.current.qKey.isPressed) moveDirection -= transform.up;

        // Apply movement
        if (moveDirection != Vector3.zero)
        {
            transform.position += moveDirection.normalized * (moveSpeed * Time.deltaTime);
        }
    }
}