using UnityEngine;
using UnityEngine.InputSystem;

/// <summary>
/// Controls the camera behavior, allowing toggling between an orbit mode and a free-look mode.
/// Requires an OrbitCamera component to be attached to the same GameObject.
/// </summary>
[RequireComponent(typeof(OrbitCamera))]
public class CameraController : MonoBehaviour
{
    /// <summary>
    /// Defines the available operational states of the camera.
    /// </summary>
    private enum CameraState { Orbit, FreeLook }

    private CameraState currentState = CameraState.Orbit;
    private OrbitCamera orbitCamera;

    [Header("Free Look Settings")]

    /// <summary>
    /// Movement speed of the camera when operating in Free Look mode.
    /// </summary>
    [Tooltip("Movement speed in Free Look mode.")]
    public float moveSpeed = 15f;

    /// <summary>
    /// Mouse look sensitivity when operating in Free Look mode.
    /// </summary>
    [Tooltip("Mouse look sensitivity in Free Look mode.")]
    public float lookSensitivity = 0.2f;

    private float pitch;
    private float yaw;

    /// <summary>
    /// Initializes the camera controller, acquiring required components and setting the initial state.
    /// </summary>
    private void Start()
    {
        orbitCamera = GetComponent<OrbitCamera>();
        orbitCamera.enabled = (currentState == CameraState.Orbit);
    }

    /// <summary>
    /// Handles input processing per frame to toggle camera modes and update free-look mechanics.
    /// </summary>
    private void Update()
    {
        if (Keyboard.current == null || Mouse.current == null) return;

        if (Keyboard.current.fKey.wasPressedThisFrame)
        {
            ToggleCameraMode();
        }

        if (currentState == CameraState.FreeLook)
        {
            HandleFreeLook();
            HandleMovement();
        }
    }

    /// <summary>
    /// Toggles the camera state between Orbit and FreeLook modes, synchronizing rotation values.
    /// </summary>
    private void ToggleCameraMode()
    {
        currentState = currentState == CameraState.Orbit ? CameraState.FreeLook : CameraState.Orbit;
        orbitCamera.enabled = (currentState == CameraState.Orbit);

        if (currentState == CameraState.FreeLook)
        {
            Vector3 euler = transform.eulerAngles;
            pitch = euler.x;
            yaw = euler.y;
        }
    }

    /// <summary>
    /// Processes mouse input to adjust the camera's pitch and yaw when in FreeLook mode.
    /// </summary>
    private void HandleFreeLook()
    {
        if (Mouse.current.rightButton.isPressed)
        {
            Vector2 mouseDelta = Mouse.current.delta.ReadValue();

            yaw += mouseDelta.x * lookSensitivity;
            pitch -= mouseDelta.y * lookSensitivity;
            pitch = Mathf.Clamp(pitch, -89f, 89f);

            transform.rotation = Quaternion.Euler(pitch, yaw, 0f);
        }
    }

    /// <summary>
    /// Processes keyboard input to translate the camera through space when in FreeLook mode.
    /// </summary>
    private void HandleMovement()
    {
        Vector3 moveDirection = Vector3.zero;

        if (Keyboard.current.wKey.isPressed) moveDirection += transform.forward;
        if (Keyboard.current.sKey.isPressed) moveDirection -= transform.forward;

        if (Keyboard.current.aKey.isPressed) moveDirection -= transform.right;
        if (Keyboard.current.dKey.isPressed) moveDirection += transform.right;

        if (Keyboard.current.eKey.isPressed) moveDirection += transform.up;
        if (Keyboard.current.qKey.isPressed) moveDirection -= transform.up;

        if (moveDirection != Vector3.zero)
        {
            transform.position += moveDirection.normalized * (moveSpeed * Time.deltaTime);
        }
    }
}