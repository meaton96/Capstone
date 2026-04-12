using UnityEngine;
using UnityEngine.InputSystem;
using System.Collections.Generic;
using Assets.Scripts.Simulation.AGV;
using Assets.Scripts.Logging;

/// @brief Attaches the camera to AGVs within the AGVPool.
///
/// @details Toggles control away from CameraController and OrbitCamera when active. 
/// When enabled, the camera smoothly tracks a specific AGV's position and rotation 
/// based on a defined offset.
[RequireComponent(typeof(CameraController))]
[RequireComponent(typeof(OrbitCamera))]
public class AGVFollowCamera : MonoBehaviour
{
    [Header("References")]
    [SerializeField] private AGVPool agvPool;
    private CameraController manualController;
    private OrbitCamera orbitCamera;

    [Header("Follow Settings")]
    [Tooltip("Camera offset relative to the AGV's position and rotation.")]
    public Vector3 followOffset = new Vector3(0, 4f, -6f);

    [Tooltip("How smoothly the camera catches up to the AGV.")]
    public float smoothSpeed = 10f;

    private int currentAgvIndex = 0;
    private bool isFollowing = false;

    /// @brief Initializes references to the sibling camera control components.
    private void Awake()
    {
        manualController = GetComponent<CameraController>();
        orbitCamera = GetComponent<OrbitCamera>();
    }

    /// @brief Polls for user input to toggle camera modes or swap AGV targets.
    ///
    /// @details Monitors the @c Keyboard for the 'C' key to enter/exit follow mode 
    /// and uses Arrow/Tab keys to cycle through the fleet list.
    private void Update()
    {
        if (Keyboard.current == null) return;

        if (Keyboard.current.cKey.wasPressedThisFrame)
        {
            ToggleFollowMode();
        }

        if (isFollowing)
        {
            if (Keyboard.current.rightArrowKey.wasPressedThisFrame || Keyboard.current.tabKey.wasPressedThisFrame)
                ChangeAGV(1);
            if (Keyboard.current.leftArrowKey.wasPressedThisFrame)
                ChangeAGV(-1);
        }
    }

    /// @brief Performs the camera transformation updates after all movement logic is processed.
    ///
    /// @details Calculates a world-space target position by applying @c followOffset 
    /// to the AGV's local rotation. Uses @c Lerp for position smoothing and 
    /// @c LookAt to maintain focus on the target.
    private void LateUpdate()
    {
        if (!isFollowing || agvPool == null) return;

        IReadOnlyList<AGVController> fleet = agvPool.AllAGVs;
        if (fleet == null || fleet.Count == 0) return;

        currentAgvIndex = Mathf.Clamp(currentAgvIndex, 0, fleet.Count - 1);
        Transform target = fleet[currentAgvIndex].transform;

        Vector3 targetPosition = target.position + (target.rotation * followOffset);
        transform.position = Vector3.Lerp(transform.position, targetPosition, smoothSpeed * Time.deltaTime);

        transform.LookAt(target.position + Vector3.up * 1.2f);
    }

    /// @brief Switches the camera between manual navigation and automated AGV tracking.
    ///
    /// @details Updates the @c enabled state of @c CameraController and @c OrbitCamera 
    /// to prevent multiple scripts from competing for the camera's Transform. 
    /// Logs the transition to the @c SimLogger.
    private void ToggleFollowMode()
    {
        isFollowing = !isFollowing;

        if (manualController != null) manualController.enabled = !isFollowing;

        if (orbitCamera != null)
        {
            orbitCamera.enabled = !isFollowing;
        }

        SimLogger.Medium(isFollowing ? $"[Camera] Following AGV_{currentAgvIndex}" : "[Camera] Manual Control Restored");
    }

    /// @brief Increments or decrements the current AGV target index.
    ///
    /// @param direction The integer value to add to the index (usually 1 or -1).
    ///
    /// @details Performs a modulo operation on the @c currentAgvIndex to ensure 
    /// the selection wraps around when reaching the beginning or end of the fleet list.
    private void ChangeAGV(int direction)
    {
        IReadOnlyList<AGVController> fleet = agvPool.AllAGVs;
        if (fleet == null || fleet.Count <= 1) return;

        currentAgvIndex = (currentAgvIndex + direction + fleet.Count) % fleet.Count;
        SimLogger.Medium($"[Camera] Switched to following: {fleet[currentAgvIndex].gameObject.name}");
    }
}