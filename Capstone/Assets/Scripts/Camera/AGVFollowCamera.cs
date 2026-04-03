using UnityEngine;
using UnityEngine.InputSystem;
using System.Collections.Generic;
using Assets.Scripts.Simulation.AGV;
using Assets.Scripts.Logging;

/// <summary>
/// Attaches the camera to AGVs within the AGVPool.
/// Toggles control away from CameraController/OrbitCamera when active.
/// </summary>
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
    void Awake()
    {
        manualController = GetComponent<CameraController>();
        orbitCamera = GetComponent<OrbitCamera>();
    }
    private void Update()
    {
        if (Keyboard.current == null) return;

        // Toggle AGV Follow Mode with 'C' (for Camera/Capture)
        if (Keyboard.current.cKey.wasPressedThisFrame)
        {
            ToggleFollowMode();
        }

        if (isFollowing)
        {
            // Swap between AGVs using Left/Right arrows or Tab
            if (Keyboard.current.rightArrowKey.wasPressedThisFrame || Keyboard.current.tabKey.wasPressedThisFrame)
                ChangeAGV(1);
            if (Keyboard.current.leftArrowKey.wasPressedThisFrame)
                ChangeAGV(-1);
        }
    }

    private void LateUpdate()
    {
        if (!isFollowing || agvPool == null) return;

        List<AGVController> fleet = agvPool.Fleet;
        if (fleet == null || fleet.Count == 0) return;

        // Ensure index stays valid if fleet size changes dynamically
        currentAgvIndex = Mathf.Clamp(currentAgvIndex, 0, fleet.Count - 1);
        Transform target = fleet[currentAgvIndex].transform;

        // Calculate the target position based on the AGV's current orientation
        // This keeps the camera behind the AGV even when it turns.
        Vector3 targetPosition = target.position + (target.rotation * followOffset);

        // Interpolate position for a professional "cinematic" feel
        transform.position = Vector3.Lerp(transform.position, targetPosition, smoothSpeed * Time.deltaTime);

        // Always look at the AGV (aiming slightly above the pivot)
        transform.LookAt(target.position + Vector3.up * 1.2f);
    }

    private void ToggleFollowMode()
    {
        isFollowing = !isFollowing;

        // Disable manual scripts so they don't fight over the camera Transform
        if (manualController != null) manualController.enabled = !isFollowing;

        // If we stop following, OrbitCamera usually takes back over via CameraController logic
        if (!isFollowing && orbitCamera != null)
        {
            orbitCamera.enabled = true;
        }
        else if (isFollowing && orbitCamera != null)
        {
            orbitCamera.enabled = false;
        }

        SimLogger.Medium(isFollowing ? $"[Camera] Following AGV_{currentAgvIndex}" : "[Camera] Manual Control Restored");
    }

    private void ChangeAGV(int direction)
    {
        List<AGVController> fleet = agvPool.Fleet;
        if (fleet == null || fleet.Count <= 1) return;

        // Loop the index around the fleet list
        currentAgvIndex = (currentAgvIndex + direction + fleet.Count) % fleet.Count;
        SimLogger.Medium($"[Camera] Switched to following: {fleet[currentAgvIndex].gameObject.name}");
    }
}