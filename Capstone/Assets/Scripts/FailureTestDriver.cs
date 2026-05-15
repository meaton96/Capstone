using UnityEngine;
using UnityEngine.InputSystem; // Added for New Input System
using Assets.Scripts.Simulation.Machines;
using Assets.Scripts.Simulation.Stochastic;
using Assets.Scripts.Simulation.Types;
using System.Linq;

public class FailureTestDriver : MonoBehaviour
{
#if UNITY_EDITOR
    [Header("Force a failure on this machine")]
    public int TargetMachineId = 0;

    [Header("Read-only monitoring")]
    [SerializeField] private string healthState;
    [SerializeField] private float ttfCountdown;
    [SerializeField] private float repairRemaining;
    [SerializeField] private float sampledRepair;

    private PhysicalMachine _target;

    private void Update()
    {
        if (_target == null)
            _target = FindFirstMachineById(TargetMachineId);

        if (_target == null) return;

        // Mirror internal state to inspector
        healthState = _target.HealthState.ToString();
        repairRemaining = _target.RemainingRepairTime;
        sampledRepair = _target.SampledRepairDuration;

        // Reflection for private TTF
        var field = typeof(PhysicalMachine)
            .GetField("_ttfCountdown", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
        if (field != null)
            ttfCountdown = (float)field.GetValue(_target);

        // --- NEW INPUT SYSTEM ---
        // Checks if the 'P' key was pressed this frame
        if (Keyboard.current != null && Keyboard.current.pKey.wasPressedThisFrame)
        {
            ForceFailure();
        }
    }

    private void ForceFailure()
    {
        if (_target == null) return;
        _target.DEBUG_ForceFailure();
        Debug.Log($"[TestDriver] Forced failure on machine {TargetMachineId}");
    }

    private PhysicalMachine FindFirstMachineById(int id)
    {
        // Modern replacement for FindObjectsOfType<T>()
        // FindObjectsSortMode.None is faster than the old method because it doesn't sort by InstanceID
        var machines = FindObjectsByType<PhysicalMachine>(FindObjectsSortMode.None);

        return machines.FirstOrDefault(m => m.MachineId == id);
    }
#endif
}