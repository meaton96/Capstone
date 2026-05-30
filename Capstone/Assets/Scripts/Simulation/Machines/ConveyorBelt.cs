using System.Collections.Generic;
using UnityEngine;
using Assets.Scripts.Simulation.Jobs;

namespace Assets.Scripts.Simulation.Machines
{
    /// @brief A linear conveyor belt that smoothly moves job visuals between an input end and an output end.
    /// @details Jobs pack toward the output, advancing automatically when space opens up ahead.
    /// Orientation is determined by the transform's forward vector, while the flow direction is
    /// toggled via the @ref reverseFlow flag.
    ///
    /// @remarks This component manages a fixed-capacity list of belt entries, each representing
    /// a job visual positioned along the belt spine. Items are packed toward the output end
    /// (slot 0) and automatically shift forward when space opens up.
    public class ConveyorBelt : MonoBehaviour
    {
        [SerializeField] private int capacity = 3;
        [SerializeField] private float slotSpacing = 0.5f;
        [SerializeField] private float beltSpeed = 3f;
        [SerializeField] private float heightOffset = 0.5f;

        [Tooltip("FALSE (outgoing): items enter at origin and flow out.\nTRUE (incoming): items enter at far end and flow back.")]
        [SerializeField] private bool reverseFlow = false;

        /// @brief Represents a single job entry on the conveyor belt.
        ///
        /// @details Stores the job's unique identifier, its associated visual component,
        /// and the current/target world positions for smooth interpolation.
        private class BeltEntry
        {
            public int JobId;
            public JobVisual Visual;
            public Vector3 CurrentWorldPos;
            public Vector3 TargetWorldPos;
        }

        private readonly List<BeltEntry> entries = new List<BeltEntry>();

        public int Count => entries.Count;
        public bool IsFull => entries.Count >= capacity;
        public bool IsEmpty => entries.Count == 0;
        public int Capacity { get { return capacity; } set { capacity = value; } }

        /// @brief Calculates the total physical length of the belt in world units.
        public float BeltLength => (capacity - 1) * slotSpacing;

        /// @brief The world position at the transform's origin (local zero point), offset vertically.
        private Vector3 OriginEnd => transform.position + Vector3.up * heightOffset;

        /// @brief The world position at the far end of the belt, offset vertically.
        private Vector3 FarEnd => transform.position + transform.forward * BeltLength + Vector3.up * heightOffset;

        public Vector3 InputEndPosition => reverseFlow ? FarEnd : OriginEnd;

        public Vector3 OutputEndPosition => reverseFlow ? OriginEnd : FarEnd;

        public string DumpBeltJobs()
        {
            string s = "[Conveyor] ";
            for (int i = 0; i < entries.Count; i++)
            {
                BeltEntry job = entries[i];
                s += $"id: {job.JobId}, visual: {job.Visual.name}\n";
            }
            return s;
        }

        /// @brief Calculates the world-space coordinate for a specific belt slot.
        ///
        /// @details Slot 0 is always the output end, and slot (capacity-1) is the input end.
        /// The physical mapping of these indices to @ref OriginEnd or @ref FarEnd is
        /// determined by the @ref reverseFlow state.
        ///
        /// @param slotIndex The index of the slot to query (0 to capacity-1).
        /// @return The world-space Vector3 position of the specified slot.
        private Vector3 GetSlotWorldPosition(int slotIndex)
        {
            if (reverseFlow)
            {
                float dist = slotIndex * slotSpacing;
                return transform.position + transform.forward * dist + Vector3.up * heightOffset;
            }
            else
            {
                float dist = (capacity - 1 - slotIndex) * slotSpacing;
                return transform.position + transform.forward * dist + Vector3.up * heightOffset;
            }
        }

        /// @brief Maps a list entry index to its target world slot position.
        ///
        /// @param entryIndex The index of the entry in the packed list.
        /// @return The world position the entry should move toward.
        private Vector3 GetTargetForEntry(int entryIndex)
        {
            return GetSlotWorldPosition(entryIndex);
        }

        /// @brief Attempts to place a job at the input end of the belt.
        ///
        /// @details If the belt has available capacity and the job ID is not already present,
        /// a new belt entry is created. The provided @p visual is snapped to the input
        /// position and flagged as being handled by a conveyor.
        ///
        /// @param jobId The unique ID of the job to enqueue.
        /// @param visual The visual component associated with the job.
        /// @return True if the job was successfully enqueued; otherwise, false (belt is full or duplicate).
        /// @post Job count increases by one; @ref entries list is updated with the new entry.
        public bool TryEnqueue(int jobId, JobVisual visual)
        {
            if (IsFull) return false;
            if (Contains(jobId)) return false;

            int newIndex = entries.Count;
            Vector3 target = GetTargetForEntry(newIndex);

            var entry = new BeltEntry
            {
                JobId = jobId,
                Visual = visual,
                CurrentWorldPos = InputEndPosition,
                TargetWorldPos = target,
            };
            entries.Add(entry);

            if (visual != null)
            {
                visual.SetOnConveyor(true);
                visual.SnapToPosition(InputEndPosition);
            }

            return true;
        }

        /// @brief Retrieves the ID of the job at the output end without removing it.
        /// @return The job ID at the front of the belt, or -1 if the belt is empty.
        public int PeekFront() => entries.Count > 0 ? entries[0].JobId : -1;

        /// @brief Retrieves the visual of the job at the output end without removing it.
        /// @return The JobVisual component at the front of the belt, or null if the belt is empty.
        public JobVisual PeekFrontVisual() => entries.Count > 0 ? entries[0].Visual : null;

        /// @brief Removes and returns the job at the output end of the belt.
        ///
        /// @details Dequeues the front entry, releases the associated visual from conveyor
        /// control via @c SetOnConveyor(false), and triggers a target recalculation so
        /// remaining jobs shift forward toward the output end.
        ///
        /// @pre Belt must not be empty.
        /// @post The @ref entries list count decreases by one; remaining items update their @ref TargetWorldPos.
        /// @return A tuple containing the removed job ID and its associated JobVisual.
        public (int jobId, JobVisual visual) DequeueFront()
        {
            if (entries.Count == 0) return (-1, null);

            BeltEntry front = entries[0];
            if (front.Visual != null)
                front.Visual.SetOnConveyor(false);

            entries.RemoveAt(0);
            RecalculateTargets();

            return (front.JobId, front.Visual);
        }

        /// @brief Removes a specific job ID from any position on the belt.
        ///
        /// @details Locates the entry matching @p jobId, removes it from the list,
        /// and repacks the remaining items toward the output end by recalculating targets.
        ///
        /// @param jobId The ID of the job to remove.
        /// @return The JobVisual associated with the removed job, or null if the job ID was not found.
        public JobVisual RemoveJob(int jobId)
        {
            for (int i = 0; i < entries.Count; i++)
            {
                if (entries[i].JobId != jobId) continue;

                JobVisual visual = entries[i].Visual;
                if (visual != null) visual.SetOnConveyor(false);

                entries.RemoveAt(i);
                RecalculateTargets();
                return visual;
            }
            return null;
        }

        /// @brief Checks if a specific job ID is currently managed by this belt.
        ///
        /// @param jobId The job ID to search for.
        /// @return True if the specified ID exists in the current @ref entries list; otherwise, false.
        public bool Contains(int jobId)
        {
            for (int i = 0; i < entries.Count; i++)
                if (entries[i].JobId == jobId) return true;
            return false;
        }

        /// @brief Generates an ordered list of all job IDs currently on the belt.
        /// @return A list of job IDs ordered from output end (index 0) to input end.
        public List<int> GetJobIds()
        {
            var ids = new List<int>(entries.Count);
            foreach (var e in entries) ids.Add(e.JobId);
            return ids;
        }

        /// @brief Forcefully clears all jobs from the belt.
        ///
        /// @post All visuals are released from conveyor control via @c SetOnConveyor(false)
        /// and the @ref entries list is emptied.
        public void Clear()
        {
            foreach (var e in entries)
                if (e.Visual != null) e.Visual.SetOnConveyor(false);
            entries.Clear();
        }

        /// @brief Updates target positions for all entries based on their current list index.
        ///
        /// @details Called after any entry is added or removed to ensure all remaining
        /// items have correct slot positions for smooth interpolation.
        private void RecalculateTargets()
        {
            for (int i = 0; i < entries.Count; i++)
                entries[i].TargetWorldPos = GetTargetForEntry(i);
        }

        /// @brief Unity Update loop driving the smooth sliding of items along the belt.
        ///
        /// @details Iterates through all entries and moves their world positions toward
        /// their calculated slot targets at a constant speed (@ref beltSpeed). Entries
        /// that reach their target are snapped to the exact position.
        private void Update()
        {
            if (entries.Count == 0) return;

            float step = beltSpeed * Time.deltaTime;

            for (int i = 0; i < entries.Count; i++)
            {
                BeltEntry e = entries[i];

                if ((e.CurrentWorldPos - e.TargetWorldPos).sqrMagnitude < 0.0001f)
                {
                    e.CurrentWorldPos = e.TargetWorldPos;
                    if (e.Visual != null)
                        e.Visual.transform.position = e.TargetWorldPos;
                    continue;
                }

                e.CurrentWorldPos = Vector3.MoveTowards(e.CurrentWorldPos, e.TargetWorldPos, step);

                if (e.Visual != null)
                    e.Visual.transform.position = e.CurrentWorldPos;
            }
        }

        /// @brief Renders debug gizmos for the belt spine, slots, and flow direction in the Unity Editor.
        ///
        /// @details Draws the belt spine as a line, slot positions as wire cubes (color-coded:
        /// red for output, green for input, blue for intermediate), a yellow flow direction
        /// arrow, and wire spheres at the input/output endpoints.
        private void OnDrawGizmos()
        {
            if (capacity <= 0) return;

            Vector3 origin = transform.position + Vector3.up * heightOffset;
            Vector3 far = transform.position + transform.forward * BeltLength + Vector3.up * heightOffset;

            Gizmos.color = new Color(0.3f, 0.8f, 1f, 0.6f);
            Gizmos.DrawLine(origin, far);

            for (int i = 0; i < capacity; i++)
            {
                Vector3 pos = GetSlotWorldPosition(i);
                bool isOutput = (i == 0);
                bool isInput = (i == capacity - 1);

                if (isOutput) Gizmos.color = new Color(1f, 0.3f, 0.3f, 0.8f);
                else if (isInput) Gizmos.color = new Color(0.3f, 1f, 0.3f, 0.8f);
                else Gizmos.color = new Color(0.3f, 0.8f, 1f, 0.4f);

                Gizmos.DrawWireCube(pos, Vector3.one * 0.25f);
            }

            Vector3 mid = (origin + far) * 0.5f;
            Vector3 flowDir = reverseFlow ? -transform.forward : transform.forward;
            Gizmos.color = Color.yellow;
            Gizmos.DrawRay(mid, flowDir * slotSpacing * 0.6f);

            Gizmos.color = new Color(0.3f, 1f, 0.3f, 0.9f);
            Gizmos.DrawWireSphere(InputEndPosition, 0.15f);

            Gizmos.color = new Color(1f, 0.3f, 0.3f, 0.9f);
            Gizmos.DrawWireSphere(OutputEndPosition, 0.15f);
        }
    }
}