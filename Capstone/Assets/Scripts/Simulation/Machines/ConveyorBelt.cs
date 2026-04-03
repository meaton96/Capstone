using System.Collections.Generic;
using UnityEngine;

namespace Assets.Scripts.Simulation.Machines
{
    /// @brief A linear conveyor belt that smoothly moves job visuals between an input end and an output end.
    /// @details Jobs pack toward the output, advancing automatically when space opens up ahead. 
    /// Orientation is determined by the transform's forward vector, while the flow direction is 
    /// toggled via the @ref reverseFlow flag.
    public class ConveyorBelt : MonoBehaviour
    {
        [SerializeField] private int capacity = 3;
        [SerializeField] private float slotSpacing = 0.5f;
        [SerializeField] private float beltSpeed = 3f;
        [SerializeField] private float heightOffset = 0.5f;

        [Tooltip("FALSE (outgoing): items enter at origin and flow out.\nTRUE (incoming): items enter at far end and flow back.")]
        [SerializeField] private bool reverseFlow = false;

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

        public float BeltLength => (capacity - 1) * slotSpacing;

        private Vector3 OriginEnd => transform.position + Vector3.up * heightOffset;

        private Vector3 FarEnd => transform.position + transform.forward * BeltLength + Vector3.up * heightOffset;

        public Vector3 InputEndPosition => reverseFlow ? FarEnd : OriginEnd;

        public Vector3 OutputEndPosition => reverseFlow ? OriginEnd : FarEnd;

        /// @brief Calculates the world-space coordinate for a specific belt slot.
        ///
        /// @details Slot 0 is always the output end, and slot (capacity-1) is the input end. 
        /// The physical mapping of these indices to @ref OriginEnd or @ref FarEnd is 
        /// determined by the @ref reverseFlow state.
        ///
        /// @param slotIndex The index of the slot to query.
        /// @return The world-space Vector3 position of the slot.
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

        /// @brief Maps a list entry index to a target world slot.
        /// @param entryIndex The index of the entry in the packed list.
        /// @return The world position the entry should move toward.
        private Vector3 GetTargetForEntry(int entryIndex)
        {
            return GetSlotWorldPosition(entryIndex);
        }

        /// @brief Attempts to place a job at the input end of the belt.
        ///
        /// @details If the belt has capacity and the job is not a duplicate, a new entry 
        /// is created. The @p visual is snapped to the input position and flagged 
        /// as being handled by a conveyor.
        ///
        /// @param jobId The unique ID of the job.
        /// @param visual The visual component associated with the job.
        ///
        /// @return True if the job was successfully enqueued; otherwise, false.
        /// @post Job count increases by one; @ref entries is updated.
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
        /// @return The job ID, or -1 if the belt is empty.
        public int PeekFront() => entries.Count > 0 ? entries[0].JobId : -1;

        /// @brief Retrieves the visual of the job at the output end without removing it.
        /// @return The JobVisual component, or null if the belt is empty.
        public JobVisual PeekFrontVisual() => entries.Count > 0 ? entries[0].Visual : null;

        /// @brief Removes and returns the job at the output end of the belt.
        ///
        /// @details Dequeues the front entry, releases the visual from conveyor control, 
        /// and triggers a target recalculation so remaining jobs shift forward.
        ///
        /// @pre Belt must not be empty.
        /// @post The @ref entries list count decreases; remaining items update their @ref TargetWorldPos.
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
        /// @details Locates the entry matching @p jobId, removes it, and repacks the 
        /// remaining items toward the output end.
        ///
        /// @param jobId The ID of the job to remove.
        /// @return The visual associated with the removed job, or null if not found.
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

        /// @brief Checks if a specific job is currently managed by this belt.
        /// @param jobId The ID to search for.
        /// @return True if the ID exists in the current @ref entries list.
        public bool Contains(int jobId)
        {
            for (int i = 0; i < entries.Count; i++)
                if (entries[i].JobId == jobId) return true;
            return false;
        }

        /// @brief Generates a list of all job IDs currently on the belt.
        /// @return An ordered list of IDs from output to input end.
        public List<int> GetJobIds()
        {
            var ids = new List<int>(entries.Count);
            foreach (var e in entries) ids.Add(e.JobId);
            return ids;
        }

        /// @brief Forcefully clears all jobs from the belt.
        /// @post Visuals are released from conveyor control and the @ref entries list is emptied.
        public void Clear()
        {
            foreach (var e in entries)
                if (e.Visual != null) e.Visual.SetOnConveyor(false);
            entries.Clear();
        }

        /// @brief Updates target positions for all entries based on their current list index.
        private void RecalculateTargets()
        {
            for (int i = 0; i < entries.Count; i++)
                entries[i].TargetWorldPos = GetTargetForEntry(i);
        }

        /// @brief Unity update loop driving the smooth sliding of items along the belt.
        /// @details Iterates through entries and moves their world positions toward 
        /// their calculated slot targets at a constant speed.
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

        /// @brief Renders the belt spine, slots, and flow direction in the Unity Editor.
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