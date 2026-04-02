using System.Collections.Generic;
using UnityEngine;

namespace Assets.Scripts.Simulation.Machines
{
    /// <summary>
    /// A linear conveyor belt that smoothly moves job visuals between an input
    /// end and an output end. Jobs pack toward the output, advancing automatically
    /// when space opens up ahead.
    ///
    /// <para><b>Scene Setup:</b></para>
    /// Place the GameObject near the machine body and point its local Z+
    /// (forward / blue arrow) <b>away</b> from the machine.
    ///
    /// <para><b>For an incoming belt</b> set <see cref="reverseFlow"/> = true.
    /// Items enter at the far end (where AGVs arrive) and slide back toward
    /// the machine (the origin).</para>
    ///
    /// <para><b>For an outgoing belt</b> leave <see cref="reverseFlow"/> = false.
    /// Items enter at the origin (where the machine drops them) and slide out
    /// toward the far end (where AGVs pick up).</para>
    ///
    /// Both conveyors can share the same orientation — origin at machine,
    /// forward pointing away — just flip the checkbox.
    /// </summary>
    public class ConveyorBelt : MonoBehaviour
    {
        // ─────────────────────────────────────────────────────────
        //  Inspector
        // ─────────────────────────────────────────────────────────

        [Header("Belt Configuration")]
        [Tooltip("Maximum jobs the belt can hold at once.")]
        [SerializeField] private int capacity = 3;

        [Tooltip("World-space distance between adjacent slot centers.")]
        [SerializeField] private float slotSpacing = 0.5f;

        [Tooltip("Speed at which visuals slide along the belt (units/sec).")]
        [SerializeField] private float beltSpeed = 3f;

        [Tooltip("Height offset above the belt surface for job tokens.")]
        [SerializeField] private float heightOffset = 0.5f;

        [Header("Flow Direction")]
        [Tooltip("When FALSE (outgoing): items enter at the origin and flow " +
                 "outward along forward toward the far end.\n\n" +
                 "When TRUE (incoming): items enter at the far end and flow " +
                 "back toward the origin (the machine).")]
        [SerializeField] private bool reverseFlow = false;

        // ─────────────────────────────────────────────────────────
        //  Internal Data
        // ─────────────────────────────────────────────────────────

        private class BeltEntry
        {
            public int JobId;
            public JobVisual Visual;
            public Vector3 CurrentWorldPos;
            public Vector3 TargetWorldPos;
        }

        // Ordered front-to-back: entries[0] is nearest the OUTPUT end.
        private readonly List<BeltEntry> entries = new List<BeltEntry>();

        // ─────────────────────────────────────────────────────────
        //  Properties
        // ─────────────────────────────────────────────────────────

        public int Count => entries.Count;
        public bool IsFull => entries.Count >= capacity;
        public bool IsEmpty => entries.Count == 0;
        public int Capacity { get { return capacity; } set { capacity = value; } }

        /// <summary>Total world-space length of the belt.</summary>
        public float BeltLength => (capacity - 1) * slotSpacing;

        /// <summary>World position at the origin end (transform.position + height).</summary>
        private Vector3 OriginEnd =>
            transform.position + Vector3.up * heightOffset;

        /// <summary>World position at the far end (along forward + height).</summary>
        private Vector3 FarEnd =>
            transform.position
            + transform.forward * BeltLength
            + Vector3.up * heightOffset;

        /// <summary>
        /// World position where new items enter the belt.
        /// reverseFlow OFF → origin.  reverseFlow ON → far end.
        /// </summary>
        public Vector3 InputEndPosition => reverseFlow ? FarEnd : OriginEnd;

        /// <summary>
        /// World position where items leave the belt.
        /// reverseFlow OFF → far end.  reverseFlow ON → origin.
        /// </summary>
        public Vector3 OutputEndPosition => reverseFlow ? OriginEnd : FarEnd;

        // ─────────────────────────────────────────────────────────
        //  Slot Geometry
        // ─────────────────────────────────────────────────────────

        /// <summary>
        /// Returns the world position of the given slot index.
        /// Slot 0 is always at the OUTPUT end; slot (capacity-1) at the INPUT end.
        /// The <see cref="reverseFlow"/> flag controls which physical end those map to.
        /// </summary>
        private Vector3 GetSlotWorldPosition(int slotIndex)
        {
            if (reverseFlow)
            {
                // OUTPUT = origin (near machine), INPUT = far end.
                // Slot 0 (output) at origin; higher slots toward far end.
                float dist = slotIndex * slotSpacing;
                return transform.position
                       + transform.forward * dist
                       + Vector3.up * heightOffset;
            }
            else
            {
                // OUTPUT = far end, INPUT = origin.
                // Slot 0 (output) at far end; higher slots toward origin.
                float dist = (capacity - 1 - slotIndex) * slotSpacing;
                return transform.position
                       + transform.forward * dist
                       + Vector3.up * heightOffset;
            }
        }

        /// <summary>Entry at list index i targets slot i (packed toward output).</summary>
        private Vector3 GetTargetForEntry(int entryIndex)
        {
            return GetSlotWorldPosition(entryIndex);
        }

        // ─────────────────────────────────────────────────────────
        //  Public API — Enqueue / Dequeue
        // ─────────────────────────────────────────────────────────

        /// <summary>
        /// Places a job at the input end of the belt.
        /// The visual is snapped to the input position and will smoothly
        /// slide toward the output end as space allows.
        /// </summary>
        /// <returns>True if the job was accepted; false if full or duplicate.</returns>
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

        /// <summary>Peek at the front (output-end) job without removing.</summary>
        public int PeekFront()
        {
            return entries.Count > 0 ? entries[0].JobId : -1;
        }

        /// <summary>Peek at the front visual without removing.</summary>
        public JobVisual PeekFrontVisual()
        {
            return entries.Count > 0 ? entries[0].Visual : null;
        }

        /// <summary>
        /// Removes and returns the front (output-end) job.
        /// All remaining jobs shift one slot toward the output.
        /// </summary>
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

        /// <summary>
        /// Removes a specific job by ID from anywhere on the belt.
        /// Remaining jobs repack toward the output end.
        /// </summary>
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

        /// <summary>True if the given job is currently on this belt.</summary>
        public bool Contains(int jobId)
        {
            for (int i = 0; i < entries.Count; i++)
                if (entries[i].JobId == jobId) return true;
            return false;
        }

        /// <summary>All job IDs on the belt, ordered output to input.</summary>
        public List<int> GetJobIds()
        {
            var ids = new List<int>(entries.Count);
            foreach (var e in entries) ids.Add(e.JobId);
            return ids;
        }

        /// <summary>Removes all jobs and releases visuals from conveyor control.</summary>
        public void Clear()
        {
            foreach (var e in entries)
                if (e.Visual != null) e.Visual.SetOnConveyor(false);
            entries.Clear();
        }

        // ─────────────────────────────────────────────────────────
        //  Movement
        // ─────────────────────────────────────────────────────────

        private void RecalculateTargets()
        {
            for (int i = 0; i < entries.Count; i++)
                entries[i].TargetWorldPos = GetTargetForEntry(i);
        }

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

                e.CurrentWorldPos = Vector3.MoveTowards(
                    e.CurrentWorldPos, e.TargetWorldPos, step);

                if (e.Visual != null)
                    e.Visual.transform.position = e.CurrentWorldPos;
            }
        }

        // ─────────────────────────────────────────────────────────
        //  Editor Gizmos
        // ─────────────────────────────────────────────────────────

        private void OnDrawGizmos()
        {
            if (capacity <= 0) return;

            Vector3 origin = transform.position + Vector3.up * heightOffset;
            Vector3 far = transform.position
                + transform.forward * BeltLength
                + Vector3.up * heightOffset;

            // Belt spine
            Gizmos.color = new Color(0.3f, 0.8f, 1f, 0.6f);
            Gizmos.DrawLine(origin, far);

            // Slots — color-coded by role
            for (int i = 0; i < capacity; i++)
            {
                Vector3 pos = GetSlotWorldPosition(i);
                bool isOutput = (i == 0);
                bool isInput = (i == capacity - 1);

                if (isOutput)
                    Gizmos.color = new Color(1f, 0.3f, 0.3f, 0.8f);   // red = output
                else if (isInput)
                    Gizmos.color = new Color(0.3f, 1f, 0.3f, 0.8f);   // green = input
                else
                    Gizmos.color = new Color(0.3f, 0.8f, 1f, 0.4f);   // cyan = middle

                Gizmos.DrawWireCube(pos, Vector3.one * 0.25f);
            }

            // Flow arrow in item movement direction
            Vector3 mid = (origin + far) * 0.5f;
            Vector3 flowDir = reverseFlow ? -transform.forward : transform.forward;
            Gizmos.color = Color.yellow;
            Gizmos.DrawRay(mid, flowDir * slotSpacing * 0.6f);

            // Spheres marking input/output ends
            Gizmos.color = new Color(0.3f, 1f, 0.3f, 0.9f);
            Gizmos.DrawWireSphere(InputEndPosition, 0.15f);  // green = input

            Gizmos.color = new Color(1f, 0.3f, 0.3f, 0.9f);
            Gizmos.DrawWireSphere(OutputEndPosition, 0.15f); // red = output
        }
    }
}