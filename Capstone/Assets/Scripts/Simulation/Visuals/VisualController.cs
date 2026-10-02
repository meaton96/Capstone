using System;
using UnityEngine;
using Assets.Scripts.Simulation.AGV;
using Assets.Scripts.Simulation.Jobs;
using Assets.Scripts.Simulation.Machines;

namespace Assets.Scripts.Simulation.Visuals
{
    /// @brief Single source of truth for which prefabs the simulation instantiates.
    ///
    /// @details Holds the original primitive prefabs and the Kenney Prefab Variants side by side, plus
    /// one toggle that picks between them. Spawners (@c FactoryLayoutManager, @c JobStore, @c AGVPool)
    /// read their prefabs from here instead of holding their own references. A Kenney slot left empty
    /// falls back to the primitive prefab, so the variants can be added one at a time.
    ///
    /// Visual-only: the variants keep the same colliders and components as their base prefabs.
    /// The toggle can be forced from the command line with @c -kenneyvisuals or @c -primitivevisuals
    /// (used for headless A/B runs).
    public class VisualController : MonoBehaviour
    {
        private static VisualController instance;

        /// @brief The scene's controller, found on first access if @c Awake has not run yet.
        public static VisualController Instance
        {
            get
            {
                if (instance == null)
                    instance = FindFirstObjectByType<VisualController>();
                return instance;
            }
        }

        [Serializable]
        public class PrefabSet
        {
            public PhysicalMachine machine;
            public PhysicalMachine doubleSidedMachine;
            public GameObject conveyor;
            public GameObject wall;
            public AGVController agv;
            public JobVisual job;
        }

        [Tooltip("Use the Kenney Prefab Variants. Slots left empty fall back to the primitive prefab.")]
        [SerializeField] private bool useKenneyVisuals;

        [Header("Primitive (original prefabs)")]
        [SerializeField] private PrefabSet primitive = new PrefabSet();

        [Header("Kenney (Prefab Variants)")]
        [SerializeField] private PrefabSet kenney = new PrefabSet();

        [Header("Primitive-only")]
        [Tooltip("Applied to every renderer of the incoming belt. Ignored when the Kenney conveyor is in use, " +
                 "since it would overwrite the Kenney material.")]
        [SerializeField] private Material incomingBeltMaterial;

        /// @brief True when the Kenney variants are selected (inspector toggle, or a command-line override).
        public bool UseKenneyVisuals => useKenneyVisuals;

        public PhysicalMachine MachinePrefab => Pick(kenney.machine, primitive.machine);
        public PhysicalMachine DoubleSidedMachinePrefab => Pick(kenney.doubleSidedMachine, primitive.doubleSidedMachine);
        public GameObject ConveyorPrefab => Pick(kenney.conveyor, primitive.conveyor);
        public GameObject WallPrefab => Pick(kenney.wall, primitive.wall);
        public AGVController AgvPrefab => Pick(kenney.agv, primitive.agv);
        public JobVisual JobPrefab => Pick(kenney.job, primitive.job);

        /// @brief Belt material override for the incoming conveyor, or null when none should be applied.
        public Material IncomingBeltMaterial => UseKenneyConveyor ? null : incomingBeltMaterial;

        private bool UseKenneyConveyor => useKenneyVisuals && kenney.conveyor != null;

        private void Awake()
        {
            instance = this;

            string[] args = Environment.GetCommandLineArgs();
            if (Array.IndexOf(args, "-kenneyvisuals") >= 0) useKenneyVisuals = true;
            else if (Array.IndexOf(args, "-primitivevisuals") >= 0) useKenneyVisuals = false;
        }

        private T Pick<T>(T kenneyPrefab, T primitivePrefab) where T : UnityEngine.Object
        {
            return useKenneyVisuals && kenneyPrefab != null ? kenneyPrefab : primitivePrefab;
        }
    }
}
