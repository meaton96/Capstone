#if UNITY_EDITOR
using System.Collections;
using System.IO;
using System.Linq;
using Assets.Scripts.Simulation;
using Assets.Scripts.Simulation.FactoryLayout;
using Assets.Scripts.Simulation.Types;
using UnityEditor;
using UnityEngine;

namespace Assets.Scripts.Editor
{
    /// @brief Top-down PNG captures of the factory floor for the thesis figures.
    ///
    /// @details The Linux players are dedicated-server builds and cannot render, so the figures
    ///          are taken in the editor. Enter Play mode and start any simulation from the start
    ///          menu (that loads a config), then use:
    ///          - Tools/Thesis Figures/Capture All Layouts (A-O): rebuilds the floor once per
    ///            built layout letter from the current config (single tile, lane parking) and
    ///            saves layout_&lt;id&gt;.png;
    ///          - Tools/Thesis Figures/Capture Current Floor: saves whatever floor is spawned now
    ///            (use it for tiled / linked floors set up through the menu).
    ///          Output: &lt;repo&gt;/docs/Thesis/figures/layouts/. The camera looks straight down with
    ///          world +Z at the top of the image and is fitted to the bounds of every renderer
    ///          under the FactoryLayoutManager.
    public static class LayoutScreenshotter
    {
        private const int LongSidePixels = 2400;
        private const float Margin = 1.04f;

        private static string OutputDir =>
            Path.GetFullPath(Path.Combine(Application.dataPath, "..", "..", "docs", "Thesis", "figures", "layouts"));

        [MenuItem("Tools/Thesis Figures/Capture All Layouts (A-O)")]
        private static void CaptureAllLayouts()
        {
            if (!CheckReady()) return;
            Runner().StartCoroutine(CaptureAllRoutine());
        }

        [MenuItem("Tools/Thesis Figures/Capture Current Floor")]
        private static void CaptureCurrentFloor()
        {
            if (!CheckReady()) return;
            var cfg = FactoryOrchestrator.Instance.CurrentConfig;
            string name = $"floor_{cfg.Layout.Id}";
            if (cfg.Tiling.Tiles > 1)
                name += $"_tiles{cfg.Tiling.Tiles}" + (cfg.Tiling.JobsOpen ? "_linked" : "");
            Runner().StartCoroutine(CaptureRoutine(name));
        }

        [MenuItem("Tools/Thesis Figures/Capture All Layouts (A-O)", true)]
        [MenuItem("Tools/Thesis Figures/Capture Current Floor", true)]
        private static bool ValidatePlaying() => Application.isPlaying;

        private static bool CheckReady()
        {
            if (FactoryOrchestrator.Instance == null || FactoryOrchestrator.Instance.CurrentConfig == null)
            {
                Debug.LogError("[LayoutScreenshotter] Start a simulation from the start menu first, so a config is loaded.");
                return false;
            }
            return true;
        }

        private static IEnumerator CaptureAllRoutine()
        {
            var orch = FactoryOrchestrator.Instance;
            FJSSPConfig original = orch.CurrentConfig;
            foreach (string id in LayoutSpec.BuiltIds)
            {
                FJSSPConfig cfg = original.CloneWithSeed(original.Seed);
                cfg.Layout = LayoutSpec.FromPreset(id);
                cfg.Tiling = TilingSpec.Single;
                cfg.parkingMethod = "lane";
                orch.LoadConfig(cfg);
                orch.SpawnFactory();
                yield return CaptureRoutine($"layout_{id}");
            }
            // Leave the scene on the config the user started with.
            orch.LoadConfig(original);
            orch.SpawnFactory();
            Debug.Log($"[LayoutScreenshotter] Captured {LayoutSpec.BuiltIds.Length} layouts to {OutputDir}");
        }

        private static IEnumerator CaptureRoutine(string fileStem)
        {
            // Let destroyed objects go and the new floor's renderers settle.
            yield return null;
            yield return new WaitForEndOfFrame();

            var layout = Object.FindFirstObjectByType<FactoryLayoutManager>();
            if (layout == null) { Debug.LogError("[LayoutScreenshotter] No FactoryLayoutManager in scene."); yield break; }

            var renderers = layout.GetComponentsInChildren<Renderer>(false).Where(r => r.enabled).ToArray();
            if (renderers.Length == 0) { Debug.LogError("[LayoutScreenshotter] Floor has no renderers."); yield break; }
            Bounds b = renderers[0].bounds;
            foreach (var r in renderers) b.Encapsulate(r.bounds);

            float width = b.size.x * Margin, depth = b.size.z * Margin;
            int px = width >= depth ? LongSidePixels : Mathf.RoundToInt(LongSidePixels * width / depth);
            int pz = width >= depth ? Mathf.RoundToInt(LongSidePixels * depth / width) : LongSidePixels;

            var go = new GameObject("ThesisFigureCamera");
            var cam = go.AddComponent<Camera>();
            cam.orthographic = true;
            cam.orthographicSize = depth / 2f;
            cam.transform.position = new Vector3(b.center.x, b.max.y + 50f, b.center.z);
            cam.transform.rotation = Quaternion.Euler(90f, 0f, 0f);
            cam.nearClipPlane = 0.1f;
            cam.farClipPlane = b.size.y + 200f;
            cam.clearFlags = CameraClearFlags.SolidColor;
            cam.backgroundColor = Color.white;
            cam.aspect = (float)px / pz;

            var rt = new RenderTexture(px, pz, 24) { antiAliasing = 8 };
            cam.targetTexture = rt;
            cam.Render();

            var prev = RenderTexture.active;
            RenderTexture.active = rt;
            var tex = new Texture2D(px, pz, TextureFormat.RGB24, false);
            tex.ReadPixels(new Rect(0, 0, px, pz), 0, 0);
            tex.Apply();
            RenderTexture.active = prev;

            Directory.CreateDirectory(OutputDir);
            string path = Path.Combine(OutputDir, fileStem + ".png");
            File.WriteAllBytes(path, tex.EncodeToPNG());
            Debug.Log($"[LayoutScreenshotter] {path} ({px}x{pz}, floor {b.size.x:F1} x {b.size.z:F1})");

            cam.targetTexture = null;
            Object.DestroyImmediate(rt);
            Object.DestroyImmediate(tex);
            Object.DestroyImmediate(go);
        }

        /// Coroutines need a MonoBehaviour; a hidden host lives for the rest of Play mode.
        private static CoroutineHost Runner()
        {
            var host = Object.FindFirstObjectByType<CoroutineHost>();
            if (host != null) return host;
            var go = new GameObject("LayoutScreenshotterHost") { hideFlags = HideFlags.HideAndDontSave };
            return go.AddComponent<CoroutineHost>();
        }

        private sealed class CoroutineHost : MonoBehaviour { }
    }
}
#endif
