#if UNITY_EDITOR
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using Newtonsoft.Json.Linq;
using UnityEngine;
using Debug = UnityEngine.Debug;

namespace Assets.Scripts.Editor
{
    /// @brief Writes BUILD_MANIFEST.json next to a built player (thesis section 7.2).
    ///
    /// Unity players are not bit-for-bit reproducible, so a committed build cannot be checked by rebuilding it.
    /// Instead the build records its provenance: the git commit it was built from, whether the project source had
    /// uncommitted changes (and a hash of those changes), the Unity version, and the SHA-256 of every file the
    /// player loads. env/player_manifest.py checks a player against its manifest before training or evaluation.
    ///
    /// Files the player writes at runtime (ML-Agents timers, Results/) and data copied beside it (BatchConfigs/,
    /// covered by the per-episode config hash instead) are left out, so running the player does not break it.
    public static class BuildManifest
    {
        public const string FileName = "BUILD_MANIFEST.json";

        /// @brief Top-level entries of the build folder that are not part of the player.
        private static readonly string[] ExcludedTop = { FileName, "Results", "BatchConfigs", "__pycache__" };

        /// @brief Build-folder paths the player itself writes to, or that are never loaded.
        private static readonly string[] ExcludedPrefixes = { "capstone_Data/ML-Agents/" };

        /// @brief Project folders whose state defines the build (relative to the repo root).
        private static readonly string[] SourcePaths = { "Capstone/Assets", "Capstone/Packages", "Capstone/ProjectSettings" };

        public static void Write(string executablePath)
        {
            string buildDir = Path.GetDirectoryName(Path.GetFullPath(executablePath));
            string repoRoot = Path.GetFullPath(Path.Combine(Application.dataPath, "..", ".."));

            string status = Git(repoRoot, "status --porcelain -- " + string.Join(" ", SourcePaths));
            var dirtyFiles = status.Split('\n').Select(l => l.TrimEnd()).Where(l => l.Length > 3)
                                   .Select(l => l.Substring(3)).ToList();
            // Hash of the uncommitted changes, so two dirty builds from the same commit can still be told apart.
            string diff = Git(repoRoot, "diff HEAD -- " + string.Join(" ", SourcePaths));

            var files = new JObject();
            foreach (string path in Directory.EnumerateFiles(buildDir, "*", SearchOption.AllDirectories)
                                             .Select(p => Path.GetRelativePath(buildDir, p).Replace('\\', '/'))
                                             .Where(Included)
                                             .OrderBy(p => p, StringComparer.Ordinal))
                files[path] = Sha256File(Path.Combine(buildDir, path));

            var manifest = new JObject
            {
                ["schema"] = 1,
                ["built_at"] = DateTime.UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ"),
                ["unity_version"] = Application.unityVersion,
                ["executable"] = Path.GetFileName(executablePath),
                ["source_commit"] = Git(repoRoot, "rev-parse HEAD").Trim(),
                ["source_dirty"] = dirtyFiles.Count > 0,
                ["source_dirty_files"] = new JArray(dirtyFiles),
                ["source_diff_sha256"] = dirtyFiles.Count > 0 ? Sha256Text(diff) : "",
                ["files"] = files,
            };
            File.WriteAllText(Path.Combine(buildDir, FileName), manifest.ToString() + "\n");
            Debug.Log($"[BuildManifest] {files.Count} files, commit {manifest["source_commit"]}" +
                      (dirtyFiles.Count > 0 ? $" + {dirtyFiles.Count} uncommitted change(s)" : "") +
                      $" -> {Path.Combine(buildDir, FileName)}");
        }

        private static bool Included(string relPath)
        {
            string top = relPath.Split('/')[0];
            if (ExcludedTop.Contains(top) || top.EndsWith("_DoNotShip")) return false;
            return !ExcludedPrefixes.Any(relPath.StartsWith);
        }

        private static string Sha256File(string path)
        {
            using var sha = SHA256.Create();
            using var stream = File.OpenRead(path);
            return Hex(sha.ComputeHash(stream));
        }

        private static string Sha256Text(string text)
        {
            using var sha = SHA256.Create();
            return Hex(sha.ComputeHash(System.Text.Encoding.UTF8.GetBytes(text)));
        }

        private static string Hex(byte[] bytes) => string.Concat(bytes.Select(b => b.ToString("x2")));

        /// @brief Runs git in the repo; returns stdout, or "unknown" (logged) if git is unavailable.
        private static string Git(string repoRoot, string args)
        {
            try
            {
                var psi = new ProcessStartInfo("git", args)
                {
                    WorkingDirectory = repoRoot,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true,
                };
                using var proc = Process.Start(psi);
                string output = proc.StandardOutput.ReadToEnd();
                proc.WaitForExit();
                if (proc.ExitCode != 0) throw new Exception(proc.StandardError.ReadToEnd());
                return output;
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"[BuildManifest] git {args} failed: {ex.Message}");
                return "unknown";
            }
        }
    }
}
#endif
