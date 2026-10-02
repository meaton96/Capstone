#if UNITY_EDITOR
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text.RegularExpressions;
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
    /// The player's files are an allowlist (Included): the executable, the shared libraries beside it, and
    /// capstone_Data/ except ML-Agents/, where the player writes timers at runtime. The libraries count because the
    /// executable's RUNPATH is $ORIGIN, so the loader looks in the build folder before the system folders. Nothing
    /// else in the folder is part of the player (runner scripts, slurm/, logs/, Results/, BatchConfigs/, Burst's
    /// *_DoNotShip debug output), so it can change without invalidating the build.
    public static class BuildManifest
    {
        public const string FileName = "BUILD_MANIFEST.json";

        /// @brief Manifest format. Schema 1 (before 2026-10-02) hashed every file in the folder except a few excluded
        ///        ones; env/player_manifest.py compares only the allowlisted entries of those manifests.
        private const int Schema = 2;

        // Keep in step with env/player_manifest.py: the player's own files (see Included).
        private const string Executable = "capstone.x86_64";
        private const string DataDir = "capstone_Data/";
        /// @brief Written by the player while it runs (ML-Agents timers).
        private const string RuntimeDir = "capstone_Data/ML-Agents/";
        /// @brief Shared libraries at the top level: UnityPlayer.so, libdecor-0.so.0, libdecor-cairo.so.
        private static readonly Regex SharedLibrary = new Regex(@"^[^/]+\.so(\.[0-9]+)*$");

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
            foreach (string path in PlayerFiles(buildDir))
                files[path] = Sha256File(Path.Combine(buildDir, path));

            var manifest = new JObject
            {
                ["schema"] = Schema,
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

        /// @brief The player's files in @p buildDir ('/'-separated, ordinal order). Only the top level and
        ///        capstone_Data/ are walked, so Results/ and the other helper folders are never scanned.
        private static List<string> PlayerFiles(string buildDir)
        {
            string dataDir = Path.Combine(buildDir, DataDir.TrimEnd('/'));
            IEnumerable<string> paths = Directory.EnumerateFiles(buildDir);
            if (Directory.Exists(dataDir))
                paths = paths.Concat(Directory.EnumerateFiles(dataDir, "*", SearchOption.AllDirectories));
            return paths.Select(p => Path.GetRelativePath(buildDir, p).Replace('\\', '/'))
                        .Where(Included)
                        .OrderBy(p => p, StringComparer.Ordinal)
                        .ToList();
        }

        /// @brief Whether @p relPath (relative to the build folder, '/'-separated) is one of the player's files.
        private static bool Included(string relPath)
        {
            if (relPath.StartsWith(DataDir, StringComparison.Ordinal))
                return !relPath.StartsWith(RuntimeDir, StringComparison.Ordinal);
            return relPath == Executable || SharedLibrary.IsMatch(relPath);
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
