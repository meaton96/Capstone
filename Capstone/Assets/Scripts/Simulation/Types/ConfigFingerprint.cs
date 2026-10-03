using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;

namespace Assets.Scripts.Simulation.Types
{
    /// <summary>
    /// Stable fingerprint of what an episode actually ran (thesis section 7.1): every result row carries the hash of
    /// the applied config (after CLI overrides), so it can be traced back to the exact scenario. The canonical text is
    /// written once per distinct hash (applied_configs.jsonl) so a hash can be turned back into the config.
    /// </summary>
    /// <remarks>
    /// Canonical form: every public field and readable property, recursively, one "path=value" line each, sorted by
    /// path, numbers in invariant round-trip format. Reflection rather than a hand-written list, so a field added to
    /// FJSSPConfig or StochasticConfig later is covered without touching this file.
    /// </remarks>
    /// <summary>
    /// Leaves a field out of <see cref="ConfigFingerprint.Canonical"/> while it is null, so adding an optional field
    /// does not change the hash of every config or instance that does not use it (e.g. FJSSPJobDefinition.DueDate).
    /// </summary>
    [AttributeUsage(AttributeTargets.Field)]
    public sealed class FingerprintOmitIfNullAttribute : Attribute { }

    public static class ConfigFingerprint
    {
        /// <summary>Hex characters kept from the SHA-256 digest (64 bits: collisions are not a concern here).</summary>
        private const int HashLength = 16;
        private const int MaxDepth = 6;

        public static string Hash(string canonical)
        {
            using var sha = SHA256.Create();
            byte[] digest = sha.ComputeHash(Encoding.UTF8.GetBytes(canonical));
            var sb = new StringBuilder(HashLength);
            for (int i = 0; i < HashLength / 2; i++) sb.Append(digest[i].ToString("x2"));
            return sb.ToString();
        }

        /// <summary>Canonical text of any object graph (a config, or the initial job definitions).</summary>
        public static string Canonical(object value)
        {
            var lines = new List<string>();
            Append(lines, "", value, 0);
            lines.Sort(StringComparer.Ordinal);
            return string.Join("\n", lines);
        }

        private static void Append(List<string> lines, string path, object value, int depth)
        {
            if (value == null) { lines.Add($"{path}=null"); return; }
            if (TryScalar(value, out string scalar)) { lines.Add($"{path}={scalar}"); return; }
            if (depth >= MaxDepth) { lines.Add($"{path}={value}"); return; }

            if (value is IDictionary dict)
            {
                var entries = new List<(string key, object val)>();
                foreach (DictionaryEntry e in dict)
                    entries.Add((TryScalar(e.Key, out string k) ? k : e.Key.ToString(), e.Value));
                lines.Add($"{path}.count={entries.Count}");
                foreach (var (key, val) in entries)
                    Append(lines, $"{path}[{key}]", val, depth + 1);
                return;
            }
            if (value is IEnumerable seq)
            {
                int i = 0;
                foreach (object item in seq)
                    Append(lines, $"{path}[{i++:D5}]", item, depth + 1);
                lines.Add($"{path}.count={i}");
                return;
            }

            Type type = value.GetType();
            foreach (FieldInfo f in type.GetFields(BindingFlags.Public | BindingFlags.Instance))
            {
                object fv = f.GetValue(value);
                if (fv == null && f.IsDefined(typeof(FingerprintOmitIfNullAttribute), false)) continue;
                Append(lines, Join(path, f.Name), fv, depth + 1);
            }
            foreach (PropertyInfo p in type.GetProperties(BindingFlags.Public | BindingFlags.Instance))
            {
                if (!p.CanRead || p.GetIndexParameters().Length > 0) continue;
                object v;
                try { v = p.GetValue(value); }
                catch (TargetInvocationException) { continue; }
                Append(lines, Join(path, p.Name), v, depth + 1);
            }
        }

        private static string Join(string path, string name) => path.Length == 0 ? name : $"{path}.{name}";

        private static bool TryScalar(object value, out string text)
        {
            switch (value)
            {
                case string s: text = s.Replace("\n", "\\n"); return true;
                case bool b: text = b ? "true" : "false"; return true;
                case float f: text = f.ToString("R", CultureInfo.InvariantCulture); return true;
                case double d: text = d.ToString("R", CultureInfo.InvariantCulture); return true;
                case Enum e: text = e.ToString(); return true;
                case IFormattable n when value.GetType().IsPrimitive || value is decimal:
                    text = n.ToString(null, CultureInfo.InvariantCulture); return true;
                default: text = null; return false;
            }
        }
    }
}
