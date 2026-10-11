using System;
using System.Globalization;
using System.Text;
using Assets.Scripts.Simulation.Logging;

namespace Assets.Scripts.Simulation
{
    /// <summary>
    /// Python rule mode (2026-10-11, dev-pyrules; env/rules_api). With the launch flag <c>-pyrules</c>, every decision
    /// the agent is asked for also sends Python the raw candidate view (FactoryOrchestrator.BuildPyRuleView, on
    /// Channels.DecisionViewChannel, queued in SchedulingAgent.CollectObservations so it travels with that step), and
    /// Python may answer with <c>{"pick": {"job": id, "machine": id, "decision": n, "jobRule": name,
    /// "machineRule": name}}</c> on the config channel (Channels.EpisodeConfigChannel), which is applied instead of the
    /// rule the action stands for (FactoryOrchestrator.StepPick). A pick is per decision, not sticky; a decision with no
    /// pick runs the action's rule as before, so the RL action branches and existing checkpoints are unchanged.
    ///
    /// Without the flag nothing here runs: the view channel is not registered, no view is built or sent, and a pick
    /// message stops the player. An invalid pick (wrong decision index, a job not in the pool / queue, a machine that is
    /// not a candidate of the picked job) stops the player too (Fail): never a silent fallback to another choice.
    /// </summary>
    public static class PyRules
    {
        /// <summary>True when the player was launched with -pyrules (read once from the command line).</summary>
        public static readonly bool Enabled = Array.IndexOf(Environment.GetCommandLineArgs(), "-pyrules") >= 0;

        /// <summary>Version of the view message; env/rules_api/unity_adapter.py checks it.</summary>
        public const int ViewVersion = 1;

        /// <summary>A choice made by a Python rule for one decision.</summary>
        public class Pick
        {
            public int JobId = -1;
            public int MachineId = -1;      // routing: the machine; dispatch: -1 or the deciding machine
            public int Decision = -1;       // DecisionRequest.DecisionIndex of the decision it answers
            public string JobRule = "";     // names for decision_log.csv ("py:<name>")
            public string MachineRule = "";
        }

        /// <summary>Stops the player on a Python-rule protocol error (as the rule override's unknown name does).</summary>
        public static void Fail(string reason)
        {
            SimLogger.Error($"[PyRules] {reason}; quitting.");
            UnityEngine.Application.Quit(3);
            throw new InvalidOperationException($"[PyRules] {reason}");
        }

        // ── JSON numbers: round-trip formats (float G9, double G17), non-finite values as JSON5 tokens Python reads ──

        internal static void Num(StringBuilder sb, float v)
        {
            if (float.IsNaN(v)) sb.Append("NaN");
            else if (float.IsPositiveInfinity(v)) sb.Append("Infinity");
            else if (float.IsNegativeInfinity(v)) sb.Append("-Infinity");
            else sb.Append(v.ToString("G9", CultureInfo.InvariantCulture));
        }

        internal static void Num(StringBuilder sb, double v)
        {
            if (double.IsNaN(v)) sb.Append("NaN");
            else if (double.IsPositiveInfinity(v)) sb.Append("Infinity");
            else if (double.IsNegativeInfinity(v)) sb.Append("-Infinity");
            else sb.Append(v.ToString("G17", CultureInfo.InvariantCulture));
        }

        internal static void Str(StringBuilder sb, string s)
        {
            sb.Append('"');
            foreach (char c in s ?? "")
            {
                if (c == '"' || c == '\\') sb.Append('\\').Append(c);
                else if (c < ' ') sb.Append("\\u").Append(((int)c).ToString("x4"));
                else sb.Append(c);
            }
            sb.Append('"');
        }
    }
}
