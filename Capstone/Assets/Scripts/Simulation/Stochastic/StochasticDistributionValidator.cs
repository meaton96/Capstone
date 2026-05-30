using System;
using System.Collections;
using UnityEngine;
using Assets.Scripts.Simulation.Types;
using Assets.Scripts.Simulation.Logging;

namespace Assets.Scripts.Simulation.Stochastic
{
    /// <summary>
    /// Headless unit tests for validating StochasticEventManager distribution correctness.
    ///
    /// This class performs statistical validation by generating N=50,000 samples per
    /// distribution and comparing empirical statistics (mean, standard deviation) against
    /// their theoretical values within a configurable tolerance band (default 3%).
    ///
    /// <para>Usage — command line:</para>
    ///   ./capstone.exe -batchmode -nographics -validatestochastic
    ///
    /// <para>Exit codes:</para>
    ///   0 — all tests passed
    ///   1 — one or more tests failed (see log for details)
    ///
    /// <para>Integration notes:</para>
    /// Attach to any persistent GameObject. The validator activates only when the
    /// -validatestochastic CLI flag is present, ensuring no interference with normal
    /// batch simulation runs.
    /// </summary>
    public class StochasticDistributionValidator : MonoBehaviour
    {
        /// <summary>Number of Monte-Carlo samples per distribution test.</summary>
        private const int N = 50_000;

        /// <summary>Relative error tolerance (3%) for mean and standard deviation checks.</summary>
        private const float TOLERANCE = 0.03f;

        /// <summary>Overall test result — true when all individual checks have passed.</summary>
        private bool _passed;

        /// <summary>
        /// Entry point — checks for the -validatestochastic CLI flag and launches
        /// the test suite if present.
        /// </summary>
        private void Start()
        {
            if (!HasCLIFlag("-validatestochastic")) return;

            SimLogger.Low("[DistValidator] Starting distribution validation suite...");
            StartCoroutine(RunAllTests());
        }

        /// <summary>
        /// Executes the full test suite sequentially. Tests each stochastic distribution
        /// (Weibull, LogNormal, Exponential) and verifies seed reproducibility.
        /// </summary>
        /// <returns>Coroutine that completes when all tests finish.</returns>
        private IEnumerator RunAllTests()
        {
            _passed = true;

            // Test 1: Weibull(k=1.5, λ=900) — machine time-to-failure
            // Theoretical mean = λ × Γ(1 + 1/k) = 900 × Γ(1.667) ≈ 812.4
            // Theoretical std  = λ × sqrt(Γ(1+2/k) − Γ²(1+1/k)) ≈ 551.6
            yield return null;
            {
                var cfg = MakeConfig(machineFailures: true, weibullK: 1.5f, weibullLambda: 900f);
                StochasticEventManager.Instance.Initialize(cfg);

                double sum = 0, sumSq = 0;
                for (int i = 0; i < N; i++)
                {
                    float s = StochasticEventManager.Instance.SampleMachineTTF();
                    sum += s; sumSq += s * s;
                }
                double mean = sum / N;
                double std = Math.Sqrt(sumSq / N - mean * mean);
                double expectedMean = 812.4;
                double expectedStd = 551.6;
                CheckStat("Weibull(1.5,900) mean", mean, expectedMean);
                CheckStat("Weibull(1.5,900) std", std, expectedStd);
            }

            yield return null;

            // Test 2: LogNormal(μ=4.0, σ=0.5) — machine repair duration
            // Theoretical mean = exp(μ + σ²/2) = exp(4.125) ≈ 61.9
            // Theoretical std  = sqrt((exp(σ²) − 1) × exp(2μ + σ²)) ≈ 33.0
            {
                var cfg = MakeConfig(machineFailures: true);
                StochasticEventManager.Instance.Initialize(cfg);

                double sum = 0, sumSq = 0;
                for (int i = 0; i < N; i++)
                {
                    float s = StochasticEventManager.Instance.SampleMachineRepair();
                    sum += s; sumSq += s * s;
                }
                double mean = sum / N;
                double std = Math.Sqrt(sumSq / N - mean * mean);
                double expectedMean = 61.9;
                double expectedStd = 33.0;
                CheckStat("LogNormal(4.0,0.5) mean", mean, expectedMean);
                CheckStat("LogNormal(4.0,0.5) std", std, expectedStd);
            }

            yield return null;

            // Test 3: Exponential(λ=0.005) — job inter-arrival time
            // Theoretical mean = 1/λ = 200, std = 1/λ = 200
            {
                var cfg = MakeConfig(dynamicArrivals: true, arrivalLambda: 0.005f);
                StochasticEventManager.Instance.Initialize(cfg);

                double sum = 0, sumSq = 0;
                for (int i = 0; i < N; i++)
                {
                    float s = StochasticEventManager.Instance.SampleInterArrivalTime();
                    sum += s; sumSq += s * s;
                }
                double mean = sum / N;
                double std = Math.Sqrt(sumSq / N - mean * mean);
                double expectedMean = 200.0;
                double expectedStd = 200.0;
                CheckStat("Exponential(0.005) mean", mean, expectedMean);
                CheckStat("Exponential(0.005) std", std, expectedStd);
            }

            yield return null;

            // Test 4: Seed reproducibility — two managers with the same seed
            // must produce identical sample streams.
            {
                var cfg1 = MakeConfig(machineFailures: true);
                var cfg2 = MakeConfig(machineFailures: true);

                float[] stream1 = new float[100];
                float[] stream2 = new float[100];

                StochasticEventManager.Instance.Initialize(cfg1);
                for (int i = 0; i < 100; i++)
                    stream1[i] = StochasticEventManager.Instance.SampleMachineTTF();

                StochasticEventManager.Instance.Initialize(cfg2);
                for (int i = 0; i < 100; i++)
                    stream2[i] = StochasticEventManager.Instance.SampleMachineTTF();

                bool identical = true;
                for (int i = 0; i < 100; i++)
                    if (Math.Abs(stream1[i] - stream2[i]) > 1e-4f) { identical = false; break; }

                if (identical)
                    SimLogger.Low("[DistValidator] PASS  Seed reproducibility: identical streams confirmed.");
                else
                {
                    SimLogger.LogError("[DistValidator] FAIL  Seed reproducibility: streams diverged.");
                    _passed = false;
                }
            }

            yield return null;

            // Report final results and exit with appropriate code.
            if (_passed)
            {
                SimLogger.Low("[DistValidator] All tests PASSED.");
                Application.Quit(0);
            }
            else
            {
                SimLogger.LogError("[DistValidator] One or more tests FAILED. See log for details.");
                Application.Quit(1);
            }
        }

        /// <summary>
        /// Evaluates a single statistical metric (mean, std, etc.) by comparing an
        /// empirical value against its theoretical expectation. Reports PASS or FAIL
        /// based on relative error within the configured tolerance.
        /// </summary>
        /// <param name="label">Descriptive name of the statistic (e.g., "Weibull(1.5,900) mean").</param>
        /// <param name="actual">The empirically observed value.</param>
        /// <param name="expected">The theoretical expected value.</param>
        private void CheckStat(string label, double actual, double expected)
        {
            double relErr = Math.Abs(actual - expected) / expected;
            bool pass = relErr <= TOLERANCE;
            string tag = pass ? "PASS" : "FAIL";
            string msg = $"[DistValidator] {tag}  {label}: " +
                         $"actual={actual:F2}  expected={expected:F2}  " +
                         $"relErr={relErr * 100:F1}%  (tol={TOLERANCE * 100:F0}%)";

            if (pass)
                SimLogger.Low(msg);
            else
            {
                SimLogger.LogError(msg);
                _passed = false;
            }
        }

        /// <summary>
        /// Constructs a fully-populated FJSSPConfig for test scenarios with
        /// configurable stochastic parameters. All parameters default to sensible
        /// test values so callers need only override what they need.
        /// </summary>
        /// <param name="machineFailures">Enable or disable machine failure simulation.</param>
        /// <param name="weibullK">Weibull shape parameter k (default 1.5).</param>
        /// <param name="weibullLambda">Weibull scale parameter λ for machines (default 900).</param>
        /// <param name="repairLogMu">LogNormal μ for machine repair durations (default 4.0).</param>
        /// <param name="repairLogSigma">LogNormal σ for machine repair durations (default 0.5).</param>
        /// <param name="agvFailures">Enable or disable AGV failure simulation.</param>
        /// <param name="dynamicArrivals">Enable or disable dynamic job arrivals.</param>
        /// <param name="arrivalLambda">Exponential rate λ for inter-arrival times (default 0.005).</param>
        /// <param name="seed">RNG seed for reproducibility (default 99).</param>
        /// <returns>A configured FJSSPConfig ready for StochasticEventManager.Initialize().</returns>
        private static FJSSPConfig MakeConfig(
            bool machineFailures = false,
            float weibullK = 1.5f,
            float weibullLambda = 900f,
            float repairLogMu = 4.0f,
            float repairLogSigma = 0.5f,
            bool agvFailures = false,
            bool dynamicArrivals = false,
            float arrivalLambda = 0.005f,
            int seed = 99)
        {
            return new FJSSPConfig
            {
                Seed = seed,
                Stochastic = new StochasticConfig
                {
                    MachineFailuresEnabled = machineFailures,
                    WeibullK = weibullK,
                    WeibullLambda = weibullLambda,
                    RepairLogMu = repairLogMu,
                    RepairLogSigma = repairLogSigma,
                    AGVFailuresEnabled = agvFailures,
                    DynamicArrivalsEnabled = dynamicArrivals,
                    ArrivalLambda = arrivalLambda,
                }
            };
        }

        /// <summary>
        /// Checks whether a specific CLI flag is present in the process command-line arguments.
        /// </summary>
        /// <param name="flag">The flag string to search for (case-insensitive).</param>
        /// <returns>True if the flag is found in the arguments.</returns>
        private static bool HasCLIFlag(string flag)
        {
            foreach (string arg in Environment.GetCommandLineArgs())
                if (string.Equals(arg, flag, StringComparison.OrdinalIgnoreCase))
                    return true;
            return false;
        }
    }
}