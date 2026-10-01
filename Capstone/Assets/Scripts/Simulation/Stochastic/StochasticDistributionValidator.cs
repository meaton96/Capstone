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

            // Test 5: AGV full life, Weibull(k=1.5, λ=8400) operating seconds (after a repair)
            // mean = λ Γ(1 + 1/k) ≈ 7583.1, std = λ sqrt(Γ(1 + 2/k) − Γ²(1 + 1/k)) ≈ 5148.7
            {
                var cfg = MakeConfig(agvFailures: true);
                StochasticEventManager.Instance.Initialize(cfg);
                System.Random rng = StochasticEventManager.Instance.CreateAGVStream(0);
                SampleMoments(N, () => StochasticEventManager.Instance.SampleAGVTTF(rng), out double mean, out double std);
                CheckStat("AGV Weibull(1.5,8400) mean", mean, 7583.1);
                CheckStat("AGV Weibull(1.5,8400) std", std, 5148.7);
            }

            yield return null;

            // Test 6: AGV residual life at episode start (equilibrium / forward-recurrence distribution)
            // mean = E[T²] / (2 E[T]) = λ Γ(1 + 2/k) / (2 Γ(1 + 1/k)) ≈ 5539.4
            // E[R²] = E[T³] / (3 E[T]) = λ² Γ(1 + 3/k) / (3 Γ(1 + 1/k)), std ≈ 4628.4
            {
                var cfg = MakeConfig(agvFailures: true);
                StochasticEventManager.Instance.Initialize(cfg);
                System.Random rng = StochasticEventManager.Instance.CreateAGVStream(0);
                SampleMoments(N, () => StochasticEventManager.Instance.SampleAGVResidualTTF(rng), out double mean, out double std);
                CheckStat("AGV residual life mean", mean, 5539.4);
                CheckStat("AGV residual life std", std, 4628.4);
            }

            yield return null;

            // Test 7: AGV repair, LogNormal(μ=4.6, σ=0.5): mean = exp(μ + σ²/2) ≈ 112.73,
            // std = sqrt((exp(σ²) − 1) exp(2μ + σ²)) ≈ 60.08
            {
                var cfg = MakeConfig(agvFailures: true);
                StochasticEventManager.Instance.Initialize(cfg);
                System.Random rng = StochasticEventManager.Instance.CreateAGVStream(0);
                SampleMoments(N, () => StochasticEventManager.Instance.SampleAGVRepair(rng), out double mean, out double std);
                CheckStat("AGV LogNormal(4.6,0.5) mean", mean, 112.73);
                CheckStat("AGV LogNormal(4.6,0.5) std", std, 60.08);
            }

            yield return null;

            // Test 8: stream independence. AGV draws must not shift the machine failure stream (so turning AGV
            // failures on leaves a seed's machine failures unchanged), each AGV's stream must be reproducible
            // from (seed, id), and two AGVs must not share a stream.
            {
                var cfg = MakeConfig(machineFailures: true, agvFailures: true);
                var sem = StochasticEventManager.Instance;

                sem.Initialize(cfg);
                float[] machinesAlone = new float[100];
                for (int i = 0; i < 100; i++) machinesAlone[i] = sem.SampleMachineTTF();

                sem.Initialize(cfg);
                System.Random agvA = sem.CreateAGVStream(3);
                float[] machinesInterleaved = new float[100];
                float[] agv3 = new float[100];
                for (int i = 0; i < 100; i++)
                {
                    agv3[i] = sem.SampleAGVTTF(agvA);
                    machinesInterleaved[i] = sem.SampleMachineTTF();
                }

                sem.Initialize(cfg);
                System.Random agvB = sem.CreateAGVStream(3);
                System.Random agvOther = sem.CreateAGVStream(4);
                bool reproducible = true, distinct = false;
                for (int i = 0; i < 100; i++)
                {
                    if (Math.Abs(sem.SampleAGVTTF(agvB) - agv3[i]) > 1e-3f) reproducible = false;
                    if (Math.Abs(sem.SampleAGVTTF(agvOther) - agv3[i]) > 1e-3f) distinct = true;
                }

                bool machineUnchanged = true;
                for (int i = 0; i < 100; i++)
                    if (Math.Abs(machinesAlone[i] - machinesInterleaved[i]) > 1e-4f) { machineUnchanged = false; break; }

                Report("AGV draws leave the machine failure stream unchanged", machineUnchanged);
                Report("Per-AGV stream reproducible from (seed, id)", reproducible);
                Report("Different AGVs get different streams", distinct);
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

        /// <summary>Mean and standard deviation of @p n draws from @p sample.</summary>
        private static void SampleMoments(int n, Func<float> sample, out double mean, out double std)
        {
            double sum = 0, sumSq = 0;
            for (int i = 0; i < n; i++)
            {
                double x = sample();
                sum += x; sumSq += x * x;
            }
            mean = sum / n;
            std = Math.Sqrt(Math.Max(0.0, sumSq / n - mean * mean));
        }

        /// <summary>Logs a PASS/FAIL line for a yes/no check.</summary>
        private void Report(string label, bool ok)
        {
            if (ok) SimLogger.Low($"[DistValidator] PASS  {label}.");
            else
            {
                SimLogger.LogError($"[DistValidator] FAIL  {label}.");
                _passed = false;
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
                    AGVWeibullK = 1.5f,
                    AGVWeibullLambda = 8400f,
                    AGVRepairLogMu = 4.6f,
                    AGVRepairLogSigma = 0.5f,
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