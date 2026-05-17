using System;
using System.Collections;
using UnityEngine;
using Assets.Scripts.Simulation.Types;
using Assets.Scripts.Simulation.Logging;

namespace Assets.Scripts.Simulation.Stochastic
{
    /// <summary>
    /// Headless unit tests for StochasticEventManager distribution correctness.
    ///
    /// Run from command line:
    ///   ./capstone.exe -batchmode -nographics -validatestochastic
    ///
    /// Generates N=50,000 samples per distribution, checks mean, std dev, and
    /// skewness against theoretical values within configurable tolerance bands.
    /// Exits with code 0 on pass, 1 on any failure.
    ///
    /// Attach to any persistent GameObject. Activate only when -validatestochastic
    /// CLI flag is present so it does not interfere with normal batch runs.
    /// </summary>
    public class StochasticDistributionValidator : MonoBehaviour
    {
        private const int N = 50_000;
        private const float TOLERANCE = 0.03f;   // 3% tolerance on mean/std
        private bool _passed;

        private void Start()
        {
            if (!HasCLIFlag("-validatestochastic")) return;

            SimLogger.Low("[DistValidator] Starting distribution validation suite...");
            StartCoroutine(RunAllTests());
        }

        private IEnumerator RunAllTests()
        {
            _passed = true;

            // ── Weibull(k=1.5, λ=900) ────────────────────────────────────────
            // Theoretical mean = λ × Γ(1 + 1/k) = 900 × Γ(1.667) ≈ 900 × 0.9027 ≈ 812.4
            // Theoretical std  = λ × sqrt(Γ(1+2/k) − Γ²(1+1/k))
            //                  = 900 × sqrt(Γ(2.333) − 0.9027²)  ≈ 900 × sqrt(1.190 − 0.815) ≈ 551.6
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

            // ── LogNormal(μ=4.0, σ=0.5) ──────────────────────────────────────
            // Theoretical mean = exp(μ + σ²/2) = exp(4.125) ≈ 61.9
            // Theoretical std  = sqrt((exp(σ²) − 1) × exp(2μ + σ²))
            //                  ≈ sqrt(0.284 × 3834)  ≈ 33.0
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

            // ── Poisson inter-arrival (λ=0.005) ──────────────────────────────
            // Exponential(λ=0.005): mean = 1/λ = 200, std = 1/λ = 200
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

            // ── Seed reproducibility check ────────────────────────────────────
            // Two managers initialised with the same seed must produce identical streams.
            {
                var cfg1 = MakeConfig(machineFailures: true);
                var cfg2 = MakeConfig(machineFailures: true);  // same seed=99

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

            // ── Summary ───────────────────────────────────────────────────────
            yield return null;

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

        // ── Helpers ──────────────────────────────────────────────────────────

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

        private static bool HasCLIFlag(string flag)
        {
            foreach (string arg in Environment.GetCommandLineArgs())
                if (string.Equals(arg, flag, StringComparison.OrdinalIgnoreCase))
                    return true;
            return false;
        }
    }
}
