using System;
using System.Collections.Generic;
using Assets.Scripts.Simulation.Types;
using Assets.Scripts.Simulation.Machines;

namespace Assets.Scripts.Simulation
{
    /// <summary>
    /// Provides factory methods for creating default FJSSP (Flexible Job Shop Scheduling Problem) configurations.
    /// </summary>
    public static class DefaultConfigFactory
    {
        /// <summary>
        /// Builds a default deterministic FJSSP configuration with standard machine types and processing times.
        /// </summary>
        /// <returns>A configured FJSSPConfig instance with default deterministic settings.</returns>
        public static FJSSPConfig BuildDefault()
        {
            MachineType[] types = (MachineType[])Enum.GetValues(typeof(MachineType));
            var layout = new MachineType[types.Length];
            for (int i = 0; i < types.Length; i++) layout[i] = types[i];

            return new FJSSPConfig
            {
                Seed = 42,
                JobCount = 15,
                MachinesPerType = 1,
                MachineTypeLayout = layout,
                MinProcTime = 1f,
                MaxProcTime = 30f,
                MinOpsPerJob = 2,
                MaxOpsPerJob = 4,
                MaxArrivalTime = 0f,
                AGVCount = 3,
                ProcTimeParams = new Dictionary<MachineType, (float mu, float sigma)>
                {
                    { MachineType.Mill,     (9f,  1f)  },
                    { MachineType.Lathe,    (7f,  1f)  },
                    { MachineType.Weld,     (15f, 2f)  },
                    { MachineType.Inspect,  (6f,  1f)  },
                    { MachineType.Assemble, (24f, 4f)  },
                },
                dispatchingRule = DispatchingRule.SRT_SRWT,
                MachineFlexibilityProbability = 0f,
            };
        }

        /// <summary>
        /// Builds a default stochastic FJSSP configuration with machine failure parameters enabled.
        /// </summary>
        /// <returns>A configured FJSSPConfig instance with default stochastic settings including failure distributions.</returns>
        public static FJSSPConfig BuildDefaultStochastic()
        {
            MachineType[] types = (MachineType[])Enum.GetValues(typeof(MachineType));
            var layout = new MachineType[types.Length];
            for (int i = 0; i < types.Length; i++) layout[i] = types[i];

            return new FJSSPConfig
            {
                Seed = 42,
                JobCount = 5,
                MachinesPerType = 1,
                MachineTypeLayout = layout,
                MinProcTime = 1f,
                MaxProcTime = 30f,
                MinOpsPerJob = 2,
                MaxOpsPerJob = 4,
                MaxArrivalTime = 0f,
                AGVCount = 3,
                ProcTimeParams = new Dictionary<MachineType, (float mu, float sigma)>
                {
                    { MachineType.Mill,     (9f,  1f)  },
                    { MachineType.Lathe,    (7f,  1f)  },
                    { MachineType.Weld,     (15f, 2f)  },
                    { MachineType.Inspect,  (6f,  1f)  },
                    { MachineType.Assemble, (24f, 4f)  },
                },
                dispatchingRule = DispatchingRule.SRT_SRWT,
                MachineFlexibilityProbability = 0f,
                Stochastic = new StochasticConfig
                {
                    MachineFailuresEnabled = true,
                    WeibullK = 1.5f,
                    WeibullLambda = 2000f,
                    RepairLogMu = 2.0f,
                    RepairLogSigma = 0.5f,
                    DynamicArrivalsEnabled = false,
                }
            };
        }
    }
}