using System;
using System.Collections.Generic;
using Assets.Scripts.Simulation.Types;
using Assets.Scripts.Simulation.Machines;

namespace Assets.Scripts.Simulation
{
    public static class DefaultConfigFactory
    {
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