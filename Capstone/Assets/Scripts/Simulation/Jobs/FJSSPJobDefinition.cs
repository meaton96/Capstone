using System.Collections.Generic;
using Assets.Scripts.Simulation.Machines;

namespace Assets.Scripts.Simulation.Jobs
{
    public class FJSSPJobDefinition
    {
        public int JobId;
        public float ArrivalTime;           // 0 = available at episode start
        public MachineType[] OperationSequence;  // ordered list of op types needed
        public Dictionary<int, float>[] EligibleMachinesPerOp; // [opIndex] → {machineId: procTime}
    }
}