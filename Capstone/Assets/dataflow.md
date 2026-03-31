Architecture & Information Flow: Unity Job-Shop Simulation
1. Data Ingestion & Initialization (The Setup)

The simulation starts by parsing the mathematical rules and turning them into physical 3D objects.

    The Trigger: Unity's Start() calls SimulationBridge.StartEpisode().

    Parsing: The Bridge reads the Taillard JSON TextAsset. It converts the duration_matrix and machines_matrix into a TaillardInstance data object.

    Building the Floor:

        The Bridge passes the MachineCount to the FactoryLayoutManager.

        The Layout Manager calculates a grid, instantiates PhysicalMachine prefabs, and computes the 64-D distanceMatrix for the agent's observation space.

        Physics: As the machines spawn, their NavMesh Obstacle components carve holes into the factory NavMesh.

    Spawning the Jobs:

        The Bridge passes the TaillardInstance to the JobManager.

        For every job, JobManager creates a logical JobTracker (storing the exact machine sequence and durations) and instantiates a physical JobVisual (the orb) in the starting area.

        The AGVPool spawns the fleet of AGVController NavMesh agents.

    The First Dispatch: The Bridge loops through all jobs, looks at the first machine they need to visit, and asks the AGVPool to dispatch AGVs to pick them up.

2. The Physical Transport Loop (The Movement)

Jobs do not teleport. Information flows through Unity's spatial NavMesh and collision system.

    Pickup: An AGVController receives a dispatch order. It drives to the JobVisual, parents the orb to itself, and tells JobManager.BeginTransit().

        Information: The JobTracker state changes to InTransit. The visual color changes to Blue.

    Navigating: The AGV uses Unity's pathfinding to avoid machines and other AGVs, driving toward the target PhysicalMachine.

    The Drop-off (The Physics Handshake): * The AGV arrives and unparents the JobVisual orb.

        The orb falls into the BoxCollider (Trigger) of the PhysicalMachine.

        Unity's physics engine fires OnTriggerEnter on the machine.

    Validation: The PhysicalMachine asks the JobManager: "I just collided with Job 4. Does Job 4's Taillard routing matrix say it belongs here right now?" * If yes: It is added to the PhysicalQueue. The overhead UI updates to Q: 1.

3. The Decision Loop (The ML-Agent)

Information flows from the physical traffic jams up to the neural network.

    The Request: If a PhysicalMachine is currently Idle and its PhysicalQueue is >0, it tells the SimulationBridge it needs instructions.

    The Traffic Cop: Because 5 machines might ask for instructions at the exact same millisecond, the SimulationBridge puts the machine into a pendingDecisions queue.

    Waking the Agent: The Bridge's Update() loop pops a machine off the queue, builds a DecisionRequest (snapshotting the machine ID and queue contents), and fires OnDecisionRequired.

        The SchedulingAgent wakes up and calls RequestDecision().

    Collecting Observations: The ML-Agent asks the SimulationBridge and JobManager for the state of the world:

        Local State: How long are the jobs in the current machine's queue?

        Global State: Where are all 20 jobs physically located? What percentage of them are done? (GetJobScalarsFlat).

        Spatial State: What is the distance matrix between all machines?

    The Action: The neural network outputs an integer (0-7) representing a heuristic rule (e.g., SPT, LPT).

    Execution: The Bridge maps the integer to a heuristic, evaluates the mathematical Taillard durations of the jobs in the PhysicalQueue, and picks the winner. It tells the PhysicalMachine: "Start processing Job 4."

4. Processing & Reward Calculation (The Clock)

Information flows from real-time execution back into the reinforcement learning system.

    The Timer: The PhysicalMachine removes Job 4 from the queue, turns its MachineVisual Yellow, and starts a Unity Coroutine.

        The Coroutine yields for the exact amount of scaled Time.time dictated by the Taillard duration matrix. (e.g., 64 seconds in sim-time).

        The progress bar fills frame-by-frame.

    The Reward: Immediately after making the decision, the SimulationBridge calculates how much Unity Time.time elapsed since the last decision.

        Because physical AGV traffic jams or poor scheduling choices waste real time, the elapsed Time.time is converted into a negative penalty and fed to the Agent (AddReward).

    Operation Complete: The Coroutine finishes. The PhysicalMachine tells the Bridge.

        The Bridge tells the JobManager to mark the operation complete.

        The Bridge looks at Job 4's JobTracker to find its next destination in the machines_matrix.

        The Bridge asks the AGVPool for an Uber. We return to Phase 2.