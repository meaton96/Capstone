High-Level Overview

These two files form the Brain and the Nervous System of your simulation.

    SimulationBridge.cs acts as the central orchestrator (the nervous system). It manages the episode lifecycle, translates physical events (like a machine finishing) into discrete decision requests, applies chosen actions, and calculates the simulation reward.

    SchedulingAgent.cs acts as the Reinforcement Learning (RL) interface (the brain). It wraps the Unity ML-Agents framework, converts the bridge's decision requests into fixed-width numerical observations, and returns an action (a dispatching rule) back to the bridge.

Component Breakdown
1. SimulationBridge (The Orchestrator)

This is a robust singleton that manages the boundary between continuous physics and discrete ML-Agents decision-making.

    State Management: Holds the CurrentConfig (FJSSPConfig) and manages the simulation lifecycle (SpawnFactory(), StartSimulation(), StopEpisode()).

    Decision Queues: Maintains internal queues (pendingDecisions for dispatching, pendingRoutingJobs for routing) to ensure the agent is only queried for one decision at a time during the Update() loop.

    Action Execution (Step): Takes an action index from the agent, maps it to a Composite Priority Dispatching Rule (PDR), and executes it:

        Dispatching: Picks the best job from a machine's physical queue (e.g., Shortest Processing Time) and commands the PhysicalMachine to start processing.

        Routing: Picks the best candidate machine for a newly finished job (e.g., Shortest Queue) and dispatches an AGV.

    Reward Calculation: Computes a normalized reward based on the negative change in makespan between decision steps.

2. SchedulingAgent (The RL Interface)

This script acts as the translator between ML-Agents (Academy) and the SimulationBridge.

    Observation Collector (CollectObservations): Listens to the bridge's CurrentDecision. It flattens dynamic simulation state (queue lengths, candidate machines, operation times) into a fixed-size VectorSensor array, handling the padding logic so the neural network always receives the expected input size.

    Action Receiver (OnActionReceived): Receives the neural network's chosen discrete action (or the heuristic's choice), passes it to bridge.Step(), and registers the resulting reward using AddReward().

    Lifecycle Gating (IsArmed): Prevents ML-Agents from endlessly looping episodes in the background until the UI or batch runner explicitly arms it.

The Core Interaction Loop (The Decision Cycle)

For your combined diagram, this is the critical data flow between these two scripts:

    Trigger: A physical event occurs (e.g., OnMachineFinished or OnJobArrivedInQueue).

    Queue: SimulationBridge realizes a decision is needed and adds the entity to its pending queues.

    Request: During Update(), SimulationBridge dequeues an item, builds a DecisionRequest, and fires the OnDecisionRequired event.

    Observe: SchedulingAgent hears the event, calls RequestDecision(), and translates the DecisionRequest into a tensor array (CollectObservations).

    Act: ML-Agents evaluates the observation and returns an integer action to SchedulingAgent.OnActionReceived().

    Execute & Reward: The agent passes the action to SimulationBridge.Step(), which updates the physical simulation, calculates the reward, and returns it to the agent.

External Dependencies (Hooks for Final Diagram)

When connecting this module to your other folders, look out for these connections:

    Inbound Data/Events (Listens to):

        PhysicalMachine (fires arrival and finish events).

        Config/UI (triggers file loading and factory spawning).

    Outbound Commands (Controls):

        FactoryLayoutManager (told to build/clear the floor; queried for machine states).

        JobManager (queried for processing times and job states; told to mark ops as started/completed).

        AGVPool (told to dispatch AGVs during routing).

        TrafficZoneManager (told to build the zone graph upon factory spawn).