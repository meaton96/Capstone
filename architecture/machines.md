High-Level Overview

If the jobs are the "Work" and the layout is the "Stage," the Machines are the "Actors" of your simulation. This module handles the physical queueing of items, the simulation of time passing while work is done, and all the visual feedback required for a human observer to understand what the AI is doing.

This component perfectly separates the mechanical logic (queues and timers) from the cosmetic presentation (colors and UI).
Component Breakdown
1. The Core Logic (PhysicalMachine.cs)

This is the central brain of a single workstation.

    State & Flow Management: Tracks whether the machine is Idle, Busy, or Blocked. It orchestrates the movement of jobs from incoming buffers, into the machine itself (via a timed Coroutine), and out to the outgoing buffers.

    Double-Sided Routing: Elegantly handles load-balancing for machines that have multiple incoming/outgoing belts. It automatically picks the closest belt for an arriving AGV or falls back to a secondary belt if the primary is full.

    The Bridge Connection: It acts as the physical trigger for the SimulationBridge. When a job hits the belt, it yells "I have work waiting!" (OnJobArrivedInQueue). When the timer finishes, it yells "I need this job routed away!" (OnMachineFinished).

2. The Physical Buffers (ConveyorBelt.cs)

This script acts as the machine's physical queueing system.

    Smooth Interpolation: Instead of instantly snapping jobs into a list, it mathematically calculates physical "slots" in 3D space and smoothly lerps the JobVisuals forward as space opens up ahead of them.

    Directional Logic: Supports both standard and reverse flow, allowing the same script to power both the incoming (flowing toward the machine) and outgoing (flowing away from the machine) belts, as well as the main factory I/O belts.

3. The Presentation Layer (MachineVisual.cs, MachineType.cs, MachineState.cs)

This handles all UI and aesthetics, ensuring the logic layer remains clean.

    Identity & Status: Maps the MachineType (Mill, Lathe, etc.) to specific colors so the factory floor is readable at a glance. Maps the MachineState to an overhead UI (Idle, Busy, Blocked) and updates a progress bar during operation.

    The "Flash": Contains a specific RecordDecisionPoint method that flashes the machine's indicator light white for a split second. This provides vital visual feedback to the user, showing exactly when and where the ML-Agent made a scheduling decision.

The Lifecycle Flow (Processing a Job)

For your combined diagram, this is the sequence of events that happens at a machine:

    Drop-off: An AGV arrives and calls ReceiveJob(). The job is placed on an incomingConveyor.

    Notification: The machine immediately fires OnJobArrivedInQueue to the SimulationBridge.

    Dispatch (Fired by Bridge): The bridge tells the machine to StartProcessing(). The machine pulls the job off the belt and state becomes Busy.

    Work: ProcessJobRoutine runs a timer, updating the MachineVisual progress bar.

    Ejection: The timer finishes. If the outgoing belt is full, the machine becomes Blocked. Otherwise, it pushes the job to the outgoingConveyor.

    Routing Trigger: The machine fires OnMachineFinished to the SimulationBridge, prompting the agent to find a new destination for the job.

    Pick-up: An AGV arrives and calls ReleaseFromOutgoing(), removing the job from the belt.

External Dependencies (Hooks for Final Diagram)

When connecting the Machines module to the rest of the architecture:

    Inbound Commands (Listens to):

        SimulationBridge: Tells it exactly which job to process and for how long.

        AGVPool / AGVController: Requests drop-offs (ReceiveJob) and pick-ups (ReleaseFromOutgoing).

    Outbound Events (Triggers):

        SimulationBridge: Relies entirely on the machine to tell it when jobs arrive in queues and when operations finish.

    Queried Data (Provides Info To):

        FactoryLayoutManager: Instantiates these and holds the master array of them.

        SimulationBridge: Queries the PhysicalQueue (how many items are on the incoming belts) to build the ML-Agents observation vector.