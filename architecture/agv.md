High-Level Overview

The AGV module is the "Logistics and Transport" layer of your simulation. It handles the physical movement of jobs between machines while respecting the rules set by the TrafficZoneManager.

It uses a decentralized control model: the AGVPool acts as the dispatcher (handing out jobs), but once an AGVController receives a job, it autonomously negotiates its path, reserves its own zones, and executes its own state machine.
Component Breakdown
1. The Dispatcher (AGVPool.cs)

This script manages the fleet of AGVs and acts as the entry point for transport requests.

    Initialization: Spawns a configured number of AGVs at the parking area when the factory is built.

    Queuing System: When SimulationBridge requests a transport, the pool checks for an available AGV. If none are free (i.e., all are State != Idle), the request is saved in a Queue<DispatchRequest>.

    Event-Driven Drain: When any AGV finishes a job, it fires a callback (OnAnyAGVBecameIdle) back to the pool, automatically triggering the pool to dequeue the next request and hand it to the newly freed AGV.

    Staggered Dispatch: Includes a clever coroutine (TryDispatchStaggered) to prevent CPU spikes when 20 jobs are injected into the system at time t=0, staggering the heavy A*/BFS pathfinding calculations across multiple frames.

2. The Autonomous Agent (AGVController.cs)

This script drives a single AGV unit. It strictly ignores Unity's built-in physics/NavMesh movement and instead implements a custom "Turn-Then-Move" kinematic model to ensure precise alignment with factory lanes and docking stations.

    The State Machine: Transitions linearly through a defined lifecycle:
    Idle ➔ NavigatingToPickup ➔ Aligning ➔ ExecutingPickup ➔ NavigatingToDropoff ➔ Aligning ➔ ExecutingDropoff ➔ Idle.

    Zone Navigation & Deadlock Prevention: * When moving along its planned route, it attempts to reserve the next zone in its path using trafficMgr.TryReserve().

        If the zone is full, the AGV enters a special WaitingForZone state and pauses its movement, retrying the reservation every physics tick.

        As it enters a new zone, it tells the Traffic Manager to release its reservation on the previous zone, freeing up space behind it.

    Dock Handshakes: Instead of just getting "close enough" to a machine, the AGV resolves the specific DockPoint for the target machine. It navigates to an ApproachPosition, aligns its rotation to face the conveyor, drives to the HandshakePosition, and waits for a specific handshakeTimer to simulate physical loading/unloading time.

    Visual Parenting: Physically grabs the JobVisual and parents it to its carryPos transform during the transit phase.

The Lifecycle Flow (A Delivery Mission)

For your combined diagram, this is how a transport request is executed:

    Request: SimulationBridge determines a job needs to move and calls AGVPool.TryDispatch().

    Assignment: AGVPool finds an idle AGVController and calls Dispatch().

    Pathfinding: The AGV queries the TrafficZoneManager for the DockPoint of the target machine and calculates a BFS route of zones.

    Transit: The AGV executes its custom movement, reserving zones ahead and releasing zones behind as it travels to the pickup dock.

    Pickup (Handshake): AGV aligns, waits for the timer, and pulls the job off the machine's outgoingConveyor, setting the job to InTransit.

    Delivery Transit: AGV calculates a new route to the destination machine and drives there.

    Drop-off (Handshake): AGV aligns, waits, and drops the job onto the destination machine's incomingConveyor.

    Completion: AGV notifies JobManager the transit is complete, sets its state to Idle, and triggers the AGVPool callback to check for more work.

External Dependencies (Hooks for Final Diagram)

When connecting the AGV module to the rest of the architecture:

    Inbound Commands (Listens to):

        SimulationBridge: Demands that jobs be moved.

    Outbound Commands (Controls):

        JobManager: The AGV tells the JobManager when a job physically enters/exits the InTransit state.

        PhysicalMachine: The AGV tells the machines to ReleaseFromOutgoing() and ReceiveJob() during handshakes.

    Queried Data (Relies On):

        TrafficZoneManager: The AGV is completely subservient to this. It relies on it for GetRoute(), TryReserve(), Release(), and FindDockForMachine().