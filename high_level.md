Environment & Infrastructure

These components handle the generation of the physical factory and the logical traffic network.

    [x] FactoryLayoutManager.cs

        Purpose: The physical architect. It calculates grid spacing, instantiates machine prefabs, builds boundary walls for NavMesh carving, and manages I/O belt placement.

    [x] TrafficZoneManager.cs

        Purpose: The logical traffic controller. It divides the floor into a directed graph of reservable zones. It manages one-way flow constraints and provides BFS pathfinding data for the AGV fleet.

2. Logistics & Fleet Management

These components handle the transport of materials between workstations.

    [x] AGVPool.cs

        Purpose: Fleet manager. Handles the instantiation of the AGV fleet and maintains a request queue. If all AGVs are busy, it buffers dispatch requests and assigns them as vehicles become idle.

    [x] AGVController.cs

        Purpose: Individual vehicle intelligence. Implements the "Turn-Then-Move" physical model, handles zone-ahead reservations to prevent deadlocks, and performs "handshakes" with conveyors for job pickup/dropoff.

3. Machine Workstations

These components represent the processing units where jobs are transformed.

    [x] PhysicalMachine.cs

        Purpose: The central workstation anchor. It coordinates between the logic of the machine, the input/output conveyors, and the visual feedback. It handles double-sided load balancing.

    [x] ConveyorBelt.cs

        Purpose: Buffer management. A linear system that moves job visuals toward an output end. It tracks capacity and "packs" items as space opens up.

    [x] MachineVisual.cs

        Purpose: Feedback Layer. Handles mesh color changes (Idle/Busy/Blocked/Failed) and manages the overhead World-Space UI (labels and progress bars).

4. Job & Entity Tracking

These components manage the lifecycle and data of the scheduling units (the Taillard jobs).

    [x] JobManager.cs

        Purpose: The authoritative database. It initializes trackers from JSON/Taillard data, records timestamps for wait/transit times, and drives the state transitions for every job in the episode.

    [x] JobVisual.cs

        Purpose: The physical token. A lightweight script attached to job prefabs that handles smooth interpolation (lerping) between targets and manages visual state colors.

High-Level Data Flow

    Initialization: FactoryLayoutManager builds the floor → TrafficZoneManager builds the graph → JobManager creates job entities.

    Logistics Request: A machine or the incoming belt requests a pickup → AGVPool assigns an AGVController.

    Navigation: AGVController queries TrafficZoneManager for a route → traverses zones via TryReserve().

    Processing: AGV drops job at PhysicalMachine → ConveyorBelt queues it → PhysicalMachine runs processing timer.

    Completion: JobManager updates stats → Job is routed to next machine or the factory exit.