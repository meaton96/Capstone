# Job Shop Scheduling Simulator

A real-time 3D logistics simulation of the Flexible Job Shop Scheduling Problem (FJSP), where an AI agent manages a fleet of Automated Guided Vehicles (AGVs) and machine queues to minimize factory makespan.

**[⬇ Download Latest Release](https://github.com/meaton96/Capstone/releases/latest)**

*Note: Builds are available for both Windows (PC) and macOS.*

---

## Getting Started

1. Download and unzip the release relevant to your operating system.
2. Run the executable file (`Capstone.exe` for Windows or the `.app` file for macOS).
3. Configure the simulation parameters via the main menu (problem size, AGV count, etc.).
4. Click **Start Sim** to begin the episode.

---

## The Factory Floor

The simulation features a dynamic environment governed by physical constraints. Unlike standard scheduling models, jobs in this simulator must be physically transported between workstations.

### 1. The Machines & Conveyors
The simulator has transitioned to a **Flexible Job Shop (FJSP)** model. To distinguish between different machine capabilities, workstations are now **color-coded by type**.

Each machine features an **Indicator Light** that communicates its current operational state using the following status scheme:

| Indicator Color | Meaning |
|---|---|
| Green | **Idle** — Waiting for a job to arrive in the queue. |
| Yellow | **Busy** — Currently processing a job (progress bar visible). |
| Orange | **Blocked** — Processing complete, but the outgoing belt is full. |
| Red | **Failed** — Machine is broken and requires maintenance. |
| Blue | **Repair** — Maintenance is currently being performed. |

**Workstation Components:**
* **Incoming Belts:** Hold jobs delivered by AGVs waiting for processing.
* **Outgoing Belts:** Hold finished jobs waiting for AGV pickup.
* **Double-Sided Logic:** High-capacity machines (center rows) feature conveyors on both sides to prevent bottlenecks.

### 2. The AGV Fleet (Automated Guided Vehicles)
The fleet handles all logistics. These robots follow a **"Turn-Then-Move"** model, meaning they rotate in place to align with their path before driving forward, mimicking real-world industrial AGVs.

* **Pathfinding:** AGVs use BFS (Breadth-First Search) to navigate a directed graph of traffic zones.
* **Deadlock Prevention:** The floor is divided into reservable **Traffic Zones**. An AGV will only enter a zone (like a narrow aisle) if it has successfully reserved space, ensuring head-on collisions are avoided.
* **The Handshake:** When an AGV arrives at a dock, it must align its orientation to the conveyor and wait for a brief **Handshake Duration** to simulate the physical transfer of the job.

### 3. Traffic Flow
To maintain efficiency, the factory uses a **One-Way Traffic** system:
* **Row Aisles:** Narrow paths between machines with alternating flow (East/West).
* **Spine Aisles:** Wide peripheral lanes at the top and bottom for high-speed travel.
* **Connector Aisles:** Vertical lanes (North/South) allowing AGVs to switch between rows.
* **Floor Markers:** Yellow arrows indicate the legal direction of travel for each lane.

---

## Job Lifecycle
Each **Job Token** follows a strict path from entry to exit:
1. **Entry:** Jobs spawn at the **Incoming Belt** at the top-left of the factory.
2. **Transport:** An AGV picks up the job and navigates to the first machine in its sequence.
3. **Queuing:** The job sits on the machine's incoming conveyor.
4. **Processing:** The machine pulls the job inside (the token becomes invisible during this phase).
5. **Pickup:** Once finished, the job moves to the outgoing conveyor to wait for the next AGV.
6. **Exit:** After the final operation, an AGV delivers the job to the **Outgoing Belt** at the bottom-right.

---

## User Interface & Controls

### The HUD
- **Sim Time:** The total elapsed time for the current schedule.
- **Last Rule:** The Dispatching Rule (PDR) currently being used by the agent.
- **Decisions:** Cumulative count of scheduling choices made.
- **Jobs Done:** Progress counter for completed vs. total jobs.

### Controls
| Key | Action |
|---|---|
| **F** | Toggle Free Camera mode. |
| **W, A, S, D** | Standard Fly Camera movement. |
| **E / R** | Fly Camera vertical elevation (Up / Down). |
| **C** | Toggle AGV Follow Camera. |
| **Left / Right Arrows** | Cycle through the AGV fleet while in Follow Camera mode. |
| **Speed Slider** | Adjust the time scale (from slow-motion to high-speed). |
| **Stop Button** | Immediately terminates the episode and returns to the menu. |
| **Gizmos (Dev)** | Visualizes AGV paths (Green for pickup, Orange for dropoff) and Zone occupancy. |

---

### Results
Baseline results are stored as CSV in the /results folder. Included are results for the randomly generated job sets (Guassian selection) and Brandimarte job sets mk01-mk15 across all 8 PDRs and random selection. Also included is a sensitivity sweep varying the number of AGVs and Machines across different job and operation counts.

## Testing the Code Base

The following sections outline the testing procedures ranging from high-level simulation verification to environment unit testing.

### A. Standalone Simulation Testing
The simplest method to verify the system is by downloading the standalone build. This allows for manual verification of the simulation logic, agent behavior, and physical constraints.
* Download the standalone build from the releases section.
* Modify configurations via the UI (problem size, number of AGVs, etc.) to test scalability.
* Use the **Fly Camera (F)** and **Follow Camera (C)** controls to inspect AGV pathfinding and machine state transitions.

### B. Python Environment Testing
To verify the ML-Agents environment and the Python-side logic:
1. Clone the repository.
2. Navigate to the environment folder: `cd repo/env`.
3. Build and start the containers: `docker-compose up --build` (this will take several minutes)
4. Execute the test suite within the container:
   `docker exec -it unity-ml-agents-test pytest`



### C. Bridge Connectivity Testing (Smoke Test)

This confirms the "Round Trip" data flow: Python starts the Unity engine, observations are received, actions are sent back, and rewards are processed.

Run the following commands:
```bash
    docker exec -it unity-ml-agents-test chmod +x linux_server/capstone.x86_64

    docker exec -it unity-ml-agents-test mlagents-learn /code/smoke_test.yaml --env=/code/linux_server/capstone.x86_64 --run-id=smoke01 --no-graphics
```
Success Criteria: The test is successful if the terminal displays the Unity logo and begins logging Step counts and Mean Reward values. Steps print every ~1 minute, rewards print every ~20k steps (4 minutes) 

Control+C once satisfied (entire training loop takes ~1.5hours)

(might be longer on slower cpu)


