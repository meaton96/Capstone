# Job Shop Scheduling Simulator

A real-time 3D logistics simulation of the Job Shop Scheduling Problem (JSP), where an AI agent manages a fleet of Automated Guided Vehicles (AGVs) and machine queues to minimize factory makespan.

**[⬇ Download Latest Release](https://github.com/meaton96/Capstone/releases/latest)**

---

## Getting Started

1. Download and unzip the release.
2. Run `Capstone.exe`.
3. Select a **Taillard Instance** from the dropdown.
4. Click **Start Sim** to begin the episode.

---

## The Factory Floor

The simulation features a dynamic environment governed by physical constraints. Unlike standard scheduling models, jobs in this simulator must be physically transported between workstations.

### 1. The Machines & Conveyors
Machines are the core of the factory. Each workstation is equipped with **Conveyor Belts** for incoming and outgoing buffers. 

* **Incoming Belts:** Hold jobs delivered by AGVs waiting for processing.
* **Outgoing Belts:** Hold finished jobs waiting for AGV pickup.
* **Double-Sided Logic:** High-capacity machines (center rows) feature conveyors on both sides to prevent bottlenecks.

**Machine States (Visual Cues):**
| Colour | Meaning |
|---|---|
| 🟢 **Green** | **Idle** — Waiting for a job to arrive in the queue. |
| 🟡 **Yellow** | **Busy** — Currently processing a job (progress bar visible). |
| 🟠 **Orange** | **Blocked** — Processing complete, but the outgoing belt is full. |
| 🔴 **Red** | **Failed** — Machine is broken and requires maintenance. |
| 🔵 **Blue** | **Repair** — Maintenance is currently being performed. |

### 2. The AGV Fleet (Automated Guided Vehicles)
The fleet handles all logistics. These robots follow a **"Turn-Then-Move"** model, meaning they rotate in place to align with their path before driving forward, mimicking real-world industrial AGVs.

* **Pathfinding:** AGVs use BFS (Breadth-First Search) to navigate a directed graph of traffic zones.
* **Deadlock Prevention:** The floor is divided into reservable **Traffic Zones**. An AGV will only enter a zone (like a narrow aisle) if it has successfully reserved space, ensuring head-on collisions are impossible.
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
1.  **Entry:** Jobs spawn at the **Incoming Belt** at the top-left of the factory.
2.  **Transport:** An AGV picks up the job and navigates to the first machine in its sequence.
3.  **Queuing:** The job sits on the machine's incoming conveyor.
4.  **Processing:** The machine pulls the job inside (the token becomes invisible during this phase).
5.  **Pickup:** Once finished, the job moves to the outgoing conveyor to wait for the next AGV.
6.  **Exit:** After the final operation, an AGV delivers the job to the **Outgoing Belt** at the bottom-right.

---

## User Interface & Controls

### The HUD
- **Sim Time:** The total elapsed time for the current schedule.
- **Last Rule:** The Dispatching Rule (PDR) currently being used by the agent.
- **Decisions:** Cumulative count of scheduling choices made.
- **Jobs Done:** Progress counter for completed vs. total jobs.

### Controls
| Control | Action |
|---|---|
| **Speed Slider** | Adjust the time scale (from slow-motion to high-speed). |
| **Stop Button** | Immediately terminates the episode and returns to the menu. |
| **Gizmos (Dev Only)** | Visualizes the AGV paths (Green for pickup, Orange for dropoff) and Zone occupancy (Red blocks). |

---

## What is a Taillard Instance?
The scheduling problems in this simulator utilize the **Taillard (1993)** benchmark dataset. Each instance specifies:
1.  **Job Count & Machine Count** (e.g., 20 jobs on 15 machines).
2.  **Processing Order:** The specific sequence of machines each job must visit.
3.  **Durations:** Exactly how long each operation takes.

The goal of the AI agent is to minimize the **Makespan**—the total time taken to move every job through its entire sequence and out of the factory.