# Job Shop Scheduling Simulator

A real-time 3D visualisation of a job shop scheduling problem, where an AI agent learns to assign jobs to machines as efficiently as possible.

**[⬇ Download Latest Release](https://github.com/YOUR_USERNAME/YOUR_REPO/releases/latest)**

---

## Getting Started

1. Download and unzip the release
2. Run `JobShopSim.exe`
3. Select an instance from the dropdown and click **Start Sim**

No installation required.

---

## What You're Looking At

Once the simulation is running you'll see a factory floor with a set of machines laid out in a grid. Coloured tokens representing jobs move between them.

**Machines** sit on the floor as blocks and change colour based on their current state:

| Colour | Meaning |
|---|---|
| 🟢 Green | Idle, waiting for a job |
| 🟡 Yellow | Busy processing a job |

The label above each machine shows its ID, current status, and how many jobs are queued up waiting.

**Job tokens** are the small objects that move across the floor between machines. When a job is being processed it sits inside the machine and isn't visible — you'll see it again once it exits and heads to its next destination.

The **HUD** in the corner shows:
- **Sim Time** — how long the current run has been going (lower is better)
- **Last Rule** — the dispatching rule the agent most recently applied
- **Decisions** — how many scheduling choices have been made this episode
- **Jobs Done** — completed jobs out of the total

Use the **Speed** slider to slow down or speed up the simulation. Hit **Stop** at any time to return to the menu and choose a different instance.

---

## What is a Taillard Instance?

The scheduling problems in this simulator come from a well-known benchmark dataset published by Éric Taillard in 1993. Each instance describes a **Job Shop Problem**: a set of jobs, each of which must be processed on a fixed sequence of machines in a specific order. Every operation has a known duration and must wait for both the machine and the previous operation in the job to be free.

The goal is to minimise the **makespan** — the time from when the first job starts to when the last job finishes.

Taillard instances are widely used in operations research to compare scheduling algorithms. Each file encodes the machine visit order and processing times for every job, along with the best known optimal makespan so results can be benchmarked.

---

## Controls

| Control | Action |
|---|---|
| Speed slider | Slow down or speed up the simulation |
| Stop button | End the current run and return to the menu |
| Dropdown | Select which Taillard instance to load |
| Start Sim | Begin the simulation with the selected instance |