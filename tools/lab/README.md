# Lab: experiment tracker and runner (spec, 2026-10-05)

One place where experiments are planned, queued, run, watched and finished, for the user (web page) and for Claude
(CLI). It replaced hand-maintained `docs/WIP/RUNNING_EXPERIMENTS.md` on 2026-10-04. Code: `lab.py` (stdlib only),
`index.html`, tests `test_lab.py`. State: `state/` (gitignored): `lab.db` (SQLite), `token`, `logs/<name>.log`,
`serve.log`.

## How to reach it

| who | how |
|---|---|
| User | web page **http://127.0.0.1:8765** (live clock, Running / Queued / Future / Failed / Done, progress bars, ETA countdown, event feed with analysis output, add form, start / hold / cancel / requeue); Discord messages; `docs/WIP/LAB_STATUS.md` |
| Claude | CLI `python3 tools/lab/lab.py ...` (below); `lab.py wait` in a **background** Bash call for notifications |
| Service | systemd user unit `capstone-lab` (`~/.config/systemd/user/capstone-lab.service`): `systemctl --user status|restart capstone-lab`. Starts at login; `KillMode=process`, so a restart never kills the experiments it launched. Not lingering: it stops at logout unless `loginctl enable-linger ubuntupc` (the user's call). |

**Why a CLI and not an MCP server (decided 10-05):** the valuable call is waiting for a run to finish, and an MCP tool
call blocks the session until it returns, while a background `lab.py wait` gives one notification and keeps the
session free. Reads and edits are one short CLI call each and work from any session; an MCP server would add a
registration to keep in sync for no new capability.

## Data model (`experiments` table)

| field | meaning |
|---|---|
| `name` | `<purpose>-<what>[-<variant>]` (CLAUDE.md naming); primary key |
| `status` | `future` -> `queued` -> `running` -> `done` / `failed` / `cancelled` |
| `location` | `local` (a process on this machine) or `cluster` (Slurm jobs over SSH) |
| `launch_cmd`, `cwd` | shell command run from the repo root (`cwd` relative); must resume (skip finished outputs) when rerun |
| `workers` | local worker slots it occupies (one Unity player ~ one slot; 0 for GPU-only jobs); `max_local_workers` = 20 |
| `proc_pattern` | `pgrep -f` pattern for runs the lab did not start (imported); else the lab's own child pid is watched |
| `done_glob`, `done_count` | done when the glob (repo-relative) matches at least `done_count` files |
| `slurm_jobs`, `pull_cmd` | cluster: job ids (parsed from "Submitted batch job N" in this launch's output); command run when they leave `squeue` |
| `analysis_cmd` | run on finish; its output goes into the event, Discord and Claude's inbox |
| `waits_on`, `waits_note` | experiment names that must be `done`; free text that blocks auto-queueing until cleared |
| `eta`, `check_at` | ISO times: overdue alert while running past `eta`; one-off reminder at `check_at` |
| `priority`, `auto_start` | lower starts first; 0 = never auto-start |
| `results_path`, `writeup`, `notes` | pointers and prose |

`events` table: every state change; kinds `finished`, `failed`, `timer`, `overdue`, `promoted`, `cluster` are unread
(inbox) until `wait` / `inbox` reads them.

## Service loop (`lab.py serve`, every 30 s)

1. **Running, local:** done when the process is gone and the done-check holds; failed when it is gone without it.
2. **Running, cluster:** when none of `slurm_jobs` is in `squeue --me`: run `pull_cmd`, apply the done-check. SSH goes
   through the shared ControlMaster (`~/.ssh/cm-%C`, alias `rit-research`, BatchMode); it keeps the master alive while
   cluster runs exist. **It cannot pass Duo:** if the cluster is unreachable it notifies once; the user logs in
   (`ssh rit-research`) and it resumes.
3. **On finish:** `analysis_cmd`, event, Discord (`env/notify.py`, webhook `~/.capstone_webhook`).
4. **Promote:** a `future` whose `waits_on` are all `done` and whose `waits_note` is empty becomes `queued` (+ Discord).
5. **Start:** `queued` with a launch command, in priority order, while local slots allow (cluster ones at once);
   output to `state/logs/<name>.log`, own process group.
6. **Timers:** `check_at` and overdue fire once each.
7. Regenerate `docs/WIP/LAB_STATUS.md` when anything changed.

## CLI

```bash
python3 tools/lab/lab.py status [--all]          # everything by status (done/cancelled with --all)
python3 tools/lab/lab.py show NAME [--events 5] [--tail 15]   # record, progress, events, log tail
python3 tools/lab/lab.py add NAME --status future|queued [--location local|cluster] [--launch CMD] [--workers N] \
    [--done-glob GLOB --done-count N] [--analysis CMD] [--pull CMD] [--waits-on A,B] [--waits-note TEXT] \
    [--eta ISO] [--check-at ISO] [--priority N] [--auto-start 0|1] [--results PATH] [--writeup PATH] [--note TEXT]
python3 tools/lab/lab.py set NAME [same flags] [--status S]   # edit any field
python3 tools/lab/lab.py start|cancel|requeue|done NAME       # cancel = SIGTERM to the process group / scancel
python3 tools/lab/lab.py wait [--timeout HOURS]  # blocks until an unread inbox event; prints and marks it read
python3 tools/lab/lab.py inbox [--keep]          # unread events now
```

## Web API (localhost only)

`GET /` page; `GET /api/state` all experiments (+ progress), last 60 events, slot use. `POST /api/add`, `/api/set`,
`/api/action` (`start`, `cancel`, `queued`, `future`, `done`, `read`) need header `X-Lab-Token` = `state/token` and a
localhost `Host` (other web pages in the browser cannot drive it).

## Claude workflow (new session)

1. `lab.py status`, then `lab.py show <name>` for anything running; verify with reality when in doubt
   (`pgrep -af '[c]apstone.x86_64'`, `squeue --me`).
2. Start `lab.py wait` with `run_in_background`; when it fires, read the event, write up (results README +
   `lab.py set <name> --writeup <path>`), and start the wait again.
3. New work: `add` with a launch command that resumes, a done-check, slots and an analysis command; `--status queued`
   to run it, `future` with `waits_on` / `waits_note` to park it. Never start a long run outside the lab.
4. Cluster launches: a local wrapper script that ssh's to the cluster and runs the submit script (example:
   `results/train-due-twin/submit.sh`); pull commands must expand `$HOME` (no single-quoted `$HOME`).

## Known limitations

- Pull/analysis commands run synchronously inside a tick (long pulls delay other checks).
- A run launched before a service restart is followed by pid / pattern, not by its exit status.
- No per-GPU accounting: GPU jobs use `workers 0`.
