# Capstone: working notes for Claude

## Experiment naming

Name every new experiment, results folder and cluster run `<purpose>-<what>[-<variant>]`, lowercase, hyphen-separated.

| purpose | use for |
|---|---|
| `rq1` ... `rq4` | data for that research question (RQ2 action space, RQ3 conditions/generalization, RQ4 disruptions) |
| `agv` | AGV physical study: collisions, layouts, fleet size, reservation protocols |
| `twin` | Unity vs event-based DES twin (sim-to-real) |
| `base` | fixed-rule (PDR) baselines that policies are scored against |
| `train` | PPO training runs (cluster folders become `results/<name>_s<seed>`) |
| `eval` | evaluating a trained checkpoint |
| `dev` | smoke tests, build regressions, tuning that is not reported |

- **Variant suffixes:** `-l<λ>` breakdown scale, `-p<p>-m<m>` flexibility, `-agv<N>` fleet size, `-lay<X>` layout,
  `-hold` / `-rel` reservation protocol, `-reg` reproduces an earlier result on a new build, `-check` quick sanity
  run, `-qfifo` reruns FIFO cells on a strict-FIFO build (FIFO by time in queue since 2026-10-02). Seeds go in the
  folder (`s<seed>`), not the experiment name.
- **Never use a bare capital letter or letter+digit as a name** (G1, D4): layouts are A-O, so these read as layouts.
- **Where data goes:** `evaluate.py` runs to `results/<name>/`; batch-runner sweeps to `<player dir>/Results/<name>/`.
- **Outdated results (moved 2026-10-03):** everything produced before the C# audit-fix build lives in
  `results/outdated/<name>/` and its findings docs in `docs/experiments/outdated/` (README in each says why). Do not
  compare new runs against them as baselines. `results/` keeps only `scripts/`, live/planned experiments and new runs.
  Player sweeps likewise moved to `<player dir>/Results/outdated/<name>/` (`linux_server_des/Results/G1` is a
  symlink kept for the rq2-twin-due floor input).
- **Old names are not renamed** (AGVR, G1, FLEX_p*, eval_pdr12_heads, ...): scripts, findings docs and memory point at
  those folders (under `results/outdated/` since 2026-10-03). Refer to them by the new name with the old one in parentheses, e.g. "twin-gap (G1)". The registry in
  `docs/WIP/EXPERIMENT_REGISTRY.md` maps old to new.

## Tracking experiments: the lab (`tools/lab/`, since 2026-10-04)

Experiments are tracked and run by the lab service, not by hand-edited docs. Spec (data model, service loop, CLI, web API, new-session workflow): `tools/lab/README.md`. `tools/lab/lab.py serve` (keep it
running; web page http://127.0.0.1:8765) holds every experiment in `tools/lab/state/lab.db` with a status
(future -> queued -> running -> done / failed / cancelled), watches running ones (local process + done-glob, or Slurm
job ids over the shared SSH connection), pulls and analyzes on finish, notifies Discord, starts the next queued
experiment when worker slots free up, promotes futures whose `waits_on` experiments are done, fires `check_at` /
ETA timers, and regenerates `docs/WIP/LAB_STATUS.md`.

- **Plan an experiment:** `lab.py add <name> --status future --waits-on <names> --waits-note "<what else>" --note ...`
  (a free-text `waits_note` blocks auto-queueing until cleared with `lab.py set <name> --waits-note ""`).
- **Queue / start:** give it a launch command, done-check, worker slots and analysis command, then `--status queued`
  (the service starts it) or `lab.py start <name>`. A launcher must skip finished outputs, so rerunning resumes.
  Never start a long run outside the lab: add it so it is tracked.
- **Check on experiments:** `lab.py status`; verify against reality as before (`pgrep -af`, `squeue --me`) when in
  doubt, and fix the record with `lab.py set`.
- **Results to Claude:** run `lab.py wait` in the background (one notification per finished / failed / timer event,
  with the analysis output); `lab.py inbox` shows unread events.
- **On finish:** check the results are written up (findings doc or results `README.md`); record it with
  `lab.py set <name> --writeup <path>`. Still keep `EXPERIMENT_REGISTRY.md` (old -> new names, history) and the
  prose sections of `FUTURE_EXPERIMENTS.md` (known issues, writeups owed, open decisions) current by hand.
- `RUNNING_EXPERIMENTS.md` is retired (see LAB_STATUS.md). `docs/` is gitignored; the lab database is not in git.
- The service cannot pass Duo: if the cluster connection drops it notifies; log in once (`ssh rit-research`).

## Measuring headroom and targets

- Measure RL headroom, training targets and rule rankings on the quantity the reward optimizes (episode `return` /
  time in system including unfinished jobs). Total flow of *exited* jobs is censored (finishing fewer jobs lowers
  it); report it only next to jobs finished, never as the target.
- Measure switching headroom with `env/switch_oracle.py` (simulated), not by recombining fixed-rule segments, and
  check it on the regime a run will train on before starting the run.
- Background: the 8.1% "headroom" of 09-27 was this error; the real figure was ~1%
  (`docs/experiments/HEADROOM_METRIC_CORRECTION_1002.md`).
