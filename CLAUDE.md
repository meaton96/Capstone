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
- **Where data goes:** `evaluate.py` runs to `results/<name>/`; batch-runner sweeps to `linux_server/Results/<name>/`.
- **One player build (since 2026-10-11):** `linux_server/` is the current Unity player, locally and on the cluster
  (`~/capstone/linux_server`, glibc-patched; also the Slurm working dir for logs). Rebuild into it, not into a new
  `linux_server_*` folder; record the build in its `BUILD_NOTES.md`. `linux_server_archive/linux_server_v4` (H15 heads)
  is kept only to rerun H15-era checkpoints; `linux_server_archive/notes/` holds the notes of deleted builds. Player
  sweeps of the deleted builds were merged into `linux_server/Results/` (suffix `__des` / `__dev` on name clashes).
- **Outdated results (moved 2026-10-03):** everything produced before the C# audit-fix build lives in
  `results/outdated/<name>/` and its findings docs in `docs/experiments/outdated/` (README in each says why). Do not
  compare new runs against them as baselines. `results/` keeps only `scripts/`, live/planned experiments and new runs.
  Player sweeps likewise moved to `linux_server/Results/outdated/<name>/` (`linux_server/Results/G1` is a
  symlink kept for the rq2-twin-due floor input).
- **Index and archive (since 2026-10-11):** every results folder has a row in `results/INDEX.tsv` (rendered to
  `results/INDEX.md` by `results/scripts/make_results_index.py`; add a row for each new experiment). Finished smoke
  tests, checks and superseded runs move with `git mv` to `results/archive/<name>` (fix references in docs, READMEs,
  memory and the lab db). Thesis-facing table of contents: `docs/experiments/THESIS_RESULTS_INDEX.md`; update it when a
  result lands or is superseded.
- **Old names are not renamed** (AGVR, G1, FLEX_p*, eval_pdr12_heads, ...): scripts, findings docs and memory point at
  those folders (under `results/outdated/` since 2026-10-03). Refer to them by the new name with the old one in parentheses, e.g. "twin-gap (G1)". The registry in
  `docs/WIP/EXPERIMENT_REGISTRY.md` maps old to new.

## Tracking experiments: the lab (labwatch, since 2026-10-13; `tools/lab/` 10-04 to 10-13)

Experiments are tracked and run by the lab service, not by hand-edited docs. The code is its own repo, labwatch
(`~/Documents/Github/labwatch`; its README is the spec: CLI, fields, service loop, page, remote API), installed as
`lab` (`~/.local/bin`). Capstone's lab directory is `tools/lab/state/` (`lab.db`, `token`, `logs/`, `config.json`:
20 local slots, Slurm via `rit-research`, writes `docs/WIP/LAB_STATUS.md`); the repo-root symlink `.lab` points at
it, so `lab ...` works from anywhere in the repo, and `python3 tools/lab/lab.py ...` forwards to `lab`. Service:
systemd user unit `capstone-lab` (page http://127.0.0.1:8765). It starts queued experiments when slots free up,
watches them (local process + done-glob / exit code, or Slurm ids), pulls and analyzes on finish, promotes futures
whose `waits_on` are done, fires `check_at` / ETA timers.

- **Watch the lab** (user says so, or after queueing something): `lab status`, then `lab wait` with
  `run_in_background`; when it exits handle each event (write up finished runs, diagnose failures) and start
  `lab wait` again. Exit code 3 = service down: tell the user.
- **Plan an experiment:** `lab add <name> --status future --waits-on <names> --waits-note "<what else>" --note ...`
  (a free-text `waits_note` blocks auto-queueing until cleared with `lab set <name> --waits-note ""`).
- **Queue / start:** give it a launch command, done-check, worker slots and analysis command, then `--status queued`
  (the service starts it at once if slots are free) or `lab start <name>`. A launcher must skip finished outputs, so
  rerunning resumes. Never start a long run outside the lab (nohup, `&`, a background Bash call): it cannot be
  watched or cancelled, and runs started from Claude's Bash tool die when the session's tasks are stopped.
- **Cancel:** `lab cancel <name>` stops the whole process tree, including Unity players in their own session (the
  old tools/lab cancel missed them), or scancels the Slurm jobs.
- **Check on experiments:** `lab status` / `lab show <name>`; verify against reality (`pgrep -af '[c]apstone.x86_64'`,
  `squeue --me`) when in doubt, and fix the record with `lab set`.
- **On finish:** check the results are written up (findings doc or results `README.md`); record it with
  `lab set <name> --writeup <path>`. Still keep `EXPERIMENT_REGISTRY.md` (old -> new names, history) and the
  prose sections of `FUTURE_EXPERIMENTS.md` (known issues, writeups owed, open decisions) current by hand.
- `RUNNING_EXPERIMENTS.md` is retired (see LAB_STATUS.md). `docs/` is gitignored; the lab database is not in git.
- The service cannot pass Duo: if the cluster connection drops it sends a `cluster` event; log in once
  (`ssh rit-research`).

## Measuring headroom and targets

- Measure RL headroom, training targets and rule rankings on the quantity the reward optimizes (episode `return` /
  time in system including unfinished jobs). Total flow of *exited* jobs is censored (finishing fewer jobs lowers
  it); report it only next to jobs finished, never as the target.
- Measure switching headroom with `env/switch_oracle.py` (simulated), not by recombining fixed-rule segments, and
  check it on the regime a run will train on before starting the run.
- Background: the 8.1% "headroom" of 09-27 was this error; the real figure was ~1%
  (`docs/experiments/HEADROOM_METRIC_CORRECTION_1002.md`).
