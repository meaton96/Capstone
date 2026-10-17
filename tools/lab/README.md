# Lab (moved)

The experiment tracker now lives in its own repo, **labwatch** (`~/Documents/Github/labwatch`, installed as
`lab` in `~/.local/bin` with `uv tool install -e`). Its README covers the CLI, the page, the service, the Claude
workflow and the remote API.

Capstone's lab directory is still `tools/lab/state/` (`lab.db`, `token`, `logs/`, `serve.log`, and
`config.json` with the Capstone settings: 20 local slots, `rit-research` for Slurm, `docs/WIP/LAB_STATUS.md`).
The repo-root symlink `.lab -> tools/lab/state` lets `lab` find it from anywhere in the repo. `lab.py` here only
forwards old `python3 tools/lab/lab.py ...` calls to `lab`.

Service: systemd user unit `capstone-lab` (`systemctl --user status|restart capstone-lab`); the unit from
before the move is saved as `state/capstone-lab.service.pre-labwatch.bak`.
