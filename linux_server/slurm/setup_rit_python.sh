#!/bin/bash
# One-time setup ON THE RIT LOGIN NODE (needs internet; compute nodes may not have it):
#   cd ~/capstone/linux_server && bash slurm/setup_rit_python.sh
# Creates ~/capstone/.venv with Python 3.10.12 (mlagents-envs 1.1.0 needs 3.10.x) and the training packages,
# then runs the Python unit tests as a check. Uses an existing python3.10 if one is on PATH; otherwise installs
# uv (https://astral.sh/uv, a single binary in ~/.local/bin) to fetch a standalone Python 3.10.12.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"      # ~/capstone
VENV="$REPO/.venv"

UV="$(command -v uv 2>/dev/null || true)"
[[ -z "$UV" && -x "$HOME/.local/bin/uv" ]] && UV="$HOME/.local/bin/uv"

if [[ -x "$VENV/bin/python" ]]; then
    echo "[setup] reusing $VENV ($("$VENV/bin/python" --version)); delete it to rebuild."
elif command -v python3.10 >/dev/null 2>&1; then
    echo "[setup] using $(command -v python3.10) ($(python3.10 --version))"
    python3.10 -m venv "$VENV"
else
    if [[ -z "$UV" ]]; then
        echo "[setup] no python3.10 on PATH; installing uv to ~/.local/bin"
        curl -LsSf https://astral.sh/uv/install.sh | sh
        UV="$HOME/.local/bin/uv"
    fi
    "$UV" venv --python 3.10.12 "$VENV"
fi

# Install from the hash-locked env/requirements-rit.lock (regenerate with env/lock_requirements.sh): every package,
# torch included, must match a recorded SHA-256 or the install fails. uv needs unsafe-best-match to see both the PyPI
# and PyTorch CPU indexes; the hashes, not index order, decide what is installed.
LOCK="$REPO/env/requirements-rit.lock"
if [[ -n "$UV" ]]; then
    "$UV" pip install --python "$VENV/bin/python" --require-hashes --no-deps \
        --index-strategy unsafe-best-match -r "$LOCK"
else
    "$VENV/bin/python" -m pip install --require-hashes --no-deps -r "$LOCK"
fi

echo "[setup] $("$VENV/bin/python" --version); running unit tests"
cd "$REPO/env"
"$VENV/bin/python" -m pytest tests -q
