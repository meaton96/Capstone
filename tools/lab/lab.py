#!/usr/bin/env python3
"""The lab moved to its own repo, labwatch (~/Documents/Github/labwatch), on 2026-10-13. This forwards old
`python3 tools/lab/lab.py ...` calls to the `lab` command with this project's lab directory (tools/lab/state)."""
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("LAB_DIR", str(Path(__file__).resolve().parent / "state"))
exe = shutil.which("lab") or str(Path.home() / ".local/bin/lab")
os.execv(exe, [exe, *sys.argv[1:]])
