#!/bin/bash
# dev-size-floors (10-19): export the floor-size study's layout-D floors from the current player (linux_server/),
# 3 at a time for 30-200 machines, then 1000 machines alone (long timeout), then compare.py. Exports that exist are
# skipped (export_floor.sh), so a rerun resumes.
set -u
cd "$(dirname "$0")"
H=$(pwd)
printf '%s\n' 30 50 75 100 150 200 | xargs -P 3 -I{} bash export_floor.sh {} 3600 >> launch.out 2>&1
bash export_floor.sh 1000 14400 >> launch.out 2>&1
../../.venv/bin/python compare.py > compare.out 2>&1
date -Is > DONE
