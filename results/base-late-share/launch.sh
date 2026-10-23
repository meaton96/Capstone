#!/bin/bash
# base-late-share (10-22): local lab run, one task per setting (7), all in parallel. Finished settings are skipped.
set -uo pipefail
cd "$(dirname "$0")/../.."
H=results/base-late-share
mkdir -p $H/logs
for s in $(.venv/bin/python $H/run.py list); do
    .venv/bin/python $H/run.py $s > $H/logs/$s.log 2>&1 &
done
wait
echo "done: $(ls $H/out/*.json 2>/dev/null | wc -l) / 7 ($(date -Is))"
